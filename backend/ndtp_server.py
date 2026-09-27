"""Asyncio NDTP TCP server: one connection per device, frames go to the state store and packet listeners.

The server never answers the device (the emulator does not need a handshake reply). Any error inside one
connection (garbage, bad CRC, unknown cell, abrupt disconnect) is counted and never propagates beyond it.

Anti-DoS limits: a connection is closed after ``max_crc_errors`` CRC mismatches or ``max_garbage_bytes``
discarded bytes since its last valid frame, when it sends no valid frame within ``first_frame_timeout_s`` of
accept, or stays silent for ``read_timeout_s``; beyond ``max_connections`` new connections are closed at once.

Extension point: pass extra ``listeners`` (e.g. the future stop-passage detector or feature builder); each is
called synchronously with every decoded :class:`~shared.ndtp.RealtimePacket` and the resolved unit id.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from backend.state import StateStore
from shared.ndtp import Frame, FrameDecoder, NdtpError, RealtimePacket, decode_handshake, decode_realtime

log = logging.getLogger(__name__)

PacketListener = Callable[[int, RealtimePacket], None]
"""Callback ``(unit_id, packet)`` invoked for every decoded realtime packet."""

_READ_CHUNK = 64 * 1024


class RateMeter:
    """Events-per-second over a sliding window of one-second buckets.

    Args:
        window_s: Window length in seconds.
        clock: Monotonic clock in seconds.
    """

    def __init__(self, window_s: int = 10, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self.clock = clock
        self._buckets: deque[list[int]] = deque()

    def _trim(self, now_s: int) -> None:
        while self._buckets and self._buckets[0][0] <= now_s - self.window_s:
            self._buckets.popleft()

    def add(self, count: int = 1) -> None:
        """Record ``count`` events now."""
        now_s = int(self.clock())
        if self._buckets and self._buckets[-1][0] == now_s:
            self._buckets[-1][1] += count
        else:
            self._buckets.append([now_s, count])
        self._trim(now_s)

    def rate(self) -> float:
        """Average events per second over the window."""
        self._trim(int(self.clock()))
        return sum(c for _, c in self._buckets) / self.window_s


@dataclass(slots=True)
class IngestStats:
    """Cumulative ingest counters (exported by ``/api/ingest/stats`` and ``/metrics``)."""

    connections_total: int = 0
    connections_active: int = 0
    disconnects: int = 0
    read_timeouts: int = 0
    bytes_received: int = 0
    bytes_discarded: int = 0
    frames: int = 0
    handshakes: int = 0
    realtime_packets: int = 0
    nav_records: int = 0
    other_frames: int = 0
    crc_errors: int = 0
    bad_headers: int = 0
    parse_errors: int = 0
    unknown_cells: int = 0
    listener_errors: int = 0
    abusive_disconnects: int = 0
    first_frame_timeouts: int = 0
    rejected_connections: int = 0
    packets_rate: RateMeter = field(default_factory=RateMeter)


@dataclass(slots=True)
class ConnectionInfo:
    """One open device connection.

    Attributes:
        conn_id: Sequential connection id.
        peer: Remote ``host:port``.
        connected_at: Server time the connection was accepted.
        unit_id: Device id once known (handshake or first realtime packet).
        frames: Valid frames received on this connection.
        last_frame_at: Server time of the last valid frame.
        garbage: Bytes discarded since the last valid frame (or accept).
    """

    conn_id: int
    peer: str
    connected_at: datetime
    unit_id: int | None = None
    frames: int = 0
    last_frame_at: datetime | None = None
    garbage: int = 0


class NdtpServer:
    """NDTP ingest server.

    Args:
        store: State store updated with every Nav00.
        host: Bind address.
        port: TCP port (0 picks a free one; see :attr:`port`).
        max_data_size: Largest accepted NPL ``dataSize``.
        max_crc_errors: CRC mismatches tolerated since the last valid frame before the connection
            is closed (protects the event loop from floods of plausible false headers).
        read_timeout_s: Close a connection silent for this long.
        first_frame_timeout_s: Close a connection that sent no valid frame this long after accept.
        max_garbage_bytes: Bytes discarded since the last valid frame (or accept) before the connection is
            closed as abusive (a stream without the ``0x7E7E`` signature never costs a CRC).
        max_connections: Open connections at most; further ones are closed right after accept.
        listeners: Extra packet consumers (extension point for the detector / features).
        time_map: Maps a fix time before anything sees it (a realtime stream onto the plan day,
            :func:`backend.clock.to_plan_day`); ``None`` — as sent.
    """

    def __init__(
        self,
        store: StateStore,
        host: str = "0.0.0.0",
        port: int = 9201,
        *,
        max_data_size: int = 8192,
        max_crc_errors: int = 32,
        read_timeout_s: float = 300.0,
        first_frame_timeout_s: float = 10.0,
        max_garbage_bytes: int = 65_536,
        max_connections: int = 1024,
        listeners: Iterable[PacketListener] = (),
        time_map: Callable[[datetime], datetime] | None = None,
    ) -> None:
        self.store = store
        self.time_map = time_map
        self._mapped_logged = False
        self.host = host
        self._port = port
        self.max_data_size = max_data_size
        self.max_crc_errors = max_crc_errors
        self.read_timeout_s = read_timeout_s
        self.first_frame_timeout_s = first_frame_timeout_s
        self.max_garbage_bytes = max_garbage_bytes
        self.max_connections = max_connections
        self.listeners: list[PacketListener] = list(listeners)
        self.stats = IngestStats()
        self.connections: dict[int, ConnectionInfo] = {}
        self.started_at: datetime | None = None
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_conn_id = 0

    @property
    def port(self) -> int:
        """Actual listening port (resolved after :meth:`start` when created with port 0)."""
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self._port

    @property
    def listening(self) -> bool:
        """Whether the server accepts connections."""
        return self._server is not None and self._server.is_serving()

    async def start(self) -> None:
        """Bind and start accepting connections."""
        self._server = await asyncio.start_server(self._handle, self.host, self._port)
        self.started_at = datetime.now(UTC)
        log.info("NDTP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop accepting, close all connections and wait for handlers to finish."""
        if self._server is None:
            return
        self._server.close()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._server.wait_closed(), 5)
        self._server = None
        log.info("NDTP server stopped")

    # ---- connection handling ---------------------------------------------------------------------

    async def _reject(self, writer: asyncio.StreamWriter) -> None:
        self.stats.rejected_connections += 1
        if self.stats.rejected_connections % 100 == 1:  # a flood must not flood the log too
            log.warning(
                "NDTP connection limit %d reached: connection from %s closed (%d rejected so far)",
                self.max_connections,
                writer.get_extra_info("peername"),
                self.stats.rejected_connections,
            )
        writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer.wait_closed(), 1)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self.connections) >= self.max_connections:
            await self._reject(writer)
            return
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        peername = writer.get_extra_info("peername")
        peer = f"{peername[0]}:{peername[1]}" if isinstance(peername, tuple) else str(peername)
        self._next_conn_id += 1
        conn = ConnectionInfo(conn_id=self._next_conn_id, peer=peer, connected_at=datetime.now(UTC))
        self.connections[conn.conn_id] = conn
        self.stats.connections_total += 1
        self.stats.connections_active += 1
        decoder = FrameDecoder(self.max_data_size, self.max_crc_errors)
        first_frame_by = time.monotonic() + self.first_frame_timeout_s
        log.info("NDTP connection #%d from %s", conn.conn_id, peer)
        try:
            while True:
                timeout = self.read_timeout_s
                if not conn.frames:  # until the first valid frame, only first_frame_timeout_s in total
                    timeout = min(timeout, first_frame_by - time.monotonic())
                try:
                    if timeout <= 0:
                        raise TimeoutError
                    data = await asyncio.wait_for(reader.read(_READ_CHUNK), timeout)
                except TimeoutError:
                    if conn.frames:
                        self.stats.read_timeouts += 1
                        log.warning(
                            "NDTP connection #%d idle %.0f s, closing", conn.conn_id, self.read_timeout_s
                        )
                    else:
                        self.stats.first_frame_timeouts += 1
                        log.warning(
                            "NDTP connection #%d from %s: no valid frame in %.0f s, closing",
                            conn.conn_id,
                            conn.peer,
                            self.first_frame_timeout_s,
                        )
                    break
                if not data:
                    break
                self._feed(conn, decoder, data)
                if decoder.abusive or conn.garbage > self.max_garbage_bytes:
                    self.stats.abusive_disconnects += 1
                    log.warning(
                        "NDTP connection #%d from %s: %s, closing",
                        conn.conn_id,
                        conn.peer,
                        "too many CRC errors" if decoder.abusive else f"{conn.garbage} bytes of garbage",
                    )
                    break
        except (ConnectionError, OSError) as exc:
            log.info("NDTP connection #%d dropped: %s", conn.conn_id, exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("NDTP connection #%d failed", conn.conn_id)
        finally:
            self.stats.connections_active -= 1
            self.stats.disconnects += 1
            self.connections.pop(conn.conn_id, None)
            if conn.unit_id is not None:
                self.store.on_disconnect(conn.unit_id)
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), 1)
            if task is not None:
                self._tasks.discard(task)
            log.info("NDTP connection #%d (unit %s) closed", conn.conn_id, conn.unit_id)

    def _feed(self, conn: ConnectionInfo, decoder: FrameDecoder, data: bytes) -> None:
        stats = self.stats
        stats.bytes_received += len(data)
        crc_before, bad_before, disc_before = decoder.crc_errors, decoder.bad_headers, decoder.bytes_discarded
        frames = decoder.feed(data)
        stats.crc_errors += decoder.crc_errors - crc_before
        stats.bad_headers += decoder.bad_headers - bad_before
        discarded = decoder.bytes_discarded - disc_before
        stats.bytes_discarded += discarded
        # garbage since the last valid frame (what follows the last frame of a chunk is forgiven)
        conn.garbage = 0 if frames else conn.garbage + discarded
        for frame in frames:
            stats.frames += 1
            conn.frames += 1
            conn.last_frame_at = datetime.now(UTC)
            try:
                self._on_frame(conn, frame)
            except (NdtpError, struct.error) as exc:
                stats.parse_errors += 1
                log.debug("NDTP connection #%d: bad frame: %s", conn.conn_id, exc)

    def _bind_unit(self, conn: ConnectionInfo, unit_id: int) -> None:
        if conn.unit_id == unit_id:
            return
        if conn.unit_id is not None:
            self.store.on_disconnect(conn.unit_id)
        conn.unit_id = unit_id
        self.store.on_connect(unit_id)

    def _on_frame(self, conn: ConnectionInfo, frame: Frame) -> None:
        stats = self.stats
        if frame.is_handshake:
            hs = decode_handshake(frame)
            stats.handshakes += 1
            self._bind_unit(conn, hs.peer_address or frame.unit_id)
            return
        if not frame.is_realtime:
            stats.other_frames += 1
            return
        if conn.unit_id is None:  # realtime without a handshake: trust NPL.peerAddress
            self._bind_unit(conn, frame.unit_id)
        unit_id = conn.unit_id if conn.unit_id is not None else frame.unit_id
        packet = decode_realtime(frame)
        if self.time_map is not None and packet.nav is not None:
            ts = self.time_map(packet.nav.timestamp)
            if ts != packet.nav.timestamp:
                if not self._mapped_logged:
                    self._mapped_logged = True
                    log.info(
                        "realtime stream: fix times go onto the plan day (%s → %s)", packet.nav.timestamp, ts
                    )
                packet = replace(packet, nav=replace(packet.nav, timestamp=ts))
        stats.realtime_packets += 1
        stats.packets_rate.add()
        if packet.unknown_cell_type is not None:
            stats.unknown_cells += 1
        if packet.truncated:
            stats.parse_errors += 1
        if packet.nav is not None:
            stats.nav_records += 1
            self.store.on_nav(unit_id, packet.nav, packet.irma[0] if packet.irma else None)
        else:
            self.store.on_packet_without_nav(unit_id)
        for listener in self.listeners:
            try:
                listener(unit_id, packet)
            except Exception:
                stats.listener_errors += 1
                log.exception("NDTP packet listener %r failed", listener)
