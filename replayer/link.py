"""One NDTP connection per device: handshake, realtime packets in order, reconnect with backoff.

A :class:`UnitLink` owns a bounded queue of packets handed to it by the scheduler and a task that keeps a TCP
connection to the receiver:

1. connect, retrying with exponential backoff and jitter while the receiver is down;
2. send the handshake ``NPH_SGC_CONN_REQUEST``;
3. send realtime frames in queue order. ``requestId`` is one counter per device shared with the handshake
   (as the official emulator does) and keeps growing across reconnects, so a receiver can see gaps.

The receiver never answers, so a watcher reads the socket only to notice EOF or a reset at once. While the
connection is down, packets keep queueing by default (``buffer``: like a terminal's black box that sends the
backlog after reconnecting) and go out right after the new handshake. The queue is bounded and drops its
oldest packets when full, so a long outage or a slow receiver costs bounded memory.

What "sent" means: NDTP as used here has no acknowledgements, so a frame counts as sent once it has been
handed to the kernel (written to the socket), not when the receiver has read it. The link keeps that gap
small:

* the transport's high-water mark is zero, so ``drain()`` returns only when the frame has left the asyncio
  buffer; a frame is never counted while it still sits in user space, and a write timeout (hung or half-open
  receiver) aborts the connection and sends the unflushed frame again after reconnecting — no duplicate from a
  late flush of the old transport;
* after the handshake the link pauses for ``settle_s`` (0.1 s) before the first frame: a receiver that rejects
  connections (accepts and closes at once) is noticed before any frame is written into it, and the replay
  does not treat it as up;
* the connection state (EOF, reset) is checked before every frame and again after one pass of the event loop
  that follows its write; if the close shows up right after a write, that frame stays in flight and goes out
  again on the next connection (at least once; the predictor drops duplicates).

Frames written before a close or reset reaches the replayer are lost and still counted as sent: without
acknowledgements nobody can tell. At the realtime pace (a few frames per second per device) a clean kill of
the receiver is noticed before the next frame, so at most a frame written at that very moment is lost. While
a backlog is being sent after a reconnect, a receiver that dies mid-burst takes the frames written in the
last moments with it (up to a couple of dozen per device). A hung receiver (accepts but does not read)
swallows whatever fits into the kernel buffers of a connection: at the realtime pace that takes minutes, and
if it recovers it reads them all. Only when a write stays unflushed for ``write_timeout_s`` is the connection
reset and reopened; the reset discards those buffers (hundreds to thousands of frames per connection,
depending on buffer sizes), and every new connection to a still hung receiver fills another set, all
counted as sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from replayer.clock import Clock, SystemClock
from replayer.stats import DROP_DISCONNECTED, DROP_OVERFLOW, DROP_STOPPED, ReplayStats
from shared.ndtp import NavRecord, encode_handshake, encode_realtime

log = logging.getLogger(__name__)

_READ_CHUNK = 4096
_MAX_REQUEST_ID = 0xFFFFFFFF
_CLOSE_GRACE_S = 1.0


@dataclass(frozen=True, slots=True)
class LinkSettings:
    """Connection parameters shared by all devices of a replay.

    Attributes:
        host: Receiver host.
        port: Receiver NDTP port.
        max_queue: Packets kept per device while the connection is down or slow; the oldest are dropped.
        buffer_while_down: Queue packets while disconnected (``buffer``) or drop them (``skip``).
        backoff_initial_s: First reconnect delay.
        backoff_max_s: Largest reconnect delay.
        connect_timeout_s: TCP connect timeout.
        write_timeout_s: A write that cannot be flushed for this long means a dead receiver.
        stable_after_s: A connection that lived this long resets the backoff.
        settle_s: Pause after the handshake before the first frame. A receiver that rejects a connection
            closes it right after accepting it; the pause lets that close arrive first, so no frame is written
            into a connection that is already going away.
    """

    host: str
    port: int
    max_queue: int = 5000
    buffer_while_down: bool = True
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 5.0
    connect_timeout_s: float = 5.0
    write_timeout_s: float = 10.0
    stable_after_s: float = 5.0
    settle_s: float = 0.1


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _gone(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, lost: asyncio.Event) -> bool:
    """Whether the receiver has closed or reset the connection.

    ``at_eof()``/``is_closing()`` see a FIN or a reset as soon as the transport has processed it, one pass of
    the event loop before the watcher task sets ``lost``.
    """
    return lost.is_set() or reader.at_eof() or writer.is_closing()


async def _close_writer(writer: asyncio.StreamWriter, *, abort: bool = False) -> None:
    """Close a connection: gracefully (flush, FIN) or at once (RST, the buffer is discarded).

    A graceful close that does not finish within :data:`_CLOSE_GRACE_S` is aborted too, so a hung receiver
    never leaves a transport lingering in the event loop.
    """
    if abort:
        writer.transport.abort()
    else:
        writer.close()
    try:
        with contextlib.suppress(OSError, TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), _CLOSE_GRACE_S)
    finally:
        writer.transport.abort()  # a no-op once closed; drops a transport that could not flush in time


class UnitLink:
    """NDTP connection of one device.

    Args:
        unit_id: Device id (``NPL.peerAddress`` and handshake ``peerAddress``).
        settings: Connection parameters.
        stats: Shared counters.
        clock: Clock used to measure how late packets are sent.
        on_connected: Called with ``unit_id`` after every successful handshake.

    Attributes:
        connected: Whether a connection is open.
        connections: Connections opened so far (``connections - 1`` reconnects).
        sent: Realtime frames written to the socket (handed to the kernel; delivery is not acknowledged).
        dropped: Packets dropped (overflow, skip policy, stop).
        last_error: Last connection error, for status output.
    """

    def __init__(
        self,
        unit_id: int,
        settings: LinkSettings,
        *,
        stats: ReplayStats | None = None,
        clock: Clock | None = None,
        on_connected: Callable[[int], None] | None = None,
    ) -> None:
        self.unit_id = unit_id
        self.settings = settings
        self.stats = stats if stats is not None else ReplayStats()
        self.clock = clock or SystemClock()
        self._on_connected = on_connected
        self._queue: deque[tuple[NavRecord, float]] = deque()
        self._inflight: tuple[NavRecord, float] | None = None
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._closing = False
        self._drain = False
        self._task: asyncio.Task[None] | None = None
        self._request_id = 0
        self.connected = False
        self.connections = 0
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None

    # ---- scheduler side -----------------------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Packets waiting to be written (including one being retried)."""
        return len(self._queue) + (self._inflight is not None)

    def oldest_due(self) -> float | None:
        """Monotonic due time of the oldest unsent packet, or ``None`` when nothing is pending."""
        if self._inflight is not None:
            return self._inflight[1]
        return self._queue[0][1] if self._queue else None

    def enqueue(self, nav: NavRecord, due: float) -> None:
        """Hand a packet to the connection.

        Args:
            nav: Navigation record to send.
            due: Monotonic time the packet was scheduled for (to measure lag).
        """
        if self._closing:
            self._count_drop(DROP_STOPPED)
            return
        if not self.connected and not self.settings.buffer_while_down:
            self._count_drop(DROP_DISCONNECTED)
            return
        if len(self._queue) >= self.settings.max_queue:
            self._queue.popleft()
            self._count_drop(DROP_OVERFLOW)
        self._queue.append((nav, due))
        self._wake.set()

    def start(self) -> None:
        """Start the connection task (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"ndtp-unit-{self.unit_id}")

    async def close(self, drain_timeout_s: float = 0.0) -> None:
        """Stop the connection gracefully.

        Args:
            drain_timeout_s: If positive, keep sending queued packets (reconnecting if needed) for up to this
                long before closing. Whatever is still queued afterwards is counted as dropped.
        """
        self._closing = True
        self._drain = drain_timeout_s > 0
        if not self._drain:
            self._stop.set()
        self._wake.set()
        task = self._task
        if task is not None and not task.done():
            # without draining the task still gets a moment to finish its write and close the socket itself
            await asyncio.wait({task}, timeout=drain_timeout_s if self._drain else _CLOSE_GRACE_S)
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
        left = self.pending
        if left:
            self._queue.clear()
            self._inflight = None
            self._count_drop(DROP_STOPPED, left)

    # ---- connection task ----------------------------------------------------------------------------

    def _count_drop(self, reason: str, count: int = 1) -> None:
        self.dropped += count
        self.stats.drop(reason, count)

    def _next_request_id(self) -> int:
        self._request_id = self._request_id % _MAX_REQUEST_ID + 1
        return self._request_id

    def _should_exit(self) -> bool:
        return self._closing and (not self._drain or self.pending == 0)

    async def _pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), seconds * random.uniform(0.5, 1.0))

    async def _run(self) -> None:
        s = self.settings
        backoff = s.backoff_initial_s
        while not self._should_exit():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(s.host, s.port), s.connect_timeout_s
                )
            except OSError as exc:
                self.stats.connect_failures += 1
                self.last_error = _describe(exc)
                log.debug(
                    "unit %d: connect to %s:%d failed: %s", self.unit_id, s.host, s.port, self.last_error
                )
                await self._pause(backoff)
                backoff = min(backoff * 2, s.backoff_max_s)
                continue
            opened = time.monotonic()
            clean = False  # a lost connection, a write timeout or a cancellation is aborted (RST)
            try:
                await self._session(reader, writer)
                clean = True
            except OSError as exc:
                self.last_error = _describe(exc)
            finally:
                await _close_writer(writer, abort=not clean)
            if self._should_exit():
                break
            self.stats.disconnects += 1
            log.info("unit %d: connection lost (%s), reconnecting", self.unit_id, self.last_error)
            if time.monotonic() - opened >= s.stable_after_s:
                backoff = s.backoff_initial_s
            await self._pause(backoff)
            backoff = min(backoff * 2, s.backoff_max_s)

    async def _session(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lost = asyncio.Event()
        watcher = asyncio.create_task(self._watch(reader, lost))
        # zero high-water mark: drain() waits until the asyncio buffer is empty (the frame is in the kernel)
        writer.transport.set_write_buffer_limits(high=0)
        self.connected = True
        self.connections += 1
        self.stats.connects += 1
        self.stats.connections_active += 1
        if self.connections > 1:
            self.stats.reconnects += 1
            log.info("unit %d: reconnected (connection #%d)", self.unit_id, self.connections)
        try:
            await self._write(writer, encode_handshake(self.unit_id, self._next_request_id()))
            self.stats.handshakes += 1
            if self.settings.settle_s > 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(lost.wait(), self.settings.settle_s)
            if _gone(reader, writer, lost):
                raise ConnectionResetError("closed by the receiver right after accepting")
            if self._on_connected is not None:
                self._on_connected(self.unit_id)
            while True:
                if self._closing and not self._drain:
                    return
                if _gone(reader, writer, lost):
                    raise ConnectionResetError("closed by the receiver")
                if self._inflight is None:
                    if not self._queue:
                        if self._closing:
                            return
                        self._wake.clear()
                        await self._wake.wait()
                        continue
                    self._inflight = self._queue.popleft()
                nav, due = self._inflight
                await self._write(writer, encode_realtime(self.unit_id, self._next_request_id(), nav))
                # one pass of the event loop lets the watcher report EOF before the frame is counted; a frame
                # written after the receiver closed stays in flight and is sent again on the next connection
                await asyncio.sleep(0)
                if _gone(reader, writer, lost):
                    raise ConnectionResetError("closed by the receiver")
                self._inflight = None
                self.sent += 1
                self.stats.packets_sent += 1
                self.stats.last_send_lag_s = max(0.0, self.clock.monotonic() - due)
        finally:
            watcher.cancel()
            self.connected = False
            self.stats.connections_active -= 1

    async def _watch(self, reader: asyncio.StreamReader, lost: asyncio.Event) -> None:
        """Read (and ignore) whatever the receiver sends; flag the connection as lost on EOF or error."""
        with contextlib.suppress(OSError):
            while await reader.read(_READ_CHUNK):
                pass
        lost.set()
        self._wake.set()

    async def _write(self, writer: asyncio.StreamWriter, frame: bytes) -> None:
        """Write one frame and wait until it has left the asyncio buffer.

        Raises:
            ConnectionResetError: If the connection is closing or was reset.
            TimeoutError: If the receiver has not taken the frame within ``write_timeout_s``.
        """
        if writer.is_closing():
            raise ConnectionResetError("transport is closing")
        writer.write(frame)
        timeout = self.settings.write_timeout_s
        try:
            await asyncio.wait_for(writer.drain(), timeout)
        except TimeoutError:
            raise TimeoutError(f"receiver did not take a frame within {timeout:g} s") from None
