"""Redis bus: telemetry stream, hot vehicle state, pub/sub channels and the consumer group.

Key layout (all keys are prefixed with ``foresight:``):

* ``foresight:telemetry`` — Stream of navigation events (:class:`TelemetryEvent`), ``MAXLEN ~ 200k``;
  consumer group ``predictors``;
* ``foresight:vehicle:{unit_id}`` — Hash with the latest state of a vehicle (TTL ~600 s);
* ``foresight:vehicles`` — ZSET index of vehicles, score = last packet time (Unix seconds);
* ``foresight:ingest:stats`` — Hash with the latest ingest counters;
* ``foresight:stream_time`` — String, the stream clock (ISO 8601);
* ``foresight:stream_epoch`` — String, the epoch of the stream clock (changes when the clock jumps);
* ``foresight:stream_devices`` — String, devices active on the stream clock when it was last saved;
* channel ``foresight:vehicles`` — JSON deltas of changed vehicles (for WebSocket clients);
* channel ``foresight:alerts`` — alerts of the predictor,
  ``{"type": "alert", "stream_time", "alert": AlertOut}``;
* channel ``foresight:incidents`` — ``{"type": "incident", "stream_time", "action": open|update|close,
  "incident": IncidentOut}``;
* channel ``foresight:predictions`` — ``{"type": "prediction_closed", "stream_time", "prediction":
  PredictionOut}`` when a forecast gets its fact (the messages have the shape of the WebSocket messages of the
  API contract, so the api can relay them as they are);
* ``foresight:forecast`` — Hash ``tr_id → JSON`` of the latest forecast state of every vehicle (risk, nearest
  forecast, incident), rewritten every tick; ``foresight:forecast:status`` — its stream time and ML state.

Producer (:class:`TelemetryPublisher`, ingest) never blocks the NDTP server: events go to a bounded
in-memory buffer and are written in pipelined batches; while Redis is down (or refuses writes: OOM, MISCONF,
READONLY...) the buffer keeps the newest events (evictions are counted) and is written out after recovery.
Delivery is at-least-once: after an ambiguous failure a batch may be written twice, consumers drop exact
repeats. Every event carries the ingest's stream clock and its epoch (see :mod:`backend.clock`), so the
predictor follows the same clock.

Consumer (:class:`StreamConsumer`, predictor) reads the stream in a consumer group, ACKs processed batches,
re-reads its own pending entries after a restart or reconnect and claims entries of dead consumers with
``XAUTOCLAIM``; consumers that stay idle without pending entries are removed from the group. At start it can
read back the already processed history (``XREVRANGE``) to refill in-memory state.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import (
    ClusterDownError,
    OutOfMemoryError,
    ReadOnlyError,
    RedisError,
    ResponseError,
    TryAgainError,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from backend.ndtp_server import RateMeter
from backend.runtime import Backoff, BoundedBuffer, DependencyStatus
from backend.state import StateStore, VehicleState
from shared.ndtp import IrmaRecord, RealtimePacket

log = logging.getLogger(__name__)

PREFIX = "foresight"
STREAM_TELEMETRY = f"{PREFIX}:telemetry"
VEHICLE_INDEX = f"{PREFIX}:vehicles"
INGEST_STATS_KEY = f"{PREFIX}:ingest:stats"
STREAM_TIME_KEY = f"{PREFIX}:stream_time"
STREAM_EPOCH_KEY = f"{PREFIX}:stream_epoch"
STREAM_DEVICES_KEY = f"{PREFIX}:stream_devices"
CHANNEL_VEHICLES = f"{PREFIX}:vehicles"
CHANNEL_ALERTS = f"{PREFIX}:alerts"
CHANNEL_INCIDENTS = f"{PREFIX}:incidents"
CHANNEL_PREDICTIONS = f"{PREFIX}:predictions"
FORECAST_KEY = f"{PREFIX}:forecast"
FORECAST_STATUS_KEY = f"{PREFIX}:forecast:status"
ROUTES_KEY = f"{PREFIX}:routes"
"""Route network of the predictor's plan (``GET /api/routes``), JSON, written by the predictor at start."""

REDIS_ERRORS: tuple[type[BaseException], ...] = (
    RedisConnectionError,
    RedisTimeoutError,
    OSError,
    TimeoutError,
)
"""Errors that mean "Redis is unreachable": retried with backoff, the service degrades."""

_TRANSIENT_REPLIES: tuple[type[BaseException], ...] = (
    OutOfMemoryError,
    ReadOnlyError,
    TryAgainError,
    ClusterDownError,
)
# error codes redis-py leaves in the message of a plain ResponseError
_TRANSIENT_CODES = frozenset(
    {"OOM", "MISCONF", "READONLY", "TRYAGAIN", "LOADING", "MASTERDOWN", "NOREPLICAS"}
)


def is_transient(exc: BaseException) -> bool:
    """Whether a Redis error is worth retrying: Redis is unreachable or refuses writes for now.

    Besides connection errors this covers error replies that clear up when Redis recovers: ``OOM``
    (``maxmemory``), ``MISCONF`` (failed RDB/AOF write), ``READONLY`` (a replica during failover),
    ``TRYAGAIN``, ``LOADING``, ``MASTERDOWN``, ``CLUSTERDOWN``, ``NOREPLICAS``. Other error replies
    (``WRONGTYPE``, syntax) are permanent.
    """
    if isinstance(exc, REDIS_ERRORS + _TRANSIENT_REPLIES):
        return True
    return isinstance(exc, ResponseError) and str(exc).split(" ", 1)[0] in _TRANSIENT_CODES


RedisFactory = Callable[[], aioredis.Redis]
"""Creates a Redis client (called inside the running event loop)."""


def vehicle_key(unit_id: int | str) -> str:
    """Key of the hot-state hash of one vehicle."""
    return f"{PREFIX}:vehicle:{unit_id}"


def redis_factory(url: str, timeout_s: float = 5.0) -> RedisFactory:
    """Build a factory of Redis clients with short timeouts and a single immediate retry.

    The retry covers a stale pooled connection after a Redis restart; longer outages are handled by the
    services with :class:`~backend.runtime.Backoff`.

    Args:
        url: Redis URL.
        timeout_s: Socket connect/read timeout (must exceed the XREADGROUP BLOCK time).

    Returns:
        A zero-argument factory.
    """

    def make() -> aioredis.Redis:
        return aioredis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=timeout_s,
            socket_connect_timeout=timeout_s,
            retry=Retry(NoBackoff(), 1),
            health_check_interval=30,
        )

    return make


# --------------------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------------------


def _num(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _bool(value: bool) -> str:
    return "1" if value else "0"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "t", "yes")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromtimestamp(float(value), UTC)


def doors_json(irma: IrmaRecord) -> str:
    """Serialise an Irma04 record (door state) to compact JSON."""
    return json.dumps(
        {
            "zone": irma.zone,
            "odometer": irma.odometer,
            "door_in": list(irma.door_in),
            "door_out": list(irma.door_out),
            "door_present": list(irma.door_present),
            "door_closed": list(irma.door_closed),
        },
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One navigation event on the ``foresight:telemetry`` stream.

    Attributes:
        unit_id: NDTP device id.
        tr_id: Vehicle id from the schedule, ``None`` for an unknown device.
        ts: Fix time, Unix seconds (UTC).
        lon: Longitude, degrees.
        lat: Latitude, degrees.
        speed: Average speed, km/h.
        course: Course, degrees.
        valid: GPS fix validity.
        doors: Door state as JSON (see :func:`doors_json`), if the packet carried Irma04.
        received_at: Ingest wall-clock time, Unix seconds (end-to-end latency).
        epoch: Epoch of the ingest's stream clock (0: unknown, e.g. the fix time was garbage before any
            valid one).
        clock: The ingest's stream clock after this event, Unix seconds (``None`` if not set yet).
    """

    unit_id: int
    tr_id: int | None
    ts: float
    lon: float
    lat: float
    speed: int = 0
    course: int = 0
    valid: bool = True
    doors: str | None = None
    received_at: float = 0.0
    epoch: int = 0
    clock: float | None = None

    def to_fields(self) -> dict[str, str]:
        """Stream entry fields (all strings; ``tr_id`` is empty for unknown devices)."""
        fields = {
            "unit_id": str(self.unit_id),
            "tr_id": "" if self.tr_id is None else str(self.tr_id),
            "ts": _num(self.ts),
            "lon": f"{self.lon:.7f}",
            "lat": f"{self.lat:.7f}",
            "speed": str(self.speed),
            "course": str(self.course),
            "valid": _bool(self.valid),
            "rx": f"{self.received_at:.3f}",
        }
        if self.doors is not None:
            fields["doors"] = self.doors
        if self.epoch:
            fields["epoch"] = str(self.epoch)
        if self.clock is not None:
            fields["clock"] = _num(self.clock)
        return fields

    @classmethod
    def from_fields(cls, fields: Mapping[str, str]) -> TelemetryEvent:
        """Parse stream entry fields.

        Raises:
            KeyError: A mandatory field is missing.
            ValueError: A field is malformed.
        """
        tr_id = fields.get("tr_id", "")
        clock = fields.get("clock")
        return cls(
            unit_id=int(fields["unit_id"]),
            tr_id=int(tr_id) if tr_id else None,
            ts=float(fields["ts"]),
            lon=float(fields["lon"]),
            lat=float(fields["lat"]),
            speed=int(fields.get("speed") or 0),
            course=int(fields.get("course") or 0),
            valid=_parse_bool(fields.get("valid", "1")),
            doors=fields.get("doors") or None,
            received_at=float(fields.get("rx") or 0.0),
            epoch=int(fields.get("epoch") or 0),
            clock=float(clock) if clock else None,
        )


def vehicle_fields(vehicle: VehicleState, tr_id: int | None, updated_ms: int) -> dict[str, str]:
    """Hot-state hash fields of one vehicle (see :class:`VehicleRecord` for the reverse)."""
    fields = {
        "unit_id": str(vehicle.unit_id),
        "tr_id": "" if tr_id is None else str(tr_id),
        "connected": _bool(vehicle.connections > 0),
        "packets": str(vehicle.packets),
        "reconnects": str(vehicle.reconnects),
        "updated_ms": str(updated_ms),
    }
    if vehicle.first_seen is not None:
        fields["first_seen"] = f"{vehicle.first_seen.timestamp():.3f}"
    if vehicle.last_packet_at is not None:
        fields["last_packet_at"] = f"{vehicle.last_packet_at.timestamp():.3f}"
    nav = vehicle.nav
    if nav is not None:
        fields.update(
            ts=_num(nav.timestamp.timestamp()),
            lon=f"{nav.lon:.7f}",
            lat=f"{nav.lat:.7f}",
            valid=_bool(nav.valid),
            speed=str(nav.speed_avg),
            speed_max=str(nav.speed_max),
            course=str(nav.course),
            altitude=str(nav.altitude),
            nsat=str(nav.nsat),
        )
        if vehicle.last_nav_at is not None:
            fields["received_at"] = f"{vehicle.last_nav_at.timestamp():.3f}"
    good = vehicle.valid_nav
    if good is not None:  # the last valid position (the map keeps it while the fix is invalid)
        fields.update(
            pos_lon=f"{good.lon:.7f}", pos_lat=f"{good.lat:.7f}", pos_ts=_num(good.timestamp.timestamp())
        )
    if vehicle.doors is not None:
        fields["doors"] = doors_json(vehicle.doors)
    return fields


def _opt_int(fields: Mapping[str, str], name: str) -> int | None:
    value = fields.get(name)
    return int(value) if value not in (None, "") else None


def _opt_float(fields: Mapping[str, str], name: str) -> float | None:
    value = fields.get(name)
    return float(value) if value not in (None, "") else None


@dataclass(slots=True)
class VehicleRecord:
    """Vehicle state as read from the hot-state hash (api side).

    Attributes:
        unit_id: NDTP device id.
        tr_id: Vehicle id from the schedule, ``None`` if unknown.
        connected: Whether a TCP connection of the device is open on the ingest.
        packets: Realtime packets received.
        reconnects: Reconnections (handshakes after the first).
        last_packet_at: Ingest time of the last frame.
        event_time: Fix time of the last position.
        received_at: Ingest time of the last position.
        lat: Latitude.
        lon: Longitude.
        valid: GPS fix validity.
        speed_kmh: Average speed.
        speed_max_kmh: Maximum speed.
        course_deg: Course.
        altitude_m: Altitude.
        satellites: Number of satellites.
        doors: Door state (dict with the :class:`~backend.schemas.DoorsOut` fields).
        pos_lat: Latitude of the last valid fix (``None`` — no valid fix yet).
        pos_lon: Longitude of the last valid fix.
        pos_time: Fix time of the last valid fix.
        updated_ms: Ingest wall-clock time of the write, ms (orders deltas against snapshots).
        version: Local cache version of the last change.
    """

    unit_id: int
    tr_id: int | None = None
    connected: bool = False
    packets: int = 0
    reconnects: int = 0
    last_packet_at: datetime | None = None
    event_time: datetime | None = None
    received_at: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    valid: bool | None = None
    speed_kmh: int | None = None
    speed_max_kmh: int | None = None
    course_deg: int | None = None
    altitude_m: int | None = None
    satellites: int | None = None
    doors: dict[str, Any] | None = None
    pos_lat: float | None = None
    pos_lon: float | None = None
    pos_time: datetime | None = None
    updated_ms: int = 0
    version: int = 0

    @classmethod
    def from_fields(cls, fields: Mapping[str, str]) -> VehicleRecord:
        """Parse hot-state hash fields.

        Raises:
            KeyError: ``unit_id`` is missing.
            ValueError: A field is malformed.
        """
        valid = fields.get("valid")
        doors = fields.get("doors")
        return cls(
            unit_id=int(fields["unit_id"]),
            tr_id=_opt_int(fields, "tr_id"),
            connected=_parse_bool(fields.get("connected", "0")),
            packets=int(fields.get("packets") or 0),
            reconnects=int(fields.get("reconnects") or 0),
            last_packet_at=_parse_ts(fields.get("last_packet_at")),
            event_time=_parse_ts(fields.get("ts")),
            received_at=_parse_ts(fields.get("received_at")),
            lat=_opt_float(fields, "lat"),
            lon=_opt_float(fields, "lon"),
            valid=_parse_bool(valid) if valid not in (None, "") else None,
            speed_kmh=_opt_int(fields, "speed"),
            speed_max_kmh=_opt_int(fields, "speed_max"),
            course_deg=_opt_int(fields, "course"),
            altitude_m=_opt_int(fields, "altitude"),
            satellites=_opt_int(fields, "nsat"),
            doors=json.loads(doors) if doors else None,
            pos_lat=_opt_float(fields, "pos_lat"),
            pos_lon=_opt_float(fields, "pos_lon"),
            pos_time=_parse_ts(fields.get("pos_ts")),
            updated_ms=int(fields.get("updated_ms") or 0),
        )

    def same_state(self, other: VehicleRecord) -> bool:
        """Whether two records describe the same state (ignoring the write time and cache version)."""
        return all(
            getattr(self, f.name) == getattr(other, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("updated_ms", "version")
        )


# --------------------------------------------------------------------------------------------------
# Producer (ingest)
# --------------------------------------------------------------------------------------------------

StatsProvider = Callable[[], Mapping[str, str]]
"""Returns flat string fields of the ingest stats hash."""


class TelemetryPublisher:
    """Writes navigation events and hot vehicle state to Redis without ever blocking the NDTP server.

    :meth:`on_packet` (an NDTP packet listener) only appends to a bounded buffer; :meth:`run` flushes it
    in pipelined batches together with the hot state of vehicles changed since the last flush
    (``HSET`` + ``EXPIRE`` + ``ZADD``, one ``PUBLISH`` with the delta), the stream clock and periodic stats.
    The hot state is re-derived from the :class:`~backend.state.StateStore` after an outage, so only
    stream events need buffering.

    Args:
        store: Ingest state store (source of the hot state).
        factory: Redis client factory.
        status: Redis dependency status to report to.
        unit_map: ``unit_id -> tr_id``.
        max_buffer: Events kept while Redis is down (older ones are evicted and counted).
        batch_size: Events per pipeline.
        stream_maxlen: Approximate stream length limit.
        vehicle_ttl_s: TTL of vehicle hashes; the ZSET index is trimmed by the same age.
        flush_interval_s: Idle wake-up period.
        stats_interval_s: Period of the stats hash update.
        stats_provider: Source of the stats hash fields (``None`` disables stats).
        backoff: Reconnect delays.
    """

    #: Time to collect a batch after the first event of a burst.
    LINGER_S = 0.01

    def __init__(
        self,
        store: StateStore,
        factory: RedisFactory,
        status: DependencyStatus,
        *,
        unit_map: Mapping[int, int] | None = None,
        max_buffer: int = 100_000,
        batch_size: int = 1000,
        stream_maxlen: int = 200_000,
        vehicle_ttl_s: int = 600,
        flush_interval_s: float = 0.25,
        stats_interval_s: float = 2.0,
        stats_provider: StatsProvider | None = None,
        backoff: Backoff | None = None,
    ) -> None:
        self.store = store
        self.factory = factory
        self.status = status
        self.unit_map: Mapping[int, int] = unit_map or {}
        self.buffer: BoundedBuffer[TelemetryEvent] = BoundedBuffer(max_buffer)
        self.batch_size = batch_size
        self.stream_maxlen = stream_maxlen
        self.vehicle_ttl_s = vehicle_ttl_s
        self.flush_interval_s = flush_interval_s
        self.stats_interval_s = stats_interval_s
        self.stats_provider = stats_provider
        self.backoff = backoff or Backoff()
        self.published = 0
        self.state_writes = 0
        self.flushes = 0
        self.flush_errors = 0
        self.command_errors = 0
        self.unmapped = 0
        self.last_flush_s = 0.0
        self._client: aioredis.Redis | None = None
        self._wake = asyncio.Event()
        self._synced_version = -1
        self._synced_clock: tuple[datetime | None, int] | None = None
        self._last_stats = float("-inf")

    @property
    def client(self) -> aioredis.Redis:
        """The Redis client (created on first use inside the event loop)."""
        if self._client is None:
            self._client = self.factory()
        return self._client

    @property
    def buffered(self) -> int:
        """Events waiting to be written."""
        return len(self.buffer)

    @property
    def evicted(self) -> int:
        """Events dropped because the buffer overflowed."""
        return self.buffer.evicted

    def tr_id(self, unit_id: int) -> int | None:
        """``tr_id`` of a device or ``None`` if unknown."""
        return self.unit_map.get(unit_id)

    def on_packet(self, unit_id: int, packet: RealtimePacket) -> None:
        """NDTP packet listener: enqueue the Nav00 of the packet as a :class:`TelemetryEvent`.

        Runs after the state store has fed the fix to the stream clock, so the event carries the clock and
        epoch that include it.
        """
        nav = packet.nav
        if nav is None:
            return
        tr_id = self.tr_id(unit_id)
        if tr_id is None:
            self.unmapped += 1
        clock = self.store.stream_clock
        self.enqueue(
            TelemetryEvent(
                unit_id=unit_id,
                tr_id=tr_id,
                ts=nav.timestamp.timestamp(),
                lon=nav.lon,
                lat=nav.lat,
                speed=nav.speed_avg,
                course=nav.course,
                valid=nav.valid,
                doors=doors_json(packet.irma[0]) if packet.irma else None,
                received_at=time.time(),
                epoch=clock.epoch,
                clock=clock.now,
            )
        )

    def enqueue(self, event: TelemetryEvent) -> None:
        """Buffer an event for the stream (never blocks)."""
        self.buffer.append(event)
        self._wake.set()

    async def run(self) -> None:
        """Flush loop: runs until cancelled, retrying with backoff while Redis is down.

        A new event wakes the loop, which lingers ``LINGER_S`` to collect a batch (one pipeline per burst
        instead of one per packet); state-only changes (connects, disconnects) go out every
        ``flush_interval_s``.
        """
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), self.flush_interval_s)
                if len(self.buffer) < self.batch_size:
                    await asyncio.sleep(self.LINGER_S)
            except TimeoutError:
                pass
            self._wake.clear()
            while True:
                try:
                    await self.flush()
                except Exception as exc:
                    self.flush_errors += 1
                    if is_transient(exc):  # unreachable or refusing writes: keep the batch, retry later
                        self.status.mark_down(exc)
                        self.resync()
                        await asyncio.sleep(self.backoff.next())
                        continue
                    # a bug must not stop the publisher: log and retry later
                    log.exception("Redis publisher flush failed")
                    await asyncio.sleep(self.backoff.next())
                    break
                self.backoff.reset()
                if not self.buffer:
                    break

    async def flush(self) -> int:
        """Write one batch of events plus changed hot state, stream clock and due stats.

        Returns:
            Number of stream events written.

        Raises:
            redis.exceptions.RedisError: Redis is unreachable or refused writes for now (see
                :func:`is_transient`); the batch stays in the buffer.
        """
        now = time.time()
        last_seq, events = self.buffer.peek(self.batch_size)
        version = self.store.version
        changed = self.store.changed_since(self._synced_version) if version != self._synced_version else []
        stream_time = self.store.stream_time
        clock = (stream_time, self.store.stream_clock.epoch)
        clock_due = stream_time is not None and clock != self._synced_clock
        stats_due = self.stats_provider is not None and (
            time.monotonic() - self._last_stats >= self.stats_interval_s
        )
        if not events and not changed and not stats_due and not clock_due:
            return 0

        pipe = self.client.pipeline(transaction=False)
        for event in events:
            pipe.xadd(STREAM_TELEMETRY, event.to_fields(), maxlen=self.stream_maxlen, approximate=True)
        if changed:
            updated_ms = int(now * 1000)
            payload = []
            for vehicle in changed:
                fields = vehicle_fields(vehicle, self.tr_id(vehicle.unit_id), updated_ms)
                key = vehicle_key(vehicle.unit_id)
                pipe.hset(key, mapping=fields)
                pipe.expire(key, self.vehicle_ttl_s)
                seen = vehicle.last_packet_at.timestamp() if vehicle.last_packet_at else now
                pipe.zadd(VEHICLE_INDEX, {str(vehicle.unit_id): seen})
                payload.append(fields)
            message = {
                "type": "delta",
                "stream_time": stream_time.isoformat() if stream_time else None,
                "epoch": clock[1],
                "vehicles": payload,
            }
            pipe.publish(CHANNEL_VEHICLES, json.dumps(message, separators=(",", ":")))
        if clock_due and stream_time is not None:
            pipe.set(STREAM_TIME_KEY, stream_time.isoformat())
            pipe.set(STREAM_EPOCH_KEY, str(clock[1]))
            pipe.set(STREAM_DEVICES_KEY, str(self.store.stream_clock.active_devices(now)))
        if stats_due and self.stats_provider is not None:
            pipe.hset(INGEST_STATS_KEY, mapping=dict(self.stats_provider()))
            pipe.zremrangebyscore(VEHICLE_INDEX, "-inf", now - self.vehicle_ttl_s)

        started = time.perf_counter()
        results = await pipe.execute(raise_on_error=False)
        self.last_flush_s = time.perf_counter() - started
        errors = [r for r in results if isinstance(r, RedisError)]
        transient = [r for r in errors if is_transient(r)]
        if transient:  # Redis is down or refuses writes (OOM, MISCONF, READONLY...): retry the whole batch
            raise transient[0]
        if errors:  # e.g. WRONGTYPE: retrying would not help, count and move on
            self.command_errors += len(errors)
            log.error("Redis rejected %d commands, first: %s", len(errors), errors[0])

        self.buffer.commit(last_seq)
        self.published += len(events)
        self.state_writes += len(changed)
        self.flushes += 1
        self._synced_version = version
        self._synced_clock = clock
        if stats_due:
            self._last_stats = time.monotonic()
        self.status.mark_ok()
        return len(events)

    def resync(self) -> None:
        """Rewrite the hot state of every vehicle, the stream clock and the stats on the next flush.

        Called after a failed flush: Redis may come back without data (restart without AOF), and vehicles
        that stay silent would otherwise vanish from the hot state until their next packet.
        """
        self._synced_version = -1
        self._synced_clock = None
        self._last_stats = float("-inf")

    async def load_clock(self, timeout_s: float = 2.0) -> bool:
        """Continue the stream clock saved in Redis (call before accepting NDTP).

        After a restart of the ingest the clock keeps its epoch, so the predictor sees no jump. The devices
        that were active on the clock before the restart count in its quorum while they reconnect (see
        :meth:`~backend.clock.StreamClock.restore`). Without Redis the clock starts a new epoch with the first
        fix; the predictor adopts it silently when it is close to its own clock.

        Args:
            timeout_s: Give up after this long (Redis may be down at startup).

        Returns:
            Whether a saved clock was restored.
        """
        try:
            pipe = self.client.pipeline(transaction=False)
            pipe.get(STREAM_TIME_KEY)
            pipe.get(STREAM_EPOCH_KEY)
            pipe.get(STREAM_DEVICES_KEY)
            raw_time, raw_epoch, raw_devices = await asyncio.wait_for(pipe.execute(), timeout_s)
            now = datetime.fromisoformat(raw_time).timestamp() if raw_time else None
            epoch = int(raw_epoch) if raw_epoch else 0
            devices = int(raw_devices) if raw_devices else 0
        except (*REDIS_ERRORS, RedisError, ValueError) as exc:
            log.info("stream clock not restored: %r", exc)
            return False
        if now is None or not epoch:
            return False
        self.store.stream_clock.restore(now, epoch, devices=devices)
        log.info(
            "stream clock restored: %s (epoch %d, %d devices expected back)",
            self.store.stream_time,
            epoch,
            devices,
        )
        return True

    async def drain(self, timeout_s: float) -> None:
        """Try to write everything buffered before shutdown (best effort, bounded by ``timeout_s``)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                await asyncio.wait_for(self.flush(), max(deadline - time.monotonic(), 0.01))
            except (*REDIS_ERRORS, RedisError) as exc:
                log.warning("Redis flush on shutdown failed: %s (%d events lost)", exc, self.buffered)
                return
            if not self.buffer:
                return

    async def close(self) -> None:
        """Close the Redis client."""
        if self._client is not None:
            await _close_quietly(self._client)
            self._client = None


async def _close_quietly(client: aioredis.Redis) -> None:
    try:
        await asyncio.wait_for(client.aclose(), 2)
    except Exception as exc:  # the connection may already be gone
        log.debug("closing Redis client: %s", exc)


# --------------------------------------------------------------------------------------------------
# Consumer (predictor)
# --------------------------------------------------------------------------------------------------

StreamBatch = list[tuple[str, TelemetryEvent]]
BatchHandler = Callable[[StreamBatch], Awaitable[None]]
"""Processes a batch of ``(entry_id, event)``; entries are ACKed after it returns (or raises)."""

HistoryHandler = Callable[[list[TelemetryEvent]], bool]
"""Takes already processed events, newest first; returns ``True`` to get older ones."""


def _prev_id(entry_id: str) -> str:
    """The largest stream id below ``entry_id`` (an exclusive upper bound for ``XREVRANGE``)."""
    ms, _, seq = entry_id.partition("-")
    if int(seq or 0) > 0:
        return f"{ms}-{int(seq) - 1}"
    return f"{int(ms) - 1}-18446744073709551615"


def _entries(response: Any) -> list[tuple[str, dict[str, str]]]:
    """Extract ``(id, fields)`` from an XREADGROUP reply (RESP2: ``[[stream, [(id, fields), ...]]]``).

    A pending entry that was trimmed from the stream comes back with ``None`` fields; it becomes an empty
    dict and is later counted as malformed and ACKed.
    """
    if not response:
        return []
    return [(entry_id, fields or {}) for _, batch in response for entry_id, fields in batch]


class StreamConsumer:
    """Consumer-group reader of the telemetry stream with pending-entry recovery.

    After (re)start and after every reconnect it first re-reads its own pending entries (delivered but not
    ACKed before a crash); every ``claim_every_s`` it claims entries idle longer than ``claim_idle_ms`` from
    other (dead) consumers with ``XAUTOCLAIM`` and then removes consumers idle longer than ``gc_idle_ms``
    without pending entries (e.g. old container names). Malformed entries are counted and ACKed; a failing
    handler is logged and the batch is ACKed anyway (a poison entry must not block the stream).

    With a ``history`` handler, the first successful start of an existing group reads back the entries the
    group has already delivered (``XREVRANGE`` from its last-delivered id, newest first) so the service can
    refill its in-memory state; they are not ACKed or re-processed.

    Args:
        factory: Redis client factory.
        handler: Batch processor.
        status: Redis dependency status to report to.
        group: Consumer group name.
        consumer: Consumer name (stable across restarts of the same instance).
        batch_size: Entries per read.
        block_ms: XREADGROUP BLOCK time.
        claim_idle_ms: Idle time after which pending entries of other consumers are claimed.
        claim_every_s: Period of XAUTOCLAIM sweeps.
        gc_idle_ms: Idle time after which other consumers without pending entries are deleted (0: never).
        start_id: Stream id the group starts from when it is created.
        lag_every_s: Period of XINFO GROUPS polling (lag metrics).
        history: Receiver of the already processed history at start (``None``: no read-back).
        history_max: Largest number of history entries to read back.
        backoff: Reconnect delays.
    """

    #: Pause after an empty read (guards against busy-looping if BLOCK returns immediately).
    IDLE_SLEEP_S = 0.01

    def __init__(
        self,
        factory: RedisFactory,
        handler: BatchHandler,
        status: DependencyStatus,
        *,
        group: str = "predictors",
        consumer: str = "predictor",
        batch_size: int = 500,
        block_ms: int = 1000,
        claim_idle_ms: int = 30_000,
        claim_every_s: float = 10.0,
        gc_idle_ms: int = 600_000,
        start_id: str = "0",
        lag_every_s: float = 2.0,
        history: HistoryHandler | None = None,
        history_max: int = 200_000,
        backoff: Backoff | None = None,
    ) -> None:
        self.factory = factory
        self.handler = handler
        self.status = status
        self.group = group
        self.consumer = consumer
        self.batch_size = batch_size
        self.block_ms = block_ms
        self.claim_idle_ms = claim_idle_ms
        self.claim_every_s = claim_every_s
        self.gc_idle_ms = gc_idle_ms
        self.start_id = start_id
        self.lag_every_s = lag_every_s
        self.history = history
        self.history_max = history_max
        self.backoff = backoff or Backoff()
        self.processed = 0
        self.acked = 0
        self.malformed = 0
        self.handler_errors = 0
        self.claimed = 0
        self.recovered = 0
        self.read_errors = 0
        self.history_entries = 0
        self.consumers_removed = 0
        self.lag: int | None = None
        self.pending: int | None = None
        self.stream_length: int | None = None
        self.last_id: str | None = None
        self.rate = RateMeter()
        self._client: aioredis.Redis | None = None
        self._group_ready = False
        self._history_due = history is not None and history_max > 0
        self._recover_pending = True
        self._pending_cursor = "0"
        self._claim_cursor = "0-0"
        self._next_claim = 0.0
        self._next_lag = 0.0

    @property
    def client(self) -> aioredis.Redis:
        """The Redis client (created on first use inside the event loop)."""
        if self._client is None:
            self._client = self.factory()
        return self._client

    async def ensure_group(self) -> bool:
        """Create the consumer group (and the stream) unless it exists.

        Returns:
            ``True`` if the group was created now (it has no processed history).
        """
        created = False
        try:
            await self.client.xgroup_create(STREAM_TELEMETRY, self.group, id=self.start_id, mkstream=True)
            created = True
            log.info(
                "created consumer group %s on %s from id %s", self.group, STREAM_TELEMETRY, self.start_id
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True
        self._recover_pending = True
        self._pending_cursor = "0"
        return created

    async def poll_once(self) -> int:
        """One iteration: ensure the group (reading back history once), read a batch, process and ACK it.

        Returns:
            Number of entries handled.
        """
        if not self._group_ready and await self.ensure_group():
            self._history_due = False
        if self._history_due:
            await self.read_history()
            self._history_due = False
        entries = await self._read()
        if entries:
            await self._process(entries)
        await self._maybe_poll_lag()
        self.status.mark_ok()
        return len(entries)

    async def run(self) -> None:
        """Read loop: runs until cancelled, reconnecting with backoff."""
        while True:
            try:
                if not await self.poll_once():
                    # BLOCK normally waits on the server; never spin if a read returns empty at once
                    await asyncio.sleep(self.IDLE_SLEEP_S)
                self.backoff.reset()
            except ResponseError as exc:
                if "NOGROUP" in str(exc):  # the stream or group vanished (Redis restarted without data)
                    log.warning("consumer group is gone, recreating: %s", exc)
                    self._group_ready = False
                    continue
                self.read_errors += 1
                if is_transient(exc):  # OOM, READONLY, LOADING...: Redis is not usable right now
                    self._outage(exc)
                else:
                    log.error("Redis error in consumer: %s", exc)
                await asyncio.sleep(self.backoff.next())
            except REDIS_ERRORS as exc:
                self.read_errors += 1
                self._outage(exc)
                await asyncio.sleep(self.backoff.next())
            except Exception:  # a bug must not stop consumption: log and retry later
                self.read_errors += 1
                log.exception("telemetry consumer failed")
                await asyncio.sleep(self.backoff.next())

    def _outage(self, exc: BaseException) -> None:
        self.status.mark_down(exc)
        self._recover_pending = True  # an unacked batch may be pending: re-read it first
        self._pending_cursor = "0"

    async def read_history(self) -> int:
        """Hand the entries the group has already delivered to the ``history`` handler, newest first.

        Reads back from the group's last-delivered id with ``XREVRANGE`` in batches until the handler has
        enough or ``history_max`` entries were read. Nothing is ACKed; pending entries among them are
        re-delivered later as usual (the handler of the live stream must tolerate repeats).

        Returns:
            Number of entries read.
        """
        if self.history is None:
            return 0
        last = None
        for info in await self.client.xinfo_groups(STREAM_TELEMETRY):
            if info.get("name") == self.group:
                last = str(info.get("last-delivered-id") or "")
        if not last or last in ("0", "0-0"):
            return 0
        upper, read = last, 0
        while read < self.history_max:
            count = min(self.batch_size, self.history_max - read)
            rows = await self.client.xrevrange(STREAM_TELEMETRY, max=upper, min="-", count=count)
            if not rows:
                break
            read += len(rows)
            events = []
            for _, fields in rows:
                try:
                    events.append(TelemetryEvent.from_fields(fields or {}))
                except (KeyError, ValueError, TypeError):
                    continue
            try:
                more = self.history(events)
            except Exception:  # the live stream matters more than the history: go on without it
                log.exception("history handler failed")
                break
            if not more or len(rows) < count:
                break
            upper = _prev_id(rows[-1][0])
        self.history_entries += read
        log.info("read back %d processed entries of group %s up to %s", read, self.group, last)
        return read

    async def _read(self) -> list[tuple[str, dict[str, str]]]:
        client = self.client
        if self._recover_pending:
            response = await client.xreadgroup(
                self.group, self.consumer, {STREAM_TELEMETRY: self._pending_cursor}, count=self.batch_size
            )
            entries = _entries(response)
            if entries:
                self._pending_cursor = entries[-1][0]
                self.recovered += len(entries)
                return entries
            self._recover_pending = False
        if self.claim_idle_ms and time.monotonic() >= self._next_claim:
            reply = await client.xautoclaim(
                STREAM_TELEMETRY,
                self.group,
                self.consumer,
                min_idle_time=self.claim_idle_ms,
                start_id=self._claim_cursor,
                count=self.batch_size,
            )
            next_id, claimed = str(reply[0]), [(i, f) for i, f in reply[1] if f]
            if next_id in ("0-0", "0"):
                self._claim_cursor = "0-0"
                self._next_claim = time.monotonic() + self.claim_every_s
                if self.gc_idle_ms:
                    await self.remove_idle_consumers()
            else:
                self._claim_cursor = next_id
            if claimed:
                self.claimed += len(claimed)
                log.info("claimed %d pending entries from other consumers", len(claimed))
                return claimed
        response = await client.xreadgroup(
            self.group, self.consumer, {STREAM_TELEMETRY: ">"}, count=self.batch_size, block=self.block_ms
        )
        return _entries(response)

    async def remove_idle_consumers(self) -> int:
        """Delete other consumers idle longer than ``gc_idle_ms`` that hold no pending entries.

        A recreated container gets a new host name, so its old consumer would stay in the group forever.
        Consumers with pending entries are kept: ``XAUTOCLAIM`` moves those entries first.

        Returns:
            Number of consumers removed.
        """
        removed = 0
        for info in await self.client.xinfo_consumers(STREAM_TELEMETRY, self.group):
            name = str(info.get("name"))
            idle = int(info.get("idle") or 0)
            if name == self.consumer or int(info.get("pending") or 0) or idle < self.gc_idle_ms:
                continue
            removed += 1
            await self.client.xgroup_delconsumer(STREAM_TELEMETRY, self.group, name)
            log.info("removed consumer %s from group %s (idle %.0f s)", name, self.group, idle / 1000)
        self.consumers_removed += removed
        return removed

    async def _process(self, entries: list[tuple[str, dict[str, str]]]) -> None:
        batch: StreamBatch = []
        for entry_id, fields in entries:
            try:
                batch.append((entry_id, TelemetryEvent.from_fields(fields)))
            except (KeyError, ValueError, TypeError):
                self.malformed += 1
        if batch:
            try:
                await self.handler(batch)
            except Exception:
                self.handler_errors += 1
                log.exception("telemetry handler failed on %d events", len(batch))
        ids = [entry_id for entry_id, _ in entries]
        self.acked += await self.client.xack(STREAM_TELEMETRY, self.group, *ids)
        self.processed += len(batch)
        self.rate.add(len(batch))
        self.last_id = ids[-1]

    async def _maybe_poll_lag(self) -> None:
        if time.monotonic() < self._next_lag:
            return
        self._next_lag = time.monotonic() + self.lag_every_s
        await self.poll_lag()

    async def poll_lag(self) -> None:
        """Refresh :attr:`lag`, :attr:`pending` and :attr:`stream_length` from ``XINFO GROUPS``/``XLEN``."""
        pipe = self.client.pipeline(transaction=False)
        pipe.xinfo_groups(STREAM_TELEMETRY)
        pipe.xlen(STREAM_TELEMETRY)
        groups, length = await pipe.execute()
        self.stream_length = int(length)
        for info in groups:
            if info.get("name") == self.group:
                lag = info.get("lag")
                self.lag = int(lag) if lag is not None else None
                self.pending = int(info.get("pending") or 0)

    async def close(self) -> None:
        """Close the Redis client."""
        if self._client is not None:
            await _close_quietly(self._client)
            self._client = None
