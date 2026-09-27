"""Predictor service: consumes the telemetry stream, keeps track windows and runs prediction ticks.

Run with ``python -m backend.predictor``. HTTP (port 8002 by default): ``/health``, ``/metrics``,
``/api/predictor/stats``.

The prediction pipeline (docs/architecture.md §5):

* a consumer group reads ``foresight:telemetry`` in batches (:class:`~backend.bus.StreamConsumer`); at start
  the already processed part of the stream is read back to refill the windows;
* a :class:`~backend.clock.StreamClock` follows the ingest's stream clock carried by every event, so the
  predictor and the dashboard share one "now" and reset together when the replayer restarts;
* :class:`TrackWindows` keeps the last ``track_window_s`` of every vehicle (``tr_id``) in memory, in fix-time
  order like the offline data (late points are inserted, not dropped);
* :class:`TickScheduler` fires a tick every ``tick_period_s`` of stream time; the tick handler is the
  forecast engine (:class:`backend.forecast.ForecastEngine`: stop detector, features, ``ml-service`` or the
  fallback, forecasts, alerts, incidents, online check against the fact).

A tick runs inline, before the event that crossed its boundary is added, so it sees only telemetry up to its
time. If the predictor falls behind the stream (a batch already reaches a later boundary), the stale tick is
skipped and counted: the tick of the latest boundary runs instead, the queue does not grow.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

from backend import __version__
from backend.bus import RedisFactory, StreamBatch, StreamConsumer, TelemetryEvent, redis_factory
from backend.clock import ClockJump, Fix, StreamClock
from backend.config import Settings
from backend.db import BufferedWriter, Database, ServiceEventLog
from backend.forecast import ForecastEngine, ModelBackend, Publisher
from backend.metrics import (
    ConsumerCollector,
    DependencyCollector,
    FunctionCollector,
    WriterCollector,
    build_registry,
)
from backend.mlclient import ML_DEPENDENCY, MLClient
from backend.runtime import Backoff, DependencyStatus, TaskSet, dependencies_ok, run_http
from backend.schedule import Schedule
from backend.schemas import DependencyOut, ForecastStatsOut, HealthOut, PredictorStatsOut
from shared.routes import load_segments

__all__ = [
    "PredictorCore",
    "PredictorService",
    "StreamClock",
    "TickContext",
    "TickScheduler",
    "TrackPoint",
    "TrackWindows",
    "create_app",
    "log_tick",
]

log = logging.getLogger(__name__)

SERVICE = "predictor"


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """One telemetry point in a track window (equal points are exact repeats)."""

    ts: float
    lon: float
    lat: float
    speed: int
    course: int
    valid: bool
    unit_id: int


def _ts(point: TrackPoint) -> float:
    return point.ts


class TrackWindows:
    """Last ``window_s`` seconds (stream time) of telemetry per ``tr_id``, ordered by fix time.

    Late points (older than the newest point of the track: black-box data, reordered packets) are inserted in
    time order, as the offline data sorts them (``shared/data.py``). Only an exact repeat (same time,
    position, speed, course, validity and device — an at-least-once redelivery) is dropped; points of one
    second with different content are all kept, the feature code cleans them the same way as offline.

    Off-timeline points (see :mod:`backend.clock`) are also *held* per track. When the clock then jumps to
    their timeline (a source restart), :meth:`rebase` puts them back, so the first points of the new pass that
    arrived before the jump was confirmed are not lost.

    Args:
        window_s: Window length, seconds of stream time.
    """

    #: Off-timeline points held per track.
    HOLD_MAX = 64

    def __init__(self, window_s: float = 1800.0) -> None:
        self.window_s = window_s
        self.late = 0
        self.duplicates = 0
        self.expired = 0
        self._tracks: dict[int, list[TrackPoint]] = {}
        self._held: dict[int, deque[TrackPoint]] = {}
        self._points = 0

    def add(self, tr_id: int, point: TrackPoint, *, now: float | None = None, count: bool = True) -> bool:
        """Insert a point in fix-time order.

        Args:
            tr_id: Vehicle.
            point: The point.
            now: Stream clock; points older than ``now - window_s`` are dropped as expired.
            count: Update the late / duplicate / expired counters (off when refilling from history).

        Returns:
            ``False`` if the point was dropped (an exact repeat or older than the window).
        """
        if now is not None and point.ts < now - self.window_s:
            self.expired += count
            return False
        track = self._tracks.setdefault(tr_id, [])
        if not track or point.ts > track[-1].ts:
            track.append(point)
        else:
            lo = bisect.bisect_left(track, point.ts, key=_ts)
            hi = bisect.bisect_right(track, point.ts, lo=lo, key=_ts)
            if point in track[lo:hi]:
                self.duplicates += count
                return False
            if point.ts < track[-1].ts:
                self.late += count
            track.insert(hi, point)
        self._points += 1
        return True

    def hold(self, tr_id: int, point: TrackPoint) -> None:
        """Keep an off-timeline point in case the clock jumps to its timeline."""
        held = self._held.get(tr_id)
        if held is None:
            held = self._held[tr_id] = deque(maxlen=self.HOLD_MAX)
        held.append(point)

    def settle(self, tr_id: int) -> None:
        """The track is on the timeline again: forget its held points (they were only late)."""
        self._held.pop(tr_id, None)

    def rebase(self, now: float, *, back: bool) -> int:
        """The clock jumped to a new timeline: start over (if it went back) and put the held points back.

        Args:
            now: The clock after the jump.
            back: The clock went back (a source restart): the old tracks belong to another pass.

        Returns:
            Number of held points put into the windows.
        """
        if back:
            self._tracks.clear()
            self._points = 0
        held, self._held = self._held, {}
        restored = 0
        for tr_id, points in held.items():
            for point in points:
                if now - self.window_s <= point.ts <= now and self.add(tr_id, point, count=False):
                    restored += 1
        return restored

    def trim(self, now: float) -> int:
        """Drop points older than ``now - window_s`` and empty tracks; returns dropped points."""
        cutoff = now - self.window_s
        removed = 0
        for tr_id in list(self._tracks):
            track = self._tracks[tr_id]
            old = bisect.bisect_left(track, cutoff, key=_ts)
            if old:
                del track[:old]
                removed += old
            if not track:
                del self._tracks[tr_id]
        self._points -= removed
        return removed

    def clear(self) -> None:
        """Forget all tracks and held points."""
        self._tracks.clear()
        self._held.clear()
        self._points = 0

    def track(self, tr_id: int, until: float | None = None) -> tuple[TrackPoint, ...]:
        """Points of one vehicle, oldest first; with ``until`` only those with ``ts <= until``."""
        track = self._tracks.get(tr_id, [])
        if until is not None:
            track = track[: bisect.bisect_right(track, until, key=_ts)]
        return tuple(track)

    def __iter__(self) -> Iterator[int]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def points(self) -> int:
        """Points in all windows."""
        return self._points

    @property
    def held(self) -> int:
        """Off-timeline points held."""
        return sum(len(points) for points in self._held.values())


class TickScheduler:
    """Fires at every multiple of ``period_s`` of stream time.

    The first observation only aligns the schedule. If the stream jumps over several boundaries (catch-up
    after an outage), only the latest one fires: predictions are never issued for the past.

    Args:
        period_s: Tick period, seconds of stream time.
    """

    def __init__(self, period_s: float = 30.0) -> None:
        self.period_s = period_s
        self.last: float | None = None
        self.skipped = 0

    def due(self, now: float) -> float | None:
        """Return the tick time (Unix seconds) if a boundary was crossed, else ``None``."""
        boundary = math.floor(now / self.period_s) * self.period_s
        if self.last is None:
            self.last = boundary
            return None
        if boundary <= self.last:
            return None
        self.skipped += max(round((boundary - self.last) / self.period_s) - 1, 0)
        self.last = boundary
        return boundary

    def reset(self) -> None:
        """Forget the schedule (stream clock jump)."""
        self.last = None


@dataclass(frozen=True, slots=True)
class TickContext:
    """What a tick handler gets.

    Attributes:
        stream_time: Tick time (a multiple of the tick period, stream time).
        windows: Track windows (read-only use).
        events: Events processed so far.
        epoch: Epoch of the stream clock (changes when the clock jumps to a new timeline).
    """

    stream_time: datetime
    windows: TrackWindows
    events: int
    epoch: int = 0

    def track(self, tr_id: int) -> tuple[TrackPoint, ...]:
        """Points of one vehicle known at the tick with ``ts <= stream_time`` (the honest view)."""
        return self.windows.track(tr_id, until=self.stream_time.timestamp())


TickHandler = Callable[[TickContext], Awaitable[None]]
"""Extension point: called on every tick (features -> ML -> predictions/alerts in the next block)."""


async def log_tick(ctx: TickContext) -> None:
    """Default tick handler: only log the state of the windows."""
    log.info(
        "tick %s: %d active vehicles, %d points in windows, %d events so far",
        ctx.stream_time.isoformat(),
        len(ctx.windows),
        ctx.windows.points,
        ctx.events,
    )


class PredictorCore:
    """Stream clock, track windows and the tick loop, fed by batches from the consumer.

    Every event carries the ingest's clock and epoch (:meth:`StreamClock.follow`); an event without them
    (written by something else than the ingest) goes through the same rule locally
    (:meth:`StreamClock.observe`).

    Args:
        window_s: Track window, seconds of stream time.
        tick_period_s: Tick period, seconds of stream time.
        clock: Stream clock (default: default settings of :class:`~backend.clock.StreamClock`).
        on_tick: Tick handler.
        lag_s: A tick is stale (skipped) when the batch already reaches a later boundary and the event that
            crossed its boundary was received by the ingest more than this many wall seconds ago.
    """

    def __init__(
        self,
        *,
        window_s: float = 1800.0,
        tick_period_s: float = 30.0,
        clock: StreamClock | None = None,
        on_tick: TickHandler = log_tick,
        lag_s: float = 1.0,
    ) -> None:
        self.clock = clock or StreamClock()
        self.windows = TrackWindows(window_s)
        self.ticks = TickScheduler(tick_period_s)
        self.on_tick = on_tick
        self.lag_s = lag_s
        self.events = 0
        self.unmapped = 0
        self.ahead = 0
        self.restored = 0
        self.ticks_run = 0
        self.tick_errors = 0
        self.ticks_lagging = 0
        self.last_tick: datetime | None = None
        self.last_tick_duration_s: float | None = None
        self.latency = Histogram(
            "foresight_predictor_event_latency_seconds",
            "Ingest receive -> predictor processing latency of an event.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=None,
        )
        self.tick_seconds = Histogram(
            "foresight_predictor_tick_seconds",
            "Duration of prediction ticks (every tick, unlike the gauge of the last one).",
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=None,
        )

    def _latest_boundaries(self, batch: StreamBatch) -> dict[int, float]:
        """Latest tick boundary reached by the batch, per clock epoch (the ingest's clock of the events)."""
        latest: dict[int, float] = {}
        for _, event in batch:
            ref = event.ts if event.clock is None else event.clock
            if ref > latest.get(event.epoch, -math.inf):
                latest[event.epoch] = ref
        period = self.ticks.period_s
        return {epoch: math.floor(ref / period) * period for epoch, ref in latest.items()}

    async def handle(self, batch: StreamBatch) -> None:
        """Process a batch of stream events (the consumer's handler).

        Ticks are checked per event: when an event moves the stream clock over a tick boundary, the tick
        runs *before* the event is added, so a tick at time ``T`` sees only telemetry with ``ts < T``.
        Late points join their tracks in time order; points ahead of the stream (a device with a skewed
        clock) are held, never shown before their time. A tick is stale — the predictor is behind real time
        and catching up — when the same batch already reaches a later boundary and the event that crossed
        this one was received more than ``lag_s`` ago: it is skipped and counted, so a slow tick never builds
        a queue. Events without a receive time (history, tests, the offline simulation) never skip a tick.
        """
        now = time.time()
        boundaries = self._latest_boundaries(batch)
        for _, event in batch:
            self.events += 1
            if event.received_at:
                self.latency.observe(max(now - event.received_at, 0.0))
            fix = self._observe(event, now)
            if fix is Fix.GARBAGE or self.clock.now is None:
                continue  # a corrupted timestamp: neither clock nor windows
            clock_now = self.clock.now
            tick = self.ticks.due(clock_now)
            if tick is not None:
                self.windows.trim(tick)
                behind = event.received_at > 0 and now - event.received_at > self.lag_s
                if behind and tick < boundaries.get(event.epoch, -math.inf):
                    self.ticks_lagging += 1
                    self.ticks.skipped += 1
                else:
                    await self.run_tick(tick)
            if event.tr_id is None:
                self.unmapped += 1
                continue
            point = TrackPoint(
                event.ts, event.lon, event.lat, event.speed, event.course, event.valid, event.unit_id
            )
            if fix is Fix.OFF:
                # a black-box point far behind, or the first points of a new pass before the jump: counted
                # as off-timeline only (not as late / duplicate / expired), held in case the clock jumps
                self.windows.hold(event.tr_id, point)
                if event.ts > clock_now:
                    self.ahead += 1
                else:
                    self.windows.add(event.tr_id, point, now=clock_now, count=False)
                continue
            self.windows.settle(event.tr_id)
            self.windows.add(event.tr_id, point, now=clock_now)
        if self.clock.now is not None:
            self.windows.trim(self.clock.now)

    def _observe(self, event: TelemetryEvent, wall_now: float) -> Fix:
        """Move the clock with an event; on a jump rebase the windows and the tick schedule."""
        wall = event.received_at or wall_now
        before = self.clock.now
        if event.epoch:
            clock_ts = event.ts if event.clock is None else event.clock
            fix = self.clock.follow(event.ts, clock_ts, event.epoch, wall)
        else:
            fix = self.clock.observe(event.ts, event.unit_id, wall)
        if fix is not Fix.JUMP or self.clock.now is None:
            return fix
        back = before is not None and self.clock.now < before
        restored = self.windows.rebase(self.clock.now, back=back)
        self.ticks.reset()
        log.info(
            "stream clock jumped %s to %s: %s, %d held points put back",
            "back" if back else "forward",
            self.clock.time,
            "windows start over" if back else "windows kept",
            restored,
        )
        return Fix.ON if self.clock.on_timeline(event.ts) else Fix.OFF

    def restore(self, events: list[TelemetryEvent]) -> bool:
        """History handler: refill the clock and the windows from processed events, newest first.

        No ticks run and no counters move. The first event sets the clock and aligns the tick schedule; older
        events go into the windows while they are in the same epoch and within the window. Right before a jump
        the stream holds the first points of the new pass under the old epoch (off its clock, sent before the
        ingest confirmed the jump); they are taken as well, as the live windows did (see
        :meth:`TrackWindows.rebase`).

        Args:
            events: Events, newest first.

        Returns:
            ``True`` while older events are still needed.
        """
        for event in events:
            if not self.clock.plausible(event.ts, event.received_at or None):
                continue
            ref = event.ts if event.clock is None else event.clock
            if self.clock.now is None:
                self.clock.restore(ref, event.epoch)
                self.ticks.due(ref)
            elif event.epoch != self.clock.epoch:
                if abs(event.ts - ref) <= self.clock.jump_s:
                    return False  # on the old timeline: another pass
                self._refill(event)  # off the old clock: the start of this pass; fix times are not ordered
                continue
            elif ref < self.clock.now - self.windows.window_s:
                return False  # the clock of an epoch only grows: everything older is out of the window
            self._refill(event)
        return True

    def _refill(self, event: TelemetryEvent) -> None:
        now = self.clock.now
        if event.tr_id is None or now is None or event.ts > now:
            return
        point = TrackPoint(
            event.ts, event.lon, event.lat, event.speed, event.course, event.valid, event.unit_id
        )
        if self.windows.add(event.tr_id, point, now=now, count=False):
            self.restored += 1

    async def run_tick(self, tick_ts: float) -> None:
        """Run the tick handler for stream time ``tick_ts``; failures are logged and counted."""
        stream_time = datetime.fromtimestamp(tick_ts, UTC)
        started = time.perf_counter()
        try:
            await self.on_tick(TickContext(stream_time, self.windows, self.events, self.clock.epoch))
        except Exception:
            self.tick_errors += 1
            log.exception("tick %s failed", stream_time.isoformat())
        self.ticks_run += 1
        self.last_tick = stream_time
        self.last_tick_duration_s = time.perf_counter() - started
        self.tick_seconds.observe(self.last_tick_duration_s)

    def metrics(self) -> Iterator[Metric]:
        """Prometheus metric families of the core."""
        w, c = self.windows, self.clock
        yield CounterMetricFamily("foresight_predictor_events", "Events processed.", value=self.events)
        yield CounterMetricFamily(
            "foresight_predictor_unmapped_events", "Events without tr_id (not windowed).", value=self.unmapped
        )
        yield CounterMetricFamily(
            "foresight_predictor_late_points",
            "On-timeline points inserted before newer ones of their track.",
            value=w.late,
        )
        yield CounterMetricFamily(
            "foresight_predictor_duplicate_points", "Exact repeats dropped.", value=w.duplicates
        )
        yield CounterMetricFamily(
            "foresight_predictor_expired_points", "Points older than the window dropped.", value=w.expired
        )
        yield CounterMetricFamily(
            "foresight_predictor_ahead_points", "Points ahead of the stream clock (held).", value=self.ahead
        )
        yield CounterMetricFamily(
            "foresight_predictor_restored_points",
            "Points refilled from the stream at start.",
            value=self.restored,
        )
        yield GaugeMetricFamily("foresight_predictor_tracks", "Vehicles with a track window.", len(w))
        yield GaugeMetricFamily("foresight_predictor_window_points", "Points in all track windows.", w.points)
        yield GaugeMetricFamily("foresight_predictor_held_points", "Off-timeline points held.", w.held)
        if c.now is not None:
            yield GaugeMetricFamily(
                "foresight_predictor_stream_time_seconds", "Stream clock, Unix seconds.", c.now
            )
        yield GaugeMetricFamily("foresight_predictor_clock_epoch", "Epoch of the stream clock.", c.epoch)
        yield CounterMetricFamily(
            "foresight_predictor_clock_resets", "Stream clock jumps (source restarts).", value=c.resets
        )
        yield CounterMetricFamily(
            "foresight_predictor_clock_garbage", "Events with an implausible fix time.", value=c.garbage
        )
        yield CounterMetricFamily(
            "foresight_predictor_off_timeline", "Events off the stream timeline.", value=c.off_timeline
        )
        yield CounterMetricFamily("foresight_predictor_ticks", "Prediction ticks run.", value=self.ticks_run)
        yield CounterMetricFamily(
            "foresight_predictor_ticks_skipped", "Tick boundaries skipped.", value=self.ticks.skipped
        )
        yield CounterMetricFamily("foresight_predictor_tick_errors", "Failed ticks.", value=self.tick_errors)
        yield CounterMetricFamily(
            "foresight_predictor_ticks_lagging",
            "Ticks skipped because the batch already reached a later boundary (catching up).",
            value=self.ticks_lagging,
        )
        if self.last_tick_duration_s is not None:
            yield GaugeMetricFamily(
                "foresight_predictor_tick_duration_seconds",
                "Duration of the last tick.",
                self.last_tick_duration_s,
            )
        yield from self.tick_seconds.collect()
        yield from self.latency.collect()


class PredictorService:
    """Consumer, core, journal and dependency state of one predictor instance.

    Args:
        settings: Configuration.
        factory: Redis client factory (default: from ``settings.redis_url``).
        database: PostgreSQL access (default: from ``settings.database_url``; empty URL disables it).
        on_tick: Tick handler; by default the forecast engine (``settings.forecast_enabled``, the plan
            schedule loaded) or, without it, :func:`log_tick`.
        model: Forecast model for the engine (default: the ml-service client from ``settings.ml_url``).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        factory: RedisFactory | None = None,
        database: Database | None = None,
        on_tick: TickHandler | None = None,
        model: ModelBackend | None = None,
    ) -> None:
        self.settings = settings
        self.started_monotonic = time.monotonic()
        self.redis_status = DependencyStatus("redis")
        self.pg_status = DependencyStatus("postgres")
        factory = factory or redis_factory(settings.redis_url, settings.redis_timeout_s)
        if database is None and settings.database_url:
            database = Database(settings.database_url)
        self.writer = BufferedWriter(
            database,
            self.pg_status,
            max_buffer=settings.db_buffer_max,
            batch_size=settings.db_batch_size,
            flush_interval_s=settings.db_flush_interval_s,
            health_interval_s=settings.db_health_interval_s,
            backoff=Backoff(settings.backoff_initial_s, settings.backoff_max_s),
        )
        self.ml: MLClient | None = None
        self.engine: ForecastEngine | None = None
        self.publisher: Publisher | None = None
        if on_tick is None and settings.forecast_enabled:
            self.engine = self._build_engine(factory, model)
        if on_tick is None:
            on_tick = self.engine.tick if self.engine is not None else log_tick
        self.core = PredictorCore(
            window_s=settings.track_window_s,
            tick_period_s=settings.tick_period_s,
            clock=settings.stream_clock(on_jump=self._on_clock_jump),
            on_tick=on_tick,
            lag_s=settings.tick_lag_s,
        )
        self.consumer = StreamConsumer(
            factory,
            self.core.handle,
            self.redis_status,
            group=settings.consumer_group,
            consumer=settings.consumer,
            batch_size=settings.consumer_batch,
            block_ms=settings.consumer_block_ms,
            claim_idle_ms=settings.consumer_claim_idle_ms,
            gc_idle_ms=settings.consumer_gc_idle_ms,
            start_id=settings.consumer_start_id,
            history=self.core.restore,
            history_max=settings.history_max_entries,
            backoff=Backoff(settings.backoff_initial_s, settings.backoff_max_s),
        )
        self.events = ServiceEventLog(self.writer, SERVICE, settings.instance)
        self.events.watch(self.redis_status)
        self.events.watch(self.pg_status)
        if self.ml is not None:
            self.events.watch(self.ml.status)
        self._tasks = TaskSet()

    def _build_engine(self, factory: RedisFactory, model: ModelBackend | None) -> ForecastEngine | None:
        """The forecast engine; ``None`` (ticks only log) if the plan schedule cannot be loaded."""
        s = self.settings
        root = s.schedule_dir or s.unit_map_dir
        try:
            segments = None
            if s.routes_segments is not None:
                try:
                    segments = load_segments(s.routes_segments)
                except (OSError, ValueError) as exc:  # the map loses its road geometry, not the forecasts
                    log.warning(
                        "route geometry %s not loaded (%s): straight segments", s.routes_segments, exc
                    )
                    segments = {}
            schedule = Schedule.load(root, s.schedule_split, segments)
        except (OSError, ValueError, KeyError) as exc:
            log.error("forecasts disabled: no plan schedule %s in %s (%s)", s.schedule_split, root, exc)
            return None
        geometry = schedule.routes.geometry_source
        log.info("route geometry: %d of %d segments by GPS", geometry["gps_segments"], geometry["segments"])
        if model is None:
            self.ml = MLClient(
                s.ml_url,
                DependencyStatus(ML_DEPENDENCY),
                timeout_s=s.ml_timeout_s,
                retry_s=s.ml_retry_s,
                explain=s.ml_explain,
            )
            model = self.ml
        self.publisher = Publisher(factory)
        return ForecastEngine(s, schedule, model, self.writer, publisher=self.publisher)

    def _on_clock_jump(self, jump: ClockJump) -> None:
        self.events.write(
            "clock_reset", f"stream clock jumped to {self.core.clock.time}", details=_jump(jump)
        )

    @property
    def dependencies(self) -> list[DependencyStatus]:
        """Redis, PostgreSQL and (with the forecast engine) ml-service."""
        deps = [self.redis_status, self.pg_status]
        if self.engine is not None:
            deps.append(self.engine.model.status)
        return deps

    async def refresh_settings(self) -> None:
        """Re-read the risk / alert thresholds from PostgreSQL every ``settings_refresh_s`` (runs forever)."""
        while True:
            db = self.writer.db
            if self.engine is not None and db is not None and db.connected:
                try:
                    values = await db.fetch_settings(["risk_thresholds", "alert_thresholds"])
                    self.engine.apply_settings(values)
                except Exception as exc:  # the last known thresholds stay in force
                    log.debug("settings refresh failed: %s", exc)
            await asyncio.sleep(self.settings.settings_refresh_s)

    async def start(self) -> None:
        """Start the writer, the consumer and (with forecasts) the ML probe, the publisher, the settings."""
        self.started_monotonic = time.monotonic()
        self._tasks.spawn(self.writer.run(), "db-writer")
        self._tasks.spawn(self.consumer.run(), "stream-consumer")
        if self.ml is not None:
            self._tasks.spawn(self.ml.run(), "ml-probe")
        if self.publisher is not None:
            self._tasks.spawn(self.publisher.run(), "forecast-publisher")
        if self.engine is not None:
            self._tasks.spawn(self.refresh_settings(), "settings-refresh")
        self.events.write(
            "start",
            f"predictor {__version__} started",
            details={
                "group": self.consumer.group,
                "consumer": self.consumer.consumer,
                "forecasts": self.engine is not None,
                "schedule_split": self.settings.schedule_split,
                "ml_url": self.settings.ml_url,
            },
        )

    async def stop(self) -> None:
        """Stop consuming (unacked entries stay pending for the next start) and flush the journal."""
        await self._tasks.cancel()
        await self.consumer.close()
        if self.publisher is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.publisher.flush(), 2.0)
            await self.publisher.close()
        if self.ml is not None:
            self.ml.close()
        self.events.write("stop", "predictor stopped")
        await self.writer.drain(2.0)
        if self.writer.db is not None:
            await self.writer.db.close()

    def health(self) -> HealthOut:
        """Health body (the service is alive whenever it can answer; without ml-service it is degraded)."""
        deps = self.dependencies
        return HealthOut(
            status="ok" if dependencies_ok(deps) else "degraded",
            service=SERVICE,
            version=__version__,
            uptime_s=round(time.monotonic() - self.started_monotonic, 3),
            dependencies={s.name: DependencyOut.of(s) for s in deps},
            vehicles=len(self.core.windows),
        )

    def stats(self) -> PredictorStatsOut:
        """Consumer and core counters."""
        c, core = self.consumer, self.core
        return PredictorStatsOut(
            redis=self.redis_status.state,  # type: ignore[arg-type]
            postgres=self.pg_status.state,  # type: ignore[arg-type]
            consumer=c.consumer,
            group=c.group,
            events=core.events,
            events_per_s=round(c.rate.rate(), 3),
            unmapped_events=core.unmapped,
            malformed=c.malformed,
            recovered=c.recovered,
            claimed=c.claimed,
            history_entries=c.history_entries,
            consumers_removed=c.consumers_removed,
            lag=c.lag,
            pending=c.pending,
            stream_length=c.stream_length,
            stream_time=core.clock.time,
            clock_epoch=core.clock.epoch,
            clock_resets=core.clock.resets,
            clock_garbage=core.clock.garbage,
            off_timeline=core.clock.off_timeline,
            tracks=len(core.windows),
            window_points=core.windows.points,
            late_points=core.windows.late,
            duplicate_points=core.windows.duplicates,
            expired_points=core.windows.expired,
            ahead_points=core.ahead,
            held_points=core.windows.held,
            restored_points=core.restored,
            ticks=core.ticks_run,
            ticks_skipped=core.ticks.skipped,
            ticks_lagging=core.ticks_lagging,
            last_tick=core.last_tick,
            last_tick_duration_s=core.last_tick_duration_s,
            db_buffered=self.writer.buffered,
            db_dropped=self.writer.dropped,
            forecast=ForecastStatsOut.model_validate(
                {**self.engine.stats(), "schedule_split": self.settings.schedule_split}
            )
            if self.engine is not None
            else ForecastStatsOut(enabled=False),
        )


def _jump(jump: ClockJump) -> dict[str, object]:
    def iso(ts: float | None) -> str | None:
        return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None

    return {
        "from": iso(jump.before),
        "to": iso(jump.after),
        "epoch": jump.epoch,
        "back": jump.back,
        "reason": jump.reason,
        "devices": jump.devices,
    }


_DESCRIPTION = """
Foresight predictor: reads the `foresight:telemetry` stream in the consumer group `predictors`, keeps a
90-minute track window per vehicle and runs a prediction tick every 30 s of stream time: the stop detector
on the stream, features, `ml-service` (fallback formula when it is down), forecasts for the stops planned
10–15 min ahead, alerts, incidents, the online check against the detected fact. The stream clock is the
ingest's (carried by every event), so it resets together with the dashboard when the replayer restarts.

* `GET /health` — always 200 while alive; Redis / PostgreSQL / ml-service state in the body;
* `GET /api/predictor/stats` — consumer lag, throughput, stream clock, windows, ticks, forecasts;
* `GET /metrics` — Prometheus metrics.
"""


def create_app(settings: Settings | None = None, service: PredictorService | None = None) -> FastAPI:
    """Build the predictor HTTP application; its lifespan runs the :class:`PredictorService`.

    Args:
        settings: Configuration; defaults to :class:`Settings` from the environment.
        service: Pre-built service (tests inject fakes through it).

    Returns:
        The application; ``app.state.service`` is the service.
    """
    settings = settings or Settings()
    svc = service or PredictorService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(
        title="Foresight · Predictor", version=__version__, description=_DESCRIPTION, lifespan=lifespan
    )
    app.state.service = svc
    collectors = [
        ConsumerCollector(svc.consumer),
        FunctionCollector(svc.core.metrics),
        WriterCollector(svc.writer),
        DependencyCollector(svc.dependencies),
    ]
    if svc.engine is not None:
        collectors.append(FunctionCollector(svc.engine.metrics))
    app.state.registry = build_registry(*collectors)

    @app.get("/health", response_model=HealthOut, tags=["ops"], summary="Liveness and dependencies")
    async def health() -> HealthOut:
        return svc.health()

    @app.get(
        "/api/predictor/stats", response_model=PredictorStatsOut, tags=["ops"], summary="Predictor state"
    )
    async def stats() -> PredictorStatsOut:
        return svc.stats()

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics",
        response_class=Response,
        responses={200: {"content": {CONTENT_TYPE_LATEST: {}}, "description": "Prometheus text format"}},
    )
    async def metrics(request: Request) -> Response:
        return Response(generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    """Entry point of ``python -m backend.predictor``."""
    settings = Settings()
    run_http(
        "backend.predictor:create_app", settings.http_host, settings.predictor_http_port, settings.log_level
    )


if __name__ == "__main__":
    main()
