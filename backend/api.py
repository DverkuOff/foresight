"""API service: REST, WebSocket and Swagger «Foresight · API» over the hot state in Redis.

Run with ``python -m backend.api`` (port 8000 by default). The vehicle state comes from Redis through an
in-memory cache (:class:`~backend.hotstate.VehicleCache`) kept in sync by pub/sub deltas and periodic full
reads; when Redis is down the API keeps answering with the last known state and ``degraded: true``.

The ingest is watched through its stats in Redis (written every ``stats_interval_s``). When they are older
than ``ingest_stale_after_s`` the ingest is down: ``/health`` lists it as a failed dependency, the stats say
``available: false``, every vehicle is ``connected: false`` / ``offline`` and responses are ``degraded``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import redis.asyncio as aioredis
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.core import GaugeMetricFamily, Metric

from backend import __version__
from backend.admin import admin_router
from backend.bus import INGEST_STATS_KEY, REDIS_ERRORS, RedisFactory, VehicleRecord, redis_factory
from backend.config import Settings
from backend.db import BufferedWriter, Database, ServiceEventLog
from backend.hotstate import HotStateSync, VehicleCache
from backend.journal import Journal, epoch_start, incident_sort_key, iso
from backend.metrics import (
    DependencyCollector,
    FunctionCollector,
    WriterCollector,
    build_registry,
    vehicle_status_metric,
)
from backend.mlclient import HttpClient
from backend.perf import PerfScraper
from backend.runtime import Backoff, DependencyStatus, TaskSet, dependencies_ok, run_http
from backend.schedule import Schedule
from backend.schemas import (
    AlertListOut,
    DependencyOut,
    DoorsOut,
    HealthOut,
    HorizonOut,
    IncidentDetailOut,
    IncidentListOut,
    IngestStatsOut,
    NextStopOut,
    PerfOut,
    PredictionListOut,
    RouteOut,
    StringlineOut,
    VehicleListOut,
    VehicleOut,
    WhatIfIn,
    WhatIfOut,
    WsMessage,
)
from backend.state import LinkStatus
from backend.stringline import build_stringline, latest_tick
from backend.whatif import request_error, run_whatif

log = logging.getLogger(__name__)

SERVICE = "api"

_WS_MIN_GAP_S = 0.2  # batch bursts of packets into one WebSocket delta
_WS_CLOCK_S = 1.0  # period of the ``clock`` message (the dashboard reconnects after a long silence)
_LATE_S = 120.0  # a stop is late above this (the red threshold of the contract)
_BOOK_MAX = 5000  # incidents kept in memory from the pub/sub


def _client(name: str, url: str) -> HttpClient | None:
    """HTTP client of a neighbour service (``None``: not configured or a bad URL)."""
    if not url:
        return None
    try:
        return HttpClient(url)
    except ValueError as exc:
        log.warning("%s disabled: %s", name, exc)
        return None


class EventFeed:
    """The predictor's messages (alerts, incidents, closed forecasts) for the WebSocket clients: a bounded
    ring with sequence numbers, so each client sends what came after its last one."""

    def __init__(self, maxlen: int = 2000) -> None:
        self.items: deque[tuple[int, dict[str, Any]]] = deque(maxlen=maxlen)
        self.seq = 0
        self._changed = asyncio.Event()

    def push(self, message: dict[str, Any]) -> None:
        self.seq += 1
        self.items.append((self.seq, message))
        self._changed.set()
        self._changed = asyncio.Event()

    def since(self, seq: int) -> list[tuple[int, dict[str, Any]]]:
        """Messages after ``seq`` (the oldest are lost if a client fell more than ``maxlen`` behind)."""
        return [(n, m) for n, m in self.items if n > seq]

    def latest(self, kind: str, key: str, limit: int) -> list[dict[str, Any]]:
        """The ``key`` objects of the newest ``limit`` messages of a type (newest first)."""
        out = []
        for _, m in reversed(self.items):
            if m.get("type") == kind and isinstance(m.get(key), dict):
                out.append(m[key])
                if len(out) >= limit:
                    break
        return out

    async def wait(self, timeout: float) -> bool:
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return False
        return True


def vehicle_out(
    cache: VehicleCache, record: VehicleRecord, now: datetime, *, link_down: bool = False
) -> VehicleOut:
    """Convert a cached record into the API model.

    Args:
        cache: Cache (for status thresholds).
        record: Vehicle record.
        now: Current server time.
        link_down: The ingest is down: no device is connected, whatever the last known state says.

    Returns:
        API representation of the vehicle.
    """
    doors = None
    if record.doors:
        try:
            doors = DoorsOut.model_validate(record.doors)
        except ValueError:
            doors = None
    # the position shown is the last *valid* fix: an invalid one (often 0, 0) never moves the vehicle
    lat, lon, pos_time = record.pos_lat, record.pos_lon, record.pos_time
    if lat is None and record.valid and record.lat is not None and (record.lat, record.lon) != (0.0, 0.0):
        lat, lon, pos_time = record.lat, record.lon, record.event_time  # a hash written by an older ingest
    age = None
    if pos_time is not None and record.event_time is not None:
        age = round(max((record.event_time - pos_time).total_seconds(), 0.0), 1)
    scheduled = None
    if record.tr_id is None:
        scheduled = False
    elif cache.scheduled is not None:
        scheduled = record.tr_id in cache.scheduled
    fc = cache.forecast_of(record.tr_id) if scheduled is not False else None
    fc = fc or {}
    next_stop = None
    if isinstance(fc.get("next_stop"), dict):
        with contextlib.suppress(ValueError):
            next_stop = NextStopOut.model_validate(fc["next_stop"])
    risk = fc.get("risk") if fc.get("risk") in ("green", "yellow", "red") else "unknown"
    return VehicleOut(
        unit_id=record.unit_id,
        tr_id=record.tr_id,
        status=LinkStatus.OFFLINE if link_down else cache.status(record, now),
        connected=record.connected and not link_down,
        lat=lat,
        lon=lon,
        valid=record.valid,
        position_time=pos_time,
        position_age_s=age,
        speed_kmh=record.speed_kmh,
        speed_max_kmh=record.speed_max_kmh,
        course_deg=record.course_deg,
        altitude_m=record.altitude_m,
        satellites=record.satellites,
        event_time=record.event_time,
        received_at=record.received_at,
        last_packet_at=record.last_packet_at,
        age_s=cache.age_s(record, now),
        packets=record.packets,
        reconnects=record.reconnects,
        doors=doors,
        scheduled=scheduled,
        route_id=fc.get("route_id"),
        risk=risk,
        current_delay_s=fc.get("current_delay_s"),
        pred_delay_s=fc.get("pred_delay_s"),
        p10=fc.get("p10"),
        p90=fc.get("p90"),
        p_late=fc.get("p_late"),
        next_stop=next_stop,
        incident_id=fc.get("incident_id"),
        segment_speed_kmh=fc.get("segment_speed_kmh"),
        dwell_s=fc.get("dwell_s"),
        idle_s=fc.get("idle_s"),
        forecast_source=fc.get("source") if fc.get("source") in ("model", "fallback") else None,
    )


def ingest_stats_age_s(fields: dict[str, str] | None, now: datetime) -> float | None:
    """Seconds since the ingest produced its stats (their ``updated_at``); ``None`` if unknown."""
    if not fields or not fields.get("updated_at"):
        return None
    try:
        updated = datetime.fromisoformat(fields["updated_at"])
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return max((now - updated).total_seconds(), 0.0)


def parse_ingest_stats(
    fields: dict[str, str],
    *,
    degraded: bool,
    stale_after_s: float | None = None,
    now: datetime | None = None,
) -> IngestStatsOut:
    """Build :class:`IngestStatsOut` from the ``foresight:ingest:stats`` hash.

    Args:
        fields: The hash.
        degraded: Redis is down (the stats are the last known).
        stale_after_s: Stats older than this come from an ingest that is down: ``available`` is false, the
            live fields (listening, connections, packet rate) are zeroed, the counters stay as last known.
        now: Current time (default: now).

    Returns:
        The stats.
    """
    data: dict[str, Any] = dict(fields)
    try:
        data["connections"] = json.loads(fields.get("connections") or "[]")
    except ValueError:
        data["connections"] = []
    age = ingest_stats_age_s(fields, now or datetime.now(UTC))
    stale = stale_after_s is not None and age is not None and age > stale_after_s
    data["degraded"] = degraded or stale
    data["available"] = not stale
    data["age_s"] = round(age, 3) if age is not None else None
    if stale:
        data.update(listening=False, connections_active=0, packets_per_s=0.0, connections=[])
    return IngestStatsOut.model_validate(data)


class ApiService:
    """Hot-state cache, its Redis sync, the journal and dependency state of one api instance.

    Args:
        settings: Configuration.
        factory: Redis client factory (default: from ``settings.redis_url``).
        database: PostgreSQL access (default: from ``settings.database_url``; empty URL disables it).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        factory: RedisFactory | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings
        self.started_monotonic = time.monotonic()
        self.factory = factory or redis_factory(settings.redis_url, settings.redis_timeout_s)
        self.cache = VehicleCache(
            stale_after_s=settings.stale_after_s, offline_after_s=settings.offline_after_s
        )
        self.redis_status = DependencyStatus("redis")
        self.pg_status = DependencyStatus("postgres")
        self.redis: aioredis.Redis | None = None
        self.sync: HotStateSync | None = None
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
        self.ingest_status = DependencyStatus("ingest")
        self.events = ServiceEventLog(self.writer, SERVICE, settings.instance)
        self.events.watch(self.redis_status)
        self.events.watch(self.pg_status)
        self.events.watch(self.ingest_status)
        self.ws_clients = 0
        self._ingest_stats: dict[str, str] | None = None
        self._tasks = TaskSet()
        self.journal = Journal(self.writer.db)
        self.feed = EventFeed()
        self.incidents: dict[int, dict[str, Any]] = {}
        """Latest ``IncidentOut`` by id from the pub/sub (the journal fills the ones missed)."""
        self.perf = PerfScraper(
            {"predictor": settings.predictor_url, "ml": settings.ml_url}, interval_s=settings.perf_scrape_s
        )
        self.replayer = _client("replayer", settings.replayer_url)
        self.ml = _client("ml-service", settings.ml_url)
        self.predictor = _client("predictor", settings.predictor_url)
        # /health probes of the admin on connections of their own: a probe timing out closes its connection,
        # and it must not be the one of a long request of the same neighbour (a model activation)
        self.probes = {
            "predictor": _client("predictor", settings.predictor_url),
            "ml-service": _client("ml-service", settings.ml_url),
            "replayer": _client("replayer", settings.replayer_url),
        }
        self.schedule: Schedule | None = None

    def on_event(self, message: dict[str, Any]) -> None:
        """A message of the predictor from the pub/sub: remember incidents, pass everything to the clients."""
        if message.get("type") == "incident" and isinstance(message.get("incident"), dict):
            incident = message["incident"]
            with contextlib.suppress(TypeError, ValueError):
                self.incidents[int(incident["incident_id"])] = incident
                if len(self.incidents) > _BOOK_MAX:
                    for key in list(self.incidents)[: len(self.incidents) - _BOOK_MAX]:
                        del self.incidents[key]
        self.feed.push(message)
        self.cache.notify()  # WebSocket clients send it at once

    @property
    def epoch(self) -> int | None:
        """Timeline of the forecasts (the predictor's epoch, else the ingest's)."""
        status = self.cache.forecast_status or {}
        with contextlib.suppress(TypeError, ValueError):
            if status.get("epoch"):
                return int(status["epoch"])
        return self.cache.stream_epoch or None

    @property
    def forecast_live(self) -> bool:
        """The predictor's forecast snapshot is there and not stale."""
        status = self.cache.forecast_status or {}
        at = status.get("stream_time")
        if not at or self.cache.stream_time is None:
            return False
        with contextlib.suppress(ValueError):
            lag = (self.cache.stream_time - datetime.fromisoformat(at)).total_seconds()
            return lag <= self.cache.forecast_stale_s
        return False

    def _load_schedule(self) -> Schedule | None:
        s = self.settings
        root = Path(s.schedule_dir or s.unit_map_dir)
        try:
            return Schedule.load(root, s.schedule_split)
        except (OSError, ValueError, KeyError) as exc:
            log.warning(
                "stringline without the plan: schedule %s in %s not loaded (%s)", s.schedule_split, root, exc
            )
            return None

    async def load_schedule(self) -> None:
        """Load the plan schedule in a thread (the stringline's planned lines)."""
        self.schedule = await asyncio.to_thread(self._load_schedule)

    @property
    def degraded(self) -> bool:
        """Whether Redis is unavailable (the state served is the last known)."""
        return self.redis_status.ok is not True

    def check_ingest(self, now: datetime | None = None) -> bool | None:
        """Update the ingest's status from the age of its stats in Redis.

        Returns:
            ``True`` up; ``False`` down (no fresh stats for ``ingest_stale_after_s``, or NDTP not listening);
            ``None`` unknown (no stats yet). While Redis is down, and for ``ingest_stale_after_s`` after it
            came back (the ingest needs a moment to write again), old stats prove nothing: the state is kept.
        """
        now = now or datetime.now(UTC)
        fields = self._latest_ingest_stats()
        age = ingest_stats_age_s(fields, now)
        if fields is None or age is None:
            return self.ingest_status.ok
        limit = self.settings.ingest_stale_after_s
        if age > limit:
            since = self.redis_status.since
            redis_up_s = (now - since).total_seconds() if since is not None else 0.0
            if not self.degraded and redis_up_s > limit:
                self.ingest_status.mark_down(f"no stats from the ingest for {age:.0f} s")
        elif fields.get("listening") == "0":
            self.ingest_status.mark_down("NDTP is not listening")
        else:
            self.ingest_status.mark_ok()
        return self.ingest_status.ok

    def ingest_down(self) -> bool:
        """Whether the ingest is down: no device is connected, whatever the last known state says."""
        return self.check_ingest() is False

    def _latest_ingest_stats(self) -> dict[str, str] | None:
        candidates = [f for f in (self._ingest_stats, self.sync.ingest_stats if self.sync else None) if f]
        return max(candidates, key=lambda f: f.get("updated_at") or "", default=None)

    async def start(self) -> None:
        """Create the Redis client and start the sync and writer loops."""
        self.started_monotonic = time.monotonic()
        self.redis = self.factory()
        self.sync = HotStateSync(
            self.redis,
            self.cache,
            self.redis_status,
            resync_s=self.settings.api_resync_s,
            stats_s=self.settings.stats_interval_s,
            backoff=Backoff(self.settings.backoff_initial_s, self.settings.backoff_max_s),
            on_event=self.on_event,
        )
        self._tasks.spawn(self.writer.run(), "db-writer")
        self._tasks.spawn(self.sync.run(), "hot-state-sync")
        self._tasks.spawn(self.perf.run(), "perf-scraper")
        if self.settings.api_schedule:
            self._tasks.spawn(self.load_schedule(), "schedule")
        self.events.write("start", f"api {__version__} started")

    async def stop(self) -> None:
        """Stop the loops, flush the journal and close connections."""
        await self._tasks.cancel()
        self.perf.close()
        for client in (self.replayer, self.ml, self.predictor, *self.probes.values()):
            if client is not None:
                client.close()
        if self.redis is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.redis.aclose(), 2)
        self.events.write("stop", "api stopped")
        await self.writer.drain(2.0)
        if self.writer.db is not None:
            await self.writer.db.close()

    async def ingest_stats(self) -> IngestStatsOut:
        """Ingest stats from Redis, or the last known ones (``degraded``) when Redis is down."""
        fields: dict[str, str] | None = None
        degraded = False
        if self.redis is not None and not self.degraded:
            try:
                fields = await asyncio.wait_for(self.redis.hgetall(INGEST_STATS_KEY), 1.0)
                self._ingest_stats = fields or self._ingest_stats
            except REDIS_ERRORS as exc:
                log.debug("ingest stats from Redis failed: %s", exc)
                degraded = True
        if not fields:
            fields = self._latest_ingest_stats()
        if not fields:
            return IngestStatsOut(available=False, degraded=degraded or self.degraded)
        return parse_ingest_stats(
            fields, degraded=degraded or self.degraded, stale_after_s=self.settings.ingest_stale_after_s
        )

    def health(self) -> HealthOut:
        """Health body (always served with 200 while the process is alive)."""
        ingest_ok = self.check_ingest()
        deps = (self.redis_status, self.pg_status)
        return HealthOut(
            status="ok" if dependencies_ok(deps) and ingest_ok is not False else "degraded",
            service=SERVICE,
            version=__version__,
            uptime_s=round(time.monotonic() - self.started_monotonic, 3),
            dependencies={s.name: DependencyOut.of(s) for s in (*deps, self.ingest_status)},
            vehicles=len(self.cache),
        )

    def status_counts(self, now: datetime, *, link_down: bool) -> dict[LinkStatus, int]:
        """Vehicles per link status (all ``offline`` while the ingest is down)."""
        if not link_down:
            return self.cache.status_counts(now)
        counts = dict.fromkeys(LinkStatus, 0)
        counts[LinkStatus.OFFLINE] = len(self.cache)
        return counts

    def metrics(self) -> list[Metric]:
        """Service-specific metric families."""
        link_down = self.ingest_down()
        return [
            vehicle_status_metric(self.status_counts(datetime.now(UTC), link_down=link_down)),
            GaugeMetricFamily("foresight_api_ws_clients", "Open WebSocket connections.", self.ws_clients),
            GaugeMetricFamily(
                "foresight_api_degraded",
                "1 while serving the last known state (Redis or the ingest is down).",
                int(self.degraded or link_down),
            ),
        ]


_DESCRIPTION = """
Foresight API: the state of the fleet for the dispatcher dashboard and the admin panel.

* `GET /api/vehicles` — all vehicles with `tr_id`, the last valid position (never 0, 0 for a lost GPS fix),
  speed, course and link status (online / stale / offline), the forecast of the predictor (route, risk,
  forecast delay at the nearest stop, P10–P90, p_late, open incident) and the derived features: current
  deviation from the plan, mean speed on the segment, dwell and idle time; `scheduled: false` — a vehicle
  outside the plan schedule (no route, risk `unknown`); `degraded: true` means Redis is unavailable or the
  ingest is down and this is the last known state (while the ingest is down every vehicle is `offline`);
* `GET /api/vehicles/{unit_id}` — one vehicle;
* `GET /api/routes` — routes derived from the plan, each direction as its own line along the GPS tracks;
* `GET /api/ingest/stats` — NDTP ingest counters (published by the ingest service; `available: false` when
  they are older than `ingest_stale_after_s`, i.e. the ingest is down);
* `GET /api/incidents`, `GET /api/incidents/{id}` — problem vehicles now (red first) and the incident card
  (deviation over 30 min, forecasts ahead, alerts);
* `GET /api/alerts`, `GET /api/predictions` — alert feed and forecast journal with the check against the fact;
* `GET /api/stringline` — time × stops of a route: plan, fact by the stop detector, forecast with P10–P90;
* `GET /api/metrics/horizon` — honesty of the forecasts on the stream (online MAE vs the baseline, lead
  time, late stops warned in advance, forecasts after the fact = 0);
* `GET /api/metrics/perf` — p95 latency of the pipeline, inference and tick, consumer lag, dependencies;
* `/api/replay/*` — control of the demo stream (proxy to the replayer);
* `/api/admin/*` — thresholds (applied on the fly), model versions and their activation, journal export
  (JSON / CSV), health of all services, vehicle directory;
* `WS /ws` — snapshot on connect, then deltas of changed vehicles, periodic full snapshots, the predictor's
  `alert` / `incident` / `prediction_closed` messages and a `clock` every second;
* `GET /health` — service and dependency (Redis, PostgreSQL, ingest) state, 200 while the process is alive;
* `GET /metrics` — Prometheus metrics.
"""


def create_app(settings: Settings | None = None, service: ApiService | None = None) -> FastAPI:
    """Build the API application; its lifespan runs the :class:`ApiService`.

    Args:
        settings: Configuration; defaults to :class:`Settings` from the environment.
        service: Pre-built service (tests inject fakes through it).

    Returns:
        The application; ``app.state.service`` is the service.
    """
    settings = settings or Settings()
    svc = service or ApiService(settings)
    cache = svc.cache

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(title="Foresight · API", version=__version__, description=_DESCRIPTION, lifespan=lifespan)
    app.state.service = svc
    app.state.registry = build_registry(
        FunctionCollector(svc.metrics),
        WriterCollector(svc.writer),
        DependencyCollector([svc.redis_status, svc.pg_status, svc.ingest_status]),
    )

    @app.get("/health", response_model=HealthOut, tags=["ops"], summary="Liveness and dependencies")
    async def health() -> HealthOut:
        return svc.health()

    @app.get("/api/vehicles", response_model=VehicleListOut, tags=["vehicles"], summary="All vehicles")
    async def list_vehicles(
        status: Annotated[LinkStatus | None, Query(description="Filter by link status.")] = None,
    ) -> VehicleListOut:
        now = datetime.now(UTC)
        link_down = svc.ingest_down()
        vehicles = [vehicle_out(cache, v, now, link_down=link_down) for v in cache.all()]
        if status is not None:
            vehicles = [v for v in vehicles if v.status == status]
        return VehicleListOut(
            count=len(vehicles),
            server_time=now,
            stream_time=cache.stream_time,
            status_counts=svc.status_counts(now, link_down=link_down),
            degraded=svc.degraded or link_down,
            synced_at=cache.synced_at,
            vehicles=vehicles,
        )

    @app.get(
        "/api/vehicles/{unit_id}",
        response_model=VehicleOut,
        tags=["vehicles"],
        summary="One vehicle",
        responses={404: {"description": "Unknown unit_id"}},
    )
    async def get_vehicle(unit_id: int) -> VehicleOut:
        record = cache.get(unit_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unit {unit_id} not found")
        return vehicle_out(cache, record, datetime.now(UTC), link_down=svc.ingest_down())

    @app.get("/api/ingest/stats", response_model=IngestStatsOut, tags=["ops"], summary="NDTP ingest counters")
    async def ingest_stats() -> IngestStatsOut:
        return await svc.ingest_stats()

    @app.get(
        "/api/routes",
        response_model=list[RouteOut],
        tags=["routes"],
        summary="Routes of the plan with their lines",
    )
    async def routes() -> list[RouteOut]:
        """Routes derived from the plan schedule of the forecasts (published by the predictor): stops in the
        order of a full trip, each direction as its own line along the real GPS tracks. Empty until the
        predictor has published its plan."""
        out = []
        for item in cache.routes or []:
            with contextlib.suppress(ValueError):
                out.append(RouteOut.model_validate(item))
        return out

    # ---- journal views of the dashboard ------------------------------------------------------------

    def by_tr() -> dict[int, VehicleRecord]:
        return {r.tr_id: r for r in cache.all() if r.tr_id is not None}

    def live(incident: dict[str, Any], records: dict[int, VehicleRecord]) -> dict[str, Any]:
        """An open incident with the live state: position, current deviation, level, time to the event."""
        out = dict(incident)
        if out.get("status", "open") != "open":
            return out
        tr_id = out.get("tr_id")
        fc = cache.forecast_of(tr_id) or {}
        record = records.get(tr_id) if tr_id is not None else None
        vehicle = dict(out.get("vehicle") or {})
        if record is not None:
            lat, lon = record.pos_lat, record.pos_lon
            if lat is None and record.valid and record.lat is not None:
                lat, lon = record.lat, record.lon
            vehicle.update(lat=lat, lon=lon, course_deg=record.course_deg, speed_kmh=record.speed_kmh)
        if fc.get("current_delay_s") is not None:
            vehicle["current_delay_s"] = fc["current_delay_s"]
        out["vehicle"] = vehicle
        if out.get("kind") == "delay" and fc.get("risk") in ("green", "yellow", "red"):
            out["risk"] = fc["risk"]
        target = out.get("target_stop") or {}
        planned = target.get("planned_at")
        if planned and out.get("pred_delay_s") is not None and cache.stream_time is not None:
            with contextlib.suppress(ValueError, TypeError):
                event = datetime.fromisoformat(planned) + timedelta(seconds=float(out["pred_delay_s"]))
                out["time_to_event_s"] = round((event - cache.stream_time).total_seconds(), 1)
        return out

    def active_ids() -> list[int]:
        ids: list[int] = []
        for tr_id in list(cache.forecasts):
            fc = cache.forecast_of(tr_id) or {}
            for i in fc.get("incident_ids") or ([fc["incident_id"]] if fc.get("incident_id") else []):
                with contextlib.suppress(TypeError, ValueError):
                    if int(i) not in ids:
                        ids.append(int(i))
        return ids

    @app.get(
        "/api/incidents",
        response_model=IncidentListOut,
        tags=["incidents"],
        summary="Problem vehicles (incidents)",
    )
    async def incidents(
        status: Annotated[
            Literal["active", "all"], Query(description="active — open now; all — the timeline's journal.")
        ] = "active",
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> dict[str, Any]:
        """Open incidents of the predictor (delay forecast yellow / red, bunching) with the live position
        of the vehicle: red first, then by p_late and the forecast delay. ``status=all`` adds the closed ones
        of the current stream timeline, newest first."""
        degraded = svc.degraded or not svc.forecast_live
        records = by_tr()
        ids = active_ids()
        missing = [i for i in ids if i not in svc.incidents]
        if missing:
            try:
                svc.incidents.update(await svc.journal.incidents_by_id(missing))
            except Exception as exc:
                log.debug("incidents from the journal: %s", exc)
                degraded = True
        items = sorted(
            (live(svc.incidents[i], records) for i in ids if i in svc.incidents), key=incident_sort_key
        )
        if status == "all":
            try:
                rows = await svc.journal.incidents(since=epoch_start(svc.epoch), limit=limit)
            except Exception as exc:
                log.debug("incidents from the journal: %s", exc)
                degraded = True
                rows = sorted(svc.incidents.values(), key=lambda i: str(i.get("opened_at")), reverse=True)
            seen = set(ids)
            items += [r for r in rows if r.get("incident_id") not in seen]
        items = items[:limit]
        return {"stream_time": cache.stream_time, "degraded": degraded, "count": len(items), "items": items}

    @app.get(
        "/api/incidents/{incident_id}",
        response_model=IncidentDetailOut,
        tags=["incidents"],
        summary="Incident card",
        responses={404: {"description": "Unknown incident"}},
    )
    async def incident(incident_id: int) -> dict[str, Any]:
        """The incident with the deviation of its vehicle over the last 30 min of stream time (stop passes
        restored by the detector and the current deviation), the forecasts of its latest tick and its
        alerts."""
        found = svc.incidents.get(incident_id)
        if found is None:
            try:
                found = (await svc.journal.incidents_by_id([incident_id])).get(incident_id)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"journal unavailable: {exc}") from exc
        if found is None:
            raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")
        out = live(found, by_tr())
        tr_id = out.get("tr_id")
        now = cache.stream_time
        history: list[dict[str, Any]] = []
        forecast: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            if now is not None and tr_id is not None:
                rows = await svc.journal.passages(
                    [tr_id], now - timedelta(minutes=30), now, since=epoch_start(svc.epoch)
                )
                history = sorted(
                    (
                        {"t": r["pass_time"], "delay_s": round(float(r["delay_s"]), 1)}
                        for r in rows
                        if r["delay_s"] is not None
                    ),
                    key=lambda p: p["t"],
                )
            current = (out.get("vehicle") or {}).get("current_delay_s")
            if now is not None and current is not None and out.get("status", "open") == "open":
                history.append({"t": now, "delay_s": current})
            if tr_id is not None and out.get("status", "open") == "open":
                forecast = [
                    {
                        "stop_id": p["target_stop_id"],
                        "name": p["target_stop_name"],
                        "planned_at": p["planned_at"],
                        "pred_delay_s": p["pred_delay_s"],
                        "p10": p["p10"],
                        "p90": p["p90"],
                    }
                    # the latest tick only: older open forecasts are frozen (their stops left the window) and
                    # sort first — with a small limit they would crowd the fresh ones out
                    for p in latest_tick(
                        await svc.journal.predictions(epoch=svc.epoch, tr_id=tr_id, status="open", limit=500)
                    )
                ]
            alerts = await svc.journal.alerts(since=None, incident_id=incident_id, limit=20)
        return {**out, "history": history, "forecast": forecast, "alerts": alerts}

    @app.get("/api/alerts", response_model=AlertListOut, tags=["incidents"], summary="Alert feed")
    async def alerts(
        since: Annotated[
            datetime | None, Query(description="Only alerts issued after (stream time).")
        ] = None,
        level: Annotated[Literal["yellow", "red"] | None, Query()] = None,
        tr_id: Annotated[int | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 100,
    ) -> dict[str, Any]:
        """Alerts of the current stream timeline, newest first: one per incident and level (opening,
        escalation to red), with the forecast at the moment of the alert and — once the stop is passed —
        the fact."""
        degraded = svc.degraded
        try:
            items = await svc.journal.alerts(
                since=epoch_start(svc.epoch), issued_after=since, level=level, tr_id=tr_id, limit=limit
            )
        except Exception as exc:
            log.debug("alerts from the journal: %s", exc)
            degraded = True
            items = [
                a
                for a in svc.feed.latest("alert", "alert", 2000)
                if (level is None or a.get("level") == level) and (tr_id is None or a.get("tr_id") == tr_id)
            ][:limit]
        return {"stream_time": cache.stream_time, "degraded": degraded, "count": len(items), "items": items}

    @app.get(
        "/api/predictions", response_model=PredictionListOut, tags=["forecasts"], summary="Forecast journal"
    )
    async def predictions(
        tr_id: Annotated[int | None, Query()] = None,
        status: Annotated[Literal["open", "closed"] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 100,
    ) -> dict[str, Any]:
        """Forecasts of the current stream timeline: open ones by plan time, closed ones (checked against the
        detected fact) newest first."""
        degraded = svc.degraded
        try:
            items = await svc.journal.predictions(epoch=svc.epoch, tr_id=tr_id, status=status, limit=limit)
        except Exception as exc:
            log.debug("predictions from the journal: %s", exc)
            degraded = True
            items = []
            if status in (None, "closed"):
                items = [
                    p
                    for p in svc.feed.latest("prediction_closed", "prediction", 2000)
                    if tr_id in (None, p.get("tr_id"))
                ][:limit]
        return {"stream_time": cache.stream_time, "degraded": degraded, "count": len(items), "items": items}

    @app.get(
        "/api/stringline", response_model=StringlineOut, tags=["routes"], summary="Stringline of a route"
    )
    async def stringline(
        route_id: Annotated[str, Query(description="Route, e.g. R3.")],
        from_: Annotated[
            datetime | None, Query(alias="from", description="From (stream time); default now − 60 min.")
        ] = None,
        to: Annotated[datetime | None, Query(description="To; default now + 30 min.")] = None,
    ) -> dict[str, Any]:
        """Time × stops of a route: the plan of its vehicles, their passes restored by the detector and the
        forecast ahead (from the current deviation to the forecast of the stop 10–15 min ahead, P10–P90)."""
        route = next((r for r in cache.routes or [] if r.get("route_id") == route_id), None)
        if route is None:
            raise HTTPException(status_code=404, detail=f"route {route_id} not found")
        now = cache.stream_time
        base = now or datetime.now(UTC)
        lo = from_ or base - timedelta(minutes=60)
        hi = to or base + timedelta(minutes=30)
        if lo.tzinfo is None:
            lo = lo.replace(tzinfo=UTC)
        if hi.tzinfo is None:
            hi = hi.replace(tzinfo=UTC)
        if hi <= lo or hi - lo > timedelta(hours=12):
            raise HTTPException(status_code=422, detail="from < to, at most 12 h")
        tr_ids = [int(t) for t in route.get("tr_ids") or []]
        degraded = svc.degraded or svc.schedule is None
        passages: list[Any] = []
        forecasts: list[dict[str, Any]] = []
        try:
            passages = await svc.journal.passages(tr_ids, lo, hi, since=epoch_start(svc.epoch))
            forecasts = await svc.journal.predictions(
                epoch=svc.epoch, tr_ids=tr_ids, status="open", limit=500
            )
        except Exception as exc:
            log.debug("stringline from the journal: %s", exc)
            degraded = True
        current = {t: (cache.forecast_of(t) or {}).get("current_delay_s") for t in tr_ids}
        positions = {}
        for t in tr_ids:
            nxt = (cache.forecast_of(t) or {}).get("next_stop") or {}
            if nxt.get("stop_id") is not None:
                positions[t] = int(nxt["stop_id"])
        out = build_stringline(
            route,
            svc.schedule.vehicles if svc.schedule is not None else {},
            lo.timestamp(),
            hi.timestamp(),
            passages=passages,
            forecasts=forecasts,
            current=current,
            positions=positions,
            now=now.timestamp() if now is not None else None,
        )
        out["degraded"] = degraded
        return out

    @app.post(
        "/api/whatif", response_model=WhatIfOut, tags=["routes"], summary="What-if of a dispatcher action"
    )
    async def whatif(request: WhatIfIn) -> dict[str, Any]:
        """Headways at the stops of a route over the horizon before and after an action: a reserve vehicle
        (``add_vehicle``: from ``params.from_stop_key`` at ``params.depart_at``) or holding a vehicle
        (``hold``: ``params.tr_id`` for ``params.hold_s``). The vehicles arrive with the deviation the
        forecasts expect; the metrics are the mean passenger wait, the largest headway, late stops and
        bunching pairs."""
        route = next((r for r in cache.routes or [] if r.get("route_id") == request.route_id), None)
        if route is None:
            raise HTTPException(status_code=404, detail=f"route {request.route_id} not found")
        if svc.schedule is None:
            raise HTTPException(status_code=503, detail="the plan schedule is not loaded")
        tr_ids = [int(t) for t in route.get("tr_ids") or []]
        problem = request_error(route, request.model_dump(mode="json"))
        if problem is not None:
            raise HTTPException(status_code=422, detail=problem)
        at = request.at or cache.stream_time
        if at is None:
            raise HTTPException(status_code=409, detail="no stream time yet: give `at`")
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        forecasts: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            forecasts = await svc.journal.predictions(
                epoch=svc.epoch, tr_ids=tr_ids, status="open", limit=500
            )
        current = {t: (cache.forecast_of(t) or {}).get("current_delay_s") for t in tr_ids}
        return run_whatif(
            route,
            svc.schedule.vehicles,
            request.model_dump(mode="json"),
            at.timestamp(),
            forecasts=forecasts,
            current=current,
        )

    @app.get(
        "/api/metrics/horizon",
        response_model=HorizonOut,
        tags=["metrics"],
        summary="Honesty of the forecasts on the stream",
    )
    async def horizon(
        window_s: Annotated[
            float, Query(gt=0, le=86400, description="Window of stream time, seconds.")
        ] = 3600,
    ) -> dict[str, Any]:
        """Forecasts closed with the detected fact: online MAE against the baseline «current deviation»,
        share of late stops warned in advance, lead time (pass − issue) histogram, MAE by hour, forecasts
        and alerts issued after the fact (must be 0). The window counts from the current stream time;
        ``total`` covers the whole timeline."""
        now = cache.stream_time
        try:
            out = await svc.journal.horizon(
                epoch=svc.epoch,
                since=epoch_start(svc.epoch),
                closed_after=now - timedelta(seconds=window_s) if now is not None else None,
                late_s=_LATE_S,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"journal unavailable: {exc}") from exc
        return {**out, "window_s": window_s, "stream_time": now, "degraded": svc.degraded}

    @app.get(
        "/api/metrics/perf", response_model=PerfOut, tags=["metrics"], summary="Performance of the pipeline"
    )
    async def perf() -> dict[str, Any]:
        """Latency and throughput over the last 5 min from the services' own metrics (the api scrapes the
        predictor and ml-service): packet → forecast state p95, inference and tick p95, consumer lag, vehicles
        online and the state of the dependencies."""
        p = svc.perf
        stats = await svc.ingest_stats()
        counts = svc.status_counts(datetime.now(UTC), link_down=svc.ingest_down())

        def ms(x: float | None) -> float | None:
            return None if x is None else round(x * 1000, 1)

        ml = p.state("ml")
        if ml == "up" and p.gauge("predictor", "foresight_dependency_up{dependency=ml-service}") == 0:
            ml = "degraded"  # ml-service answers us, but not the predictor
        predictor = p.state("predictor")
        if predictor == "up" and not svc.forecast_live:
            predictor = "degraded"
        e2e = p.p("predictor", "foresight_predictor_event_latency_seconds")
        return {
            "ingest_pps": round(stats.packets_per_s, 2) if stats.available else 0.0,
            "e2e_p95_s": None if e2e is None else round(e2e, 3),
            "inference_p95_ms": ms(p.p("ml", "foresight_ml_request_duration_seconds{endpoint=/predict}")),
            "tick_p95_ms": ms(p.p("predictor", "foresight_predictor_tick_seconds")),
            "tick_last_ms": ms(p.gauge("predictor", "foresight_predictor_tick_duration_seconds")),
            "consumer_lag": p.gauge("predictor", "foresight_consumer_lag"),
            "vehicles_online": counts.get(LinkStatus.ONLINE, 0),
            "vehicles": len(cache),
            "ws_clients": svc.ws_clients,
            "window_s": p.window_s,
            "model_version": (cache.forecast_status or {}).get("model_version"),
            "forecast_source": (cache.forecast_status or {}).get("source"),
            "deps": {
                "ingest": svc.ingest_status.state,
                "redis": svc.redis_status.state,
                "predictor": predictor,
                "ml": ml,
                "postgres": svc.pg_status.state,
            },
        }

    app.include_router(admin_router(svc))

    # ---- replay control (proxy to the replayer) ---------------------------------------------------

    async def replay(method: str, path: str, body: Any = None) -> Response:
        if svc.replayer is None:
            raise HTTPException(status_code=503, detail="replayer is not configured")
        payload = None if body is None else json.dumps(body).encode()
        try:
            status, data = await asyncio.wait_for(svc.replayer.request(method, path, payload), 10.0)
        except Exception as exc:
            svc.replayer.close()
            raise HTTPException(
                status_code=502, detail=f"replayer unavailable: {exc or type(exc).__name__}"
            ) from exc
        try:
            content = json.loads(data) if data else None
        except ValueError:
            content = {"detail": data.decode("utf-8", "replace")}
        return JSONResponse(content, status_code=status)

    @app.get("/api/replay/status", tags=["replay"], summary="Demo stream status")
    async def replay_status() -> Response:
        """Status of the replayer (split, speed, position in the day, loops)."""
        return await replay("GET", "/replay/status")

    @app.post("/api/replay/start", tags=["replay"], summary="Start the demo stream")
    async def replay_start(body: Annotated[dict[str, Any] | None, Body()] = None) -> Response:
        """Start (restart) the replay of a day as real NDTP; the body is the replayer's (split, speed, start,
        until, units, loop). A restart moves the stream clock back: the ingest and the predictor detect the
        jump and begin a new timeline (the old forecasts close as ``reset``)."""
        return await replay("POST", "/replay/start", body)

    @app.post("/api/replay/stop", tags=["replay"], summary="Stop the demo stream")
    async def replay_stop() -> Response:
        return await replay("POST", "/replay/stop")

    @app.post("/api/replay/speed", tags=["replay"], summary="Change the speed of the demo stream")
    async def replay_speed(body: Annotated[dict[str, Any], Body(examples=[{"speed": 30}])]) -> Response:
        return await replay("POST", "/replay/speed", body)

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics",
        response_class=Response,
        responses={200: {"content": {CONTENT_TYPE_LATEST: {}}, "description": "Prometheus text format"}},
    )
    async def metrics(request: Request) -> Response:
        return Response(generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws")
    async def ws_state(websocket: WebSocket) -> None:
        """Push vehicle state: a snapshot on connect, then deltas (see :class:`WsMessage`)."""
        await websocket.accept()
        svc.ws_clients += 1
        interval = settings.ws_interval_s
        last_version = -1
        last_status: dict[int, LinkStatus] = {}
        last_snapshot = float("-inf")
        last_degraded = svc.degraded

        def message(
            kind: Literal["snapshot", "delta"],
            vehicles: list[VehicleOut],
            version: int,
            now: datetime,
            degraded: bool,
        ) -> dict[str, Any]:
            msg = WsMessage(
                type=kind,
                version=version,
                server_time=now,
                stream_time=cache.stream_time,
                degraded=degraded,
                vehicles=vehicles,
            )
            return msg.model_dump(mode="json")

        last_event = svc.feed.seq  # events that came before the connect are in the REST lists
        last_clock = float("-inf")

        async def pump() -> None:
            nonlocal last_version, last_status, last_snapshot, last_degraded, last_event, last_clock
            while True:
                now = datetime.now(UTC)
                version = cache.version
                link_down = svc.ingest_down()
                degraded = svc.degraded or link_down
                outs = {v.unit_id: vehicle_out(cache, v, now, link_down=link_down) for v in cache.all()}
                if (
                    time.monotonic() - last_snapshot >= settings.ws_snapshot_every_s
                    or degraded != last_degraded
                ):
                    # a snapshot also tells clients at once that the state became (or stopped being) stale
                    snapshot = message("snapshot", list(outs.values()), version, now, degraded)
                    await websocket.send_json(snapshot)
                    last_snapshot = time.monotonic()
                    last_degraded = degraded
                else:
                    changed = {v.unit_id for v in cache.changed_since(last_version)}
                    vehicles = [
                        out
                        for unit_id, out in outs.items()
                        if unit_id in changed or last_status.get(unit_id) != out.status
                    ]
                    if vehicles:
                        await websocket.send_json(message("delta", vehicles, version, now, degraded))
                last_status = {unit_id: out.status for unit_id, out in outs.items()}
                last_version = version
                for seq, event in svc.feed.since(last_event):
                    await websocket.send_json(event)
                    last_event = seq
                if time.monotonic() - last_clock >= _WS_CLOCK_S:
                    await websocket.send_json(
                        {
                            "type": "clock",
                            "stream_time": iso(cache.stream_time),
                            "epoch": svc.epoch or 0,
                            "server_time": iso(now),
                            "degraded": degraded,
                        }
                    )
                    last_clock = time.monotonic()
                # a vehicle change or a predictor event (on_event notifies the cache) wakes the loop early
                if await cache.wait_for_change(min(interval, _WS_CLOCK_S)):
                    await asyncio.sleep(_WS_MIN_GAP_S)

        async def watch_client() -> None:
            # client messages are ignored for now (future: subscriptions); returns when the client leaves
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass

        tasks = {asyncio.create_task(pump()), asyncio.create_task(watch_client())}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                expected = WebSocketDisconnect | RuntimeError | ConnectionError
                if exc is not None and not isinstance(exc, expected):
                    log.warning("WebSocket /ws failed: %r", exc)
        finally:
            try:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                with contextlib.suppress(Exception):
                    await websocket.close()
            finally:
                svc.ws_clients -= 1  # last: 0 means the handlers are done (tests wait on it before leaving)

    return app


def main() -> None:
    """Entry point of ``python -m backend.api``."""
    settings = Settings()
    run_http("backend.api:create_app", settings.http_host, settings.http_port, settings.log_level)


if __name__ == "__main__":
    main()
