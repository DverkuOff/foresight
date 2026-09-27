"""Ingest service: NDTP TCP server -> Redis (telemetry stream, hot vehicle state, stats).

Run with ``python -m backend.ingest``. HTTP (port 8001 by default): ``/health`` (503 while NDTP is not
listening), ``/metrics``, ``/api/ingest/stats``.

Redis outages never stop the NDTP intake: events wait in a bounded buffer and are written after recovery
(see :class:`~backend.bus.TelemetryPublisher`). The ingest owns the stream clock (:mod:`backend.clock`): it
detects source restarts and stamps every stream event with the clock and its epoch.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend import __version__
from backend.bus import RedisFactory, TelemetryPublisher, redis_factory
from backend.clock import ClockJump, to_plan_day
from backend.config import Settings
from backend.db import BufferedWriter, Database, ServiceEventLog
from backend.metrics import (
    ClockCollector,
    DependencyCollector,
    NdtpCollector,
    PublisherCollector,
    WriterCollector,
    build_registry,
)
from backend.ndtp_server import NdtpServer
from backend.runtime import Backoff, DependencyStatus, TaskSet, dependencies_ok, run_http
from backend.schemas import ConnectionOut, DependencyOut, HealthOut, IngestStatsOut
from backend.state import StateStore
from backend.unitmap import load_unit_map

log = logging.getLogger(__name__)

SERVICE = "ingest"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class IngestService:
    """NDTP server, state store, Redis publisher and service journal of one ingest instance.

    Args:
        settings: Configuration.
        factory: Redis client factory (default: from ``settings.redis_url``).
        unit_map: ``unit_id -> tr_id`` (default: read from the dataset, see :mod:`backend.unitmap`).
        database: PostgreSQL access (default: from ``settings.database_url``; empty URL disables it).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        factory: RedisFactory | None = None,
        unit_map: Mapping[int, int] | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings
        self.started_monotonic = time.monotonic()
        if unit_map is None:
            unit_map = load_unit_map(settings.unit_map_dir, settings.splits)
        self.unit_map = unit_map
        self.store = StateStore(
            stale_after_s=settings.stale_after_s,
            offline_after_s=settings.offline_after_s,
            stream_clock=settings.stream_clock(on_jump=self._on_clock_jump),
        )
        self.redis_status = DependencyStatus("redis")
        self.pg_status = DependencyStatus("postgres")
        self.publisher = TelemetryPublisher(
            self.store,
            factory or redis_factory(settings.redis_url, settings.redis_timeout_s),
            self.redis_status,
            unit_map=unit_map,
            max_buffer=settings.bus_buffer_max,
            batch_size=settings.bus_batch_size,
            stream_maxlen=settings.stream_maxlen,
            vehicle_ttl_s=settings.vehicle_ttl_s,
            flush_interval_s=settings.bus_flush_interval_s,
            stats_interval_s=settings.stats_interval_s,
            stats_provider=self.stats_fields,
            backoff=Backoff(settings.backoff_initial_s, settings.backoff_max_s),
        )
        self.ndtp = NdtpServer(
            self.store,
            settings.ndtp_host,
            settings.ndtp_port,
            max_data_size=settings.ndtp_max_data_size,
            max_crc_errors=settings.ndtp_max_crc_errors,
            read_timeout_s=settings.ndtp_read_timeout_s,
            first_frame_timeout_s=settings.ndtp_first_frame_timeout_s,
            max_garbage_bytes=settings.ndtp_max_garbage_bytes,
            max_connections=settings.ndtp_max_connections,
            listeners=[self.publisher.on_packet],
            time_map=(
                partial(
                    to_plan_day, day=settings.realtime_plan_day, tz_offset_s=settings.realtime_tz_offset_s
                )
                if settings.realtime_plan_day is not None
                else None
            ),
        )
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
        self.events = ServiceEventLog(self.writer, SERVICE, settings.instance)
        self.events.watch(self.redis_status)
        self.events.watch(self.pg_status)
        self._tasks = TaskSet()

    def _on_clock_jump(self, jump: ClockJump) -> None:
        clock = self.store.stream_clock
        self.events.write(
            "clock_reset",
            f"stream clock jumped {'back' if jump.back else 'forward'} to {clock.time}",
            details={
                "from": _iso(datetime.fromtimestamp(jump.before, UTC) if jump.before is not None else None),
                "to": _iso(clock.time),
                "epoch": jump.epoch,
                "reason": jump.reason,
                "devices": jump.devices,
            },
        )

    async def start(self) -> None:
        """Continue the saved stream clock, start the writer, the publisher and the NDTP server."""
        self.started_monotonic = time.monotonic()
        await self.publisher.load_clock()
        self._tasks.spawn(self.writer.run(), "db-writer")
        self._tasks.spawn(self.publisher.run(), "redis-publisher")
        if self.settings.ndtp_enabled:
            await self.ndtp.start()
        self.events.write(
            "start",
            f"ingest {__version__} started",
            details={
                "ndtp_port": self.ndtp.port,
                "units_mapped": len(self.unit_map),
                "stream_time": _iso(self.store.stream_time),
                "clock_epoch": self.store.stream_clock.epoch,
            },
        )

    async def stop(self) -> None:
        """Stop accepting NDTP, flush Redis and the journal (bounded time), release connections."""
        await self.ndtp.stop()
        await self._tasks.cancel()
        await self.publisher.drain(2.0)
        await self.publisher.close()
        self.events.write("stop", "ingest stopped")
        await self.writer.drain(2.0)
        if self.writer.db is not None:
            await self.writer.db.close()

    # ---- views -----------------------------------------------------------------------------------

    def stats(self) -> IngestStatsOut:
        """Current ingest counters."""
        s = self.ndtp.stats
        p = self.publisher
        clock = self.store.stream_clock
        # the oldest connections only: a flood of connections must not grow the stats hash without bound
        listed = itertools.islice(self.ndtp.connections.values(), self.settings.stats_connections_max)
        return IngestStatsOut(
            updated_at=datetime.now(UTC),
            stream_time=clock.time,
            clock_epoch=clock.epoch,
            clock_resets=clock.resets,
            clock_garbage=clock.garbage,
            clock_off_timeline=clock.off_timeline,
            listening=self.ndtp.listening,
            port=self.ndtp.port,
            started_at=self.ndtp.started_at,
            connections_active=s.connections_active,
            connections_total=s.connections_total,
            disconnects=s.disconnects,
            read_timeouts=s.read_timeouts,
            bytes_received=s.bytes_received,
            bytes_discarded=s.bytes_discarded,
            frames=s.frames,
            handshakes=s.handshakes,
            realtime_packets=s.realtime_packets,
            nav_records=s.nav_records,
            other_frames=s.other_frames,
            crc_errors=s.crc_errors,
            bad_headers=s.bad_headers,
            parse_errors=s.parse_errors,
            unknown_cells=s.unknown_cells,
            listener_errors=s.listener_errors,
            abusive_disconnects=s.abusive_disconnects,
            first_frame_timeouts=s.first_frame_timeouts,
            rejected_connections=s.rejected_connections,
            packets_per_s=round(s.packets_rate.rate(), 3),
            redis=self.redis_status.state,  # type: ignore[arg-type]
            bus_published=p.published,
            bus_buffered=p.buffered,
            bus_evicted=p.evicted,
            unmapped_events=p.unmapped,
            connections=[
                ConnectionOut(
                    conn_id=c.conn_id,
                    peer=c.peer,
                    unit_id=c.unit_id,
                    connected_at=c.connected_at,
                    frames=c.frames,
                    last_frame_at=c.last_frame_at,
                )
                for c in listed
            ],
        )

    def stats_fields(self) -> dict[str, str]:
        """:meth:`stats` flattened to strings for the ``foresight:ingest:stats`` hash."""
        data = self.stats().model_dump(mode="json")
        fields: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                continue
            if key == "connections":
                fields[key] = json.dumps(value, separators=(",", ":"))
            elif isinstance(value, bool):
                fields[key] = "1" if value else "0"
            else:
                fields[key] = str(value)
        fields["redis"] = "up"  # these fields are only ever read back from Redis
        return fields

    def health(self) -> tuple[HealthOut, int]:
        """Health body and HTTP status (503 while NDTP is expected but not listening)."""
        listening = self.ndtp.listening
        ndtp_ok = listening or not self.settings.ndtp_enabled
        deps_ok = dependencies_ok((self.redis_status, self.pg_status))
        body = HealthOut(
            status="ok" if ndtp_ok and deps_ok else "degraded",
            service=SERVICE,
            version=__version__,
            uptime_s=round(time.monotonic() - self.started_monotonic, 3),
            dependencies={s.name: DependencyOut.of(s) for s in (self.redis_status, self.pg_status)},
            ndtp_listening=listening,
            ndtp_port=self.ndtp.port,
            vehicles=len(self.store),
        )
        return body, 200 if ndtp_ok else 503


_DESCRIPTION = """
Foresight ingest: receives NDTP telemetry over TCP (port 9201 by default) and publishes it to Redis —
the `foresight:telemetry` stream, hot vehicle state `foresight:vehicle:{unit_id}` and ingest stats.

* `GET /health` — 503 while the NDTP server is not listening; Redis / PostgreSQL state in the body;
* `GET /api/ingest/stats` — NDTP counters and the Redis publisher buffer;
* `GET /metrics` — Prometheus metrics.
"""


def create_app(settings: Settings | None = None, service: IngestService | None = None) -> FastAPI:
    """Build the ingest HTTP application; its lifespan runs the :class:`IngestService`.

    Args:
        settings: Configuration; defaults to :class:`Settings` from the environment.
        service: Pre-built service (tests inject fakes through it).

    Returns:
        The application; ``app.state.service`` is the service.
    """
    settings = settings or Settings()
    svc = service or IngestService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(
        title="Foresight · Ingest", version=__version__, description=_DESCRIPTION, lifespan=lifespan
    )
    app.state.service = svc
    app.state.registry = build_registry(
        NdtpCollector(svc.ndtp, svc.store),
        ClockCollector(svc.store.stream_clock),
        PublisherCollector(svc.publisher),
        WriterCollector(svc.writer),
        DependencyCollector([svc.redis_status, svc.pg_status]),
    )

    @app.get(
        "/health",
        response_model=HealthOut,
        tags=["ops"],
        summary="Liveness and readiness",
        responses={503: {"model": HealthOut, "description": "NDTP server is not listening"}},
    )
    async def health(response: Response) -> HealthOut:
        body, code = svc.health()
        response.status_code = code
        return body

    @app.get("/api/ingest/stats", response_model=IngestStatsOut, tags=["ops"], summary="NDTP ingest counters")
    async def ingest_stats() -> IngestStatsOut:
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
    """Entry point of ``python -m backend.ingest``."""
    settings = Settings()
    run_http("backend.ingest:create_app", settings.http_host, settings.ingest_http_port, settings.log_level)


if __name__ == "__main__":
    main()
