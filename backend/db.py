"""PostgreSQL: idempotent schema, asyncpg pool and a buffered writer that survives database outages.

Tables (created with ``CREATE TABLE IF NOT EXISTS`` on every start, under an advisory lock so several
services can start at once; columns added later come with ``ALTER TABLE … ADD COLUMN IF NOT EXISTS``, so an
existing volume is migrated in place):

* ``predictions`` — one row per forecast «vehicle × target stop»: the first issue (``issued_at``), the latest
  update (``pred_delay_s``, ``updates``) and, once the detector confirms the target stop, the outcome;
* ``prediction_updates`` — every forecast of every tick (the log behind the online validation);
* ``alerts`` — risk alerts (``kind`` delay / bunching), updated on escalation and when the fact comes;
* ``incidents`` — open / closed incidents of the dispatcher (delay of a vehicle, bus bunching);
* ``stop_passages`` — stop passages restored by the detector on the stream (online labels for retraining);
* ``model_versions`` — registry of model versions and their quality;
* ``settings`` — tunable settings such as risk thresholds (JSON values);
* ``service_events`` — journal of service events: start, stop, dependency degradation and recovery, stream
  clock jumps.

:class:`BufferedWriter` never blocks the caller: rows go to a bounded in-memory queue and are inserted in
batches; while PostgreSQL is down they stay queued (the oldest are dropped beyond the limit and counted)
and are written after recovery. Rows of the tables in :data:`UPSERT_KEYS` that carry the key are upserts
(``ON CONFLICT (key) DO UPDATE``): the predictor rewrites the whole current row of a forecast, an alert or an
incident, so a repeated or reordered batch leaves the latest state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import asyncpg
import asyncpg.exceptions as pgerr

from backend.runtime import Backoff, BoundedBuffer, DependencyStatus

log = logging.getLogger(__name__)

SCHEMA_LOCK_ID = 7_424_242_001
"""``pg_advisory_xact_lock`` key that serialises schema creation between services."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    tr_id             BIGINT NOT NULL,
    unit_id           BIGINT,
    route_id          TEXT,
    target_stop_id    BIGINT NOT NULL,
    target_stop_name  TEXT,
    target_time_begin TIMESTAMPTZ NOT NULL,
    issued_at         TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ,
    updates           INTEGER NOT NULL DEFAULT 0,
    lead_s            DOUBLE PRECISION,
    pred_delay_s      DOUBLE PRECISION NOT NULL,
    first_pred_delay_s DOUBLE PRECISION,
    p10               DOUBLE PRECISION,
    p50               DOUBLE PRECISION,
    p90               DOUBLE PRECISION,
    p_late            DOUBLE PRECISION,
    risk              TEXT,
    cause             TEXT,
    cause_detail      JSONB,
    model_version     TEXT,
    source            TEXT,
    degraded          BOOLEAN NOT NULL DEFAULT FALSE,
    cur_dev_s         DOUBLE PRECISION,
    epoch             BIGINT,
    status            TEXT NOT NULL DEFAULT 'open',
    actual_delay_s    DOUBLE PRECISION,
    abs_error_s       DOUBLE PRECISION,
    pass_time         TIMESTAMPTZ,
    actual_lead_s     DOUBLE PRECISION,
    retroactive       BOOLEAN,
    closed_at         TIMESTAMPTZ,
    alert_level       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS route_id TEXT,
    ADD COLUMN IF NOT EXISTS target_stop_name TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updates INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS lead_s DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS first_pred_delay_s DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS risk TEXT,
    ADD COLUMN IF NOT EXISTS cause_detail JSONB,
    ADD COLUMN IF NOT EXISTS source TEXT,
    ADD COLUMN IF NOT EXISTS cur_dev_s DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS epoch BIGINT,
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'open',
    ADD COLUMN IF NOT EXISTS abs_error_s DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pass_time TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS actual_lead_s DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS retroactive BOOLEAN,
    ADD COLUMN IF NOT EXISTS alert_level TEXT;
CREATE INDEX IF NOT EXISTS predictions_issued_at_idx ON predictions (issued_at);
CREATE INDEX IF NOT EXISTS predictions_tr_id_idx ON predictions (tr_id, issued_at);
CREATE INDEX IF NOT EXISTS predictions_open_idx ON predictions (tr_id, target_stop_id)
    WHERE closed_at IS NULL;
CREATE INDEX IF NOT EXISTS predictions_closed_at_idx ON predictions (closed_at);

CREATE TABLE IF NOT EXISTS prediction_updates (
    id                BIGSERIAL PRIMARY KEY,
    prediction_id     BIGINT NOT NULL,
    tr_id             BIGINT NOT NULL,
    target_stop_id    BIGINT NOT NULL,
    target_time_begin TIMESTAMPTZ NOT NULL,
    tick_at           TIMESTAMPTZ NOT NULL,
    lead_s            DOUBLE PRECISION,
    pred_delay_s      DOUBLE PRECISION NOT NULL,
    p10               DOUBLE PRECISION,
    p50               DOUBLE PRECISION,
    p90               DOUBLE PRECISION,
    p_late            DOUBLE PRECISION,
    source            TEXT,
    cur_dev_s         DOUBLE PRECISION,
    model_version     TEXT,
    epoch             BIGINT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS prediction_updates_tick_idx ON prediction_updates (tick_at);
CREATE INDEX IF NOT EXISTS prediction_updates_tr_id_idx ON prediction_updates (tr_id, target_stop_id);
-- epoch is a millisecond wall-clock stamp (see backend/clock.py); older schemas declared it INTEGER
ALTER TABLE predictions ALTER COLUMN epoch TYPE BIGINT;
ALTER TABLE prediction_updates ALTER COLUMN epoch TYPE BIGINT;

CREATE TABLE IF NOT EXISTS alerts (
    id                BIGSERIAL PRIMARY KEY,
    kind              TEXT NOT NULL DEFAULT 'delay',
    level             TEXT,
    severity          TEXT NOT NULL DEFAULT 'warning',
    status            TEXT NOT NULL DEFAULT 'open',
    tr_id             BIGINT,
    unit_id           BIGINT,
    route_id          TEXT,
    stop_id           BIGINT,
    target_time_begin TIMESTAMPTZ,
    issued_at         TIMESTAMPTZ NOT NULL,
    prediction_id     BIGINT,
    pred_delay_s      DOUBLE PRECISION,
    p10               DOUBLE PRECISION,
    p90               DOUBLE PRECISION,
    p_late            DOUBLE PRECISION,
    cause             TEXT,
    recommendation    TEXT,
    segment           JSONB,
    details           JSONB,
    model_version     TEXT,
    degraded          BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged      BOOLEAN NOT NULL DEFAULT FALSE,
    retroactive       BOOLEAN,
    actual_delay_s    DOUBLE PRECISION,
    closed_at         TIMESTAMPTZ,
    incident_id       BIGINT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS level TEXT,
    ADD COLUMN IF NOT EXISTS route_id TEXT,
    ADD COLUMN IF NOT EXISTS acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS retroactive BOOLEAN,
    ADD COLUMN IF NOT EXISTS incident_id BIGINT;
CREATE INDEX IF NOT EXISTS alerts_issued_at_idx ON alerts (issued_at);
CREATE INDEX IF NOT EXISTS alerts_tr_id_idx ON alerts (tr_id, issued_at);

CREATE TABLE IF NOT EXISTS incidents (
    id                BIGINT PRIMARY KEY,
    kind              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'open',
    tr_id             BIGINT,
    unit_id           BIGINT,
    route_id          TEXT,
    related_tr_id     BIGINT,
    risk              TEXT,
    target_stop_id    BIGINT,
    planned_at        TIMESTAMPTZ,
    pred_delay_s      DOUBLE PRECISION,
    p10               DOUBLE PRECISION,
    p90               DOUBLE PRECISION,
    p_late            DOUBLE PRECISION,
    cause             TEXT,
    details           JSONB,
    opened_at         TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ,
    closed_at         TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS incidents_opened_at_idx ON incidents (opened_at);
CREATE INDEX IF NOT EXISTS incidents_tr_id_idx ON incidents (tr_id, opened_at);
CREATE INDEX IF NOT EXISTS incidents_open_idx ON incidents (status) WHERE closed_at IS NULL;

CREATE TABLE IF NOT EXISTS stop_passages (
    id                BIGSERIAL PRIMARY KEY,
    tr_id             BIGINT NOT NULL,
    unit_id           BIGINT,
    stop_id           BIGINT NOT NULL,
    time_begin        TIMESTAMPTZ NOT NULL,
    pass_time         TIMESTAMPTZ,
    delay_s           DOUBLE PRECISION,
    dist_m            DOUBLE PRECISION,
    confirmed_at      TIMESTAMPTZ NOT NULL,
    matched           BOOLEAN NOT NULL DEFAULT TRUE,
    source            TEXT NOT NULL DEFAULT 'detector',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS stop_passages_tr_id_idx ON stop_passages (tr_id, time_begin);
CREATE INDEX IF NOT EXISTS stop_passages_confirmed_idx ON stop_passages (confirmed_at);

CREATE TABLE IF NOT EXISTS model_versions (
    version           TEXT PRIMARY KEY,
    kind              TEXT NOT NULL DEFAULT 'catboost',
    active            BOOLEAN NOT NULL DEFAULT FALSE,
    cv_mae            DOUBLE PRECISION,
    test_mae          DOUBLE PRECISION,
    online_mae        DOUBLE PRECISION,
    artifact_uri      TEXT,
    params            JSONB,
    metrics           JSONB,
    notes             TEXT,
    trained_at        TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_active_idx ON model_versions ((TRUE)) WHERE active;

CREATE TABLE IF NOT EXISTS settings (
    key               TEXT PRIMARY KEY,
    value             JSONB NOT NULL,
    description       TEXT,
    updated_by        TEXT,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO settings (key, value, description) VALUES
    ('risk_thresholds',
     '{"green_delay_s": 60, "red_delay_s": 120, "green_p_late": 0.3, "red_p_late": 0.6}',
     'Map colours: green below both green thresholds, red above either red threshold'),
    ('alert_thresholds',
     '{"min_level": "yellow", "min_p_late": 0.0, "pred_delay_s": 120, "p_late": 0.6, "late_s": 120}',
     'Alert on the first move of a forecast to min_level (yellow / red) or higher; late_s defines p_late')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS service_events (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL,
    service           TEXT NOT NULL,
    instance          TEXT,
    kind              TEXT NOT NULL,
    dependency        TEXT,
    message           TEXT,
    details           JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS service_events_ts_idx ON service_events (ts);
CREATE INDEX IF NOT EXISTS service_events_service_idx ON service_events (service, ts);
"""


def _columns(names: str) -> frozenset[str]:
    return frozenset(names.split())


TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "predictions": _columns(
        "id tr_id unit_id route_id target_stop_id target_stop_name target_time_begin issued_at updated_at "
        "updates lead_s pred_delay_s first_pred_delay_s p10 p50 p90 p_late risk cause cause_detail "
        "model_version source degraded cur_dev_s epoch status actual_delay_s abs_error_s pass_time "
        "actual_lead_s retroactive closed_at alert_level"
    ),
    "prediction_updates": _columns(
        "prediction_id tr_id target_stop_id target_time_begin tick_at lead_s pred_delay_s p10 p50 p90 "
        "p_late source cur_dev_s model_version epoch"
    ),
    "alerts": _columns(
        "id kind level severity status tr_id unit_id route_id stop_id target_time_begin issued_at "
        "prediction_id pred_delay_s p10 p90 p_late cause recommendation segment details model_version "
        "degraded retroactive actual_delay_s closed_at incident_id"
    ),
    "incidents": _columns(
        "id kind status tr_id unit_id route_id related_tr_id risk target_stop_id planned_at pred_delay_s "
        "p10 p90 p_late cause details opened_at updated_at closed_at"
    ),
    "stop_passages": _columns(
        "tr_id unit_id stop_id time_begin pass_time delay_s dist_m confirmed_at matched source"
    ),
    "model_versions": _columns(
        "version kind active cv_mae test_mae online_mae artifact_uri params metrics notes trained_at"
    ),
    "settings": _columns("key value description updated_by updated_at"),
    "service_events": _columns("ts service instance kind dependency message details"),
}
"""Columns the :class:`BufferedWriter` may insert, per table (names are interpolated into SQL)."""

UPSERT_KEYS: dict[str, str] = {
    "predictions": "id",
    "alerts": "id",
    "incidents": "id",
    "model_versions": "version",
}
"""Tables whose rows are upserted when they carry this key column (the latest write wins); other rows —
and rows without the key — are plain inserts that skip conflicts."""

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    TimeoutError,
    asyncpg.InterfaceError,
    pgerr.PostgresConnectionError,
    pgerr.OperatorInterventionError,
    pgerr.InsufficientResourcesError,
)
"""Errors that mean "PostgreSQL is unreachable or restarting": retried later, rows stay buffered."""


def is_transient(exc: BaseException) -> bool:
    """Whether a database error is an outage (retry) rather than bad data (drop the row).

    asyncpg reports invalid query arguments as ``DataError(InterfaceError, ValueError)``; those are data
    errors even though they derive from ``InterfaceError``.
    """
    if isinstance(exc, ValueError | TypeError):
        return False
    return isinstance(exc, TRANSIENT_ERRORS)


def _adapt(value: Any) -> Any:
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, default=str, ensure_ascii=False)
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def insert_sql(table: str, columns: Sequence[str]) -> str:
    """``INSERT`` statement for validated table and column names.

    A row of a table in :data:`UPSERT_KEYS` that carries the key is an upsert: every other given column is
    overwritten on conflict (columns not given, e.g. ``acknowledged`` of an alert, are kept).
    """
    placeholders = ", ".join(f"${i}" for i in range(1, len(columns) + 1))
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    key = UPSERT_KEYS.get(table)
    rest = [c for c in columns if c != key]
    if key is None or key not in columns or not rest:
        return f"{head} ON CONFLICT DO NOTHING"
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in rest)
    return f"{head} ON CONFLICT ({key}) DO UPDATE SET {updates}"


class Database:
    """asyncpg pool that applies the schema on connect and can be dropped after an outage.

    Args:
        dsn: PostgreSQL DSN.
        min_size: Pool minimum size.
        max_size: Pool maximum size.
        timeout_s: Connect / acquire timeout.
        command_timeout_s: Statement timeout on the client side.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 4,
        timeout_s: float = 5.0,
        command_timeout_s: float = 10.0,
    ) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.timeout_s = timeout_s
        self.command_timeout_s = command_timeout_s
        self._pool: asyncpg.Pool | None = None

    @property
    def connected(self) -> bool:
        """Whether a pool exists (it may still fail on the next query)."""
        return self._pool is not None

    @property
    def pool(self) -> asyncpg.Pool:
        """The pool.

        Raises:
            asyncpg.InterfaceError: Not connected.
        """
        if self._pool is None:
            raise asyncpg.InterfaceError("database is not connected")
        return self._pool

    async def connect(self) -> None:
        """Create the pool and apply the schema (no-op when already connected)."""
        if self._pool is not None:
            return
        pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            timeout=self.timeout_s,
            command_timeout=self.command_timeout_s,
            max_inactive_connection_lifetime=60.0,
        )
        try:
            await apply_schema(pool, self.timeout_s)
        except BaseException:
            pool.terminate()
            raise
        self._pool = pool

    def reset(self) -> None:
        """Drop the pool after a connection failure; the next :meth:`connect` makes a new one."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    async def close(self) -> None:
        """Close the pool gracefully (terminate if that takes too long)."""
        pool, self._pool = self._pool, None
        if pool is None:
            return
        try:
            await asyncio.wait_for(pool.close(), 5)
        except Exception:
            pool.terminate()

    async def ping(self) -> None:
        """Run ``SELECT 1``."""
        await self.pool.fetchval("SELECT 1", timeout=self.timeout_s)

    async def insert_groups(
        self, groups: Sequence[tuple[str, tuple[str, ...], list[tuple[Any, ...]]]]
    ) -> None:
        """Insert several groups of rows in one transaction.

        Args:
            groups: ``(table, columns, rows)`` with validated names.
        """
        async with self.pool.acquire(timeout=self.timeout_s) as con, con.transaction():
            for table, columns, rows in groups:
                await con.executemany(insert_sql(table, columns), rows)

    async def insert_one(self, table: str, columns: tuple[str, ...], row: tuple[Any, ...]) -> None:
        """Insert one row (autocommit)."""
        await self.pool.execute(insert_sql(table, columns), *row, timeout=self.command_timeout_s)

    async def fetch_settings(self, keys: Sequence[str]) -> dict[str, Any]:
        """Values of the ``settings`` table (JSON decoded) for the given keys; missing keys are absent.

        Raises:
            Exception: The database is not connected or unreachable.
        """
        rows = await self.pool.fetch(
            "SELECT key, value FROM settings WHERE key = ANY($1::text[])",
            list(keys),
            timeout=self.timeout_s,
        )
        out: dict[str, Any] = {}
        for row in rows:
            value = row["value"]
            out[row["key"]] = json.loads(value) if isinstance(value, str) else value
        return out


async def apply_schema(pool: asyncpg.Pool, timeout_s: float = 5.0) -> None:
    """Create tables and indexes if missing (idempotent, serialised by an advisory lock)."""
    async with pool.acquire(timeout=timeout_s) as con, con.transaction():
        await con.execute("SELECT pg_advisory_xact_lock($1)", SCHEMA_LOCK_ID)
        await con.execute(SCHEMA_SQL)


_Row = tuple[str, tuple[str, ...], tuple[Any, ...]]


class BufferedWriter:
    """Insert rows in batches, buffering them in memory while PostgreSQL is unavailable.

    Args:
        db: Database, or ``None`` when PostgreSQL is disabled (rows are counted as dropped).
        status: PostgreSQL dependency status to report to (it also gets periodic health probes).
        max_buffer: Rows kept while the database is down; the oldest are dropped beyond it.
        batch_size: Rows per transaction.
        flush_interval_s: Idle wake-up period.
        health_interval_s: Period of ``SELECT 1`` probes when there is nothing to write.
        backoff: Reconnect delays.
    """

    def __init__(
        self,
        db: Database | None,
        status: DependencyStatus,
        *,
        max_buffer: int = 50_000,
        batch_size: int = 500,
        flush_interval_s: float = 0.5,
        health_interval_s: float = 5.0,
        backoff: Backoff | None = None,
    ) -> None:
        self.db = db
        self.status = status
        self.status.enabled = db is not None
        self.buffer: BoundedBuffer[_Row] = BoundedBuffer(max_buffer)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.health_interval_s = health_interval_s
        self.backoff = backoff or Backoff()
        self.written = 0
        self.rejected = 0
        self.disabled_drops = 0
        self.errors = 0
        self.batches = 0
        self._wake = asyncio.Event()
        self._last_ok = float("-inf")

    @property
    def buffered(self) -> int:
        """Rows waiting to be written."""
        return len(self.buffer)

    @property
    def dropped(self) -> int:
        """Rows lost: evicted from a full buffer or written while the database is disabled."""
        return self.buffer.evicted + self.disabled_drops

    def write(self, table: str, row: Mapping[str, Any]) -> None:
        """Queue one row for insertion (never blocks).

        Args:
            table: Table name (see :data:`TABLE_COLUMNS`).
            row: Column values; dicts and lists go to JSONB columns as JSON.

        Raises:
            ValueError: Unknown table or column (a programming error).
        """
        allowed = TABLE_COLUMNS.get(table)
        if allowed is None:
            raise ValueError(f"unknown table {table!r}")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"unknown columns for {table}: {sorted(unknown)}")
        if self.db is None:
            self.disabled_drops += 1
            return
        columns = tuple(row)
        self.buffer.append((table, columns, tuple(_adapt(row[c]) for c in columns)))
        self._wake.set()

    async def run(self) -> None:
        """Writer loop: runs until cancelled, reconnecting with backoff while PostgreSQL is down."""
        if self.db is None:
            return
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self.flush_interval_s)
            self._wake.clear()
            try:
                await self.flush()
            except Exception as exc:
                self.errors += 1
                if not is_transient(exc):
                    log.exception("database writer failed")
                self.status.mark_down(exc)
                self.db.reset()
                await asyncio.sleep(self.backoff.next())
                continue
            self.backoff.reset()

    async def flush(self) -> int:
        """Connect if needed and write everything buffered; probe health when idle.

        Returns:
            Rows written.

        Raises:
            Exception: A transient database error (the rows stay buffered).
        """
        assert self.db is not None
        await self.db.connect()
        written = 0
        while self.buffer:
            written += await self._flush_batch()
        if written:
            self._last_ok = time.monotonic()
        elif time.monotonic() - self._last_ok >= self.health_interval_s:
            await self.db.ping()
            self._last_ok = time.monotonic()
        self.status.mark_ok()
        return written

    async def _flush_batch(self) -> int:
        assert self.db is not None
        last_seq, rows = self.buffer.peek(self.batch_size)
        groups: list[tuple[str, tuple[str, ...], list[tuple[Any, ...]]]] = []
        for table, columns, values in rows:
            if groups and groups[-1][0] == table and groups[-1][1] == columns:
                groups[-1][2].append(values)
            else:
                groups.append((table, columns, [values]))
        try:
            await self.db.insert_groups(groups)
        except Exception as exc:
            if is_transient(exc) or not isinstance(exc, asyncpg.PostgresError | ValueError | TypeError):
                raise
            log.warning("batch of %d rows rejected (%s), inserting one by one", len(rows), exc)
            await self._insert_individually(rows)
        else:
            self.written += len(rows)
        self.buffer.commit(last_seq)
        self.batches += 1
        return len(rows)

    async def _insert_individually(self, rows: list[_Row]) -> None:
        assert self.db is not None
        for table, columns, values in rows:
            try:
                await self.db.insert_one(table, columns, values)
            except Exception as exc:
                if is_transient(exc):
                    raise  # the whole batch stays buffered; rows written so far may repeat (at-least-once)
                self.rejected += 1
                log.error("row rejected by %s: %s", table, exc)
            else:
                self.written += 1

    async def drain(self, timeout_s: float) -> None:
        """Try to write everything buffered before shutdown (best effort, bounded by ``timeout_s``)."""
        if self.db is None or not self.buffer:
            return
        try:
            await asyncio.wait_for(self.flush(), timeout_s)
        except Exception as exc:
            log.warning("database flush on shutdown failed: %s (%d rows lost)", exc, self.buffered)


class ServiceEventLog:
    """Writes the ``service_events`` journal and hooks into dependency status transitions.

    Args:
        writer: Buffered writer.
        service: Service name (``ingest``, ``predictor``, ``api``).
        instance: Instance name.
    """

    def __init__(self, writer: BufferedWriter, service: str, instance: str) -> None:
        self.writer = writer
        self.service = service
        self.instance = instance

    def write(
        self,
        kind: str,
        message: str,
        *,
        dependency: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Queue one event (``kind``: start, stop, degraded, recovered, clock_reset, info)."""
        self.writer.write(
            "service_events",
            {
                "ts": datetime.now(UTC),
                "service": self.service,
                "instance": self.instance,
                "kind": kind,
                "dependency": dependency,
                "message": message,
                "details": dict(details) if details else None,
            },
        )

    def watch(self, status: DependencyStatus) -> None:
        """Journal the outages and recoveries of a dependency (the first successful check is not logged)."""

        def hook(dep: DependencyStatus, ok: bool, detail: str | None) -> None:
            if ok and dep.outages == 0:
                return
            if ok:
                self.write("recovered", f"{dep.name} recovered after {detail or '?'} s", dependency=dep.name)
            else:
                self.write("degraded", f"{dep.name} unavailable: {detail}", dependency=dep.name)

        status.hooks.append(hook)
