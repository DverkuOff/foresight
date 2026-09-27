"""Tests for the PostgreSQL layer: schema, buffered writer (outage, overflow, bad rows), service journal.

The writer is tested against an in-memory fake; the schema is also applied to a real PostgreSQL when
``FORESIGHT_TEST_DATABASE_URL`` is set (e.g. ``postgresql://foresight:foresight@localhost:5432/foresight``).
"""

import asyncio
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest

asyncpg = pytest.importorskip("asyncpg")

from backend.db import (  # noqa: E402
    SCHEMA_SQL,
    TABLE_COLUMNS,
    BufferedWriter,
    Database,
    ServiceEventLog,
    is_transient,
)
from backend.runtime import Backoff, DependencyStatus  # noqa: E402

REAL_DB_URL = os.environ.get("FORESIGHT_TEST_DATABASE_URL")


class FakeDatabase:
    """In-memory stand-in for :class:`backend.db.Database` with switchable outages and bad rows."""

    def __init__(self) -> None:
        self.down = False
        self.rows: list[tuple[str, dict[str, Any]]] = []
        self.bad_marker = "BAD"
        self.connects = 0
        self.resets = 0
        self._connected = False

    def _check(self) -> None:
        if self.down:
            raise ConnectionRefusedError("postgres is down")

    async def connect(self) -> None:
        if not self._connected:  # like the real pool: an existing pool is not re-checked here
            self._check()
            self.connects += 1
            self._connected = True

    def reset(self) -> None:
        self.resets += 1
        self._connected = False

    async def close(self) -> None:
        self._connected = False

    async def ping(self) -> None:
        self._check()

    async def insert_groups(self, groups: list[tuple[str, tuple[str, ...], list[tuple[Any, ...]]]]) -> None:
        self._check()
        staged = []
        for table, columns, rows in groups:
            for values in rows:
                if self.bad_marker in values:
                    raise asyncpg.exceptions.DataError("invalid input")
                staged.append((table, dict(zip(columns, values, strict=True))))
        self.rows.extend(staged)  # all or nothing, like a transaction

    async def insert_one(self, table: str, columns: tuple[str, ...], row: tuple[Any, ...]) -> None:
        self._check()
        if self.bad_marker in row:
            raise asyncpg.exceptions.DataError("invalid input")
        self.rows.append((table, dict(zip(columns, row, strict=True))))


def _writer(db: FakeDatabase | None, **kw: Any) -> BufferedWriter:
    kw.setdefault("flush_interval_s", 0.01)
    kw.setdefault("health_interval_s", 0.05)
    kw.setdefault("backoff", Backoff(0.01, 0.05))
    return BufferedWriter(db, DependencyStatus("postgres"), **kw)  # type: ignore[arg-type]


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _run(coro: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(coro())  # type: ignore[arg-type]


def _event(i: int, message: str = "") -> dict[str, Any]:
    return {"ts": datetime.now(UTC), "service": "test", "kind": "info", "message": message or f"e{i}"}


def test_schema_covers_every_writer_table_and_column() -> None:
    for table, columns in TABLE_COLUMNS.items():
        match = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", SCHEMA_SQL, re.S)
        assert match, table
        defined = {line.split()[0] for line in match.group(1).strip().splitlines()}
        assert columns <= defined, (table, columns - defined)
    indexed = {
        table: " ".join(
            re.findall(rf"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS \w+ ON {table} \(([^;]*)\)", SCHEMA_SQL)
        )
        for table in TABLE_COLUMNS
    }
    for table, time_column in [
        ("predictions", "issued_at"),
        ("alerts", "issued_at"),
        ("stop_passages", "confirmed_at"),
        ("service_events", "ts"),
    ]:
        assert time_column in indexed[table], table
    for table in ("predictions", "alerts", "stop_passages"):
        assert "tr_id" in indexed[table], table
    # idempotent DDL only: every CREATE is guarded
    assert not re.search(r"CREATE (UNIQUE )?(TABLE|INDEX) (?!IF NOT EXISTS)", SCHEMA_SQL)


def test_write_validates_table_and_columns() -> None:
    writer = _writer(FakeDatabase())
    with pytest.raises(ValueError, match="unknown table"):
        writer.write("nope", {})
    with pytest.raises(ValueError, match="unknown columns"):
        writer.write("service_events", {"ts": 1, "evil); DROP TABLE x; --": 1})
    writer.write(
        "service_events", {"ts": datetime(2026, 1, 6), "service": "t", "kind": "k", "details": {"a": 1}}
    )
    _, (row,) = writer.buffer.peek(1)
    assert row[2][0].tzinfo is UTC  # naive datetimes are UTC
    assert row[2][3] == '{"a": 1}'  # dicts go to JSONB as JSON text


def test_transient_error_classification() -> None:
    assert is_transient(ConnectionRefusedError())
    assert is_transient(asyncpg.exceptions.ConnectionDoesNotExistError("gone"))
    assert is_transient(asyncpg.exceptions.CannotConnectNowError("starting"))
    assert not is_transient(asyncpg.exceptions.NotNullViolationError("null"))
    assert not is_transient(asyncpg.exceptions._base.DataError("bad argument"))  # InterfaceError + ValueError


def test_writer_buffers_while_database_is_down_and_replays_in_order() -> None:
    async def scenario() -> None:
        db = FakeDatabase()
        db.down = True
        writer = _writer(db)
        events = ServiceEventLog(writer, "test", "t-1")
        events.watch(writer.status)
        task = asyncio.create_task(writer.run())
        for i in range(5):
            writer.write("service_events", _event(i))
        await _until(lambda: writer.status.ok is False)
        assert writer.buffered == 6 and writer.written == 0  # 5 rows + the 'degraded' journal entry
        assert writer.errors >= 1 and db.resets >= 1

        db.down = False
        await _until(lambda: writer.buffered == 0 and writer.status.ok is True)
        await _until(lambda: len(db.rows) == 7)  # + the 'recovered' entry
        messages = [row["message"] for _, row in db.rows]
        assert messages[:5] == ["e0", "e1", "e2", "e3", "e4"]
        kinds = [row["kind"] for _, row in db.rows]
        assert kinds[5:] == ["degraded", "recovered"]
        assert db.rows[5][1]["dependency"] == "postgres" and db.rows[5][1]["service"] == "test"
        task.cancel()

    _run(scenario)


def test_writer_drops_oldest_rows_beyond_the_limit() -> None:
    async def scenario() -> None:
        db = FakeDatabase()
        db.down = True
        writer = _writer(db, max_buffer=3)
        for i in range(5):
            writer.write("service_events", _event(i))
        assert writer.buffered == 3 and writer.dropped == 2
        db.down = False
        await writer.flush()
        assert [row["message"] for _, row in db.rows] == ["e2", "e3", "e4"]

    _run(scenario)


def test_writer_rejects_bad_rows_one_by_one() -> None:
    async def scenario() -> None:
        db = FakeDatabase()
        writer = _writer(db, batch_size=10)
        writer.write("service_events", _event(0))
        writer.write("service_events", _event(1, message=db.bad_marker))
        writer.write("stop_passages", {"tr_id": 1, "stop_id": 2, "time_begin": datetime.now(UTC)})
        assert await writer.flush() == 3
        assert writer.rejected == 1 and writer.written == 2 and writer.buffered == 0
        assert [table for table, _ in db.rows] == ["service_events", "stop_passages"]

    _run(scenario)


def test_writer_without_database_counts_drops() -> None:
    writer = _writer(None)
    writer.write("service_events", _event(0))
    assert writer.dropped == 1 and writer.buffered == 0
    assert writer.status.state == "disabled"


def test_writer_probes_health_when_idle() -> None:
    async def scenario() -> None:
        db = FakeDatabase()
        writer = _writer(db)
        task = asyncio.create_task(writer.run())
        await _until(lambda: writer.status.ok is True)
        db.down = True  # nothing to write: the periodic probe notices the outage
        await _until(lambda: writer.status.ok is False)
        db.down = False
        await _until(lambda: writer.status.ok is True)
        task.cancel()

    _run(scenario)


@pytest.mark.skipif(not REAL_DB_URL, reason="FORESIGHT_TEST_DATABASE_URL is not set")
def test_real_database_schema_and_writer() -> None:
    async def scenario() -> None:
        marker = f"test-{uuid.uuid4().hex[:8]}"
        first, second = Database(REAL_DB_URL), Database(REAL_DB_URL)  # type: ignore[arg-type]
        await asyncio.gather(first.connect(), second.connect())  # concurrent schema creation is safe
        await second.close()
        writer = BufferedWriter(first, DependencyStatus("postgres"))
        ServiceEventLog(writer, marker, "i-1").write("start", "hello", details={"k": [1, 2]})
        now = datetime.now(UTC)
        writer.write(
            "predictions",
            {
                "tr_id": 1,
                "target_stop_id": 2,
                "target_time_begin": now,
                "issued_at": now,
                "pred_delay_s": 12.5,
                "cause": marker,
            },
        )
        assert await writer.flush() == 2
        pool = first.pool
        tables = {
            r["tablename"]
            for r in await pool.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        }
        assert set(TABLE_COLUMNS) <= tables
        row = await pool.fetchrow("SELECT kind, details FROM service_events WHERE service = $1", marker)
        assert row["kind"] == "start" and '"k"' in row["details"]
        assert await pool.fetchval("SELECT count(*) FROM predictions WHERE cause = $1", marker) == 1
        assert await pool.fetchval("SELECT value->>'red_delay_s' FROM settings WHERE key='risk_thresholds'")
        await pool.execute("DELETE FROM service_events WHERE service = $1", marker)
        await pool.execute("DELETE FROM predictions WHERE cause = $1", marker)
        await first.close()

    _run(scenario)
