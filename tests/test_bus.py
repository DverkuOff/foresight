"""Tests for the Redis bus: stream/hash/pub-sub encoding, publisher, consumer group, degradation.

Every test runs on ``fakeredis``; tests marked ``real`` also run against a real Redis when
``FORESIGHT_TEST_REDIS_URL`` points to a disposable database (it is flushed), e.g. ``redis://localhost:6379/15``.
Outage tests toggle ``FakeServer.connected`` and therefore run on fakeredis only.
"""

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from redis.exceptions import ConnectionError as RedisConnectionError  # noqa: E402
from redis.exceptions import OutOfMemoryError, ReadOnlyError, ResponseError  # noqa: E402

from backend.bus import (  # noqa: E402
    CHANNEL_VEHICLES,
    INGEST_STATS_KEY,
    STREAM_EPOCH_KEY,
    STREAM_TELEMETRY,
    STREAM_TIME_KEY,
    VEHICLE_INDEX,
    StreamBatch,
    StreamConsumer,
    TelemetryEvent,
    TelemetryPublisher,
    VehicleRecord,
    _entries,
    is_transient,
    redis_factory,
    vehicle_fields,
    vehicle_key,
)
from backend.runtime import Backoff, BoundedBuffer, DependencyStatus  # noqa: E402
from backend.state import StateStore  # noqa: E402
from shared.ndtp import IrmaRecord, NavRecord, RealtimePacket  # noqa: E402

REAL_REDIS_URL = os.environ.get("FORESIGHT_TEST_REDIS_URL")
FAST = Backoff(0.02, 0.1)


def _nav(ts: int = 1_767_690_000, lat: float = 55.75, lon: float = 37.61) -> NavRecord:
    return NavRecord(timestamp=datetime.fromtimestamp(ts, UTC), lon=lon, lat=lat, speed_avg=20, course=45)


IRMA = IrmaRecord(0, 7, 3, (1, 2, 0, 0), (0, 1, 0, 0), (True, True, False, False), (True, False, True, True))


def _packet(unit_id: int, ts: int, irma: bool = False) -> RealtimePacket:
    return RealtimePacket(unit_id=unit_id, request_id=1, nav=_nav(ts), irma=(IRMA,) if irma else ())


def _run(scenario: Callable[..., Awaitable[None]], backend: str) -> None:
    """Run an async scenario with a Redis factory of the given backend (``fake`` or ``real``)."""

    async def main() -> None:
        if backend == "real":
            if not REAL_REDIS_URL:
                pytest.skip("FORESIGHT_TEST_REDIS_URL is not set")
            factory = redis_factory(REAL_REDIS_URL, 2.0)
            client = factory()
            await client.flushdb()
            await client.aclose()
            await scenario(factory, None)
        else:
            server = fakeredis.FakeServer()
            await scenario(lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True), server)

    asyncio.run(main())


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


# ---- encoding -----------------------------------------------------------------------------------


def test_event_fields_roundtrip() -> None:
    event = TelemetryEvent(
        501, 115106, 1_767_690_000.0, 37.6173210, 55.7551234, 33, 270, True, '{"x":1}', 1.5
    )
    fields = event.to_fields()
    assert fields["tr_id"] == "115106" and fields["ts"] == "1767690000" and fields["valid"] == "1"
    assert fields["lon"] == "37.6173210" and fields["doors"] == '{"x":1}'
    assert TelemetryEvent.from_fields(fields) == event

    unknown = TelemetryEvent(7, None, 1.25, 0.0, 0.0, valid=False)
    fields = unknown.to_fields()
    assert fields["tr_id"] == "" and "doors" not in fields and fields["ts"] == "1.25"
    assert "epoch" not in fields and "clock" not in fields
    assert TelemetryEvent.from_fields(fields) == unknown
    with pytest.raises(KeyError):
        TelemetryEvent.from_fields({"unit_id": "1"})

    stamped = TelemetryEvent(
        501, 115106, 1_767_690_000.0, 37.6, 55.7, epoch=1_790_000_000_123, clock=1_767_690_004.0
    )
    fields = stamped.to_fields()
    assert fields["epoch"] == "1790000000123" and fields["clock"] == "1767690004"
    assert TelemetryEvent.from_fields(fields) == stamped


def test_transient_error_classification() -> None:
    transient = [
        RedisConnectionError("refused"),
        TimeoutError(),
        OutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
        ReadOnlyError("You can't write against a read only replica."),
        ResponseError(
            "MISCONF Redis is configured to save RDB snapshots, but it's currently unable to persist"
        ),
        ResponseError("NOREPLICAS Not enough good replicas to write."),
    ]
    permanent = [
        ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value"),
        ResponseError("ERR syntax error"),
        ValueError("bug"),
    ]
    assert all(is_transient(exc) for exc in transient)
    assert not any(is_transient(exc) for exc in permanent)


def test_vehicle_hash_roundtrip() -> None:
    store = StateStore()
    store.on_connect(501)
    store.on_nav(501, _nav(), IRMA)
    vehicle = store.get(501)
    assert vehicle is not None
    record = VehicleRecord.from_fields(vehicle_fields(vehicle, 115106, 123))
    assert (record.unit_id, record.tr_id, record.connected, record.packets) == (501, 115106, True, 1)
    assert record.event_time == _nav().timestamp and record.updated_ms == 123
    assert record.lat == pytest.approx(55.75) and record.speed_kmh == 20 and record.valid is True
    assert record.doors is not None and record.doors["door_closed"] == [True, False, True, True]

    bare = StateStore()
    bare.on_connect(9)
    record = VehicleRecord.from_fields(vehicle_fields(bare.get(9), None, 1))  # type: ignore[arg-type]
    assert record.tr_id is None and record.lat is None and record.event_time is None and record.doors is None


def test_bounded_buffer_commit_after_eviction() -> None:
    buf: BoundedBuffer[int] = BoundedBuffer(3)
    for i in range(5):
        buf.append(i)
    assert buf.evicted == 2 and len(buf) == 3
    last, items = buf.peek(2)
    assert items == [2, 3]
    buf.append(5)  # evicts 2 while the batch [2, 3] is "in flight"
    buf.commit(last)
    assert buf.peek(10)[1] == [4, 5]


# ---- publisher (ingest side) --------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_publisher_writes_stream_state_and_stats(backend: str) -> None:
    async def scenario(factory: Callable, server: object) -> None:
        store = StateStore()
        publisher = TelemetryPublisher(
            store,
            factory,
            DependencyStatus("redis"),
            unit_map={501: 115106},
            stats_provider=lambda: {"frames": "3", "listening": "1"},
            vehicle_ttl_s=600,
        )
        reader = factory()
        pubsub = reader.pubsub()
        await pubsub.subscribe(CHANNEL_VEHICLES)
        await pubsub.get_message(timeout=0.5)  # subscribe confirmation

        store.on_connect(501)
        for i, unit in enumerate((501, 501, 777)):
            if unit == 777 and store.get(777) is None:
                store.on_connect(777)
            nav = _nav(1_767_690_000 + i)
            store.on_nav(unit, nav)
            publisher.on_packet(unit, _packet(unit, 1_767_690_000 + i, irma=i == 0))
        assert publisher.buffered == 3 and publisher.unmapped == 1
        assert await publisher.flush() == 3
        assert publisher.buffered == 0 and publisher.published == 3

        entries = await reader.xrange(STREAM_TELEMETRY)
        assert [e[1]["tr_id"] for e in entries] == ["115106", "115106", ""]
        assert [e[1]["ts"] for e in entries] == ["1767690000", "1767690001", "1767690002"]
        assert "doors" in entries[0][1]
        state = await reader.hgetall(vehicle_key(501))
        assert state["tr_id"] == "115106" and state["packets"] == "2" and state["ts"] == "1767690001"
        assert 0 < await reader.ttl(vehicle_key(501)) <= 600
        assert set(await reader.zrange(VEHICLE_INDEX, 0, -1)) == {"501", "777"}
        assert (await reader.get(STREAM_TIME_KEY)).startswith("2026-01-06T09:00:02")
        assert (await reader.hgetall(INGEST_STATS_KEY))["frames"] == "3"

        message = None
        for _ in range(20):
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message:
                break
        assert message is not None
        delta = json.loads(message["data"])
        assert delta["type"] == "delta" and {v["unit_id"] for v in delta["vehicles"]} == {"501", "777"}

        # nothing changed: nothing is written; a state change alone is written without stream events
        assert await publisher.flush() == 0
        store.on_disconnect(777)
        await publisher.flush()
        assert (await reader.hgetall(vehicle_key(777)))["connected"] == "0"
        assert await reader.xlen(STREAM_TELEMETRY) == 3

        await pubsub.aclose()
        await reader.aclose()
        await publisher.close()

    _run(scenario, backend)


def test_publisher_buffers_while_redis_is_down_and_replays_in_order() -> None:
    async def scenario(factory: Callable, server: object) -> None:
        store = StateStore()
        status = DependencyStatus("redis")
        transitions: list[bool] = []
        status.hooks.append(lambda _s, ok, _d: transitions.append(ok))
        publisher = TelemetryPublisher(
            store, factory, status, max_buffer=8, flush_interval_s=0.02, backoff=Backoff(0.02, 0.05)
        )
        task = asyncio.create_task(publisher.run())
        publisher.on_packet(1, _packet(1, 1000))
        await _until(lambda: publisher.published == 1)
        assert status.ok is True

        server.connected = False  # type: ignore[attr-defined]
        for ts in range(1001, 1011):  # 10 events, the buffer keeps the newest 8
            store.on_nav(1, _nav(ts))
            publisher.on_packet(1, _packet(1, ts))
        await _until(lambda: status.ok is False)
        assert publisher.buffered == 8 and publisher.evicted == 2 and publisher.flush_errors >= 1

        server.connected = True  # type: ignore[attr-defined]
        await _until(lambda: publisher.buffered == 0)
        task.cancel()
        client = factory()
        entries = await client.xrange(STREAM_TELEMETRY)
        assert [int(e[1]["ts"]) for e in entries] == [1000, *range(1003, 1011)]  # order kept, oldest evicted
        assert (await client.hgetall(vehicle_key(1)))["ts"] == "1010"  # hot state re-synced after the outage
        assert status.ok is True and status.outages == 1 and transitions == [True, False, True]
        await client.aclose()

    _run(scenario, "fake")


def test_publisher_rewrites_all_hot_state_after_an_outage() -> None:
    async def scenario(factory: Callable, server: object) -> None:
        store = StateStore()
        for unit in (1, 2):
            store.on_connect(unit)
            store.on_nav(unit, _nav())
        status = DependencyStatus("redis")
        publisher = TelemetryPublisher(store, factory, status, flush_interval_s=0.02, backoff=FAST)
        task = asyncio.create_task(publisher.run())
        client = factory()
        await _until(lambda: status.ok is True)
        assert await client.exists(vehicle_key(1), vehicle_key(2)) == 2

        await client.flushall()  # Redis comes back without its data after the outage
        server.connected = False  # type: ignore[attr-defined]
        store.on_nav(1, _nav(1_767_690_100))  # only vehicle 1 changes meanwhile
        await _until(lambda: status.ok is False)
        server.connected = True  # type: ignore[attr-defined]
        await _until(lambda: status.ok is True)
        assert await client.exists(vehicle_key(1), vehicle_key(2)) == 2  # the silent vehicle is back too
        assert await client.get(STREAM_TIME_KEY) is not None
        task.cancel()
        await client.aclose()

    _run(scenario, "fake")


class RefusingRedis:
    """A Redis whose pipelines answer every command with the same error reply."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def pipeline(self, transaction: bool = False) -> "RefusingRedis._Pipeline":
        return RefusingRedis._Pipeline(self.error)

    async def aclose(self) -> None:
        pass

    class _Pipeline:
        def __init__(self, error: Exception) -> None:
            self.error = error
            self.commands = 0

        def __getattr__(self, name: str) -> Callable[..., None]:
            def command(*args: object, **kwargs: object) -> None:
                self.commands += 1

            return command

        async def execute(self, raise_on_error: bool = True) -> list[Exception]:
            return [self.error] * self.commands


@pytest.mark.parametrize(
    "error",
    [
        OutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
        ResponseError(
            "MISCONF Redis is configured to save RDB snapshots, but it's currently unable to persist"
        ),
        ReadOnlyError("You can't write against a read only replica."),
    ],
    ids=["oom", "misconf", "readonly"],
)
def test_publisher_keeps_the_batch_while_redis_refuses_writes(error: Exception) -> None:
    async def scenario(factory: Callable, server: object) -> None:
        store = StateStore()
        status = DependencyStatus("redis")
        publisher = TelemetryPublisher(store, factory, status, flush_interval_s=0.02, backoff=FAST)
        for ts in (1_767_690_000, 1_767_690_001):
            store.on_nav(1, _nav(ts))
            publisher.on_packet(1, _packet(1, ts))
        publisher._client = RefusingRedis(error)  # type: ignore[assignment]
        with pytest.raises(type(error)):
            await publisher.flush()
        assert publisher.buffered == 2 and publisher.published == 0 and publisher.command_errors == 0

        task = asyncio.create_task(publisher.run())
        await _until(lambda: status.ok is False)
        assert publisher.buffered == 2 and publisher.flush_errors >= 1
        publisher._client = None  # Redis accepts writes again (a fresh client from the factory)
        await _until(lambda: publisher.buffered == 0)
        task.cancel()
        client = factory()
        assert [e[1]["ts"] for e in await client.xrange(STREAM_TELEMETRY)] == ["1767690000", "1767690001"]
        assert status.ok is True and status.outages == 1
        await client.aclose()

    _run(scenario, "fake")


def test_publisher_counts_permanent_errors_and_moves_on() -> None:
    async def scenario(factory: Callable, server: object) -> None:
        publisher = TelemetryPublisher(StateStore(), factory, DependencyStatus("redis"))
        publisher.on_packet(1, _packet(1, 1_767_690_000))
        refusing = RefusingRedis(ResponseError("WRONGTYPE Operation against a key"))
        publisher._client = refusing  # type: ignore[assignment]
        await publisher.flush()  # retrying would not help: counted, the batch is dropped
        assert publisher.buffered == 0 and publisher.command_errors == 1

    _run(scenario, "fake")


def test_publisher_stamps_the_stream_clock_and_continues_it_after_a_restart() -> None:
    async def scenario(factory: Callable, server: object) -> None:
        store = StateStore()
        publisher = TelemetryPublisher(store, factory, DependencyStatus("redis"), unit_map={1: 115106})
        for ts in (1_767_690_000, 1_767_690_010, 1_767_689_000):  # the last one is 17 min late
            store.on_nav(1, _nav(ts))
            publisher.on_packet(1, _packet(1, ts))
        await publisher.flush()
        epoch = store.stream_clock.epoch
        client = factory()
        entries = [e[1] for e in await client.xrange(STREAM_TELEMETRY)]
        assert [e["epoch"] for e in entries] == [str(epoch)] * 3
        assert [e["clock"] for e in entries] == ["1767690000", "1767690010", "1767690010"]
        assert await client.get(STREAM_EPOCH_KEY) == str(epoch)
        assert (await client.get(STREAM_TIME_KEY)).startswith("2026-01-06T09:00:10")

        # the ingest restarts: the new process continues the saved clock (same epoch, no jump downstream)
        restarted = TelemetryPublisher(StateStore(), factory, DependencyStatus("redis"))
        assert await restarted.load_clock()
        assert restarted.store.stream_clock.epoch == epoch
        assert restarted.store.stream_clock.now == 1_767_690_010
        # without a saved clock (or Redis) it starts afresh with the first fix
        await client.flushall()
        fresh = TelemetryPublisher(StateStore(), factory, DependencyStatus("redis"))
        assert not await fresh.load_clock() and fresh.store.stream_clock.now is None
        for p in (publisher, restarted, fresh):
            await p.close()
        await client.aclose()

    _run(scenario, "fake")


# ---- consumer (predictor side) ------------------------------------------------------------------


async def _add_events(client: object, start: int, count: int, tr_id: int | None = 115106) -> None:
    for i in range(count):
        event = TelemetryEvent(501, tr_id, float(start + i), 37.6, 55.7)
        await client.xadd(STREAM_TELEMETRY, event.to_fields())  # type: ignore[attr-defined]


def _collector() -> tuple[list[float], Callable[[StreamBatch], Awaitable[None]]]:
    seen: list[float] = []

    async def handler(batch: StreamBatch) -> None:
        seen.extend(event.ts for _, event in batch)

    return seen, handler


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_consumer_group_reads_acks_and_reports_lag(backend: str) -> None:
    async def scenario(factory: Callable, server: object) -> None:
        client = factory()
        await _add_events(client, 1000, 7)
        await client.xadd(STREAM_TELEMETRY, {"garbage": "1"})
        seen, handler = _collector()
        consumer = StreamConsumer(
            factory, handler, DependencyStatus("redis"), consumer="c1", batch_size=3, block_ms=50
        )
        while await consumer.poll_once():
            pass
        assert seen == [float(t) for t in range(1000, 1007)]
        assert consumer.malformed == 1 and consumer.acked == 8
        await consumer.poll_lag()
        assert consumer.lag == 0 and consumer.pending == 0 and consumer.stream_length == 8
        groups = await client.xinfo_groups(STREAM_TELEMETRY)
        assert groups[0]["name"] == "predictors" and groups[0]["pending"] == 0
        await consumer.close()
        await client.aclose()

    _run(scenario, backend)


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_consumer_recovers_own_pending_and_claims_dead_consumers(backend: str) -> None:
    async def scenario(factory: Callable, server: object) -> None:
        client = factory()
        await client.xgroup_create(STREAM_TELEMETRY, "predictors", id="0", mkstream=True)
        await _add_events(client, 1000, 6)
        # "c1" and "dead" read entries and crash before ACK
        await client.xreadgroup("predictors", "c1", {STREAM_TELEMETRY: ">"}, count=2)
        await client.xreadgroup("predictors", "dead", {STREAM_TELEMETRY: ">"}, count=2)
        await asyncio.sleep(0.05)

        seen, handler = _collector()
        consumer = StreamConsumer(
            factory,
            handler,
            DependencyStatus("redis"),
            consumer="c1",
            claim_idle_ms=10,
            batch_size=10,
            block_ms=50,
        )
        for _ in range(4):
            await consumer.poll_once()
        # own pending first (1000, 1001), then claimed from "dead" (1002, 1003), then new (1004, 1005)
        assert seen == [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0]
        assert consumer.recovered == 2 and consumer.claimed == 2
        pending = await client.xpending(STREAM_TELEMETRY, "predictors")
        assert pending["pending"] == 0
        await consumer.close()
        await client.aclose()

    _run(scenario, backend)


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_consumer_removes_idle_consumers_without_pending(backend: str) -> None:
    async def scenario(factory: Callable, server: object) -> None:
        client = factory()
        await client.xgroup_create(STREAM_TELEMETRY, "predictors", id="0", mkstream=True)
        await _add_events(client, 1000, 3)
        # "gone" (an old container name) processed and ACKed everything; "busy" crashed holding an entry
        reply = await client.xreadgroup("predictors", "gone", {STREAM_TELEMETRY: ">"}, count=2)
        await client.xack(STREAM_TELEMETRY, "predictors", *[entry_id for entry_id, _ in _entries(reply)])
        await client.xreadgroup("predictors", "busy", {STREAM_TELEMETRY: ">"}, count=1)
        await asyncio.sleep(0.2)

        seen, handler = _collector()
        consumer = StreamConsumer(
            factory,
            handler,
            DependencyStatus("redis"),
            consumer="predictor-1",
            claim_idle_ms=60_000,  # "busy" keeps its entry for now
            gc_idle_ms=100,
            block_ms=20,
        )
        await consumer.poll_once()
        names = {c["name"] for c in await client.xinfo_consumers(STREAM_TELEMETRY, "predictors")}
        assert names == {"busy", "predictor-1"} and consumer.consumers_removed == 1
        await consumer.close()
        await client.aclose()

    _run(scenario, backend)


def test_consumer_survives_outage_and_lost_group() -> None:
    async def scenario(factory: Callable, server: object) -> None:
        client = factory()
        seen, handler = _collector()
        status = DependencyStatus("redis")
        consumer = StreamConsumer(factory, handler, status, block_ms=20, backoff=Backoff(0.02, 0.05))
        task = asyncio.create_task(consumer.run())
        await _add_events(client, 1000, 3)
        await _until(lambda: len(seen) == 3)

        server.connected = False  # type: ignore[attr-defined]
        await _until(lambda: status.ok is False)
        server.connected = True  # type: ignore[attr-defined]
        await _add_events(client, 1003, 3)
        await _until(lambda: len(seen) == 6)
        assert status.ok is True and status.outages == 1

        # Redis restarted without data: the group is recreated and the new stream consumed from the start
        await client.flushall()
        await _add_events(client, 2000, 2)
        await _until(lambda: len(seen) == 8)
        assert seen[-2:] == [2000.0, 2001.0]
        task.cancel()
        await consumer.close()
        await client.aclose()

    _run(scenario, "fake")
