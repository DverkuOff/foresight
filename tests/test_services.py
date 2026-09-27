"""Tests for the service layer: unit map, predictor core (clock, windows, ticks), predictor and api apps
with fake dependencies, degradation of the api when Redis or PostgreSQL is down."""

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
fakeredis = pytest.importorskip("fakeredis")

from fastapi.testclient import TestClient  # noqa: E402

from backend import __main__ as backend_main  # noqa: E402
from backend import api, predictor  # noqa: E402
from backend.bus import STREAM_TELEMETRY, StreamConsumer, TelemetryEvent, TelemetryPublisher  # noqa: E402
from backend.clock import ClockJump, Fix, StreamClock  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.hotstate import VehicleCache  # noqa: E402
from backend.predictor import (  # noqa: E402
    PredictorCore,
    TickContext,
    TickScheduler,
    TrackPoint,
    TrackWindows,
)
from backend.runtime import Backoff, DependencyStatus  # noqa: E402
from backend.state import LinkStatus, StateStore  # noqa: E402
from backend.unitmap import load_unit_map  # noqa: E402
from shared.ndtp import NavRecord  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "dataset"


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = dict(
        database_url="",
        unit_map_splits="",
        api_resync_s=0.3,
        backoff_initial_s=0.02,
        backoff_max_s=0.1,
        consumer_block_ms=50,
        db_flush_interval_s=0.02,
        db_health_interval_s=0.05,
    )
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _vehicles_message(ws: Any) -> dict[str, Any]:
    """Next ``snapshot`` / ``delta`` on ``/ws`` (the ``clock`` and the predictor's events are skipped)."""
    while True:
        message = ws.receive_json()
        if message["type"] in ("snapshot", "delta"):
            return message


# ---- unit map -----------------------------------------------------------------------------------


def _traffic(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["packet_id,tr_id,unit_id,event_time"] + [
        f"1,{tr},{unit},2026-01-06 06:00:00" for unit, tr in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_unit_map_from_csv(tmp_path: Path) -> None:
    _traffic(
        tmp_path / "test" / "traffic.csv", [("664030", "115106"), ("664030", "115106"), ("794446", "116057")]
    )
    _traffic(tmp_path / "train" / "traffic.csv", [("664030", "999"), ("1", "2"), ("", "3"), ("x", "4")])
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "traffic.csv").write_text("a,b\n1,2\n", encoding="utf-8")  # no unit_id/tr_id
    mapping = load_unit_map(tmp_path, ["test", "train", "validate", "broken"])  # missing/broken: skipped
    assert mapping == {664030: 115106, 794446: 116057, 1: 2}  # the first split wins on conflicts


@pytest.mark.skipif(not (DATASET / "test" / "traffic.csv").is_file(), reason="dataset is not available")
def test_unit_map_from_dataset_is_one_to_one() -> None:
    test_map = load_unit_map(DATASET, ["test"])
    train_map = load_unit_map(DATASET, ["train"])
    both = load_unit_map(DATASET, ["test", "train"])
    assert len(test_map) == 30 and len(train_map) == 56
    assert all(train_map[u] == t for u, t in test_map.items() if u in train_map)  # no conflicts
    assert set(both) == set(test_map) | set(train_map)
    assert len(set(both.values())) == len(both)  # one tr_id per unit and vice versa


# ---- predictor core -----------------------------------------------------------------------------


WALL = 1_790_000_000.0  # 2026-09-21: the wall clock of the clock tests
BASE = 1_767_682_800.0  # 2026-01-06 07:00:00 UTC: replayed data, a multiple of 30 s


def test_stream_clock_guards_and_resets() -> None:
    clock = StreamClock(jump_s=300, confirm=3)
    assert clock.observe(BASE, 1, WALL) is Fix.ON and clock.now == BASE and clock.epoch > 0
    epoch = clock.epoch
    # garbage: far future (u32 up to 2106), no RTC / no fix (1970, 2000), GPS week rollover (2006), NaN
    for ts in (WALL + 10 * 86_400, 0.0, 946_684_800.0, 1_150_000_000.0, float("nan")):
        assert clock.observe(ts, 1, WALL) is Fix.GARBAGE
    assert clock.garbage == 5 and clock.now == BASE and clock.plausible(BASE, WALL)
    # late (black-box) events interleaved with fresh ones never reset the clock
    for i in range(5):
        assert clock.observe(BASE - 7200, 1, WALL) is Fix.OFF
        assert clock.observe(BASE + 10 + i, 1, WALL) is Fix.ON
    assert clock.resets == 0 and clock.now == BASE + 14 and clock.off_timeline == 5
    # the source restarted (a single device here): consecutive events far behind move the clock back
    fixes = [clock.observe(t, 1, WALL) for t in (BASE - 7200, BASE - 7199, BASE - 7198)]
    assert fixes == [Fix.OFF, Fix.OFF, Fix.JUMP]
    assert clock.now == BASE - 7198 and clock.resets == 1 and clock.epoch > epoch
    assert clock.time == datetime.fromtimestamp(BASE - 7198, UTC)
    assert clock.last_jump is not None and clock.last_jump.back and clock.last_jump.before == BASE + 14


def test_stream_clock_jumps_only_when_a_quorum_of_devices_moves() -> None:
    jumps: list[ClockJump] = []
    clock = StreamClock(jump_s=300, confirm=3, quorum=0.5, active_s=60, on_jump=jumps.append)
    for dev in range(10):
        clock.observe(BASE + dev, dev, WALL)
    # device 0 dumps its black box (an hour back), device 1 runs two hours ahead: the clock ignores both
    for i in range(20):
        assert clock.observe(BASE - 3600 + i, 0, WALL + i) is Fix.OFF
        assert clock.observe(BASE + 7200 + i, 1, WALL + i) is Fix.OFF
        for dev in range(2, 10):
            assert clock.observe(BASE + 10 + i, dev, WALL + i) is Fix.ON
    assert clock.resets == 0 and clock.now == BASE + 29
    # the replayer restarts 30 min back: all devices move, the clock follows once 5 of 10 confirmed it
    fixes = [
        clock.observe(BASE - 1800 + 10 * r + dev, dev, WALL + 30 + r) for r in range(3) for dev in range(10)
    ]
    assert fixes.count(Fix.JUMP) == 1 and fixes.index(Fix.JUMP) == 24 and clock.resets == 1
    assert fixes[25:] == [Fix.ON] * 5 and clock.now == BASE - 1800 + 29
    assert jumps[0].back and jumps[0].devices == 5 and jumps[0].reason == "quorum"
    # a pause: the others went quiet long ago, device 3 alone resumes 10 min later -> forward jump
    resume = clock.now + 600
    fixes = [clock.observe(resume + i, 3, WALL + 1000 + i) for i in range(3)]
    assert fixes == [Fix.OFF, Fix.OFF, Fix.JUMP] and clock.now == resume + 2 and not jumps[-1].back


def test_stream_clock_follows_the_ingest() -> None:
    clock = StreamClock(jump_s=300)
    assert clock.follow(BASE, BASE, 111, WALL) is Fix.ON and (clock.now, clock.epoch) == (BASE, 111)
    assert clock.follow(BASE - 3600, BASE + 5, 111, WALL) is Fix.OFF and clock.now == BASE + 5  # a late point
    assert (
        clock.follow(BASE + 4, BASE + 4, 111, WALL) is Fix.ON and clock.now == BASE + 5
    )  # never back in an epoch
    # the ingest restarted without its saved clock: a new epoch close to ours is adopted silently
    assert clock.follow(BASE + 6, BASE + 6, 222, WALL) is Fix.ON and clock.resets == 0 and clock.epoch == 222
    # the ingest's clock jumped (source restart): a new epoch far back is a jump
    assert clock.follow(BASE - 1800, BASE - 1800, 333, WALL) is Fix.JUMP and clock.now == BASE - 1800
    assert clock.resets == 1 and clock.last_jump is not None and clock.last_jump.reason == "follow"
    assert clock.follow(0.0, BASE, 333, WALL) is Fix.GARBAGE and clock.garbage == 1


def test_track_windows_keep_late_points_in_order_and_drop_exact_repeats_only() -> None:
    windows = TrackWindows(window_s=60)
    point = lambda ts, lat=55.7: TrackPoint(ts, 37.6, lat, 10, 0, True, 1)  # noqa: E731
    assert windows.add(7, point(100.0)) and windows.add(7, point(130.0))
    assert not windows.add(7, point(130.0))  # an exact repeat (at-least-once delivery)
    assert windows.add(7, point(120.0))  # a late point joins its track in time order
    assert windows.add(7, point(130.0, lat=55.8))  # same second, other content: kept, as offline
    assert windows.add(8, point(50.0))
    assert not windows.add(8, point(40.0), now=105.0)  # older than the window
    assert windows.points == 5 and len(windows) == 2
    assert (windows.late, windows.duplicates, windows.expired) == (1, 1, 1)
    assert [p.ts for p in windows.track(7)] == [100.0, 120.0, 130.0, 130.0]
    assert [p.ts for p in windows.track(7, until=125.0)] == [100.0, 120.0]
    assert windows.trim(165.0) == 2  # 100 (< 105) and track 8 go away
    assert [p.ts for p in windows.track(7)] == [120.0, 130.0, 130.0] and list(windows) == [7]
    assert windows.points == 3

    # held off-timeline points come back when the clock jumps to their timeline
    windows.hold(7, point(1000.0))
    windows.hold(9, point(1001.0))
    windows.hold(9, point(5000.0))  # not on the new timeline: forgotten
    windows.settle(7)  # track 7 came back to the old timeline: its held point was only late
    windows.hold(7, point(1002.0))
    assert windows.held == 3
    assert windows.rebase(1010.0, back=True) == 2 and windows.held == 0
    assert [p.ts for p in windows.track(7)] == [1002.0] and [p.ts for p in windows.track(9)] == [1001.0]


def test_tick_scheduler_fires_on_boundaries_only_forward() -> None:
    ticks = TickScheduler(30)
    assert ticks.due(1005.0) is None  # aligns to 990
    assert ticks.due(1019.0) is None
    assert ticks.due(1020.0) == 1020.0
    assert ticks.due(1049.0) is None
    assert ticks.due(1200.0) == 1200.0 and ticks.skipped == 5  # 1050..1170 skipped, never issued late
    ticks.reset()
    assert ticks.due(10.0) is None and ticks.due(30.0) == 30.0


def test_predictor_core_windows_ticks_and_clock_reset() -> None:
    ticks: list[tuple[datetime, float]] = []

    async def on_tick(ctx: TickContext) -> None:
        latest = max(ctx.windows.track(115106)[-1].ts, 0.0)
        ticks.append((ctx.stream_time, latest))

    async def scenario() -> None:
        core = PredictorCore(window_s=600, tick_period_s=30, on_tick=on_tick)
        base = 1_767_690_000.0  # 2026-01-06 09:00:00 UTC, a multiple of 30 s
        batch = [(f"{i}-0", TelemetryEvent(501, 115106, base + 10 * i, 37.6, 55.7)) for i in range(7)]
        batch.append(("x-0", TelemetryEvent(9, None, base + 61, 37.6, 55.7)))
        batch.append(("g-0", TelemetryEvent(501, 115106, 4_000_000_000.0, 37.6, 55.7)))  # garbage time
        await core.handle(batch)
        assert core.events == 9 and core.unmapped == 1 and core.clock.garbage == 1
        assert len(core.windows) == 1 and core.windows.points == 7
        # ticks at 09:00:30 and 09:01:00; each sees only telemetry strictly before its time
        assert ticks == [
            (datetime.fromtimestamp(base + 30, UTC), base + 20),
            (datetime.fromtimestamp(base + 60, UTC), base + 50),
        ]
        assert core.ticks_run == 2 and core.last_tick_duration_s is not None

        # a replayer restart: the clock goes back, windows and ticks start over; the points that came
        # before the restart was confirmed are not lost (all 20 of the new pass are in the window)
        restart = [(f"r{i}-0", TelemetryEvent(501, 115106, base - 7200 + i, 37.6, 55.7)) for i in range(20)]
        await core.handle(restart)
        assert core.clock.resets == 1 and core.windows.points == 20
        assert [p.ts for p in core.windows.track(115106)] == [base - 7200 + i for i in range(20)]
        assert "foresight_predictor_events" in {m.name for m in core.metrics()}

    asyncio.run(scenario())


class IngestClock:
    """Stamps events like the ingest: its clock observes every fix, events carry the clock and the epoch."""

    def __init__(self, units: dict[int, int]) -> None:
        self.clock = StreamClock()
        self.units = units
        self.seq = 0

    def event(self, unit: int, ts: float, wall: float, lat: float = 55.7) -> tuple[str, TelemetryEvent]:
        self.clock.observe(ts, unit, wall)
        self.seq += 1
        event = TelemetryEvent(
            unit,
            self.units[unit],
            ts,
            37.6,
            lat,
            received_at=wall,
            epoch=self.clock.epoch,
            clock=self.clock.now,
        )
        return f"{self.seq}-0", event


def test_predictor_follows_a_replayer_restart_30_min_back() -> None:
    """The replayer runs 07:00-07:30 and restarts from 07:00: the predictor resets with the ingest."""
    ticks: list[datetime] = []

    async def on_tick(ctx: TickContext) -> None:
        ticks.append(ctx.stream_time)
        for tr_id in ctx.windows:  # the honest view: nothing after the tick time
            assert all(p.ts <= ctx.stream_time.timestamp() for p in ctx.track(tr_id))

    async def scenario() -> None:
        units = {501 + i: 115106 + i for i in range(5)}
        ingest = IngestClock(units)
        core = PredictorCore(window_s=1800, tick_period_s=30, on_tick=on_tick)
        wall = time.time()
        first = [ingest.event(u, BASE + 10 * i, wall + i / 60) for i in range(180) for u in units]
        await core.handle(first)
        assert core.clock.now == BASE + 1790 and core.windows.points == 900 and len(ticks) == 59
        wall += 31
        second = [ingest.event(u, BASE + 10 * i, wall + i / 60) for i in range(30) for u in units]
        await core.handle(second)
        # one clock: the ingest and the predictor jumped back together and agree on "now"
        assert ingest.clock.resets == 1 and core.clock.resets == 1
        assert core.clock.now == ingest.clock.now == BASE + 290 and core.clock.epoch == ingest.clock.epoch
        # every point of the new pass is in the windows, nothing of the old pass is
        assert core.windows.points == 150
        assert all(
            [p.ts for p in core.windows.track(tr)] == [BASE + 10 * i for i in range(30)]
            for tr in units.values()
        )
        # ticks go on from the new start instead of waiting for 07:30 to come back
        assert ticks[59:] == [datetime.fromtimestamp(BASE + 30 * k, UTC) for k in range(1, 10)]

    asyncio.run(scenario())


def test_predictor_keeps_late_points_and_holds_points_from_the_future() -> None:
    async def scenario() -> None:
        ingest = IngestClock({501: 115106, 502: 115107, 503: 115108})
        core = PredictorCore(window_s=1800, tick_period_s=30)
        wall = time.time()
        batch = [ingest.event(u, BASE + 10 * i, wall) for i in range(60) for u in (501, 502, 503)]
        batch.append(ingest.event(501, BASE + 400, wall, lat=55.8))  # reordered packet, 3 min late
        batch.append(ingest.event(501, BASE + 400, wall, lat=55.8))  # ...delivered twice
        batch += [ingest.event(502, BASE - 600 + i, wall) for i in range(3)]  # black box, 10+ min late
        batch.append(ingest.event(503, BASE + 7200, wall))  # a device clock 2 h ahead
        await core.handle(batch)
        assert core.clock.resets == 0 and core.clock.now == BASE + 590
        track = [p.ts for p in core.windows.track(115106)]
        assert track == sorted(track) and track.count(BASE + 400) == 2  # the late point is in, in order
        assert core.windows.track(115107)[:3] == tuple(
            TrackPoint(BASE - 600 + i, 37.6, 55.7, 0, 0, True, 502) for i in range(3)
        )
        assert BASE + 7200 not in [p.ts for p in core.windows.track(115108)]  # never shown before its time
        # the reordered point is late, the black-box and future ones are off the timeline
        assert (core.windows.late, core.windows.duplicates, core.ahead, core.clock.off_timeline) == (
            1,
            1,
            1,
            4,
        )

    asyncio.run(scenario())


def test_predictor_refills_windows_from_the_stream_on_start() -> None:
    """After a restart or re-creation the predictor reads back the last 30 min of its epoch."""

    async def scenario() -> None:
        server = fakeredis.FakeServer()
        factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
        client = factory()
        # epoch 1 went further (a previous pass) until 07:50; the new pass starts at 07:18, its first two
        # points still carry epoch 1 (sent before the ingest confirmed the jump), then epoch 2: 07:20 -> 07:40
        stream = [(1, BASE + 1500 + 60 * i, BASE + 1500 + 60 * i) for i in range(26)]
        # (with a black-box point from 06:58 among them, older than any window: skipped, not the end)
        stream += [(1, BASE + 1080, BASE + 3000), (1, BASE - 100, BASE + 3000), (1, BASE + 1140, BASE + 3000)]
        stream += [(2, BASE + 1200 + 60 * i, BASE + 1200 + 60 * i) for i in range(21)]
        for epoch, ts, clock in stream:
            event = TelemetryEvent(501, 115106, ts, 37.6, 55.7, epoch=epoch, clock=clock)
            await client.xadd(STREAM_TELEMETRY, event.to_fields())
        expected = [BASE + 1080, BASE + 1140] + [BASE + 1200 + 60 * i for i in range(21)]
        first = PredictorCore(window_s=1800)
        consumer = StreamConsumer(
            factory, first.handle, DependencyStatus("redis"), consumer="old", block_ms=20
        )
        while await consumer.poll_once():
            pass
        assert first.events == 50 and first.clock.resets == 1 and consumer.history_entries == 0
        assert [p.ts for p in first.windows.track(115106)] == expected  # the live windows

        core = PredictorCore(window_s=1800, tick_period_s=30)
        restarted = StreamConsumer(
            factory,
            core.handle,
            DependencyStatus("redis"),
            consumer="new",
            batch_size=10,
            block_ms=20,
            history=core.restore,
        )
        assert await restarted.poll_once() == 0
        assert core.clock.now == BASE + 2400 and core.clock.epoch == 2 and core.events == 0
        # the same as the live windows: this pass only (with its first points sent under epoch 1), not the
        # previous pass although its times fall into the window
        assert [p.ts for p in core.windows.track(115106)] == expected
        assert core.restored == 23 and core.ticks_run == 0 and restarted.history_entries >= 23
        # the live stream continues the refilled windows and the tick schedule
        event = TelemetryEvent(501, 115106, BASE + 2430, 37.6, 55.7, epoch=2, clock=BASE + 2430)
        await client.xadd(STREAM_TELEMETRY, event.to_fields())
        await restarted.poll_once()
        assert core.ticks_run == 1 and core.last_tick == datetime.fromtimestamp(BASE + 2430, UTC)
        assert core.windows.points == 24
        await consumer.close()
        await restarted.close()
        await client.aclose()

    asyncio.run(scenario())


# ---- predictor app ------------------------------------------------------------------------------


def test_predictor_app_consumes_the_stream() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = _settings()
    svc = predictor.PredictorService(settings, factory=factory)

    async def add_events() -> None:
        client = factory()
        for i in range(50):
            event = TelemetryEvent(501, 115106 if i % 5 else None, 1_767_690_000.0 + 2 * i, 37.6, 55.7)
            await client.xadd(STREAM_TELEMETRY, event.to_fields())
        await client.aclose()

    with TestClient(predictor.create_app(settings, svc)) as client:
        client.portal.call(add_events)  # type: ignore[union-attr]
        _wait(lambda: client.get("/api/predictor/stats").json()["events"] == 50)
        _wait(lambda: client.get("/api/predictor/stats").json()["lag"] == 0)
        stats = client.get("/api/predictor/stats").json()
        assert stats["unmapped_events"] == 10 and stats["tracks"] == 1 and stats["window_points"] == 40
        assert stats["ticks"] == 3 and stats["pending"] == 0 and stats["group"] == "predictors"
        assert stats["stream_time"].startswith("2026-01-06T09:01:38")
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["dependencies"]["redis"]["state"] == "up"
        text = client.get("/metrics").text
        assert "foresight_consumer_events_total 50.0" in text
        assert "foresight_consumer_lag 0.0" in text
        assert 'foresight_dependency_up{dependency="redis"} 1.0' in text
        assert "foresight_predictor_window_points 40.0" in text
        assert client.get("/openapi.json").json()["info"]["title"] == "Foresight · Predictor"


# ---- api degradation ----------------------------------------------------------------------------


class DownDatabase:
    """A PostgreSQL that never answers."""

    async def connect(self) -> None:
        raise ConnectionRefusedError("postgres is down")

    def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _publish_vehicle(factory: Callable[[], Any]) -> Callable[[], Any]:
    async def publish() -> None:
        store = StateStore()
        store.on_connect(501)
        nav = NavRecord(timestamp=datetime.fromtimestamp(1_767_690_000, UTC), lon=37.6, lat=55.7, speed_avg=5)
        store.on_nav(501, nav)
        publisher = TelemetryPublisher(
            store,
            factory,
            DependencyStatus("redis"),
            unit_map={501: 115106},
            stats_provider=lambda: {"frames": "4"},
        )
        await publisher.flush()
        await publisher.close()

    return publish


def test_api_serves_last_known_state_while_redis_is_down() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = _settings()
    svc = api.ApiService(settings, factory=factory)
    with TestClient(api.create_app(settings, svc)) as client:
        client.portal.call(_publish_vehicle(factory))  # type: ignore[union-attr]
        _wait(lambda: client.get("/api/vehicles").json()["count"] == 1)
        assert client.get("/api/vehicles/501").json()["tr_id"] == 115106
        assert client.get("/api/ingest/stats").json()["frames"] == 4
        with client.websocket_connect("/ws") as ws:
            assert _vehicles_message(ws)["degraded"] is False

            server.connected = False
            _wait(lambda: client.get("/health").json()["dependencies"]["redis"]["state"] == "down")
            health = client.get("/health")
            assert health.status_code == 200 and health.json()["status"] == "degraded"
            listing = client.get("/api/vehicles").json()
            assert listing["degraded"] is True and listing["count"] == 1
            assert listing["vehicles"][0]["tr_id"] == 115106 and listing["vehicles"][0][
                "lat"
            ] == pytest.approx(55.7)
            stats = client.get("/api/ingest/stats").json()
            assert stats["degraded"] is True and stats["frames"] == 4
            message = _vehicles_message(ws)  # the WebSocket announces the degradation with a snapshot
            assert (
                message["type"] == "snapshot"
                and message["degraded"] is True
                and len(message["vehicles"]) == 1
            )

            server.connected = True
            _wait(lambda: client.get("/health").json()["status"] == "ok")
            assert client.get("/api/vehicles").json()["degraded"] is False
            assert client.get("/health").json()["dependencies"]["redis"]["outages"] == 1


def test_api_stays_up_while_postgres_is_down() -> None:
    server = fakeredis.FakeServer()
    settings = _settings(database_url="postgresql://unused")
    svc = api.ApiService(
        settings,
        factory=lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
        database=DownDatabase(),  # type: ignore[arg-type]
    )
    svc.writer.backoff = Backoff(0.02, 0.05)
    with TestClient(api.create_app(settings, svc)) as client:
        _wait(lambda: client.get("/health").json()["dependencies"]["postgres"]["state"] == "down")
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "degraded"
        assert svc.writer.buffered >= 2  # 'start' and 'degraded' journal entries wait for PostgreSQL
        assert client.get("/api/vehicles").status_code == 200
        text = client.get("/metrics").text
        assert 'foresight_dependency_up{dependency="postgres"} 0.0' in text
        assert "foresight_db_buffered" in text


def test_vehicle_cache_ignores_stale_deltas() -> None:
    from backend.bus import VehicleRecord

    cache = VehicleCache(stale_after_s=30, offline_after_s=120)
    now = datetime.now(UTC)
    fresh = VehicleRecord(unit_id=1, connected=True, packets=5, last_packet_at=now, updated_ms=200)
    stale = VehicleRecord(unit_id=1, connected=True, packets=4, last_packet_at=now, updated_ms=100)
    assert cache.apply([fresh]) == 1
    assert cache.apply([stale]) == 0 and cache.get(1).packets == 5  # type: ignore[union-attr]
    assert (
        cache.apply([VehicleRecord(unit_id=1, connected=True, packets=5, last_packet_at=now, updated_ms=300)])
        == 0
    )
    assert cache.status(fresh, now) is LinkStatus.ONLINE
    assert cache.apply([], full=True) == 1 and len(cache) == 0  # gone from Redis -> gone from the cache


def test_vehicle_cache_follows_the_stream_clock_epoch() -> None:
    cache = VehicleCache()

    def at(offset: float) -> datetime:
        return datetime.fromtimestamp(BASE + offset, UTC)

    cache.set_stream_time(at(100), 5)
    cache.set_stream_time(at(90), 5)  # a late message of the same epoch: no roll-back
    assert cache.stream_time == at(100)
    cache.set_stream_time(at(-1800), 7)  # the clock jumped back (a newer epoch): taken although earlier
    assert (cache.stream_time, cache.stream_epoch) == (at(-1800), 7)
    cache.set_stream_time(at(200), 5)  # a late message of the old epoch: ignored
    cache.set_stream_time(None, 8)
    assert (cache.stream_time, cache.stream_epoch) == (at(-1800), 7)


def test_backend_main_requires_a_service_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert backend_main.main([]) == 2
    assert backend_main.main(["monolith"]) == 2
    assert "ingest,predictor,api" in capsys.readouterr().err
