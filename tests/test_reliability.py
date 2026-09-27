"""Reliability of the stream under chaos: NDTP outages, a dead ingest, redelivered entries, abusive clients.

* the stream clock keeps its epoch when the devices reconnect one by one after an ingest restart (or a network
  outage) and flush their backlogs, however long the outage; a real source restart is still detected;
* the predictor never goes back to an older epoch of a redelivered entry;
* the api tells a dead ingest from a live one by the age of its stats;
* the NDTP server closes silent, garbage-only and surplus connections and caps its stats list.
"""

import asyncio
import contextlib
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
fakeredis = pytest.importorskip("fakeredis")

from fastapi.testclient import TestClient  # noqa: E402

from backend import api, ingest  # noqa: E402
from backend.bus import STREAM_DEVICES_KEY, TelemetryEvent, TelemetryPublisher  # noqa: E402
from backend.clock import ClockJump, Fix, StreamClock  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.ndtp_server import NdtpServer  # noqa: E402
from backend.predictor import PredictorCore  # noqa: E402
from backend.runtime import DependencyStatus  # noqa: E402
from backend.state import StateStore  # noqa: E402
from shared.ndtp import NavRecord, encode_handshake, encode_realtime  # noqa: E402

WALL = 1_790_000_000.0  # 2026-09-21: the wall clock of the clock tests
BASE = 1_767_682_800.0  # 2026-01-06 07:00:00 UTC: replayed data


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = dict(
        ndtp_host="127.0.0.1",
        ndtp_port=0,
        database_url="",
        unit_map_splits="",
        ws_interval_s=0.2,
        api_resync_s=0.3,
        bus_flush_interval_s=0.02,
        stats_interval_s=0.05,
        backoff_initial_s=0.02,
        backoff_max_s=0.1,
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


def _nav(ts: float, lat: float = 55.75) -> NavRecord:
    return NavRecord(timestamp=datetime.fromtimestamp(ts, UTC), lon=37.61, lat=lat, speed_avg=20, course=90)


# ---- stream clock: reconnect waves ----------------------------------------------------------------


def _reconnect_wave(
    t0: float, devices: int, *, outage_s: float, speed: float, gap_s: float, live_s: float
) -> list[tuple[float, int, float]]:
    """Fixes ``(wall, device, ts)`` of a replay at ``speed`` whose receiver was down for ``outage_s``
    wall seconds from data time ``t0``. Device ``d`` reconnects ``gap_s * d`` later than the first one,
    flushes its whole backlog (a fix every 10 s since ``t0``) at once, then sends live fixes for ``live_s``
    wall seconds."""
    fixes: list[tuple[float, int, float]] = []
    end = t0 + (outage_s + gap_s * devices + live_s) * speed
    for dev in range(devices):
        back = outage_s + gap_s * dev  # wall seconds after the outage started
        ts = t0 + 1 + dev % 7
        while ts < end:
            wall = back if ts <= t0 + back * speed else (ts - t0) / speed
            fixes.append((WALL + wall, dev, ts))
            ts += 10
    fixes.sort(key=lambda f: f[0])  # stable: a backlog keeps its order inside the burst
    return fixes


def test_clock_keeps_its_epoch_while_devices_flush_backlogs_after_an_ingest_restart() -> None:
    """The ingest was down 14 s at x30 (7 min of data): 30 devices reconnect one after another, each
    flushing its 7-minute backlog. The first ones move the clock on; the others' backlogs are late only."""
    jumps: list[ClockJump] = []
    clock = StreamClock(jump_s=300, confirm=3, quorum=0.5, active_s=60, on_jump=jumps.append)
    t0 = BASE + 1800
    clock.restore(t0, 111, devices=30, wall=WALL)
    fixes = _reconnect_wave(t0, 30, outage_s=14, speed=30, gap_s=0.25, live_s=10)
    results = [clock.observe(ts, dev, wall) for wall, dev, ts in fixes]
    assert Fix.JUMP not in results and jumps == [] and clock.resets == 0
    assert clock.epoch == 111 and clock.now == max(ts for _, _, ts in fixes)
    assert results.count(Fix.OFF) > 30 * 10  # most backlog points are far behind the clock: late only

    # a real restart right after the wave is still detected, once
    wall = fixes[-1][0] + 1
    restart = [clock.observe(BASE + 10 * r + dev % 7, dev, wall + r) for r in range(3) for dev in range(30)]
    assert restart.count(Fix.JUMP) == 1 and clock.resets == 1 and jumps[0].back
    assert clock.now < BASE + 60


def test_clock_keeps_its_epoch_after_a_network_outage_of_any_length() -> None:
    """The ingest stayed up; the devices were cut off for 2 min at x30 (1 h of data), back one by one."""
    clock = StreamClock(jump_s=300, confirm=3, quorum=0.5, active_s=60)
    for i in range(30):  # a minute of normal traffic
        for dev in range(20):
            assert clock.observe(BASE + 10 * i + dev % 7, dev, WALL + i / 3) is not Fix.JUMP
    t0, epoch = clock.now, clock.epoch
    assert t0 is not None
    fixes = _reconnect_wave(t0, 20, outage_s=120, speed=30, gap_s=0.5, live_s=5)
    results = [clock.observe(ts, dev, wall + 10) for wall, dev, ts in fixes]
    assert Fix.JUMP not in results and clock.resets == 0 and clock.epoch == epoch


def test_clock_after_a_restore_needs_the_devices_that_were_active() -> None:
    clock = StreamClock(jump_s=300, confirm=3, quorum=0.5, active_s=60)
    clock.restore(BASE + 1800, 111, devices=20, wall=WALL)
    # the first device back dumps its black box (an hour back): 1 of 20 expected devices is no quorum
    fixes = [clock.observe(BASE + 1800 - 3600 + i, 7, WALL + 1 + i / 10) for i in range(10)]
    assert Fix.JUMP not in fixes and clock.resets == 0 and clock.now == BASE + 1800
    # the others come back on the timeline
    for dev in range(20):
        assert clock.observe(BASE + 1805 + dev, dev, WALL + 3) is Fix.ON
    # once the expected devices are no longer counted (active_s after the restore) a lone device that really
    # restarted moves the clock, as without a restore
    alone = [clock.observe(BASE + 10 * i, 3, WALL + 200 + i) for i in range(3)]
    assert alone == [Fix.OFF, Fix.OFF, Fix.JUMP] and clock.resets == 1


def test_publisher_restores_the_clock_with_the_devices_that_were_active() -> None:
    async def scenario() -> None:
        server = fakeredis.FakeServer()
        factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
        store = StateStore()
        publisher = TelemetryPublisher(store, factory, DependencyStatus("redis"))
        for dev in range(10):
            store.on_connect(dev)
            store.on_nav(dev, _nav(BASE + dev))
        for dev in range(10, 30):  # connected, but no fix: not part of the clock's quorum
            store.on_connect(dev)
        await publisher.flush()
        await publisher.close()
        client = factory()
        assert await client.get(STREAM_DEVICES_KEY) == "10"
        await client.aclose()

        # the ingest restarts; the first device back sends three black-box fixes an hour behind
        restarted = TelemetryPublisher(StateStore(), factory, DependencyStatus("redis"))
        assert await restarted.load_clock()
        clock = restarted.store.stream_clock
        assert clock.now == BASE + 9
        black_box = [clock.observe(BASE - 3600 + i, 0) for i in range(3)]
        assert black_box == [Fix.OFF] * 3 and clock.resets == 0  # 1 of the 10 active devices
        assert clock.active_devices() == 10  # saved again as 10 while they reconnect
        await restarted.close()

        # the same fixes after a restore that knows no devices would have been a quorum of one
        bare = StreamClock()
        bare.restore(BASE + 9, clock.epoch)
        assert [bare.observe(BASE - 3600 + i, 0) for i in range(3)][-1] is Fix.JUMP

    asyncio.run(scenario())


# ---- predictor: redelivered entries of an older epoch ------------------------------------------------


def test_predictor_ignores_a_redelivered_older_epoch() -> None:
    """Pending entries redelivered after a restart span the jump: the predictor must not go back to the old
    epoch (it would clear its windows twice and journal two false resets)."""

    async def scenario() -> None:
        units = {501 + i: 115106 + i for i in range(4)}
        clock = StreamClock()
        seq = 0

        def event(unit: int, ts: float, wall: float) -> tuple[str, TelemetryEvent]:
            nonlocal seq
            clock.observe(ts, unit, wall)
            seq += 1
            return f"{seq}-0", TelemetryEvent(
                unit, units[unit], ts, 37.6, 55.7, received_at=wall, epoch=clock.epoch, clock=clock.now
            )

        core = PredictorCore(window_s=1800, tick_period_s=30)
        wall = time.time()
        first = [event(u, BASE + 10 * i, wall + i / 60) for i in range(180) for u in units]
        await core.handle(first)
        second = [event(u, BASE - 1800 + 10 * i, wall + 31 + i / 60) for i in range(14) for u in units]
        await core.handle(second)
        assert core.clock.resets == 1 and core.windows.points == 56
        now, epoch, ticks = core.clock.now, core.clock.epoch, core.ticks_run
        first_of_new_epoch = [e for e in second if e[1].epoch == epoch][:3]
        assert len(first_of_new_epoch) == 3 and first[-1][1].epoch < epoch

        # redelivered around the jump (own pending entries after a restart, XAUTOCLAIM)
        await core.handle(first[-3:] + first_of_new_epoch)
        assert core.clock.resets == 1 and (core.clock.now, core.clock.epoch) == (now, epoch)
        assert core.windows.points == 56 and core.ticks_run == ticks
        assert core.windows.duplicates == 3  # the points of the new epoch are exact repeats

    asyncio.run(scenario())


# ---- api: a dead ingest ---------------------------------------------------------------------------


def _ingest_alive(factory: Callable[[], Any]) -> Callable[[], Any]:
    """What a live ingest leaves in Redis: a connected vehicle and fresh stats."""

    async def publish() -> None:
        store = StateStore()
        store.on_connect(501)
        store.on_nav(501, _nav(BASE))
        publisher = TelemetryPublisher(
            store,
            factory,
            DependencyStatus("redis"),
            unit_map={501: 115106},
            stats_provider=lambda: {
                "updated_at": datetime.now(UTC).isoformat(),
                "listening": "1",
                "connections_active": "1",
                "packets_per_s": "3.5",
                "frames": "4",
            },
        )
        await publisher.flush()
        await publisher.close()

    return publish


def test_api_reports_a_dead_ingest_and_its_recovery() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = _settings(ingest_stale_after_s=1.5)
    svc = api.ApiService(settings, factory=factory)
    with TestClient(api.create_app(settings, svc)) as client:

        def ingest_state() -> str:
            return client.get("/health").json()["dependencies"]["ingest"]["state"]

        def alive_and(predicate: Callable[[], bool]) -> Callable[[], bool]:
            def check() -> bool:  # a live ingest keeps writing its stats
                client.portal.call(_ingest_alive(factory))  # type: ignore[union-attr]
                return predicate()

            return check

        assert ingest_state() == "unknown"  # no stats yet: not a failure
        _wait(alive_and(lambda: ingest_state() == "up"))
        assert client.get("/health").json()["status"] == "ok"
        listing = client.get("/api/vehicles").json()
        assert listing["degraded"] is False and listing["vehicles"][0]["connected"] is True
        assert listing["vehicles"][0]["status"] == "online"
        stats = client.get("/api/ingest/stats").json()
        assert stats["available"] is True and stats["connections_active"] == 1 and stats["age_s"] < 1.5

        with client.websocket_connect("/ws") as ws:
            assert _vehicles_message(ws)["type"] == "snapshot"
            # kill -9 of the ingest: no more stats, the hot state in Redis still says "connected"
            _wait(lambda: ingest_state() == "down")
            for _ in range(10):  # the WebSocket announces it with a snapshot
                message = _vehicles_message(ws)
                if message["degraded"]:
                    break
            assert message["type"] == "snapshot" and message["degraded"] is True
            first = message["vehicles"][0]
            assert (first["connected"], first["status"]) == (False, "offline")

        health = client.get("/health").json()
        assert health["status"] == "degraded" and "no stats" in health["dependencies"]["ingest"]["error"]
        listing = client.get("/api/vehicles").json()
        assert listing["degraded"] is True and listing["status_counts"]["offline"] == 1
        vehicle = listing["vehicles"][0]
        assert (vehicle["connected"], vehicle["status"], vehicle["tr_id"]) == (False, "offline", 115106)
        assert client.get("/api/vehicles/501").json()["connected"] is False
        stats = client.get("/api/ingest/stats").json()
        assert stats["available"] is False and stats["degraded"] is True and stats["age_s"] > 1.5
        assert (stats["listening"], stats["connections_active"], stats["packets_per_s"]) == (False, 0, 0.0)
        assert stats["frames"] == 4  # the counters stay as last known
        text = client.get("/metrics").text
        assert 'foresight_dependency_up{dependency="ingest"} 0.0' in text
        assert "foresight_api_degraded 1.0" in text and 'foresight_vehicles{status="offline"} 1.0' in text

        # the ingest is back
        _wait(alive_and(lambda: client.get("/health").json()["status"] == "ok"))
        ingest_dep = client.get("/health").json()["dependencies"]["ingest"]
        assert ingest_dep["state"] == "up" and ingest_dep["outages"] == 1
        assert client.get("/api/vehicles").json()["vehicles"][0]["connected"] is True


# ---- NDTP server: anti-DoS limits ---------------------------------------------------------------------


async def _closed_by_server(reader: asyncio.StreamReader, timeout: float = 3.0) -> bool:
    """Whether the server closed the connection (EOF or reset) within ``timeout``."""
    try:
        return await asyncio.wait_for(reader.read(), timeout) == b""
    except ConnectionError:
        return True


def test_ndtp_server_closes_silent_garbage_and_surplus_connections() -> None:
    async def scenario() -> None:
        store = StateStore()
        server = NdtpServer(
            store, "127.0.0.1", 0, first_frame_timeout_s=0.3, max_garbage_bytes=1024, max_connections=2
        )
        await server.start()
        writers: list[asyncio.StreamWriter] = []
        try:

            async def connect() -> asyncio.StreamReader:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                writers.append(writer)
                return reader

            # a connection that never sends a valid frame is closed after first_frame_timeout_s
            started = time.monotonic()
            assert await _closed_by_server(await connect())
            assert 0.2 < time.monotonic() - started < 2.0 and server.stats.first_frame_timeouts == 1

            # a stream without a single 0x7E7E signature never costs a CRC: the garbage budget closes it
            reader = await connect()
            writers[-1].write(bytes(8192))
            await writers[-1].drain()
            assert await _closed_by_server(reader)
            assert server.stats.abusive_disconnects == 1 and server.stats.crc_errors == 0

            # a device that talks stays past the deadline; the connection limit closes the surplus at once
            await connect()
            writers[-1].write(encode_handshake(77) + encode_realtime(77, 1, _nav(BASE)))
            await writers[-1].drain()
            await connect()  # silent, counts until its deadline
            for _ in range(100):
                if len(server.connections) == 2 and store.get(77) is not None:
                    break
                await asyncio.sleep(0.01)
            assert len(server.connections) == 2
            assert await _closed_by_server(await connect(), timeout=0.2)
            assert server.stats.rejected_connections == 1 and server.stats.connections_total == 4
            await asyncio.sleep(0.5)
            vehicle = store.get(77)
            assert vehicle is not None and vehicle.connections == 1 and vehicle.packets == 1
            assert server.stats.first_frame_timeouts == 2 and len(server.connections) == 1
        finally:
            for writer in writers:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            await server.stop()

    asyncio.run(scenario())


def test_ingest_stats_list_a_bounded_number_of_connections() -> None:
    async def scenario() -> None:
        server = fakeredis.FakeServer()
        svc = ingest.IngestService(
            _settings(stats_connections_max=2),
            factory=lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
            unit_map={},
        )
        await svc.start()
        socks = []
        try:
            for unit in range(5):
                sock = socket.create_connection(("127.0.0.1", svc.ndtp.port))
                sock.sendall(encode_handshake(700 + unit))
                socks.append(sock)
            for _ in range(200):
                if svc.ndtp.stats.handshakes == 5:
                    break
                await asyncio.sleep(0.01)
            stats = svc.stats()
            assert stats.connections_active == 5 and len(stats.connections) == 2
            oldest = sorted(svc.ndtp.connections)[:2]
            assert [c.conn_id for c in stats.connections] == oldest
            assert all(c.unit_id is not None and 700 <= c.unit_id < 705 for c in stats.connections)
        finally:
            for sock in socks:
                sock.close()
            await svc.stop()

    asyncio.run(scenario())
