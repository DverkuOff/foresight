"""Tests for the backend services end to end: NDTP over TCP -> ingest -> Redis -> api REST and WebSocket.

Redis is ``fakeredis`` shared by the ingest and api of one test; both run in the event loop of the api's
``TestClient`` (the ingest is started through ``client.portal``), so pub/sub works as with a real server.
"""

import asyncio
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
fakeredis = pytest.importorskip("fakeredis")

from fastapi.testclient import TestClient  # noqa: E402

from backend import api, ingest  # noqa: E402
from backend.bus import STREAM_TELEMETRY, TelemetryEvent  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.ndtp_server import NdtpServer  # noqa: E402
from backend.predictor import PredictorCore  # noqa: E402
from backend.state import LinkStatus, StateStore  # noqa: E402
from shared.ndtp import IrmaRecord, NavRecord, encode_handshake, encode_irma, encode_realtime  # noqa: E402


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = dict(
        ndtp_host="127.0.0.1",
        ndtp_port=0,
        ws_interval_s=0.2,
        database_url="",
        unit_map_splits="",
        bus_flush_interval_s=0.02,
        stats_interval_s=0.05,
        api_resync_s=0.5,
        backoff_initial_s=0.05,
        backoff_max_s=0.2,
    )
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


def _nav(lat: float = 55.75, lon: float = 37.61, ts: int = 1_790_000_000, speed: int = 30) -> NavRecord:
    return NavRecord(
        timestamp=datetime.fromtimestamp(ts, UTC), lon=lon, lat=lat, speed_avg=speed, course=90, nsat=9
    )


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


class Stack:
    """api app (TestClient) + ingest service in the same event loop, sharing one fake Redis."""

    def __init__(self, unit_map: dict[int, int] | None = None, **settings: object) -> None:
        self.server = fakeredis.FakeServer()
        self.settings = _settings(**settings)
        self.factory = lambda: fakeredis.FakeAsyncRedis(server=self.server, decode_responses=True)
        self.api = api.ApiService(self.settings, factory=self.factory)
        self.app = api.create_app(self.settings, self.api)
        self.ingest = ingest.IngestService(self.settings, factory=self.factory, unit_map=unit_map or {})
        self.client = TestClient(self.app)

    def __enter__(self) -> "Stack":
        self.client.__enter__()
        self.client.portal.call(self.ingest.start)  # type: ignore[union-attr]
        # the api is 'degraded' until its first full read from Redis
        _wait(lambda: self.client.get("/health").json()["dependencies"]["redis"]["state"] == "up")
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.client.portal.call(self.ingest.stop)  # type: ignore[union-attr]
        finally:
            self.client.__exit__(*exc)

    def redis(self, method: str, *args: object) -> object:
        """Run one Redis command on the shared fake server inside the app's event loop."""

        async def call() -> object:
            client = self.factory()
            try:
                return await getattr(client, method)(*args)
            finally:
                await client.aclose()

        return self.client.portal.call(call)  # type: ignore[union-attr]


def test_health_and_empty_vehicles() -> None:
    with Stack() as stack:
        client = stack.client
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["service"] == "api"
        assert health["dependencies"]["redis"]["state"] == "up"
        assert health["dependencies"]["postgres"]["state"] == "disabled"
        body = client.get("/api/vehicles").json()
        assert body["count"] == 0 and body["vehicles"] == [] and body["degraded"] is False
        assert client.get("/api/vehicles/1").status_code == 404
        _wait(lambda: client.get("/api/ingest/stats").json()["available"] is True)
        stats = client.get("/api/ingest/stats").json()
        assert stats["frames"] == 0 and stats["listening"] is True and stats["redis"] == "up"
        assert "foresight_api_ws_clients" in client.get("/metrics").text
        assert client.get("/openapi.json").json()["info"]["title"] == "Foresight · API"

        # the ingest's own HTTP app
        with TestClient(ingest.create_app(_settings(), _standalone_ingest())) as ing:
            health = ing.get("/health").json()
            assert health["ndtp_listening"] is True and health["ndtp_port"] > 0
            assert ing.get("/api/ingest/stats").json()["frames"] == 0
            assert "foresight_ndtp_frames_total" in ing.get("/metrics").text
            assert ing.get("/openapi.json").json()["info"]["title"] == "Foresight · Ingest"


def _standalone_ingest() -> ingest.IngestService:
    server = fakeredis.FakeServer()
    return ingest.IngestService(
        _settings(),
        factory=lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
        unit_map={},
    )


def test_end_to_end_tcp_to_rest_and_ws() -> None:
    with Stack(unit_map={501: 115106}) as stack:
        client = stack.client
        with client.websocket_connect("/ws") as ws:
            snapshot = _vehicles_message(ws)
            assert snapshot["type"] == "snapshot" and snapshot["vehicles"] == []

            port = stack.ingest.ndtp.port
            irma = IrmaRecord(0, 10, 1, (1, 0, 0, 0), (0, 2, 0, 0), (True,) * 4, (False, True, True, True))
            stream = (
                b"garbage"
                + encode_handshake(501, 1)
                + encode_realtime(501, 2, _nav(), [encode_irma(irma)])
                + encode_realtime(501, 3, _nav(lat=55.76, ts=1_790_000_005))
            )
            with socket.create_connection(("127.0.0.1", port)) as sock:
                for i in range(0, len(stream), 11):  # split into small TCP writes
                    sock.sendall(stream[i : i + 11])
                _wait(lambda: client.get("/api/vehicles/501").json().get("packets") == 2)

                vehicle = client.get("/api/vehicles/501").json()
                assert vehicle["tr_id"] == 115106
                assert vehicle["status"] == "online" and vehicle["connected"] is True
                assert vehicle["lat"] == pytest.approx(55.76) and vehicle["lon"] == pytest.approx(37.61)
                assert vehicle["speed_kmh"] == 30 and vehicle["course_deg"] == 90 and vehicle["valid"] is True
                # a realtime fix time (21.09, 14:13:25Z) goes onto the plan day by the Moscow time of day
                assert vehicle["event_time"].startswith("2026-01-06T17:13:25")
                assert vehicle["doors"]["door_closed"] == [False, True, True, True]

                delta = _vehicles_message(ws)
                assert delta["type"] == "delta" and delta["degraded"] is False
                assert [v["unit_id"] for v in delta["vehicles"]] == [501]

                _wait(lambda: client.get("/api/ingest/stats").json()["realtime_packets"] == 2)
                stats = client.get("/api/ingest/stats").json()
                assert stats["handshakes"] == 1 and stats["realtime_packets"] == 2
                assert stats["bytes_discarded"] == len(b"garbage")
                assert stats["connections_active"] == 1
                assert stats["connections"][0]["unit_id"] == 501
                assert stats["bus_published"] == 2 and stats["bus_buffered"] == 0

            # both navigation events are on the stream with tr_id
            entries = stack.redis("xrange", STREAM_TELEMETRY)
            assert [e[1]["tr_id"] for e in entries] == ["115106", "115106"]  # type: ignore[union-attr]
            # realtime fix times (21.09, 14:13Z) on the plan day by the Moscow time of day: 06.01, 17:13
            assert [e[1]["ts"] for e in entries] == ["1767719600", "1767719605"]  # type: ignore[union-attr]
            assert "doors" in entries[0][1] and "doors" not in entries[1][1]  # type: ignore[index]
            # leave only after the handler is done: TestClient cancels a running one and that races (flaky)
            ws.close()
            _wait(lambda: stack.api.ws_clients == 0)

        # the device dropped the connection: the service stays up, the vehicle goes offline
        _wait(lambda: client.get("/api/vehicles/501").json()["connected"] is False)
        vehicle = client.get("/api/vehicles/501").json()
        assert vehicle["status"] == "offline" and vehicle["lat"] == pytest.approx(55.76)
        assert client.get("/health").json()["status"] == "ok"
        listing = client.get("/api/vehicles", params={"status": "offline"}).json()
        assert listing["count"] == 1 and listing["stream_time"].startswith("2026-01-06T17:13")
        assert 'foresight_vehicles{status="offline"} 1.0' in client.get("/metrics").text


def test_replayer_restart_moves_every_clock_back_together() -> None:
    """A replay of 07:00-07:30 restarts from 07:00: the ingest, the api and the predictor all follow."""
    base = 1_767_682_800  # 2026-01-06 07:00:00 UTC
    units = {501: 115106, 502: 115107}
    with Stack(unit_map=units) as stack:
        client = stack.client
        socks = {u: socket.create_connection(("127.0.0.1", stack.ingest.ndtp.port)) for u in units}
        sent = 0

        def send(ts: int) -> None:  # one fix per device, in step like real devices
            nonlocal sent
            for unit, sock in socks.items():
                sent += 1
                sock.sendall(encode_realtime(unit, sent, _nav(ts=ts)))
            _wait(lambda: stack.ingest.ndtp.stats.nav_records == sent)

        def api_stream_time() -> str:
            return client.get("/api/vehicles").json()["stream_time"] or ""

        try:
            for ts in range(base, base + 1801, 60):  # the first pass
                send(ts)
            _wait(lambda: api_stream_time().startswith("2026-01-06T07:30:00"))
            epoch = stack.ingest.store.stream_clock.epoch
            for ts in range(base, base + 51, 10):  # the replayer starts over
                send(ts)
            _wait(lambda: api_stream_time().startswith("2026-01-06T07:00:50"))  # back, not stuck at 07:30
            _wait(lambda: client.get("/api/ingest/stats").json()["clock_resets"] == 1)
            _wait(lambda: stack.ingest.publisher.published == sent)
        finally:
            for sock in socks.values():
                sock.close()
        new_epoch = stack.ingest.store.stream_clock.epoch
        assert new_epoch > epoch

        # a predictor fed from the stream jumps with the ingest and keeps every point of the new pass
        entries: Any = stack.redis("xrange", STREAM_TELEMETRY)
        events = [(entry_id, TelemetryEvent.from_fields(fields)) for entry_id, fields in entries]
        assert len(events) == sent and {e.epoch for _, e in events} == {epoch, new_epoch}
        core = PredictorCore(window_s=1800, tick_period_s=30)
        asyncio.run(core.handle(events))
        assert core.clock.resets == 1 and core.clock.now == base + 50
        for tr_id in units.values():
            assert [p.ts for p in core.windows.track(tr_id)] == list(range(base, base + 51, 10))


def test_ndtp_server_survives_bad_input_and_reconnects() -> None:
    async def scenario() -> None:
        store = StateStore()
        server = NdtpServer(store, "127.0.0.1", 0)
        await server.start()
        try:
            # 1) pure garbage, then an abrupt close
            _, w = await asyncio.open_connection("127.0.0.1", server.port)
            w.write(b"\x7e\x7e\xff\xff" + bytes(range(256)))
            await w.drain()
            w.close()

            # 2) a frame with a broken CRC followed by a good one, no handshake
            bad = bytearray(encode_realtime(77, 1, _nav()))
            bad[-3] ^= 0xFF
            _, w = await asyncio.open_connection("127.0.0.1", server.port)
            w.write(bytes(bad) + encode_realtime(77, 2, _nav(lat=-10.5, lon=-20.25)))
            await w.drain()
            for _ in range(100):
                if store.get(77) is not None and store.get(77).packets == 1:
                    break
                await asyncio.sleep(0.01)
            vehicle = store.get(77)
            assert vehicle is not None and vehicle.nav is not None
            assert (vehicle.nav.lat, vehicle.nav.lon) == (-10.5, -20.25)
            assert server.stats.crc_errors == 1
            w.close()
            await w.wait_closed()

            # 3) reconnect with a handshake: the state continues
            _, w = await asyncio.open_connection("127.0.0.1", server.port)
            w.write(encode_handshake(77) + encode_realtime(77, 1, _nav(lat=-10.6, lon=-20.25)))
            await w.drain()
            for _ in range(100):
                if store.get(77).packets == 2 and store.get(77).connections == 1:
                    break
                await asyncio.sleep(0.01)
            assert store.get(77).packets == 2
            assert store.status(store.get(77)) is LinkStatus.ONLINE
            assert server.listening
            w.close()
        finally:
            await server.stop()
        assert not server.listening

    asyncio.run(scenario())


def test_status_thresholds() -> None:
    now = [datetime(2026, 9, 25, 12, 0, tzinfo=UTC)]
    store = StateStore(stale_after_s=30, offline_after_s=120, clock=lambda: now[0])
    store.on_connect(1)
    store.on_nav(1, _nav())
    vehicle = store.get(1)
    assert vehicle is not None
    assert store.status(vehicle) is LinkStatus.ONLINE
    now[0] += timedelta(seconds=45)
    assert store.status(vehicle) is LinkStatus.STALE
    now[0] += timedelta(seconds=100)
    assert store.status(vehicle) is LinkStatus.OFFLINE
    store.on_nav(1, _nav())
    assert store.status(vehicle) is LinkStatus.ONLINE
    store.on_disconnect(1)
    assert store.status(vehicle) is LinkStatus.OFFLINE
    assert store.changed_since(0) == [vehicle]


def test_ingest_health_503_when_ndtp_is_not_listening() -> None:
    svc = _standalone_ingest()
    with TestClient(ingest.create_app(_settings(), svc)) as client:
        assert client.get("/health").status_code == 200
        client.portal.call(svc.ndtp.stop)  # type: ignore[union-attr]
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded" and response.json()["ndtp_listening"] is False


def test_realtime_fix_times_go_onto_the_plan_day() -> None:
    from datetime import date

    from backend.clock import to_plan_day

    day = date(2026, 1, 6)
    replayed = datetime(2026, 1, 6, 7, 30, tzinfo=UTC)  # the dataset day as it is on the wire: kept
    assert to_plan_day(replayed, day, 10_800) == replayed
    assert to_plan_day(datetime(2026, 1, 7, 0, 40, tzinfo=UTC), day, 10_800) == datetime(
        2026, 1, 7, 0, 40, tzinfo=UTC
    )
    # the emulator stamps the current UTC time: 16:28Z is 19:28 in Moscow — the plan's 19:28 of the plan day
    live = datetime(2026, 9, 27, 16, 28, 33, tzinfo=UTC)
    assert to_plan_day(live, day, 10_800) == datetime(2026, 1, 6, 19, 28, 33, tzinfo=UTC)
    # after midnight (00:40 local) — the end of the plan day
    assert to_plan_day(datetime(2026, 9, 27, 21, 40, tzinfo=UTC), day, 10_800) == datetime(
        2026, 1, 7, 0, 40, tzinfo=UTC
    )
