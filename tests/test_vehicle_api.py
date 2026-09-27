"""``VehicleOut`` of the api: the last valid position (never 0, 0 for a lost fix), the forecast of the
predictor (route, risk, P10–P90, incident) and the derived features of criterion 3, vehicles outside the
schedule, and ``GET /api/routes`` — through the hot state in Redis, as in the stack."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
fakeredis = pytest.importorskip("fakeredis")

from fastapi.testclient import TestClient  # noqa: E402

from backend import api  # noqa: E402
from backend.bus import (  # noqa: E402
    FORECAST_KEY,
    FORECAST_STATUS_KEY,
    ROUTES_KEY,
    TelemetryPublisher,
    VehicleRecord,
    vehicle_fields,
)
from backend.config import Settings  # noqa: E402
from backend.hotstate import HotStateSync, VehicleCache  # noqa: E402
from backend.runtime import DependencyStatus  # noqa: E402
from backend.state import StateStore  # noqa: E402
from shared.ndtp import NavRecord  # noqa: E402

BASE = 1_767_690_000


def _nav(offset: int, lat: float, lon: float, valid: bool = True) -> NavRecord:
    return NavRecord(
        timestamp=datetime.fromtimestamp(BASE + offset, UTC), lon=lon, lat=lat, valid=valid, speed_avg=12
    )


def _record(store: StateStore, unit_id: int, tr_id: int | None) -> VehicleRecord:
    vehicle = store.get(unit_id)
    assert vehicle is not None
    return VehicleRecord.from_fields(vehicle_fields(vehicle, tr_id, 1))


FORECAST = {
    "tr_id": 115106,
    "route_id": "R3",
    "scheduled": True,
    "risk": "red",
    "current_delay_s": 140.0,
    "pred_delay_s": 180.5,
    "p10": 120.0,
    "p90": 260.0,
    "p_late": 0.81,
    "next_stop": {"stop_id": 77, "stop_key": "37.60000,55.70000", "name": "ул. Тверская", "planned_at": None},
    "incident_id": 42,
    "source": "model",
    "segment_speed_kmh": 8.5,
    "dwell_s": 95.0,
    "idle_s": 180.0,
    "gps_age_s": 12.0,
}


def test_position_is_the_last_valid_fix_never_zero() -> None:
    store = StateStore()
    store.on_connect(501)
    store.on_nav(501, _nav(0, 0.0, 0.0, valid=False))  # no fix yet: the device sends zeros
    record = _record(store, 501, 115106)
    out = api.vehicle_out(VehicleCache(), record, datetime.now(UTC))
    assert out.lat is None and out.lon is None and out.valid is False and out.position_age_s is None
    store.on_nav(501, _nav(10, 55.7, 37.6))
    store.on_nav(501, _nav(40, 0.0, 0.0, valid=False))  # the fix is lost: the last valid position stays
    out = api.vehicle_out(VehicleCache(), _record(store, 501, 115106), datetime.now(UTC))
    assert (out.lat, out.lon, out.valid) == (pytest.approx(55.7), pytest.approx(37.6), False)
    assert out.position_age_s == 30.0 and out.position_time == datetime.fromtimestamp(BASE + 10, UTC)
    store.on_nav(501, _nav(50, 55.71, 37.61))
    out = api.vehicle_out(VehicleCache(), _record(store, 501, 115106), datetime.now(UTC))
    assert out.lat == pytest.approx(55.71) and out.position_age_s == 0.0 and out.valid is True


def test_forecast_and_derived_features_of_a_scheduled_vehicle() -> None:
    store = StateStore()
    store.on_connect(501)
    store.on_nav(501, _nav(0, 55.7, 37.6))
    store.on_connect(502)
    store.on_nav(502, _nav(0, 55.8, 37.5))
    cache = VehicleCache()
    cache.set_stream_time(datetime.fromtimestamp(BASE + 60, UTC), 1)
    status = {"stream_time": datetime.fromtimestamp(BASE + 30, UTC).isoformat(), "scheduled_tr_ids": [115106]}
    changed = cache.set_forecasts({115106: FORECAST}, status)
    out = api.vehicle_out(cache, _record(store, 501, 115106), datetime.now(UTC))
    assert (out.scheduled, out.route_id, out.risk, out.incident_id) == (True, "R3", "red", 42)
    assert (out.pred_delay_s, out.p10, out.p90, out.p_late) == (180.5, 120.0, 260.0, 0.81)
    assert out.current_delay_s == 140.0 and out.forecast_source == "model"
    assert (out.segment_speed_kmh, out.dwell_s, out.idle_s) == (8.5, 95.0, 180.0)
    assert out.next_stop is not None and out.next_stop.name == "ул. Тверская"
    assert changed == 0  # no record in the cache yet: nothing to re-version
    # a vehicle outside the plan schedule: marked, no route, risk unknown
    other = api.vehicle_out(cache, _record(store, 502, 999), datetime.now(UTC))
    assert (other.scheduled, other.route_id, other.risk, other.pred_delay_s) == (False, None, "unknown", None)
    unknown = api.vehicle_out(cache, _record(store, 502, None), datetime.now(UTC))
    assert unknown.scheduled is False and unknown.risk == "unknown"
    # the predictor stopped: its snapshot lags the stream clock — risk unknown, not an old red
    cache.set_stream_time(datetime.fromtimestamp(BASE + 30 + cache.forecast_stale_s + 60, UTC), 1)
    stale = api.vehicle_out(cache, _record(store, 501, 115106), datetime.now(UTC))
    assert stale.risk == "unknown" and stale.pred_delay_s is None and stale.scheduled is True


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


ROUTES = {
    "routes": [
        {
            "route_id": "R3",
            "name": "Маршрут R3: А — Б",
            "tr_ids": [115106],
            "stops": [
                {"stop_key": "37.60000,55.70000", "name": "А", "lat": 55.7, "lon": 37.6, "seq": 0},
                {"stop_key": "37.61000,55.70000", "name": "Б", "lat": 55.7, "lon": 37.61, "seq": 1},
            ],
            "line": [[37.6, 55.7], [37.605, 55.701], [37.61, 55.7]],
            "directions": [
                {
                    "direction": 0,
                    "name": "А → Б",
                    "stops": [
                        {"stop_key": "37.60000,55.70000", "name": "А", "lat": 55.7, "lon": 37.6, "seq": 0},
                        {"stop_key": "37.61000,55.70000", "name": "Б", "lat": 55.7, "lon": 37.61, "seq": 1},
                    ],
                    "line": [[37.6, 55.7], [37.605, 55.701], [37.61, 55.7]],
                    "segments": 1,
                    "gps_segments": 1,
                }
            ],
            "color": "#4C8DF6",
        }
    ],
    "scheduled_tr_ids": [115106],
    "geometry": {"segments": 1, "gps_segments": 1, "straight_segments": 0},
}


def test_api_serves_forecasts_and_routes_from_the_hot_state() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = Settings(database_url="", unit_map_splits="", api_resync_s=0.3)  # type: ignore[call-arg]

    async def publish() -> None:
        store = StateStore()
        store.on_connect(501)
        store.on_nav(501, _nav(0, 55.7, 37.6))
        publisher = TelemetryPublisher(store, factory, DependencyStatus("redis"), unit_map={501: 115106})
        await publisher.flush()
        await publisher.close()
        client = factory()
        stream_time = datetime.fromtimestamp(BASE, UTC).isoformat()
        await client.hset(FORECAST_KEY, mapping={"115106": json.dumps(FORECAST)})
        await client.set(
            FORECAST_STATUS_KEY, json.dumps({"stream_time": stream_time, "scheduled_tr_ids": [115106]})
        )
        await client.set(ROUTES_KEY, json.dumps(ROUTES))
        await client.aclose()

    svc = api.ApiService(settings, factory=factory)
    with TestClient(api.create_app(settings, svc)) as client:
        client.portal.call(publish)  # type: ignore[union-attr]
        _wait(lambda: client.get("/api/vehicles/501").json().get("risk") == "red")
        vehicle = client.get("/api/vehicles/501").json()
        assert vehicle["route_id"] == "R3" and vehicle["incident_id"] == 42 and vehicle["dwell_s"] == 95.0
        assert vehicle["segment_speed_kmh"] == 8.5 and vehicle["current_delay_s"] == 140.0
        routes = client.get("/api/routes").json()
        assert [r["route_id"] for r in routes] == ["R3"]
        assert routes[0]["directions"][0]["line"][1] == [37.605, 55.701] and routes[0]["line"]
        schema = client.get("/openapi.json").json()["components"]["schemas"]
        derived = {"current_delay_s", "segment_speed_kmh", "dwell_s", "idle_s", "position_age_s", "scheduled"}
        assert derived <= set(schema["VehicleOut"]["properties"])
        assert "directions" in schema["RouteOut"]["properties"]


def test_hot_state_parses_the_forecast_snapshot() -> None:
    cache = VehicleCache()
    sync = HotStateSync.__new__(HotStateSync)  # only the parser: no Redis
    sync.cache = cache
    stream_time = (datetime.fromtimestamp(BASE, UTC) + timedelta(seconds=5)).isoformat()
    fields: dict[str, Any] = {"115106": json.dumps(FORECAST), "bad": "{}", "7": "not json"}
    sync.apply_forecasts(fields, json.dumps({"stream_time": stream_time, "scheduled_tr_ids": [115106, 5]}))
    assert set(cache.forecasts) == {115106} and cache.scheduled == frozenset({115106, 5})
    cache.set_routes(ROUTES)
    assert cache.routes is not None and cache.routes[0]["route_id"] == "R3"
    cache.set_routes(None)  # nothing published: the last routes stay
    assert cache.routes is not None
