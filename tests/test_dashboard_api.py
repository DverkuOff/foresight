"""Views of the api for the dashboard: journal rows → contract objects, the stringline, the performance
figures from scraped metrics, and the predictor's messages relayed on ``/ws`` with the ``clock``."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
fakeredis = pytest.importorskip("fakeredis")

from fastapi.testclient import TestClient  # noqa: E402

from backend import api  # noqa: E402
from backend.bus import CHANNEL_ALERTS, CHANNEL_INCIDENTS, FORECAST_KEY, FORECAST_STATUS_KEY  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.journal import (  # noqa: E402
    alert_out,
    epoch_start,
    incident_out,
    incident_sort_key,
    lead_histogram,
    prediction_out,
)
from backend.perf import PerfScraper, quantile  # noqa: E402
from backend.schedule import VehiclePlan  # noqa: E402
from backend.stringline import build_stringline, plan_seqs  # noqa: E402

BASE = 1_767_690_000  # 2026-01-06 09:00:00 UTC
T0 = datetime.fromtimestamp(BASE, UTC)


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


# ---- journal rows ---------------------------------------------------------------------------------------


def test_alert_row_becomes_alert_out_with_cause_and_stop() -> None:
    cause = {"code": "dwell_long", "text": "Длительный простой", "recommendation": "Связаться", "factors": []}
    row = {
        "id": 7,
        "kind": "delay",
        "level": "red",
        "status": "open",
        "tr_id": 115106,
        "unit_id": 501,
        "route_id": "R3",
        "stop_id": 77,
        "target_time_begin": T0,
        "issued_at": T0,
        "prediction_id": 3,
        "incident_id": 5,
        "pred_delay_s": 181.26,
        "p10": None,
        "p90": 250.0,
        "p_late": 0.81234,
        "cause": "dwell_long",
        "recommendation": "Связаться",
        "details": json.dumps({"cause": cause, "stop_name": "ул. Тверская", "escalated_from": 6}),
        "acknowledged": False,
        "retroactive": False,
        "actual_delay_s": None,
        "closed_at": None,
    }
    out = alert_out(row)
    assert out["alert_id"] == 7 and out["incident_id"] == 5 and out["level"] == "red"
    assert out["cause"]["code"] == "dwell_long" and out["target_stop_name"] == "ул. Тверская"
    assert (
        out["issued_at"] == "2026-01-06T09:00:00Z" and out["pred_delay_s"] == 181.3 and out["p_late"] == 0.812
    )
    assert out["escalated_from"] == 6 and out["retroactive"] is False
    # an old row without the detail keeps a contract-shaped cause
    bare = alert_out({**row, "details": None})
    assert bare["cause"] == {"code": "dwell_long", "text": "", "recommendation": "Связаться", "factors": []}


def test_prediction_and_incident_rows() -> None:
    p = prediction_out(
        {
            "id": 3,
            "tr_id": 115106,
            "target_stop_id": 77,
            "target_time_begin": T0,
            "issued_at": T0,
            "pred_delay_s": 60.04,
            "status": "missed",
            "abs_error_s": None,
            "source": "fallback",
        }
    )
    assert p["status"] == "closed" and p["outcome"] == "missed" and p["source"] == "fallback"
    assert p["planned_at"] == "2026-01-06T09:00:00Z" and p["pred_delay_s"] == 60.0 and p["risk"] == "unknown"
    body = {"incident_id": 5, "kind": "delay", "tr_id": 115106, "risk": "red", "cause": {"code": "layover"}}
    i = incident_out(
        {"id": 5, "status": "closed", "details": json.dumps(body), "opened_at": T0, "closed_at": T0}
    )
    assert (
        i["status"] == "closed"
        and i["closed_at"] == "2026-01-06T09:00:00Z"
        and i["cause"]["code"] == "layover"
    )


def test_incident_order_and_lead_histogram() -> None:
    items = [
        {"incident_id": 1, "risk": "yellow", "p_late": 0.9, "pred_delay_s": 100},
        {"incident_id": 2, "risk": "red", "p_late": 0.5, "pred_delay_s": 300},
        {"incident_id": 3, "risk": "red", "p_late": 0.7, "pred_delay_s": 130},
        {"incident_id": 4, "risk": "unknown", "p_late": None, "pred_delay_s": None},
    ]
    assert [i["incident_id"] for i in sorted(items, key=incident_sort_key)] == [3, 2, 1, 4]
    hist = lead_histogram([650, 700, 710, 890, -5, None])  # type: ignore[list-item]
    assert hist[0] == {"from_s": -60, "to_s": 0, "count": 1}
    counts = {h["from_s"]: h["count"] for h in hist}
    assert counts[600] == 1 and counts[660] == 2 and counts[840] == 1 and counts[60] == 0
    assert epoch_start(3) is None and epoch_start(1_790_000_000_000) is not None


# ---- stringline ---------------------------------------------------------------------------------------


def _plan(tr_id: int, keys: list[str], times: list[float]) -> VehiclePlan:
    n = len(keys)
    return VehiclePlan(
        tr_id=tr_id,
        route_id="R1",
        stop_ids=np.arange(100, 100 + n),
        tb=np.array(times, dtype=float),
        slon=np.zeros(n),
        slat=np.zeros(n),
        modes=np.zeros(n, dtype=np.int8),
        valid=np.ones(n, dtype=bool),
        names=tuple(keys),
        keys=tuple(keys),
        index={100 + i: i for i in range(n)},
    )


def test_plan_seqs_follow_the_circle() -> None:
    # A is the terminal of both directions: seq 0 and 3 of the axis A B C | A D E
    seq_of = {"A": [0, 3], "B": [1], "C": [2], "D": [4], "E": [5]}
    assert plan_seqs(["A", "B", "C", "A", "D", "E", "A", "B", "X"], seq_of, 6) == [
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        1,
        None,
    ]


def test_stringline_has_plan_fact_and_forecast_band() -> None:
    route = {
        "route_id": "R1",
        "tr_ids": [1],
        "stops": [{"stop_key": k, "name": k, "seq": i} for i, k in enumerate(["A", "B", "C", "D"])],
    }
    times = [BASE + 60 * i for i in range(0, 40, 4)]  # stops every 4 min: A B C D A B C D A B
    plan = _plan(1, ["A", "B", "C", "D"] * 2 + ["A", "B"], times)
    now = BASE + 10 * 60
    passages = [{"tr_id": 1, "stop_id": 101, "pass_time": datetime.fromtimestamp(BASE + 5 * 60, UTC)}]
    forecasts = [{"tr_id": 1, "target_stop_id": 105, "pred_delay_s": 120.0, "p10": 60.0, "p90": 240.0}]
    out = build_stringline(
        route,
        {1: plan},
        BASE,
        BASE + 40 * 60,
        passages=passages,
        forecasts=forecasts,
        current={1: 60.0},
        now=now,
    )
    trip = out["trips"][0]
    assert [p["seq"] for p in trip["planned"]] == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert trip["actual"] == [{"t": "2026-01-06T09:05:00Z", "seq": 1}]
    fc = {p["seq"]: p for p in trip["forecast"] if p["t"] > "2026-01-06T09:10"}
    # the target (plan 09:20 at B, seq 1) is forecast +120 s with the band 60…240 s
    target = next(p for p in trip["forecast"] if p["t"] == "2026-01-06T09:22:00Z")
    assert (
        target["seq"] == 1
        and target["p10"] == "2026-01-06T09:21:00Z"
        and target["p90"] == "2026-01-06T09:24:00Z"
    )
    assert fc  # stops after the target hold the forecast delay
    assert out["stream_time"] == "2026-01-06T09:10:00Z" and [s["seq"] for s in out["stops"]] == [0, 1, 2, 3]
    assert trip["now"] is None  # no next stop given
    at_b = build_stringline(
        route, {1: plan}, BASE, BASE + 2400, positions={1: 103}, current={1: 60.0}, now=now
    )
    assert at_b["trips"][0]["now"] == {"t": "2026-01-06T09:10:00Z", "seq": 2.5, "delay_s": 60.0}


def test_stringline_forecast_of_a_late_vehicle_starts_at_its_next_stop() -> None:
    route = {
        "route_id": "R1",
        "tr_ids": [1],
        "stops": [{"stop_key": k, "name": k, "seq": i} for i, k in enumerate(["A", "B", "C", "D"])],
    }
    plan = _plan(1, ["A", "B", "C", "D"] * 2 + ["A", "B"], [BASE + 60 * i for i in range(0, 40, 4)])
    # 8 min late at 09:18: the next stop D was planned at 09:12 (already «in the past» by the plan)
    forecasts = [
        {"tr_id": 1, "target_stop_id": 102, "pred_delay_s": 900.0},  # behind the vehicle, never closed
        {"tr_id": 1, "target_stop_id": 106, "pred_delay_s": 300.0},
    ]
    out = build_stringline(
        route,
        {1: plan},
        BASE,
        BASE + 40 * 60,
        forecasts=forecasts,
        current={1: 480.0},
        positions={1: 103},
        now=BASE + 18 * 60,
    )
    fc = out["trips"][0]["forecast"]
    assert fc[0]["seq"] == 3 and "2026-01-06T09:18" < fc[0]["t"] < "2026-01-06T09:21"
    # the target C (plan 09:24) at +300 s; the stale forecast of C at 09:08 is ignored
    assert any(p["t"] == "2026-01-06T09:29:00Z" and p["seq"] == 2 for p in fc)
    assert all(p["t"] >= "2026-01-06T09:18:00Z" for p in fc)


def test_latest_tick_keeps_the_forecasts_updated_last() -> None:
    from backend.stringline import latest_tick

    rows = [
        {"id": 1, "updated_at": "2026-01-06T09:00:00Z"},
        {"id": 2, "updated_at": "2026-01-06T09:10:00Z"},
        {"id": 3, "updated_at": None, "issued_at": "2026-01-06T09:10:00Z"},
        {"id": 4, "updated_at": "2026-01-06T09:05:00Z"},
    ]
    assert [r["id"] for r in latest_tick(rows)] == [2, 3]
    assert latest_tick([{"id": 5}]) == [{"id": 5}]  # no times: kept


def test_stringline_forecast_ignores_the_frozen_forecasts_of_the_stops_that_left_the_window() -> None:
    route = {
        "route_id": "R1",
        "tr_ids": [1],
        "stops": [{"stop_key": k, "name": k, "seq": i} for i, k in enumerate("ABCDEFGH")],
    }
    plan = _plan(1, list("ABCDEFGH"), [BASE + 120 * i for i in range(8)])
    now = BASE + 120 + 130  # 130 s late, the next stop B (plan 09:02)

    def iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")

    forecasts = [
        # the forecast of C was last updated 8 min ago (C left the window): +10 s, frozen
        {"tr_id": 1, "target_stop_id": 102, "pred_delay_s": 10.0, "updated_at": iso(now - 480)},
        # the latest tick: H at +160 s
        {"tr_id": 1, "target_stop_id": 107, "pred_delay_s": 160.0, "updated_at": iso(now - 20)},
    ]
    out = build_stringline(
        route,
        {1: plan},
        BASE,
        BASE + 3600,
        forecasts=forecasts,
        current={1: 130.0},
        positions={1: 101},
        now=now,
    )
    at_c = next(p for p in out["trips"][0]["forecast"] if p["seq"] == 2)
    # C (plan 09:04) is between the current +130 s and +160 s, not at the frozen +10 s
    assert iso(BASE + 240 + 130) <= at_c["t"] <= iso(BASE + 240 + 160)


# ---- performance ---------------------------------------------------------------------------------------


def _exposition(tick: list[int], predict: list[int], lag: float) -> str:
    les = ["0.01", "0.1", "1.0", "+Inf"]
    lines = ["# TYPE foresight_predictor_tick_seconds histogram"]
    lines += [
        f'foresight_predictor_tick_seconds_bucket{{le="{le}"}} {c}' for le, c in zip(les, tick, strict=True)
    ]
    lines += [
        f"foresight_predictor_tick_seconds_count {tick[-1]}",
        "foresight_predictor_tick_seconds_sum 1.0",
    ]
    lines += ["# TYPE foresight_ml_request_duration_seconds histogram"]
    for endpoint, counts in (("/predict", predict), ("/health", [100, 100, 100, 100])):
        lines += [
            f'foresight_ml_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{le}"}} {c}'
            for le, c in zip(les, counts, strict=True)
        ]
    lines += ["# TYPE foresight_consumer_lag gauge", f"foresight_consumer_lag {lag}"]
    lines += ["# TYPE foresight_dependency_up gauge", 'foresight_dependency_up{dependency="ml-service"} 1.0']
    return "\n".join(lines) + "\n"


def test_quantile_interpolates_like_prometheus() -> None:
    assert quantile(0.5, {0.1: 50.0, 1.0: 100.0, float("inf"): 100.0}) == pytest.approx(0.1)
    assert quantile(0.95, {0.1: 50.0, 1.0: 100.0, float("inf"): 100.0}) == pytest.approx(0.91)
    assert quantile(0.95, {0.1: 0.0, float("inf"): 0.0}) is None


def test_perf_scraper_takes_quantiles_over_the_window() -> None:
    now = [0.0]
    scraper = PerfScraper(
        {"predictor": "http://p:8002", "ml": "http://m:8003"}, window_s=60, clock=lambda: now[0]
    )
    scraper.feed("predictor", _exposition([1000, 1000, 1000, 1000], [0, 0, 0, 0], 3))
    now[0] = 30.0
    # in the last 30 s: 100 ticks, all between 0.1 and 1 s
    scraper.feed("predictor", _exposition([1000, 1000, 1100, 1100], [0, 0, 0, 0], 7))
    assert scraper.p("predictor", "foresight_predictor_tick_seconds") == pytest.approx(0.955)
    assert scraper.gauge("predictor", "foresight_consumer_lag") == 7
    assert scraper.gauge("predictor", "foresight_dependency_up{dependency=ml-service}") == 1
    scraper.feed("ml", _exposition([0, 0, 0, 0], [10, 20, 20, 20], 0))
    assert scraper.p("ml", "foresight_ml_request_duration_seconds{endpoint=/predict}") == pytest.approx(0.091)
    assert scraper.state("ml") == "up" and scraper.state("nope") == "disabled"


# ---- the api ------------------------------------------------------------------------------------------


def test_ws_relays_predictor_events_and_the_clock_and_lists_survive_without_db() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = Settings(  # type: ignore[call-arg]
        database_url="",
        unit_map_splits="",
        api_resync_s=0.3,
        predictor_url="",
        ml_url="",
        replayer_url="",
        api_schedule=False,
    )
    incident: dict[str, Any] = {
        "incident_id": 42,
        "kind": "delay",
        "tr_id": 115106,
        "risk": "yellow",
        "pred_delay_s": 150.0,
        "p_late": 0.7,
        "target_stop": {"stop_id": 77, "name": "ул. Тверская", "planned_at": "2026-01-06T09:12:00Z"},
        "cause": {"code": "slow_segment", "text": "", "recommendation": "", "factors": []},
        "status": "open",
    }
    alert = {"alert_id": 9, "incident_id": 42, "tr_id": 115106, "level": "yellow", "kind": "delay"}

    async def publish_state() -> None:
        client = factory()
        stream_time = T0.isoformat()
        forecast = {
            "tr_id": 115106,
            "risk": "red",
            "incident_id": 42,
            "incident_ids": [42],
            "current_delay_s": 95,
        }
        await client.hset(FORECAST_KEY, mapping={"115106": json.dumps(forecast)})
        await client.set(FORECAST_STATUS_KEY, json.dumps({"stream_time": stream_time, "epoch": 5}))
        await client.aclose()

    async def publish_events() -> None:
        client = factory()
        await client.publish(
            CHANNEL_INCIDENTS, json.dumps({"type": "incident", "action": "open", "incident": incident})
        )
        await client.publish(
            CHANNEL_ALERTS, json.dumps({"type": "alert", "stream_time": None, "alert": alert})
        )
        await client.aclose()

    svc = api.ApiService(settings, factory=factory)
    with TestClient(api.create_app(settings, svc)) as client:
        client.portal.call(publish_state)  # type: ignore[union-attr]
        _wait(lambda: svc.cache.forecast_status.get("epoch") == 5)
        with client.websocket_connect("/ws") as ws:
            client.portal.call(publish_events)  # type: ignore[union-attr]
            seen: dict[str, dict[str, Any]] = {}
            deadline = time.monotonic() + 5
            while not {"incident", "alert", "clock"} <= set(seen) and time.monotonic() < deadline:
                message = ws.receive_json()
                seen[message["type"]] = message
            assert seen["incident"]["action"] == "open" and seen["incident"]["incident"]["incident_id"] == 42
            assert seen["alert"]["alert"]["alert_id"] == 9
            assert seen["clock"]["epoch"] == 5
            # leave only after the handler is done: TestClient cancels a running one and that races (flaky)
            ws.close()
            _wait(lambda: svc.ws_clients == 0)

        # no PostgreSQL: the open incident comes from the pub/sub, with the level of the vehicle now
        listing = client.get("/api/incidents").json()
        assert listing["count"] == 1
        item = listing["items"][0]
        assert (
            item["incident_id"] == 42 and item["risk"] == "red" and item["vehicle"]["current_delay_s"] == 95
        )
        alerts = client.get("/api/alerts").json()
        assert alerts["degraded"] is True and [a["alert_id"] for a in alerts["items"]] == [9]
        assert client.get("/api/incidents/42").json()["incident_id"] == 42
        assert client.get("/api/incidents/43").status_code in (404, 503)
        assert client.get("/api/stringline", params={"route_id": "R9"}).status_code == 404
        assert client.get("/api/replay/status").status_code == 503
        perf = client.get("/api/metrics/perf").json()
        assert perf["deps"]["ml"] == "disabled" and perf["deps"]["postgres"] in ("disabled", "unknown")
        schema = client.get("/openapi.json").json()
        assert {"/api/incidents", "/api/alerts", "/api/stringline", "/api/metrics/horizon"} <= set(
            schema["paths"]
        )


# ---- what-if ------------------------------------------------------------------------------------------


def test_whatif_reserve_vehicle_closes_the_gap_and_hold_shifts_arrivals() -> None:
    from backend.whatif import mean_wait, run_whatif, window_wait

    keys = ["A", "B", "C", "D"]
    route = {
        "route_id": "R1",
        "tr_ids": [1, 2],
        "stops": [{"stop_key": k, "seq": i} for i, k in enumerate(keys)],
    }
    # two vehicles half a loop apart on a 4-stop circle (20 min loop, 5 min between stops)
    plans = {
        1: _plan(1, keys * 12, [BASE + 300 * i for i in range(48)]),
        2: _plan(2, keys * 12, [BASE + 600 + 300 * i for i in range(48)]),
    }
    at = BASE + 3600
    base_req = {
        "route_id": "R1",
        "action": "add_vehicle",
        "params": {"from_stop_key": "A"},
        "horizon_min": 60,
    }
    out = run_whatif(route, plans, base_req, at)
    assert out["baseline"]["mean_wait_s"] == 300 and out["baseline"]["max_gap_s"] == 600
    assert out["baseline"]["late_stops"] == 0 and out["baseline"]["bunching_pairs"] == 0
    assert out["delta"]["mean_wait_s"] < 0 and out["scenario"]["vehicles"][-1]["tr_id"] is None
    # one vehicle on the route: the headway is its loop, counted from its previous pass before `at`
    single = run_whatif({**route, "tr_ids": [1]}, plans, base_req, at)
    assert single["baseline"]["max_gap_s"] == 1200 and single["baseline"]["mean_wait_s"] == 600
    assert single["delta"]["mean_wait_s"] < 0
    hold = run_whatif(route, plans, {**base_req, "action": "hold", "params": {"tr_id": 1, "hold_s": 180}}, at)
    b1 = hold["baseline"]["vehicles"][0]["arrivals"]
    s1 = hold["scenario"]["vehicles"][0]["arrivals"]
    assert b1[0] == s1[0] and b1[1]["t"] < s1[1]["t"]  # the next stop as planned, the later ones 3 min later
    assert all(a["seq"] == keys.index(a["stop_key"]) for a in b1)  # the place on the trip, for the chart
    assert hold["delta"]["late_stops"] > 0  # held past the plan: late (never «fewer» by leaving the horizon)
    assert hold["delta"]["mean_wait_s"] > 0  # and the passengers wait longer (the next service is later)
    assert window_wait([600, 1200], 0, 1200, 9999) == 300 and window_wait([], 0, 100, 200) == 150
    assert mean_wait([600, 600]) == 300 and mean_wait([]) == 0


def test_whatif_rejects_a_vehicle_or_a_stop_of_another_route() -> None:
    from backend.whatif import request_error

    route = {"route_id": "R1", "tr_ids": [1, 2], "stops": [{"stop_key": "A", "seq": 0}]}
    assert request_error(route, {"action": "hold", "params": {"tr_id": 2}}) is None
    assert request_error(route, {"action": "hold", "params": {}}) is None
    assert "does not run" in (request_error(route, {"action": "hold", "params": {"tr_id": 999}}) or "")
    assert request_error(route, {"action": "add_vehicle", "params": {"from_stop_key": "A"}}) is None
    assert "not on route" in (
        request_error(route, {"action": "add_vehicle", "params": {"from_stop_key": "Z"}}) or ""
    )


# ---- admin --------------------------------------------------------------------------------------------


def test_admin_without_database_and_neighbours() -> None:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = Settings(  # type: ignore[call-arg]
        database_url="",
        unit_map_splits="",
        predictor_url="",
        ml_url="",
        replayer_url="",
        api_schedule=False,
    )
    svc = api.ApiService(settings, factory=factory)
    with TestClient(api.create_app(settings, svc)) as client:
        assert client.get("/api/admin/settings").status_code == 503
        bad = {"risk": {"red_delay_s": 60, "green_delay_s": 120}, "alert": {"min_level": "yellow"}}
        assert client.put("/api/admin/settings", json=bad).status_code == 422  # green above red
        models = client.get("/api/admin/models").json()
        assert (models["active"], models["ml"], models["versions"], models["retrain"]) == (
            None,
            "disabled",
            [],
            None,
        )
        assert models["fallback"]["share"] is None  # no forecasts now
        assert client.post("/api/admin/models/retrain").status_code == 503  # ml-service not configured
        states = {s["name"]: s["state"] for s in client.get("/api/admin/services").json()["services"]}
        assert (
            states["api"] == "up" and states["ml-service"] == "disabled" and states["replayer"] == "disabled"
        )
        assert client.get("/api/admin/units").json() == []
        for name in ("%2e%2e", ".env", "a%20b"):  # a version is a directory name, never a path
            assert client.post(f"/api/admin/models/{name}/activate").status_code == 422
        assert client.post("/api/admin/models/v2/activate").status_code == 503  # ml-service not configured
        assert client.get("/api/admin/journal", params={"format": "csv"}).status_code == 503


def _admin_service(**overrides: Any) -> tuple[api.ApiService, Settings, Callable[[], Any]]:
    server = fakeredis.FakeServer()
    factory = lambda: fakeredis.FakeAsyncRedis(server=server, decode_responses=True)  # noqa: E731
    settings = Settings(  # type: ignore[call-arg]
        database_url="",
        unit_map_splits="",
        predictor_url="",
        ml_url="",
        replayer_url="",
        api_schedule=False,
        **overrides,
    )
    return api.ApiService(settings, factory=factory), settings, factory


class FakeNeighbour:
    """A neighbour service of the admin (ml-service): JSON answers by ``(method, path)``."""

    def __init__(self, answers: dict[tuple[str, str], tuple[int, Any]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    async def request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        self.calls.append((method, path))
        status, payload = self.answers[(method, path)]
        return status, json.dumps(payload).encode()

    def close(self) -> None:
        pass


class FakeJournal:
    """The journal of the admin without PostgreSQL: remembers the query of the forecasts."""

    def __init__(self) -> None:
        self.query: dict[str, Any] = {}

    async def predictions(self, **query: Any) -> list[dict[str, Any]]:
        self.query = query
        return [{"prediction_id": 2, "planned_at": "2026-01-06T09:00:00Z"}]

    async def horizon(self, **_: Any) -> dict[str, Any]:
        return {"total": {"mae_s": 81.5, "closed": 120}}


def test_admin_journal_lists_the_latest_forecasts_with_the_filters() -> None:
    svc, settings, _ = _admin_service()
    journal = FakeJournal()
    svc.journal = journal  # type: ignore[assignment]
    with TestClient(api.create_app(settings, svc)) as client:
        params = {
            "kind": "predictions",
            "tr_id": 7,
            "status": "closed",
            "from": "2026-01-06T08:00:00Z",
            "limit": 50,
        }
        answer = client.get("/api/admin/journal", params=params)
        assert answer.status_code == 200 and answer.json()["count"] == 1
    query = journal.query
    assert query["newest_first"] is True  # the latest ``limit`` forecasts, not the first ones of the day
    assert (query["tr_id"], query["status"], query["limit"]) == (7, "closed", 50)
    assert query["planned_from"] == datetime(2026, 1, 6, 8, tzinfo=UTC) and query["planned_to"] is None


def test_admin_retrain_goes_to_ml_service_and_models_show_the_fallback() -> None:
    svc, settings, factory = _admin_service(api_resync_s=0.3)
    online = {"base": "v2-holdout", "n_fit": 700, "n_eval": 300, "mae_before_s": 84.0, "mae_after_s": 80.5}
    ml = FakeNeighbour(
        {
            ("GET", "/model/versions"): (
                200,
                [
                    {"version": "v2-holdout", "active": True},
                    {"version": "v2-holdout-online1", "online": online},
                ],
            ),
            ("GET", "/model/retrain"): (200, {"state": "running", "base_version": "v2-holdout"}),
            ("POST", "/model/retrain"): (202, {"state": "running", "base_version": "v2-holdout"}),
        }
    )
    svc.ml = ml  # type: ignore[assignment]
    svc.journal = FakeJournal()  # type: ignore[assignment]

    async def publish_forecasts() -> None:
        client = factory()
        sources = {"1": "model", "2": "fallback", "3": "model", "4": "model"}
        await client.hset(
            FORECAST_KEY,
            mapping={tr: json.dumps({"tr_id": int(tr), "source": src}) for tr, src in sources.items()},
        )
        await client.aclose()

    with TestClient(api.create_app(settings, svc)) as client:
        client.portal.call(publish_forecasts)  # type: ignore[union-attr]
        _wait(lambda: len(svc.cache.forecasts) == 4)
        answer = client.post("/api/admin/models/retrain")
    assert answer.status_code == 202 and ("POST", "/model/retrain") in ml.calls
    body = answer.json()
    assert body["retrain"]["state"] == "running" and body["retrain"]["base_version"] == "v2-holdout"
    assert (
        body["versions"][0]["online_mae_s"] == 81.5 and body["versions"][1]["online"]["mae_after_s"] == 80.5
    )
    fallback = body["fallback"]
    assert fallback["share"] == 0.25 and fallback["forecasts"] == 4
    assert (fallback["coef"], fallback["intercept_s"]) == (
        settings.fallback_coef,
        settings.fallback_intercept_s,
    )


def test_admin_units_take_the_route_of_the_plan_without_a_forecast() -> None:
    from backend.bus import VehicleRecord

    svc, settings, _ = _admin_service(api_resync_s=3600)  # no full re-read of the hot state during the test
    with TestClient(api.create_app(settings, svc)) as client:
        # the first full read of Redis on start replaces the cache: write only after it (flaky otherwise)
        _wait(lambda: svc.redis_status.ok is True)
        svc.cache.set_routes(
            {"routes": [{"route_id": "R12", "tr_ids": [134494]}], "scheduled_tr_ids": [134494]}
        )
        now = datetime.now(UTC)
        svc.cache.apply([VehicleRecord(unit_id=1118231, tr_id=134494, last_packet_at=now, updated_ms=1)])
        units = client.get("/api/admin/units").json()
    assert [(u["tr_id"], u["route_id"], u["scheduled"]) for u in units] == [(134494, "R12", True)]


def test_journal_csv_spreads_nested_objects_over_columns() -> None:
    from backend.admin import _csv

    rows = [
        {"alert_id": 1, "cause": {"code": "dwell_long", "text": "Простой", "factors": [{"f": 1}]}},
        {"alert_id": 2, "cause": {"code": "unknown", "text": "Нет"}},
    ]
    lines = _csv(rows).splitlines()
    assert lines[0] == "alert_id,cause_code,cause_text,cause_factors"
    assert lines[1].startswith("1,dwell_long,Простой,") and lines[2] == "2,unknown,Нет,"


def test_ml_registry_lists_the_versions_of_the_repository() -> None:
    from pathlib import Path

    from ml.service import list_versions

    versions = {v.version: v for v in list_versions(Path(__file__).parent.parent / "models", "v2-holdout")}
    assert {"v1", "v2", "v1-holdout", "v2-holdout"} <= set(versions)
    assert versions["v2-holdout"].active and not versions["v2"].active
    assert versions["v2"].cv_mae is not None and versions["v2"].precision == "int8"


def test_stringline_actual_line_never_steps_back() -> None:
    from backend.stringline import monotonic

    pts = [{"t": str(i), "seq": s} for i, s in enumerate([3, 4, 6, 5, 7, 9, 0, 1])]
    # 5 after 6 is a detector decision out of order; 0 after 9 (of 10) is a new loop
    assert [p["seq"] for p in monotonic(pts, 10)] == [3, 4, 6, 7, 9, 0, 1]
