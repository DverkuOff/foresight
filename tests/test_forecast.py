"""Forecasts on the stream (``backend/forecast.py``): a synthetic line route through the predictor core.

Vehicles drive a straight line of stops 400 m apart, planned one minute apart; their GPS tracks pass every
stop exactly ``delay`` seconds after the plan. The stream goes through
:class:`backend.predictor.PredictorCore` (clock, windows, 30 s ticks) into
:class:`backend.forecast.ForecastEngine` with a fake model, so the tests check the whole online chain: the
(t + 10, t + 15] window, the first issue and its lead, the detector closing a forecast with the fact,
«issued after the fact» = 0, the fallback and its recovery, alerts, incidents, bus bunching and the metrics
of the contract.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from backend.bus import CHANNEL_ALERTS, CHANNEL_INCIDENTS, CHANNEL_PREDICTIONS, TelemetryEvent
from backend.config import Settings
from backend.db import UPSERT_KEYS
from backend.forecast import (
    CAUSES,
    ForecastEngine,
    IdGen,
    Thresholds,
    Track,
    cause_of,
    clean_track,
    track_metrics,
)
from backend.metrics import FunctionCollector
from backend.mlclient import MLPrediction, MLResult
from backend.predictor import PredictorCore, TrackPoint
from backend.runtime import DependencyStatus
from backend.schedule import Schedule
from shared.sequences import N_CHANNELS, SEQ_LEN

T0 = pd.Timestamp("2026-01-06 07:00:00")
T0_S = T0.value / 1e9
STEP_DEG = 400.0 / (111_320.0 * math.cos(math.radians(55.75)))  # 400 m along the parallel
N_STOPS = 60


class FakeModel:
    """ml-service stand-in: forecast = ``value`` (or the row's ``cur_dev_s``), ``down`` → fallback.

    ``by_call`` (call number → value) overrides ``value``: the forecast changes from tick to tick.
    """

    def __init__(self, value: float | None = 200.0, by_call: Callable[[int], float] | None = None) -> None:
        self.value = value
        self.by_call = by_call
        self.down = False
        self.calls = 0
        self.status = DependencyStatus("ml-service")
        self.model_version: str | None = "test"

    async def predict(self, rows: Sequence[Mapping[str, float]]) -> MLResult | None:
        self.calls += 1
        if self.down:
            self.status.ok = False
            return None
        self.status.ok = True
        preds = []
        for row in rows:
            if self.by_call is not None:
                value = self.by_call(self.calls)
            elif self.value is not None:
                value = self.value
            else:
                value = float(np.nan_to_num(row.get("cur_dev_s", 0.0)))
            preds.append(MLPrediction(value, factors=({"feature": "stop_dur", "contribution_s": 30.0},)))
        return MLResult("test", "fp32", 1.0, preds)


class FakeSequenceModel(FakeModel):
    """A model with a sequence input (ML v2): records the sequences the engine sends with the rows."""

    sequence_shape = (SEQ_LEN, N_CHANNELS)

    def __init__(self) -> None:
        super().__init__(100.0)
        self.sent: list[tuple[int, list[np.ndarray]]] = []

    async def predict(  # type: ignore[override]
        self, rows: Sequence[Mapping[str, float]], sequences: Sequence[np.ndarray] | None = None
    ) -> MLResult | None:
        if sequences is not None:
            self.sent.append((len(rows), list(sequences)))
        return await super().predict(rows)


class Sink:
    """Row collector with upserts, like PostgreSQL with ``backend.db.UPSERT_KEYS``."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.latest: dict[str, dict[Any, dict[str, Any]]] = defaultdict(dict)

    def write(self, table: str, row: Mapping[str, Any]) -> None:
        self.rows[table].append(dict(row))
        key = UPSERT_KEYS.get(table)
        if key is not None and key in row:
            self.latest[table][row[key]] = dict(row)


class Publisher:
    """Captures what the engine would publish to Redis."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []
        self.snapshots: list[tuple[dict[int, Any], dict[str, Any]]] = []
        self.routes: dict[str, Any] | None = None

    def publish(self, channel: str, message: Mapping[str, Any]) -> None:
        self.messages.append((channel, dict(message)))

    def set_routes(self, payload: Mapping[str, Any]) -> None:
        self.routes = dict(payload)

    def set_snapshot(self, vehicles: Mapping[int, Any], status: Mapping[str, Any]) -> None:
        self.snapshots.append((dict(vehicles), dict(status)))

    def of(self, channel: str) -> list[dict[str, Any]]:
        return [m for c, m in self.messages if c == channel]


def _plan(tr_id: int, offset_s: float, first_id: int) -> list[dict[str, Any]]:
    return [
        {
            "tr_id": tr_id,
            "tt_action_item_id": first_id + i,
            "time_begin": T0 + pd.Timedelta(seconds=offset_s + 60 * i),
            "stop_lon": 37.6 + STEP_DEG * i,
            "stop_lat": 55.75,
            "building_address": f"Остановка {i}",
        }
        for i in range(N_STOPS)
    ]


def _track(tr_id: int, unit: int, offset_s: float, delay_s: float, until_s: float) -> list[TelemetryEvent]:
    """Points every 10 s from 06:50: the vehicle passes stop i at plan + delay (it waits before the start)."""
    out = []
    for t in np.arange(T0_S - 600, until_s + 1, 10.0):
        pos = np.clip((t - (T0_S + offset_s + delay_s)) / 60.0, 0.0, N_STOPS - 1)
        out.append(TelemetryEvent(unit, tr_id, float(t), 37.6 + STEP_DEG * float(pos), 55.75, 24, 90, True))
    return out


def _settings(**kw: Any) -> Settings:
    base: dict[str, Any] = dict(database_url="", unit_map_splits="", ml_url="", online_min_closed=1)
    base.update(kw)
    return Settings(**base)


def _run(
    model: FakeModel,
    vehicles: Sequence[tuple[int, int, float, float]] = ((1, 11, 0.0, 120.0),),
    *,
    until_min: float = 40,
    settings: Settings | None = None,
    between: Any = None,
) -> tuple[ForecastEngine, PredictorCore, Sink, Publisher]:
    """Stream the tracks of ``(tr_id, unit_id, plan offset, delay)`` through the core and the engine."""
    rows = []
    for k, (tr_id, _, offset, _) in enumerate(vehicles):
        rows += _plan(tr_id, offset, 1000 * (k + 1))
    schedule = Schedule(pd.DataFrame(rows))
    sink, publisher = Sink(), Publisher()
    settings = settings or _settings()
    engine = ForecastEngine(settings, schedule, model, sink, publisher=publisher, ids=IdGen(1))
    core = PredictorCore(window_s=5400, tick_period_s=30, on_tick=engine.tick)
    until = T0_S + 60 * until_min
    events = sorted(
        (e for tr_id, unit, offset, delay in vehicles for e in _track(tr_id, unit, offset, delay, until)),
        key=lambda e: e.ts,
    )

    async def feed() -> None:
        for i in range(0, len(events), 20):
            if between is not None:
                between(events[i].ts)
            await core.handle([(f"{i + j}-0", e) for j, e in enumerate(events[i : i + 20])])

    asyncio.run(feed())
    return engine, core, sink, publisher


def _metrics(engine: ForecastEngine) -> str:
    registry = CollectorRegistry()
    registry.register(FunctionCollector(engine.metrics))
    return generate_latest(registry).decode()


def _series(text: str) -> dict[str, float]:
    """``name{labels}`` → value of every sample of an exposition."""
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            out[name] = float(value)
    return out


# ---- pure parts ------------------------------------------------------------------------------------------


def test_thresholds_follow_the_contract_and_the_settings_table() -> None:
    th = Thresholds()
    risks = [th.risk(p, None) for p in (-30, 59.9, 60, 120, 120.1)]
    assert risks == ["green", "green", "yellow", "yellow", "red"]
    assert th.risk(30, 0.7) == "red" and th.risk(30, 0.35) == "yellow" and th.risk(None, 0.9) == "unknown"
    assert th.alerting("yellow", None) and th.alerting("red", 0.9) and not th.alerting("green", None)
    new = Thresholds.from_settings(
        {
            "risk_thresholds": {"red_delay_s": 300, "green_delay_s": 90, "red_p_late": "bad"},
            "alert_thresholds": {"min_level": "red"},
        }
    )
    assert (new.red_delay_s, new.green_delay_s, new.red_p_late, new.min_level) == (300, 90, 0.6, "red")
    assert new.risk(200, None) == "yellow" and not new.alerting("yellow", None)
    assert Thresholds.from_settings({"alert_thresholds": {"min_level": "purple"}}).min_level == "yellow"


def test_causes_rules_then_contributions() -> None:
    assert cause_of({"gps_age": 400.0})["code"] == "gps_lost"
    assert cause_of({"stop_dur": 300.0, "spd_60": 0.0})["code"] == "dwell_long"
    assert cause_of({"pos_on_layover": 1.0, "pos_delay": 200.0})["code"] == "layover"
    assert cause_of({"spd_300": 5.0, "stopfrac_300": 0.2})["code"] == "slow_segment"
    assert cause_of({}, bunching=True)["code"] == "bunching"
    assert cause_of({"cur_best": 200.0})["code"] == "accumulated_delay"
    factors = [{"feature": "hour", "contribution_s": 50}, {"feature": "stop_dur", "contribution_s": 9}]
    by_factor = cause_of({}, factors)
    assert by_factor["code"] == "dwell_long"  # the first contribution with a cause hint
    label = "Длительность текущей стоянки"
    assert by_factor["factors"][1] == {"feature": "stop_dur", "label": label, "contribution_s": 9.0}
    unknown = cause_of({}, [{"feature": "stop_dur", "contribution_s": -20}])
    assert unknown["code"] == "unknown" and unknown["text"] == CAUSES["unknown"][0]


def test_clean_track_keeps_valid_fixes_one_per_second() -> None:
    pts = [
        TrackPoint(10.0, 37.6, 55.7, 20, 90, True, 1),
        TrackPoint(10.0, 37.7, 55.7, 20, 90, True, 1),  # same second: the first one stays
        TrackPoint(20.0, 0.0, 0.0, 0, 0, False, 1),  # invalid
        TrackPoint(30.0, 37.6, 55.7, 150, 90, True, 1),  # speed outlier (offline: > 120 km/h dropped)
        TrackPoint(40.0, 37.61, 55.7, 30, 180, True, 1),
    ]
    track = clean_track(pts)
    assert track is not None and track.ts.tolist() == [10.0, 40.0] and track.lon.tolist() == [37.6, 37.61]
    assert (track.last_course, track.last_speed, track.unit_id) == (180, 30, 1)
    assert clean_track(pts[2:4]) is None


# ---- the online chain ------------------------------------------------------------------------------------


def test_forecasts_in_the_window_first_issue_and_the_fact() -> None:
    engine, core, sink, publisher = _run(FakeModel(100.0))
    forecasts = sink.latest["predictions"].values()
    assert len(forecasts) >= 20 and core.ticks_run >= 75
    plan = {1000 + i: T0_S + 60 * i for i in range(N_STOPS)}
    for f in forecasts:
        lead = f["target_time_begin"].timestamp() - f["issued_at"].timestamp()
        assert 600 < lead <= 900  # issued only for stops planned in (t + 10 min, t + 15 min]
        assert f["lead_s"] == pytest.approx(lead)
        assert f["target_time_begin"].timestamp() == plan[f["target_stop_id"]]
    # one forecast per vehicle × stop: the first issue ~15 min ahead, then updates every tick until 10 min
    full = [f for f in forecasts if f["updates"] == 10]
    assert full and all(870 < f["lead_s"] <= 900 for f in full)
    updates = sink.rows["prediction_updates"]
    assert all(600 < u["lead_s"] <= 900 for u in updates)
    assert len(updates) == sum(f["updates"] for f in forecasts)
    # the detector closes them with the fact: passes at plan + 120 s
    closed = [f for f in forecasts if f["status"] == "closed"]
    assert len(closed) >= 10
    for f in closed:
        assert f["actual_delay_s"] == pytest.approx(120.0, abs=2.0)
        assert f["abs_error_s"] == pytest.approx(20.0, abs=2.0)
        assert f["retroactive"] is False and f["actual_lead_s"] > 600
        assert f["closed_at"] >= f["pass_time"]
    assert engine.retroactive == 0 and engine.closed == len(closed)
    assert len(publisher.of(CHANNEL_PREDICTIONS)) == len(closed)
    assert publisher.of(CHANNEL_PREDICTIONS)[0]["prediction"]["status"] == "closed"
    # the detector's online labels
    passages = sink.rows["stop_passages"]
    assert passages and all(p["matched"] for p in passages)
    assert all(p["confirmed_at"] >= p["pass_time"] for p in passages)
    mae, base, n = engine.online_mae()
    assert n == len(closed) and mae == pytest.approx(20.0, abs=2.0)
    # the baseline is the online cur_dev_s at issue (the delay at the last confirmed stop; none yet — 0)
    at_issue = np.array([np.nan if f["cur_dev_s"] is None else f["cur_dev_s"] for f in closed], dtype=float)
    assert np.isnan(at_issue).any() and (np.abs(at_issue[~np.isnan(at_issue)] - 120.0) < 2.0).all()
    expected = np.mean(np.abs(np.nan_to_num(at_issue) - np.array([f["actual_delay_s"] for f in closed])))
    assert base == pytest.approx(expected)


def test_metrics_follow_the_contract() -> None:
    engine, *_ = _run(FakeModel(100.0))  # 100 s: yellow — one delay incident of the vehicle, one alert
    s = _series(_metrics(engine))
    assert s['foresight_predictions_total{source="model"}'] == engine.forecasts_by_source["model"] > 0
    assert s['foresight_predictions_total{source="fallback"}'] == 0.0  # both series from the start
    assert s['foresight_prediction_lead_seconds_bucket{kind="prediction",le="0.0"}'] == 0.0
    assert s['foresight_prediction_lead_seconds_count{kind="prediction"}'] == engine.closed > 0
    # the lead of an alert is observed when the forecast it was raised about gets its fact
    alerted_closed = sum(a.forecast.status == "closed" for a in engine.alert_log.values())
    assert engine.alerts == 1 and s['foresight_prediction_lead_seconds_count{kind="alert"}'] == alerted_closed
    assert alerted_closed == 1
    assert s['foresight_prediction_lead_seconds_bucket{kind="prediction",le="600.0"}'] == 0.0  # all > 10 min
    assert s['foresight_prediction_abs_error_seconds_bucket{le="30.0",source="model"}'] == engine.closed
    assert s['foresight_prediction_abs_error_seconds_count{source="fallback"}'] == 0.0
    alerts = {k: v for k, v in s.items() if k.startswith("foresight_alerts_total{")}
    assert sum(v for k, v in alerts.items() if 'level="warning"' in k) == engine.alerts > 0
    assert s['foresight_alerts_total{cause="other",level="critical"}'] == 0.0  # all 14 series exist
    assert len([k for k in s if k.startswith("foresight_alerts_total{")]) == 14
    assert s["foresight_alerts_retroactive_total"] == 0.0
    assert s["foresight_online_mae_seconds"] == pytest.approx(20.0, abs=2.0)
    assert "foresight_online_baseline_mae_seconds" in s
    assert s["foresight_predictions_open"] == len(engine.open_forecasts)
    # online MAE only after online_min_closed forecasts
    quiet, *_ = _run(FakeModel(100.0), settings=_settings(online_min_closed=10_000))
    assert "foresight_online_mae_seconds" not in _metrics(quiet)


def test_alerts_incidents_and_escalation() -> None:
    model = FakeModel(90.0)  # yellow

    def between(ts: float) -> None:
        if ts >= T0_S + 60 * 20:
            model.value = 200.0  # red

    engine, _, sink, publisher = _run(model, between=between)
    alerts = publisher.of(CHANNEL_ALERTS)
    assert all(a["type"] == "alert" for a in alerts)
    # alerts per incident, not per target stop: the incident opens yellow, then escalates to red — two alerts
    # for a vehicle that had ~50 target stops in its window
    assert [a["alert"]["level"] for a in alerts] == ["yellow", "red"]
    first, second = alerts[0]["alert"], alerts[1]["alert"]
    assert set(first) >= {"alert_id", "prediction_id", "tr_id", "route_id", "level", "cause", "issued_at"}
    assert first["cause"]["code"] in CAUSES and first["acknowledged"] is False
    assert first["incident_id"] == second["incident_id"] and second["escalated_from"] == first["alert_id"]
    assert first["kind"] == "delay" and first["target_stop_name"].startswith("Остановка")
    assert len(engine.forecasts) > 20
    stored = sink.latest["alerts"]
    assert len(stored) == 2 and all(a["kind"] == "delay" for a in stored.values())
    assert all(a["incident_id"] == first["incident_id"] for a in stored.values())
    s = _series(_metrics(engine))
    by_level: defaultdict[str, float] = defaultdict(float)
    for key, value in s.items():
        if key.startswith("foresight_alerts_total{"):
            by_level["critical" if 'level="critical"' in key else "warning"] += value
    assert (by_level["warning"], by_level["critical"]) == (1.0, 1.0)
    assert s['foresight_alerts_total{cause="accumulated_delay",level="critical"}'] == 1.0  # 120 s behind plan
    assert s["foresight_alerts_retroactive_total"] == 0.0
    assert s['foresight_prediction_lead_seconds_count{kind="alert"}'] > 0
    # one open delay incident for the vehicle, updated as the forecast grows
    incidents = publisher.of(CHANNEL_INCIDENTS)
    assert incidents[0]["action"] == "open" and incidents[0]["incident"]["kind"] == "delay"
    assert {m["incident"]["incident_id"] for m in incidents} == {incidents[0]["incident"]["incident_id"]}
    assert any(m["action"] == "update" for m in incidents)
    body = incidents[-1]["incident"]
    assert body["risk"] == "red" and body["route_id"] == "R1" and body["segment"]["line"]
    assert body["vehicle"]["lat"] == pytest.approx(55.75)
    assert body["target_stop"]["name"].startswith("Остановка")
    # the snapshot for the api
    vehicles, status = publisher.snapshots[-1]
    assert vehicles[1]["risk"] == "red" and vehicles[1]["incident_id"] == body["incident_id"]
    assert status["source"] == "model" and status["degraded"] is False
    # every forecast remembers the level the dispatcher saw while its stop was 10–15 min ahead
    warned = {r["alert_level"] for r in sink.latest["predictions"].values()}
    assert warned <= {"yellow", "red"} and "red" in warned


def test_vehicle_shows_the_forecast_its_incident_follows() -> None:
    # the forecast grows every tick: stops that left the 10–15 min window keep older, smaller values
    engine, _, _, publisher = _run(FakeModel(by_call=lambda n: float(n)))
    incident = engine.delay_incidents[1].out()
    vehicles, _ = publisher.snapshots[-1]
    older = [f.pred for f in engine.forecasts.values() if f.tr_id == 1 and f.is_open]
    assert min(older) < max(older)
    assert vehicles[1]["pred_delay_s"] == incident["pred_delay_s"] == pytest.approx(max(older), abs=0.1)
    assert vehicles[1]["risk"] == incident["risk"]
    # a forecast not updated for a whole horizon (the vehicle left its stops) is not shown any more
    last = max(f.updated_at for f in engine.forecasts.values() if f.tr_id == 1)
    assert engine.vehicle_forecast(1, last + 901)["pred_delay_s"] is None


def test_hysteresis_keeps_one_alert_per_incident_and_a_new_one_after_it_closes() -> None:
    def value(call: int) -> float:
        if call <= 30:
            return 125.0 if call % 2 else 105.0  # around the red threshold: red, held by the hysteresis
        if call <= 45:
            return 30.0  # green for long enough: the incident closes
        return 200.0  # late again: a new incident

    engine, _, sink, publisher = _run(FakeModel(by_call=value), until_min=45)
    alerts = [a["alert"] for a in publisher.of(CHANNEL_ALERTS)]
    assert [a["level"] for a in alerts] == ["red", "red"]
    assert alerts[0]["incident_id"] != alerts[1]["incident_id"]
    actions = [(m["action"], m["incident"]["incident_id"]) for m in publisher.of(CHANNEL_INCIDENTS)]
    first, second = alerts[0]["incident_id"], alerts[1]["incident_id"]
    assert actions[0] == ("open", first) and ("close", first) in actions and ("open", second) in actions
    assert actions.index(("close", first)) < actions.index(("open", second))
    closed = sink.latest["incidents"][first]
    assert closed["status"] == "closed" and [x["level"] for x in closed["details"]["levels"]][0] == "red"
    # without the hysteresis the same forecasts flap the level: 105 s is yellow
    th = Thresholds()
    assert th.hold("red", 105.0, None) == "red" and th.hold("red", 99.0, None) == "yellow"
    assert th.hold("yellow", 45.0, None) == "yellow" and th.hold("yellow", 39.0, None) == "green"
    assert th.hold("green", 61.0, None) == "yellow" and th.hold("yellow", 121.0, None) == "red"
    assert th.hold("red", None, None) == "red"  # no forecast: the level stays


def test_confirm_ticks_delay_the_opening() -> None:
    def value(call: int) -> float:
        return 200.0 if call % 3 == 0 else 20.0  # a single red tick now and then: noise

    engine, _, _, publisher = _run(FakeModel(by_call=value), settings=_settings(alert_confirm_ticks=2))
    assert not publisher.of(CHANNEL_ALERTS) and not engine.delay_incidents
    engine, _, _, publisher = _run(FakeModel(by_call=value))  # default: 1 tick
    assert publisher.of(CHANNEL_ALERTS)


def test_derived_features_of_the_vehicle() -> None:
    engine, _, _, publisher = _run(FakeModel(100.0))
    vehicles, _ = publisher.snapshots[-1]
    v = vehicles[1]
    # criterion 3: deviation from the plan, speed on the segment, dwell / idle time — flat and in `features`
    assert v["current_delay_s"] == pytest.approx(120.0, abs=15.0)  # the model feature cur_best
    assert v["segment_speed_kmh"] == pytest.approx(24.0, abs=1.0)  # 400 m per minute
    assert v["dwell_s"] == 0.0 and v["idle_s"] == 0.0 and v["gps_age_s"] <= 30.0
    assert v["features"]["segment_speed_kmh"] == v["segment_speed_kmh"]
    assert v["route_id"] == "R1" and v["scheduled"] is True and v["next_stop"]["name"].startswith("Остановка")
    assert v["position"]["lat"] == pytest.approx(55.75)
    incident = engine.delay_incidents[1].out()
    # следующая остановка — ближайшая по пути, а не цель прогноза в 10–15 мин
    assert v["next_stop"]["planned_at"] < incident["target_stop"]["planned_at"]
    assert incident["vehicle"]["segment_speed_kmh"] == v["segment_speed_kmh"]
    assert set(incident["vehicle"]) >= {"lat", "lon", "current_delay_s", "dwell_s", "idle_s", "speed_kmh"}
    # a vehicle standing for 5 min, then moving 2 min at 36 km/h (10 m/s), fixes every 10 s
    ts = np.arange(0.0, 421.0, 10.0)
    moving = ts > 300
    x = np.where(moving, (ts - 300) * 10.0, 0.0) / (111_320.0 * math.cos(math.radians(55.75)))
    track = Track(ts, 37.6 + x, np.full(len(ts), 55.75), np.where(moving, 36.0, 0.0), 1, 0, 0, 0, 0)
    standing = track_metrics(track, 300.0)
    assert standing["dwell_s"] == 300.0 and standing["idle_s"] == 300.0 and standing["gps_age_s"] == 0.0
    moved = track_metrics(track, 420.0, since=300.0)
    assert moved["dwell_s"] == 0.0 and moved["segment_speed_kmh"] == pytest.approx(36.0, abs=0.5)
    assert moved["idle_s"] == 310.0  # standing in the last 10 min: the gaps after the 31 fixes at 0 km/h
    assert track_metrics(track, 425.0)["gps_age_s"] == 5.0
    assert track_metrics(None, 1.0) == dict.fromkeys(("segment_speed_kmh", "dwell_s", "idle_s", "gps_age_s"))


def test_sequences_go_with_the_rows_when_the_model_takes_them() -> None:
    model = FakeSequenceModel()
    vehicles = ((1, 11, 0.0, 120.0), (2, 12, 300.0, 0.0))
    engine, *_ = _run(model, vehicles, until_min=25, settings=_settings(ml_sequences=True))
    assert model.sent and engine.sequences_sent > 0
    for n, seqs in model.sent:
        assert len(seqs) == n and all(s.shape == (SEQ_LEN, N_CHANNELS) for s in seqs)
        assert len({id(s) for s in seqs}) <= 2  # one array per vehicle, shared by its rows
    off = FakeSequenceModel()  # off by default (docs/online-validation.md): the CatBoost part only
    _run(off, until_min=20)
    assert not off.sent and off.calls > 0


def test_fallback_when_ml_is_down_and_recovery() -> None:
    model = FakeModel(100.0)
    settings = _settings(fallback_coef=0.5, fallback_intercept_s=10.0)

    def between(ts: float) -> None:
        model.down = T0_S + 60 * 12 <= ts < T0_S + 60 * 20

    engine, _, sink, publisher = _run(model, settings=settings, between=between)
    updates = pd.DataFrame(sink.rows["prediction_updates"])
    by_source = updates.groupby("source")["pred_delay_s"]
    assert set(by_source.groups) == {"model", "fallback"}
    assert (updates.loc[updates["source"] == "model", "pred_delay_s"] == 100.0).all()
    fb = updates[updates["source"] == "fallback"]
    # fallback = intercept + coef · deviation (the vehicle is 120 s late: 10 + 0.5 · 120 ≈ 70)
    assert fb["pred_delay_s"].between(40, 100).all()
    # recovery without a restart: model forecasts again after ml-service is back
    last_tick = updates["tick_at"].max()
    assert (updates.loc[updates["tick_at"] == last_tick, "source"] == "model").all()
    degraded = [s for _, s in publisher.snapshots if s["degraded"]]
    assert degraded and degraded[0]["source"] == "fallback"
    text = _metrics(engine)
    assert f'foresight_predictions_total{{source="fallback"}} {float(len(fb))}' in text
    rows = [r for r in sink.rows["predictions"] if r["source"] == "fallback"]
    assert rows and all(r["degraded"] and r["model_version"] == "fallback" for r in rows)


def test_bunching_of_two_vehicles_of_a_route() -> None:
    # vehicle 2 runs 5 min behind vehicle 1 by plan; vehicle 1 is 280 s late: they arrive 20 s apart
    model = FakeModel(None)  # forecast = online cur_dev_s = the delay at the last confirmed stop
    engine, _, sink, publisher = _run(model, ((1, 11, 0.0, 280.0), (2, 12, 300.0, 0.0)), until_min=35)
    assert engine.schedule.routes.by_tr == {1: "R1", 2: "R1"}
    incidents = [m for m in publisher.of(CHANNEL_INCIDENTS) if m["incident"]["kind"] == "bunching"]
    assert incidents and incidents[0]["action"] == "open"
    body = incidents[0]["incident"]
    assert (body["tr_id"], body["related_tr_id"]) == (2, 1)
    assert body["bunching"]["kind"] == "bunching" and body["bunching"]["headway_s"] < 60
    assert body["bunching"]["planned_headway_s"] == pytest.approx(300.0)
    assert body["cause"]["code"] == "bunching"
    bunching_alerts = [a for a in publisher.of(CHANNEL_ALERTS) if a["alert"]["kind"] == "bunching"]
    assert len(bunching_alerts) == 1
    assert 'foresight_alerts_total{cause="bunching",level="warning"} 1.0' in _metrics(engine)
    stored = [r for r in sink.latest["incidents"].values() if r["kind"] == "bunching"]
    assert len(stored) == 1 and stored[0]["related_tr_id"] == 1


def test_a_new_timeline_closes_everything_of_the_old_one() -> None:
    engine, core, sink, publisher = _run(FakeModel(200.0), until_min=20)
    open_before = len(engine.open_forecasts)
    assert open_before and engine.delay_incidents
    engine.reset(engine.epoch + 1 if engine.epoch is not None else 1, T0_S + 1200)
    assert not engine.forecasts and not engine.delay_incidents and not engine.detectors
    reset = [r for r in sink.latest["predictions"].values() if r["status"] == "reset"]
    assert len(reset) == open_before
    assert publisher.of(CHANNEL_INCIDENTS)[-1]["action"] == "close"
