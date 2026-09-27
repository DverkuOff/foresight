"""Forecast engine of the predictor: stop detector on the stream, features, ML, forecasts, alerts, incidents.

Every tick of the stream clock (``t``, every 30 s of stream time, see :mod:`backend.predictor`) the engine:

1. takes the track of every scheduled vehicle known at ``t`` (points with ``ts ≤ t``, cleaned as offline:
   valid fixes, speed ≤ 120 km/h, one point per second) and runs the stop detector on it
   (:class:`backend.passages.VehicleDetector`); new decisions go to ``stop_passages`` (online labels) and
   close the forecasts of their stops with the fact;
2. for every *active* vehicle (a fix in the last ``active_s``) finds its planned stops with plan in
   ``(t + 10 min, t + 15 min]`` that are not decided yet and the vehicle is not standing at (a terminal
   departure it waits for is forecast: the departure is ahead), computes their
   features with the offline code (:class:`shared.features.FeatureContext` on the windows: telemetry
   ``≤ t``, passages confirmed ``≤ t``, the plan) and asks ml-service for the whole batch; when the active
   model has a sequence component (ML v2), the 20-minute telemetry sequence of every vehicle
   (:mod:`shared.sequences`, the same context, points ``≤ t``) goes with it;
3. ml-service down or slow (500 ms) → the fallback forecast ``intercept + coef · deviation`` (the deviation
   is ``dev_1``: the delay at the last stop confirmed by the detector; the coefficient is fit on train, see
   ``scripts/validate_online.py fallback-coef``), ``source = "fallback"``, ``degraded``;
4. a forecast is one «vehicle × target stop»: ``issued_at`` is its *first* issue (the honest lead), the later
   ticks update it (latest value + number of updates; every value goes to ``prediction_updates``); the risk
   follows the thresholds of the ``settings`` table; the cause comes from rules on the features and the
   contributions of ml-service;
5. alerts are raised per **incident**, not per target stop: the level of a vehicle is the risk of its nearest
   target stop in the window, held with a hysteresis (it leaves a level only below the threshold minus
   ``alert_hysteresis_s``); a ``delay`` incident opens when the level reaches yellow / red and closes after
   ``incident_clear_ticks`` ticks back at green. An alert (``alerts``, ``PUBLISH foresight:alerts``) comes
   when the incident opens and when it escalates to red — never again for the same incident, so a late
   vehicle gives one or two alerts instead of one per stop; two vehicles of a route forecast within
   ``bunching_s`` at a common stop (or a headway gap) — a ``bunching`` incident with its alert;
6. online check: when the detector confirms the target stop, the forecast is closed with the fact: error,
   lead (``pass − issued_at``), «issued after the fact» flag (must stay 0), online MAE against the baseline
   «forecast = online ``cur_dev_s`` at issue» over the last hour of stream time, precision of the alerts.

Everything is honest by construction: nothing after ``t`` is read, the plan has no fact columns, a forecast is
never issued for a stop the detector has already decided.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric

from backend.bus import (
    CHANNEL_ALERTS,
    CHANNEL_INCIDENTS,
    CHANNEL_PREDICTIONS,
    FORECAST_KEY,
    FORECAST_STATUS_KEY,
    ROUTES_KEY,
    RedisFactory,
)
from backend.mlclient import MLResult
from backend.passages import Passage, VehicleDetector
from backend.runtime import Backoff, DependencyStatus
from backend.schedule import Schedule, VehiclePlan
from ml.feature_labels import cause_hint, label
from shared.data import MAX_SPEED_KMH
from shared.features import STOP_SPEED_KMH, FeatureContext, build_features
from shared.sequences import SequenceBuilder
from shared.stops import MODE_DEPART

if TYPE_CHECKING:
    from backend.config import Settings
    from backend.predictor import TickContext, TrackPoint

log = logging.getLogger(__name__)

FEATURE_LOOKBACK_S = 3900.0
"""Telemetry given to the features: they look an hour back (stop duration), plus a margin."""

LEAD_BUCKETS = (0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 630, 660, 690, 720, 750, 780, 810, 840)
LEAD_BUCKETS += (870, 900, 960, 1020, 1080, 1140, 1200)
ERROR_BUCKETS = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900)
SOURCES = ("model", "fallback")
LEVELS = {"yellow": "warning", "red": "critical"}
RISK_RANK = {"unknown": -1, "green": 0, "yellow": 1, "red": 2}

CAUSES: dict[str, tuple[str, str]] = {
    "dwell_long": (
        "Длительный простой на остановке",
        (
            "Связаться с водителем и выяснить причину стоянки (посадка маломобильного пассажира, конфликт, "
            "неисправность).\n"
            "Если ТС неисправно — вызвать техпомощь и готовить замену из резерва."
        ),
    ),
    "slow_segment": (
        "Низкая скорость на участке",
        (
            "Затор на участке: ТС идёт по своему маршруту и объехать его не может.\n"
            "Предупредить пассажиров на остановках впереди — обновить прогноз прибытия на табло.\n"
            "Следующим рейсам заложить больше времени на этот участок (+10–20 % ко времени рейса).\n"
            "Если разрыв перед этим ТС растёт — задержать впереди идущее ТС на ⅓ интервала (раздвижка "
            "интервалов)."
        ),
    ),
    "layover": (
        "Отстой на конечной сдвигает отправление",
        (
            "Сократить стоянку на конечной и отправить ТС по расписанию.\n"
            "Если ТС пришло на конечную раньше — выдержать его до времени отправления."
        ),
    ),
    "accumulated_delay": (
        "Накопленное опоздание по рейсу",
        (
            "Сократить стоянку на ближайшей конечной, чтобы следующий рейс ушёл по расписанию.\n"
            "При опоздании больше 5 мин — отправить ТС в укороченный рейс или выпустить резервное ТС в "
            "разрыв.\n"
            "Предупредить пассажиров на остановках впереди (табло)."
        ),
    ),
    "bunching": (
        "Сбивка с соседним ТС маршрута",
        (
            "Задержать догоняющее ТС на ближайшей остановке на 2–3 мин, чтобы восстановить интервал.\n"
            "Сообщить водителям обоих ТС."
        ),
    ),
    "gps_lost": (
        "Потеря GPS-сигнала",
        (
            "Связаться с водителем и проверить бортовой терминал.\n"
            "До восстановления связи прогноз строится по последнему известному положению."
        ),
    ),
    "unknown": (
        "Причина не определена",
        ("Явной причины нет — наблюдать.\nЕсли опоздание растёт — связаться с водителем."),
    ),
}
"""Cause codes of the API contract §1: text for the dispatcher and the default recommendation."""
GAP_CAUSE = (
    "Разрыв интервала с соседним ТС маршрута",
    (
        "Выпустить резервное ТС в разрыв или сократить стоянку на конечной следующего ТС.\n"
        "При долгом разрыве — временно переключить ТС с соседнего маршрута."
    ),
)

CAUSE_METRIC = {
    "dwell_long": "dwell",
    "slow_segment": "slow_segment",
    "layover": "layover",
    "accumulated_delay": "accumulated_delay",
    "bunching": "bunching",
    "gps_lost": "gps_loss",
    "unknown": "other",
}
"""Cause code → ``cause`` label of ``foresight_alerts_total`` (docs/observability.md §3.2)."""

SNAPSHOT_FEATURES = (
    "cur_best",
    "dev_1",
    "pos_delay",
    "n_to_target",
    "dist_target",
    "gps_age",
    "stop_dur",
    "spd_300",
)

SEGMENT_MAX_S = 900.0
"""The segment speed looks back to the last passed stop, but not further than this."""
IDLE_WINDOW_S = 600.0
"""Idle time: seconds standing (speed < 2 km/h) in this last window."""
IDLE_GAP_S = 60.0
"""A GPS gap counts at most this long towards the idle time (a lost signal is not standing)."""
_M_PER_DEG = math.pi / 180.0 * 6_371_008.8


def iso(ts: float | None) -> str | None:
    """Unix seconds → ISO 8601 UTC (``2026-01-06T08:15:00Z``)."""
    if ts is None or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def dt(ts: float | None) -> datetime | None:
    """Unix seconds → aware UTC datetime (``None`` for missing)."""
    if ts is None or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, UTC)


def num(x: float | None, digits: int = 1) -> float | None:
    """Rounded finite number or ``None``."""
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), digits)


def finite(x: Any) -> bool:
    """Whether ``x`` is a finite number."""
    return x is not None and isinstance(x, int | float | np.floating) and math.isfinite(x)


def higher(a: str, b: str) -> str:
    """The higher of two risk levels."""
    return a if RISK_RANK.get(a, -1) >= RISK_RANK.get(b, -1) else b


# ---- thresholds ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """Risk and alert thresholds (``settings`` table; defaults of the API contract §1) and the hysteresis."""

    red_delay_s: float = 120.0
    red_p_late: float = 0.6
    green_delay_s: float = 60.0
    green_p_late: float = 0.3
    min_level: str = "yellow"
    min_p_late: float = 0.0
    hysteresis_s: float = 20.0
    hysteresis_p: float = 0.1

    def risk(self, pred: float | None, p_late: float | None) -> str:
        """``red`` / ``yellow`` / ``green`` (``unknown`` without a forecast)."""
        if pred is None or not math.isfinite(pred):
            return "unknown"
        if pred > self.red_delay_s or (p_late is not None and p_late > self.red_p_late):
            return "red"
        if pred < self.green_delay_s and (p_late is None or p_late < self.green_p_late):
            return "green"
        return "yellow"

    def hold(self, current: str, pred: float | None, p_late: float | None) -> str:
        """Level of a vehicle now at ``current`` given a new forecast, with the hysteresis.

        Up at once by the thresholds; down only when the forecast is below the thresholds lowered by the
        hysteresis (a red vehicle stays red at 110 s, leaves red at 99 s with the defaults).
        """
        raw = self.risk(pred, p_late)
        if RISK_RANK[raw] >= RISK_RANK.get(current, 0) or raw == "unknown":
            return raw if raw != "unknown" else current
        relaxed = Thresholds(
            red_delay_s=self.red_delay_s - self.hysteresis_s,
            red_p_late=self.red_p_late - self.hysteresis_p,
            green_delay_s=self.green_delay_s - self.hysteresis_s,
            green_p_late=self.green_p_late - self.hysteresis_p,
        ).risk(pred, p_late)
        return relaxed if RISK_RANK[relaxed] < RISK_RANK[current] else current

    def alerting(self, risk: str, p_late: float | None) -> bool:
        """Whether an incident at this level raises an alert."""
        if RISK_RANK.get(risk, -1) < RISK_RANK.get(self.min_level, 1) or risk not in LEVELS:
            return False
        return p_late is None or p_late >= self.min_p_late

    @classmethod
    def from_settings(cls, values: Mapping[str, Any], base: Thresholds | None = None) -> Thresholds:
        """Build from the ``risk_thresholds`` / ``alert_thresholds`` rows (bad values keep ``base``)."""
        base = base or cls()
        risk = values.get("risk_thresholds") or {}
        alert = values.get("alert_thresholds") or {}

        def pick(src: Mapping[str, Any], key: str, default: float) -> float:
            try:
                value = float(src.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        level = str(alert.get("min_level", base.min_level))
        return cls(
            red_delay_s=pick(risk, "red_delay_s", base.red_delay_s),
            red_p_late=pick(risk, "red_p_late", base.red_p_late),
            green_delay_s=pick(risk, "green_delay_s", base.green_delay_s),
            green_p_late=pick(risk, "green_p_late", base.green_p_late),
            min_level=level if level in LEVELS else base.min_level,
            min_p_late=pick(alert, "min_p_late", base.min_p_late),
            hysteresis_s=pick(alert, "hysteresis_s", base.hysteresis_s),
            hysteresis_p=pick(alert, "hysteresis_p", base.hysteresis_p),
        )


# ---- causes ----------------------------------------------------------------------------------------------


def cause_of(
    f: Mapping[str, float], factors: Sequence[Mapping[str, Any]] = (), *, bunching: bool = False
) -> dict[str, Any]:
    """Cause of a forecast: rules on the features, then the strongest contribution with a cause hint.

    Args:
        f: Features of the forecast point.
        factors: Top contributions from ml-service (``feature``, ``contribution_s``).
        bunching: The vehicle is in an active bunching incident.

    Returns:
        ``{code, text, recommendation, factors}`` (API contract §1).
    """

    def v(name: str) -> float:
        x = f.get(name)
        return float(x) if finite(x) else math.nan

    code = "unknown"
    if v("gps_age") > 180 or v("max_gap_10m") > 300:
        code = "gps_lost"
    elif v("stop_dur") >= 180 and v("pos_on_layover") != 1 and not v("spd_60") >= 2:
        code = "dwell_long"
    elif v("pos_on_layover") == 1 and v("pos_delay") > 60:
        code = "layover"
    elif v("own_rate") > 0.15 or (v("spd_300") < 10 and v("stopfrac_300") < 0.7):
        code = "slow_segment"
    elif bunching:
        code = "bunching"
    elif v("cur_best") >= 60 or v("dev_1") >= 60:
        code = "accumulated_delay"
    else:
        for factor in factors:
            hint = cause_hint(str(factor.get("feature")))
            if hint in CAUSES and finite(factor.get("contribution_s")) and factor["contribution_s"] > 0:
                code = hint
                break
    text, recommendation = CAUSES[code]
    out_factors = [
        {
            "feature": str(fc.get("feature")),
            "label": fc.get("label") or label(str(fc.get("feature"))),
            "contribution_s": num(fc.get("contribution_s")),
        }
        for fc in factors
    ]
    return {"code": code, "text": text, "recommendation": recommendation, "factors": out_factors}


# ---- ids, tracks -----------------------------------------------------------------------------------------


class IdGen:
    """Ids of forecasts, alerts and incidents: unique across restarts (start second, then a counter).

    The predictor needs the id before the row reaches PostgreSQL (an alert refers to its forecast while both
    are still in the write buffer), so ids are made here, not by the sequence. They stay below 2^53, so a
    JavaScript client reads them exactly: the start second (~1.8e9) times 2^20 is ~1.9e15; a restart gets
    larger ids as long as the previous run made less than a million ids per second of its life.
    """

    def __init__(self, start_s: int | None = None) -> None:
        self._next = (start_s if start_s is not None else int(time.time())) << 20

    def __call__(self) -> int:
        self._next += 1
        return self._next


@dataclass(frozen=True)
class Track:
    """Cleaned track of a vehicle known at the tick (as ``shared.data.clean_traffic`` cleans offline)."""

    ts: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    unit_id: int
    last_lon: float
    last_lat: float
    last_speed: float
    last_course: float

    def __len__(self) -> int:
        return len(self.ts)


def clean_track(points: Sequence[TrackPoint]) -> Track | None:
    """Keep valid fixes with coordinates and speed ≤ 120 km/h, one per second (the first received).

    Args:
        points: Track window points up to the tick, oldest first.

    Returns:
        The track, ``None`` if no point is left.
    """
    ts: list[float] = []
    lon: list[float] = []
    lat: list[float] = []
    speed: list[float] = []
    last = None
    for p in points:
        if not p.valid or p.speed > MAX_SPEED_KMH or (p.lon == 0 and p.lat == 0):
            continue
        if not (math.isfinite(p.lon) and math.isfinite(p.lat)):
            continue
        if ts and p.ts == ts[-1]:
            continue  # the same second: keep the first (offline: drop_duplicates keep="first")
        ts.append(p.ts)
        lon.append(p.lon)
        lat.append(p.lat)
        speed.append(float(p.speed))
        last = p
    if last is None:
        return None
    return Track(
        ts=np.asarray(ts, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
        lat=np.asarray(lat, dtype=np.float64),
        speed=np.asarray(speed, dtype=np.float64),
        unit_id=last.unit_id,
        last_lon=last.lon,
        last_lat=last.lat,
        last_speed=float(last.speed),
        last_course=float(last.course),
    )


def track_metrics(track: Track | None, t: float, since: float | None = None) -> dict[str, float | None]:
    """Derived features of the vehicle at ``t`` (criterion 3 of the task: speed on the segment, dwell).

    Args:
        track: Cleaned track (valid fixes only).
        t: Tick time; points after it are not read.
        since: Pass of the last stop by the detector (start of the current segment); ``None`` — unknown.

    Returns:
        ``segment_speed_kmh`` — mean speed from the last passed stop (at most :data:`SEGMENT_MAX_S` back) by
        the GPS path length; ``dwell_s`` — how long the vehicle has been standing (speed < 2 km/h) at its
        last fix, 0 if moving; ``idle_s`` — seconds standing in the last :data:`IDLE_WINDOW_S`;
        ``gps_age_s`` — age of the last valid fix. ``None`` where the track is too short.
    """
    out: dict[str, float | None] = dict.fromkeys(("segment_speed_kmh", "dwell_s", "idle_s", "gps_age_s"))
    if track is None:
        return out
    k = int(np.searchsorted(track.ts, t, side="right"))
    if k == 0:
        return out
    ts, lon, lat, spd = track.ts[:k], track.lon[:k], track.lat[:k], track.speed[:k]
    out["gps_age_s"] = round(float(t - ts[-1]), 1)
    moving = np.flatnonzero(spd >= STOP_SPEED_KMH)
    if len(moving) == 0:
        dwell = ts[-1] - ts[0]
    elif moving[-1] == k - 1:
        dwell = 0.0
    else:
        dwell = ts[-1] - ts[moving[-1] + 1]
    out["dwell_s"] = round(float(dwell), 1)
    a = int(np.searchsorted(ts, t - IDLE_WINDOW_S, side="left"))
    if k - a >= 2:
        gaps = np.minimum(np.diff(ts[a:]), IDLE_GAP_S)
        out["idle_s"] = round(float(np.sum(gaps[spd[a:-1] < STOP_SPEED_KMH])), 1)
    start = max(since if since is not None and math.isfinite(since) else -math.inf, t - SEGMENT_MAX_S)
    b = int(np.searchsorted(ts, start, side="left"))
    if k - b >= 2 and ts[-1] - ts[b] >= 30.0:
        coslat = np.cos(np.deg2rad(lat[b:]))
        dx = np.diff(lon[b:]) * coslat[1:] * _M_PER_DEG
        dy = np.diff(lat[b:]) * _M_PER_DEG
        dist = float(np.sum(np.hypot(dx, dy)))
        out["segment_speed_kmh"] = round(dist / float(ts[-1] - ts[b]) * 3.6, 1)
    return out


# ---- forecasts -------------------------------------------------------------------------------------------


@dataclass
class Forecast:
    """One forecast «vehicle × target stop» (a row of ``predictions``)."""

    id: int
    tr_id: int
    unit_id: int | None
    route_id: str | None
    stop_idx: int
    stop_id: int
    stop_name: str
    stop_key: str
    planned: float
    issued_at: float
    epoch: int
    cur_dev: float
    pred: float = math.nan
    first_pred: float = math.nan
    updated_at: float = math.nan
    updates: int = 0
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    p_late: float | None = None
    expected_abs_error: float | None = None
    source: str = "model"
    model_version: str | None = None
    degraded: bool = False
    risk: str = "unknown"
    cause: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, float] = field(default_factory=dict)
    warned: str = "green"
    """Highest incident level of the vehicle while this stop was in the window (what the dispatcher saw)."""
    alert_ids: list[int] = field(default_factory=list)
    status: str = "open"
    actual: float | None = None
    abs_error: float | None = None
    pass_s: float | None = None
    closed_at: float | None = None
    retroactive: bool | None = None

    @property
    def lead_s(self) -> float:
        """Plan − first issue: how early the forecast was given."""
        return self.planned - self.issued_at

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def row(self) -> dict[str, Any]:
        """The ``predictions`` row (upsert by id)."""
        return {
            "id": self.id,
            "tr_id": self.tr_id,
            "unit_id": self.unit_id,
            "route_id": self.route_id,
            "target_stop_id": self.stop_id,
            "target_stop_name": self.stop_name,
            "target_time_begin": dt(self.planned),
            "issued_at": dt(self.issued_at),
            "updated_at": dt(self.updated_at),
            "updates": self.updates,
            "lead_s": self.lead_s,
            "pred_delay_s": float(self.pred),
            "first_pred_delay_s": float(self.first_pred),
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "p_late": self.p_late,
            "risk": self.risk,
            "alert_level": self.warned,
            "cause": self.cause.get("code"),
            "cause_detail": self.cause or None,
            "model_version": self.model_version,
            "source": self.source,
            "degraded": self.degraded,
            "cur_dev_s": self.cur_dev if math.isfinite(self.cur_dev) else None,
            "epoch": self.epoch,
            "status": self.status,
            "actual_delay_s": self.actual,
            "abs_error_s": self.abs_error,
            "pass_time": dt(self.pass_s),
            "actual_lead_s": None if self.pass_s is None else self.pass_s - self.issued_at,
            "retroactive": self.retroactive,
            "closed_at": dt(self.closed_at),
        }

    def out(self) -> dict[str, Any]:
        """``PredictionOut`` (API contract §2)."""
        return {
            "prediction_id": self.id,
            "tr_id": self.tr_id,
            "unit_id": self.unit_id,
            "route_id": self.route_id,
            "target_stop_id": self.stop_id,
            "target_stop_name": self.stop_name,
            "planned_at": iso(self.planned),
            "issued_at": iso(self.issued_at),
            "lead_s": num(self.lead_s),
            "pred_delay_s": num(self.pred),
            "p10": num(self.p10),
            "p50": num(self.p50),
            "p90": num(self.p90),
            "p_late": num(self.p_late, 3),
            "risk": self.risk,
            "model_version": self.model_version,
            "source": self.source,
            "status": "open" if self.is_open else "closed",
            "actual_delay_s": num(self.actual),
            "abs_error_s": num(self.abs_error),
            "closed_at": iso(self.closed_at),
        }


@dataclass
class Alert:
    """An alert: an incident opened or escalated (a row of ``alerts``, ``AlertOut``).

    The forecast values are those at the moment of the alert (what the dispatcher was told); the fact and the
    «after the fact» flag come when the detector passes the alert's target stop.
    """

    id: int
    kind: str
    level: str
    incident_id: int
    forecast: Forecast
    issued_at: float
    cause: dict[str, Any]
    stop_id: int
    stop_name: str
    planned: float
    pred: float
    p10: float | None = None
    p90: float | None = None
    p_late: float | None = None
    escalated_from: int | None = None
    related_tr_id: int | None = None
    status: str = "open"
    closed_at: float | None = None

    @property
    def retroactive(self) -> bool | None:
        """Issued at or after the pass of its stop (``None``: no fact yet or not a delay alert)."""
        f = self.forecast
        if self.kind != "delay" or f.pass_s is None:
            return None
        return self.issued_at >= f.pass_s

    def row(self) -> dict[str, Any]:
        """The ``alerts`` row (upsert by id)."""
        f = self.forecast
        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "severity": LEVELS.get(self.level, "warning"),
            "status": self.status,
            "tr_id": f.tr_id,
            "unit_id": f.unit_id,
            "route_id": f.route_id,
            "stop_id": self.stop_id,
            "target_time_begin": dt(self.planned),
            "issued_at": dt(self.issued_at),
            "prediction_id": f.id,
            "incident_id": self.incident_id,
            "pred_delay_s": float(self.pred),
            "p10": self.p10,
            "p90": self.p90,
            "p_late": self.p_late,
            "cause": self.cause.get("code"),
            "recommendation": self.cause.get("recommendation"),
            "details": {
                "cause": self.cause,
                "text": self.cause.get("text"),
                "stop_name": self.stop_name,
                "source": f.source,
                "incident_id": self.incident_id,
                "escalated_from": self.escalated_from,
                "related_tr_id": self.related_tr_id,
            },
            "model_version": f.model_version,
            "degraded": f.degraded,
            "retroactive": self.retroactive,
            "actual_delay_s": f.actual if self.kind == "delay" and self.stop_id == f.stop_id else None,
            "closed_at": dt(self.closed_at),
        }

    def out(self) -> dict[str, Any]:
        """``AlertOut`` (API contract §2) plus ``kind``, ``incident_id``, the stop and the escalation."""
        f = self.forecast
        return {
            "alert_id": self.id,
            "prediction_id": f.id,
            "incident_id": self.incident_id,
            "kind": self.kind,
            "tr_id": f.tr_id,
            "route_id": f.route_id,
            "level": self.level,
            "cause": self.cause,
            "issued_at": iso(self.issued_at),
            "planned_at": iso(self.planned),
            "target_stop_id": self.stop_id,
            "target_stop_name": self.stop_name,
            "pred_delay_s": num(self.pred),
            "p_late": num(self.p_late, 3),
            "escalated_from": self.escalated_from,
            "related_tr_id": self.related_tr_id,
            "acknowledged": False,
        }


@dataclass
class Incident:
    """An open or closed incident (``incidents`` row, ``IncidentOut``)."""

    id: int
    kind: str
    tr_id: int
    opened_at: float
    related_tr_id: int | None = None
    status: str = "open"
    closed_at: float | None = None
    updated_at: float | None = None
    body: dict[str, Any] = field(default_factory=dict)
    signature: tuple[Any, ...] = ()
    missed: int = 0
    level: str = "green"
    alerted: str = "green"
    """Highest level an alert of this incident has been raised at (a new alert only above it)."""
    clear: int = 0
    seen_at: float = 0.0
    """Last tick the vehicle had a target stop in the window (a delay incident without one for a horizon
    closes)."""
    alert_ids: list[int] = field(default_factory=list)
    levels: list[tuple[float, str]] = field(default_factory=list)

    def set_level(self, level: str, t: float) -> None:
        """Change the level and log the change (``details.levels`` of the row)."""
        if level != self.level:
            self.level = level
            self.levels.append((t, level))

    def row(self) -> dict[str, Any]:
        """The ``incidents`` row (upsert by id)."""
        b = self.body
        target = b.get("target_stop") or {}
        details = self.out()
        details["levels"] = [{"t": iso(t), "level": level} for t, level in self.levels]
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "tr_id": self.tr_id,
            "unit_id": b.get("unit_id"),
            "route_id": b.get("route_id"),
            "related_tr_id": self.related_tr_id,
            "risk": b.get("risk"),
            "target_stop_id": target.get("stop_id"),
            "planned_at": dt(target.get("_planned_s")),
            "pred_delay_s": b.get("pred_delay_s"),
            "p10": b.get("p10"),
            "p90": b.get("p90"),
            "p_late": b.get("p_late"),
            "cause": (b.get("cause") or {}).get("code"),
            "details": details,
            "opened_at": dt(self.opened_at),
            "updated_at": dt(self.updated_at),
            "closed_at": dt(self.closed_at),
        }

    def out(self) -> dict[str, Any]:
        """``IncidentOut`` (API contract §2) plus ``status``, ``opened_at``, ``closed_at``, ``alert_ids``."""
        body = {k: v for k, v in self.body.items() if not k.startswith("_")}
        if "target_stop" in body:
            body["target_stop"] = {k: v for k, v in body["target_stop"].items() if not k.startswith("_")}
        return {
            "incident_id": self.id,
            "kind": self.kind,
            **body,
            "related_tr_id": self.related_tr_id,
            "status": self.status,
            "opened_at": iso(self.opened_at),
            "closed_at": iso(self.closed_at),
            "alert_ids": list(self.alert_ids),
        }


# ---- outputs ---------------------------------------------------------------------------------------------


class Sink(Protocol):
    """Where the engine's rows go (``backend.db.BufferedWriter`` or a test / simulation collector)."""

    def write(self, table: str, row: Mapping[str, Any]) -> None: ...


class ModelBackend(Protocol):
    """Forecast model: ml-service over HTTP (:class:`backend.mlclient.MLClient`) or a local bundle.

    A backend whose model has a sequence component exposes ``sequence_shape`` (``(steps, channels)``); the
    engine then passes ``sequences=`` (one array per row, the same object for the rows of one vehicle).
    """

    status: DependencyStatus
    model_version: str | None

    async def predict(self, rows: Sequence[Mapping[str, float]]) -> MLResult | None: ...


class Publisher:
    """Best-effort Redis output: pub/sub messages (bounded queue) and the latest forecast snapshot.

    Runs in its own task, so a slow or dead Redis never delays a tick. While Redis is down the newest
    ``max_messages`` messages are kept and sent after recovery; the snapshot is only the latest one. The route
    network (static for a schedule) is written once and again after every reconnect.

    Args:
        factory: Redis client factory (``None``: nothing is sent, messages are only counted).
        ttl_s: TTL of the snapshot keys.
        max_messages: Messages kept while Redis is unavailable.
    """

    def __init__(self, factory: RedisFactory | None, ttl_s: int = 600, max_messages: int = 5000) -> None:
        self.factory = factory
        self.ttl_s = ttl_s
        self.messages: deque[tuple[str, str]] = deque(maxlen=max_messages)
        self.snapshot: tuple[dict[str, str], str] | None = None
        self.routes: str | None = None
        self._routes_sent = False
        self.published = 0
        self.errors = 0
        self._wake = asyncio.Event()
        self._client: Any = None

    def publish(self, channel: str, message: Mapping[str, Any]) -> None:
        """Queue one JSON message for a channel."""
        self.messages.append((channel, json.dumps(message, ensure_ascii=False, default=str)))
        self._wake.set()

    def set_snapshot(self, vehicles: Mapping[int, Mapping[str, Any]], status: Mapping[str, Any]) -> None:
        """Replace the forecast snapshot (``foresight:forecast`` hash and its status)."""
        fields = {str(k): json.dumps(v, ensure_ascii=False, default=str) for k, v in vehicles.items()}
        self.snapshot = (fields, json.dumps(status, ensure_ascii=False, default=str))
        self._wake.set()

    def set_routes(self, payload: Mapping[str, Any]) -> None:
        """The route network for ``GET /api/routes`` (``foresight:routes``)."""
        self.routes = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        self._routes_sent = False
        self._wake.set()

    async def flush(self) -> int:
        """Send the queued messages and the snapshot in one transaction; returns messages sent."""
        if self.factory is None:
            self.messages.clear()
            self.snapshot = None
            return 0
        if self._client is None:
            self._client = self.factory()
            self._routes_sent = False
        batch = list(self.messages)[:1000]
        snapshot = self.snapshot
        routes = self.routes if not self._routes_sent else None
        if not batch and snapshot is None and routes is None:
            return 0
        pipe = self._client.pipeline(transaction=True)
        for channel, message in batch:
            pipe.publish(channel, message)
        if snapshot is not None:
            fields, status = snapshot
            pipe.delete(FORECAST_KEY)
            if fields:
                pipe.hset(FORECAST_KEY, mapping=fields)
                pipe.expire(FORECAST_KEY, self.ttl_s)
            pipe.set(FORECAST_STATUS_KEY, status, ex=self.ttl_s)
        if routes is not None:
            pipe.set(ROUTES_KEY, routes)
        await asyncio.wait_for(pipe.execute(), 3.0)
        for _ in batch:
            self.messages.popleft()
        if self.snapshot is snapshot:
            self.snapshot = None
        if routes is not None and self.routes is routes:
            self._routes_sent = True
        self.published += len(batch)
        return len(batch)

    async def run(self) -> None:
        """Sender loop (runs until cancelled)."""
        backoff = Backoff(0.5, 5.0)
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), 1.0)
            self._wake.clear()
            try:
                await self.flush()
            except Exception as exc:
                self.errors += 1
                log.debug("forecast publisher: %s", exc)
                if self._client is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(self._client.aclose(), 1.0)
                    self._client = None
                await asyncio.sleep(backoff.next())
                continue
            backoff.reset()

    async def close(self) -> None:
        """Close the Redis client."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._client.aclose(), 2.0)
            self._client = None


# ---- the engine ------------------------------------------------------------------------------------------


@dataclass
class TickTimings:
    """Durations of the parts of the last tick, seconds."""

    total: float = 0.0
    detector: float = 0.0
    features: float = 0.0
    ml: float = 0.0


@dataclass
class AlertQuality:
    """Alerts checked against the fact of their target stop (the detector's pass), and late stops warned."""

    red: int = 0
    red_true: int = 0
    yellow: int = 0
    yellow_true: int = 0
    late: int = 0
    late_warned_red: int = 0
    late_warned: int = 0

    def out(self) -> dict[str, float | int | None]:
        def share(a: int, b: int) -> float | None:
            return round(a / b, 3) if b else None

        return {
            "red_checked": self.red,
            "red_precision": share(self.red_true, self.red),
            "yellow_checked": self.yellow,
            "yellow_precision": share(self.yellow_true, self.yellow),
            "late_stops": self.late,
            "late_recall_red": share(self.late_warned_red, self.late),
            "late_recall_any": share(self.late_warned, self.late),
        }


class ForecastEngine:
    """The forecast pipeline behind the predictor's tick (see the module docstring).

    Args:
        settings: Configuration (``forecast`` block of :class:`backend.config.Settings`).
        schedule: Plan schedule.
        model: Forecast model (ml-service client or a local bundle).
        sink: Row sink (PostgreSQL writer).
        publisher: Redis output (``None``: none).
        ids: Id generator.
    """

    def __init__(
        self,
        settings: Settings,
        schedule: Schedule,
        model: ModelBackend,
        sink: Sink,
        *,
        publisher: Publisher | None = None,
        ids: IdGen | None = None,
    ) -> None:
        self.s = settings
        self.schedule = schedule
        self.model = model
        self.sink = sink
        self.publisher = publisher
        self.ids = ids or IdGen()
        self.thresholds = Thresholds(
            hysteresis_s=settings.alert_hysteresis_s, hysteresis_p=settings.alert_hysteresis_p
        )
        self.epoch: int | None = None
        self.now: float | None = None
        self.detectors: dict[int, VehicleDetector] = {}
        self.forecasts: dict[tuple[int, int], Forecast] = {}
        self.alert_log: dict[int, Alert] = {}
        self.delay_incidents: dict[int, Incident] = {}
        self.bunching: dict[tuple[int, int], Incident] = {}
        self._rising: dict[int, tuple[str, int]] = {}
        self.window: deque[tuple[float, float, float]] = deque()
        self.timings = TickTimings()
        self.tracks: dict[int, Track] = {}
        self.quality = AlertQuality()
        self.on_features: Callable[[float, list[tuple[int, int]], pd.DataFrame], None] | None = None
        """Observer of every tick's feature rows (the online validation compares them with offline)."""
        # counters
        self.issued = 0
        self.closed = 0
        self.missed = 0
        self.expired = 0
        self.retroactive = 0
        self.alerts = 0
        self.alerts_retroactive = 0
        self.passages = 0
        self.passages_matched = 0
        self.skipped_near = 0
        self.sequences_sent = 0
        self.last_targets = 0
        self.last_active = 0
        self.last_source: str | None = None
        self.forecasts_by_source = dict.fromkeys(SOURCES, 0)
        # Prometheus (docs/observability.md §3.2); every series exists from the start, at zero
        self.m_predictions = Counter(
            "foresight_predictions",
            "Forecasts issued or updated (vehicle × target stop per tick), by source.",
            ["source"],
            registry=None,
        )
        self.m_lead = Histogram(
            "foresight_prediction_lead_seconds",
            "Lead of closed forecasts: detector pass of the target stop − issue time.",
            ["kind"],
            buckets=LEAD_BUCKETS,
            registry=None,
        )
        self.m_abs = Histogram(
            "foresight_prediction_abs_error_seconds",
            "|forecast − fact| of closed forecasts, by source of the last forecast.",
            ["source"],
            buckets=ERROR_BUCKETS,
            registry=None,
        )
        self.m_alerts = Counter(
            "foresight_alerts",
            "Alerts (incident opened or escalated, bunching) by level and cause.",
            ["level", "cause"],
            registry=None,
        )
        self.m_retro = Counter(
            "foresight_alerts_retroactive",
            "Alerts issued at or after the actual pass (must stay 0).",
            registry=None,
        )
        for source in SOURCES:
            self.m_predictions.labels(source=source)
            self.m_abs.labels(source=source)
        for kind in ("prediction", "alert"):
            self.m_lead.labels(kind=kind)
        for level in LEVELS.values():
            for cause in CAUSE_METRIC.values():
                self.m_alerts.labels(level=level, cause=cause)
        if publisher is not None:
            publisher.set_routes(self.routes_payload())

    # ---- state -------------------------------------------------------------------------------------

    def reset(self, epoch: int | None, now: float | None = None) -> None:
        """New stream timeline (replayer restart, another source): forget the vehicles' state.

        Open forecasts, alerts and incidents of the old timeline are closed as ``reset`` (they can never
        get their fact).
        """
        for f in self.forecasts.values():
            if f.is_open:
                self._finish(f, "reset", now if now is not None else f.updated_at)
        for incident in [*self.delay_incidents.values(), *self.bunching.values()]:
            self._close_incident(incident, now if now is not None else incident.opened_at)
        self.detectors.clear()
        self.forecasts.clear()
        self.alert_log.clear()
        self.delay_incidents.clear()
        self.bunching.clear()
        self._rising.clear()
        self.window.clear()
        self.epoch = epoch

    def apply_settings(self, values: Mapping[str, Any]) -> None:
        """Take new thresholds from the ``settings`` table."""
        new = Thresholds.from_settings(values, self.thresholds)
        if new != self.thresholds:
            log.info("thresholds: %s", new)
        self.thresholds = new

    @property
    def open_forecasts(self) -> list[Forecast]:
        """Forecasts waiting for their fact."""
        return [f for f in self.forecasts.values() if f.is_open]

    def routes_payload(self) -> dict[str, Any]:
        """The route network of the plan for ``GET /api/routes`` (see :mod:`shared.routes`)."""
        routes = self.schedule.routes
        return {
            "routes": [r.to_dict() for r in routes.routes],
            "scheduled_tr_ids": sorted(self.schedule.vehicles),
            "geometry": routes.geometry_source,
        }

    # ---- the tick ----------------------------------------------------------------------------------

    async def tick(self, ctx: TickContext) -> None:
        """One tick at stream time ``ctx.stream_time`` (see the module docstring)."""
        started = time.perf_counter()
        t = ctx.stream_time.timestamp()
        if ctx.epoch != self.epoch:
            self.reset(ctx.epoch, t)
        self.now = t
        tracks: dict[int, Track] = {}
        for tr_id in list(ctx.windows):
            if self.schedule.get(tr_id) is None:
                continue
            track = clean_track(ctx.track(tr_id))
            if track is not None:
                tracks[tr_id] = track
        self.tracks = tracks
        for tr_id, track in tracks.items():
            det = self.detectors.get(tr_id)
            if det is None:
                det = self.detectors[tr_id] = VehicleDetector(self.schedule.vehicles[tr_id])
            for passage in det.update(t, track.ts, track.lon, track.lat):
                self._on_passage(tr_id, track.unit_id, passage, t)
        t_det = time.perf_counter()
        self._expire(t)
        active = {tr: tr_track for tr, tr_track in tracks.items() if t - tr_track.ts[-1] <= self.s.active_s}
        self.last_active = len(active)
        targets = self._targets(t, active)
        self.last_targets = len(targets)
        t_feat = t_ml = time.perf_counter()
        if targets:
            features, fctx = self._features(t, targets, active)
            dist = features["dist_target"].to_numpy(dtype=np.float64)
            near = np.isfinite(dist) & (dist < self.s.near_stop_m)
            # a terminal departure is safe: the vehicle waits there, its pass (the departure) is still ahead
            plans = self.schedule.vehicles
            depart = np.array([plans[tr].modes[i] == MODE_DEPART for tr, i in targets], dtype=bool)
            keep = np.flatnonzero(~(near & ~depart)).tolist()
            self.skipped_near += len(targets) - len(keep)
            targets = [targets[i] for i in keep]
            features = features.iloc[keep]
            sequences = self._sequences(t, targets, fctx) if targets else None
            t_feat = time.perf_counter()
            if targets:
                if self.on_features is not None:
                    self.on_features(t, targets, features)
                records = features.to_dict("records")
                if sequences is not None:
                    result = await self.model.predict(records, sequences=sequences)  # type: ignore[call-arg]
                else:
                    result = await self.model.predict(records)
                t_ml = time.perf_counter()
                self._apply(t, targets, records, result, active)
            else:
                t_ml = time.perf_counter()
        self._incidents(t, active)
        self._bunching(t, active)
        for f in self.forecasts.values():  # the current row of every forecast of this tick (upsert by id)
            if f.is_open and f.updated_at == t:
                self.sink.write("predictions", f.row())
        self._publish_snapshot(t)
        end = time.perf_counter()
        self.timings = TickTimings(end - started, t_det - started, t_feat - t_det, t_ml - t_feat)

    # ---- detector ----------------------------------------------------------------------------------

    def _on_passage(self, tr_id: int, unit_id: int, p: Passage, t: float) -> None:
        self.passages += 1
        self.passages_matched += p.matched
        if p.covered:
            self.sink.write(
                "stop_passages",
                {
                    "tr_id": tr_id,
                    "unit_id": unit_id,
                    "stop_id": p.stop_id,
                    "time_begin": dt(p.time_begin),
                    "pass_time": dt(p.pass_s) if p.matched else None,
                    "delay_s": p.delay_s if p.matched else None,
                    "dist_m": p.dist_m if p.matched else None,
                    "confirmed_at": dt(p.confirmed_at),
                    "matched": p.matched,
                    "source": "detector",
                },
            )
        f = self.forecasts.get((tr_id, p.stop_id))
        if f is None or not f.is_open:
            return
        if not p.matched:
            self._finish(f, "missed", p.confirmed_at)
            return
        f.pass_s = p.pass_s
        f.actual = p.pass_s - f.planned
        f.abs_error = abs(f.pred - f.actual)
        f.retroactive = f.issued_at >= p.pass_s
        self.retroactive += f.retroactive
        self.m_lead.labels(kind="prediction").observe(p.pass_s - f.issued_at)
        self.m_abs.labels(source=f.source).observe(f.abs_error)
        th = self.thresholds
        for alert in self._alerts_of(f):
            self.m_lead.labels(kind="alert").observe(p.pass_s - alert.issued_at)
            if alert.issued_at >= p.pass_s:
                self.m_retro.inc()
                self.alerts_retroactive += 1
            if alert.level == "red":
                self.quality.red += 1
                self.quality.red_true += f.actual > th.red_delay_s
            else:
                self.quality.yellow += 1
                self.quality.yellow_true += f.actual >= th.green_delay_s
        if f.actual > th.red_delay_s:
            self.quality.late += 1
            self.quality.late_warned_red += f.warned == "red"
            self.quality.late_warned += RISK_RANK.get(f.warned, 0) >= 1
        base = f.cur_dev if math.isfinite(f.cur_dev) else 0.0
        self.window.append((p.confirmed_at, f.abs_error, abs(base - f.actual)))
        self._finish(f, "closed", p.confirmed_at)
        if self.publisher is not None:
            self.publisher.publish(
                CHANNEL_PREDICTIONS,
                {"type": "prediction_closed", "stream_time": iso(t), "prediction": f.out()},
            )

    def _alerts_of(self, f: Forecast) -> list[Alert]:
        return [self.alert_log[i] for i in f.alert_ids if i in self.alert_log]

    def _finish(self, f: Forecast, status: str, at: float) -> None:
        f.status = status
        f.closed_at = at
        if status == "closed":
            self.closed += 1
        elif status == "missed":
            self.missed += 1
        elif status == "expired":
            self.expired += 1
        self.sink.write("predictions", f.row())
        for alert in self._alerts_of(f):
            self.sink.write("alerts", alert.row())

    def _expire(self, t: float) -> None:
        for key, f in list(self.forecasts.items()):
            if f.is_open and t - f.planned > self.s.expire_after_s:
                self._finish(f, "expired", t)
            if not f.is_open and t - f.planned > 2 * self.s.expire_after_s:
                del self.forecasts[key]
        for alert_id, alert in list(self.alert_log.items()):
            f = alert.forecast
            if alert.status != "open" and not f.is_open and t - f.planned > 2 * self.s.expire_after_s:
                del self.alert_log[alert_id]
        cutoff = t - self.s.online_window_s
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    # ---- targets and features ----------------------------------------------------------------------

    def _targets(self, t: float, active: Mapping[int, Track]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for tr_id in sorted(active):
            plan = self.schedule.vehicles[tr_id]
            det = self.detectors[tr_id]
            for idx in plan.window(t + self.s.horizon_min_s, t + self.s.horizon_max_s):
                if plan.valid[idx] and not det.decided[idx]:
                    out.append((tr_id, int(idx)))
        return out

    def cur_dev(self, tr_id: int, t: float) -> float:
        """Online analogue of ``cur_dev_s`` (see ``Settings.cur_dev_mode``)."""
        mode = self.s.cur_dev_mode
        det = self.detectors.get(tr_id)
        if mode == "nan" or det is None:
            return math.nan
        if mode == "median3":
            return det.median_deviation(t, 3)
        return det.last_deviation(t, self.s.cur_dev_lag_s, regular_only=mode == "last_regular")

    def _features(
        self, t: float, targets: list[tuple[int, int]], tracks: Mapping[int, Track]
    ) -> tuple[pd.DataFrame, FeatureContext]:
        """Features of the targets with the offline code on the windows known at ``t`` (and the context)."""
        ids = sorted({tr for tr, _ in targets})
        parts = []
        for tr_id in ids:
            tr = tracks[tr_id]
            a = int(np.searchsorted(tr.ts, t - FEATURE_LOOKBACK_S, side="left"))
            n = len(tr.ts) - a
            parts.append(
                pd.DataFrame(
                    {
                        "tr_id": np.full(n, tr_id, dtype=np.int64),
                        "event_time": pd.to_datetime(np.round(tr.ts[a:] * 1e9).astype(np.int64), unit="ns"),
                        "lon": tr.lon[a:],
                        "lat": tr.lat[a:],
                        "speed": tr.speed[a:],
                    }
                )
            )
        traffic = pd.concat(parts, ignore_index=True)
        pas = []
        for tr_id in ids:
            det = self.detectors[tr_id]
            idx = det.decided_indices()
            if len(idx) == 0:
                continue
            pas.append(
                pd.DataFrame(
                    {
                        "tr_id": np.full(len(idx), tr_id, dtype=np.int64),
                        "tt_action_item_id": det.plan.stop_ids[idx],
                        "pass_time": pd.to_datetime(det.pass_s[idx], unit="s"),
                        "confirmed_at": pd.to_datetime(det.conf_s[idx], unit="s"),
                    }
                )
            )
        passages = (
            pd.concat(pas, ignore_index=True)
            if pas
            else pd.DataFrame(
                {
                    "tr_id": pd.Series(dtype=np.int64),
                    "tt_action_item_id": pd.Series(dtype=np.int64),
                    "pass_time": pd.Series(dtype="datetime64[ns]"),
                    "confirmed_at": pd.Series(dtype="datetime64[ns]"),
                }
            )
        )
        ctx = FeatureContext(traffic, self.schedule.frame_of(set(ids)), passages)
        plans = self.schedule.vehicles
        points = pd.DataFrame(
            {
                "tr_id": [tr for tr, _ in targets],
                "T": pd.to_datetime(np.full(len(targets), round(t * 1e9), dtype=np.int64), unit="ns"),
                "target_stop_id": [int(plans[tr].stop_ids[i]) for tr, i in targets],
                "target_time_begin": pd.to_datetime(
                    np.array([round(plans[tr].tb[i] * 1e9) for tr, i in targets], dtype=np.int64), unit="ns"
                ),
                "cur_dev_s": [self.cur_dev(tr, t) for tr, _ in targets],
            }
        )
        return build_features(points, ctx), ctx

    def _sequences(
        self, t: float, targets: list[tuple[int, int]], ctx: FeatureContext
    ) -> list[np.ndarray] | None:
        """Telemetry sequences of the targets (one per vehicle, shared by its rows) if the model takes them.

        The same context as the features: points ``≤ t``, passages confirmed ``≤ t`` (``shared.sequences``).
        """
        shape = getattr(self.model, "sequence_shape", None)
        if not self.s.ml_sequences or shape is None:
            return None
        builder = SequenceBuilder(ctx)
        by_tr = {tr: builder.point(tr, t) for tr in sorted({tr for tr, _ in targets})}
        if any(arr.shape != tuple(shape) for arr in by_tr.values()):
            return None  # the model expects another sequence format: forecast without it
        self.sequences_sent += len(by_tr)
        return [by_tr[tr] for tr, _ in targets]

    def fallback(self, features: Mapping[str, float], cur_dev: float) -> tuple[float, str]:
        """Fallback forecast ``intercept + coef · deviation``; returns the value and the deviation used.

        The deviation is ``fallback_feature`` (default ``dev_1``: the delay at the last stop confirmed by the
        detector), else the online ``cur_dev_s``; without either — the intercept alone.
        """
        a, k = self.s.fallback_intercept_s, self.s.fallback_coef
        x = features.get(self.s.fallback_feature)
        if finite(x):
            return a + k * float(x), self.s.fallback_feature
        if math.isfinite(cur_dev):
            return a + k * cur_dev, "cur_dev_s"
        return a, "none"

    # ---- forecasts ---------------------------------------------------------------------------------

    def _apply(
        self,
        t: float,
        targets: list[tuple[int, int]],
        records: list[dict[str, float]],
        result: MLResult | None,
        tracks: Mapping[int, Track],
    ) -> None:
        source = "model" if result is not None else "fallback"
        self.last_source = source
        in_bunching = {inc.tr_id for inc in self.bunching.values()} | {
            inc.related_tr_id for inc in self.bunching.values() if inc.related_tr_id is not None
        }
        for i, ((tr_id, idx), feats) in enumerate(zip(targets, records, strict=True)):
            plan = self.schedule.vehicles[tr_id]
            stop_id = int(plan.stop_ids[idx])
            key = (tr_id, stop_id)
            f = self.forecasts.get(key)
            cur_dev = float(feats.get("cur_dev_s", math.nan))
            cur_dev = cur_dev if math.isfinite(cur_dev) else math.nan
            if f is None or not f.is_open:
                f = Forecast(
                    id=self.ids(),
                    tr_id=tr_id,
                    unit_id=tracks[tr_id].unit_id,
                    route_id=plan.route_id,
                    stop_idx=idx,
                    stop_id=stop_id,
                    stop_name=plan.names[idx],
                    stop_key=plan.keys[idx],
                    planned=float(plan.tb[idx]),
                    issued_at=t,
                    epoch=self.epoch or 0,
                    cur_dev=cur_dev,
                )
                self.forecasts[key] = f
                self.issued += 1
            if result is not None:
                ml = result.predictions[i]
                f.pred = ml.pred_delay_s
                f.p10, f.p50, f.p90, f.p_late = ml.p10, ml.p50, ml.p90, ml.p_late
                f.expected_abs_error = ml.expected_abs_error_s
                f.model_version = result.model_version or f.model_version
                factors: Sequence[Mapping[str, Any]] = ml.factors
            else:
                f.pred, base = self.fallback(feats, cur_dev)
                f.p10 = f.p50 = f.p90 = f.p_late = f.expected_abs_error = None
                f.model_version = "fallback"
                factors = []
                if base != "none":
                    factors = [{"feature": base, "contribution_s": feats.get(base, math.nan)}]
            f.source = source
            f.degraded = result is None
            if f.updates == 0:
                f.first_pred = f.pred
            f.updates += 1
            f.updated_at = t
            f.risk = self.thresholds.risk(f.pred, f.p_late)
            f.cause = cause_of(feats, factors, bunching=tr_id in in_bunching)
            f.snapshot = {k: float(feats[k]) for k in SNAPSHOT_FEATURES if finite(feats.get(k))}
            self.m_predictions.labels(source=source).inc()
            self.forecasts_by_source[source] += 1
            if self.s.log_updates:
                self.sink.write(
                    "prediction_updates",
                    {
                        "prediction_id": f.id,
                        "tr_id": tr_id,
                        "target_stop_id": stop_id,
                        "target_time_begin": dt(f.planned),
                        "tick_at": dt(t),
                        "lead_s": f.planned - t,
                        "pred_delay_s": float(f.pred),
                        "p10": f.p10,
                        "p50": f.p50,
                        "p90": f.p90,
                        "p_late": f.p_late,
                        "source": source,
                        "cur_dev_s": cur_dev if math.isfinite(cur_dev) else None,
                        "model_version": f.model_version,
                        "epoch": f.epoch,
                    },
                )

    # ---- incidents and alerts ----------------------------------------------------------------------

    def _vehicle(self, tr_id: int, f: Forecast | None) -> dict[str, Any]:
        """``IncidentOut.vehicle``: last valid position and the derived features of the vehicle."""
        track = self.tracks.get(tr_id)
        metrics = self.vehicle_metrics(tr_id, self.now or 0.0, f)
        if track is None:
            empty = dict.fromkeys(("lat", "lon", "course_deg", "speed_kmh"))
            return {**empty, **metrics}
        return {
            "lat": round(track.last_lat, 6),
            "lon": round(track.last_lon, 6),
            "course_deg": int(track.last_course),
            "speed_kmh": int(track.last_speed),
            **metrics,
        }

    def vehicle_metrics(self, tr_id: int, t: float, f: Forecast | None = None) -> dict[str, float | None]:
        """Derived features of a vehicle (criterion 3): current delay, segment speed, dwell, idle, GPS age.

        ``current_delay_s`` is the deviation from the plan now: the model's best estimate of it
        (``cur_best``: detector and position on the plan) from the vehicle's nearest forecast, else the online
        ``cur_dev_s`` (median delay of the last three stops passed).
        """
        current = None
        if f is not None:
            current = f.snapshot.get("cur_best", f.snapshot.get("dev_1"))
        if current is None:
            dev = self.cur_dev(tr_id, t)
            current = dev if math.isfinite(dev) else None
        det = self.detectors.get(tr_id)
        since = None
        if det is not None and det.anchor is not None:
            since = float(det.pass_s[det.anchor])
        return {"current_delay_s": num(current), **track_metrics(self.tracks.get(tr_id), t, since)}

    def _segment(self, plan: VehiclePlan, f: Forecast) -> dict[str, Any]:
        n_to = f.snapshot.get("n_to_target")
        start = f.stop_idx - int(n_to) if n_to is not None and n_to >= 0 else f.stop_idx
        start = max(0, min(start, f.stop_idx))
        idx = [i for i in range(start, f.stop_idx + 1) if plan.valid[i]]
        keys = [plan.keys[i] for i in idx]
        line = self.schedule.routes.path(keys) if len(keys) >= 2 else None
        if not line:
            line = [[round(float(plan.slon[i]), 6), round(float(plan.slat[i]), 6)] for i in idx]
        return {"from_stop": plan.names[start], "to_stop": plan.names[f.stop_idx], "line": line}

    def _incident_body(self, f: Forecast, t: float, risk: str, cause: dict[str, Any]) -> dict[str, Any]:
        plan = self.schedule.vehicles[f.tr_id]
        route = self.schedule.routes.route(f.route_id)
        return {
            "tr_id": f.tr_id,
            "unit_id": f.unit_id,
            "route_id": f.route_id,
            "route_name": route.name if route else None,
            "risk": risk,
            "target_stop": {
                "stop_id": f.stop_id,
                "name": f.stop_name,
                "lat": num(float(plan.slat[f.stop_idx]), 6),
                "lon": num(float(plan.slon[f.stop_idx]), 6),
                "planned_at": iso(f.planned),
                "_planned_s": f.planned,
            },
            "pred_delay_s": num(f.pred),
            "p10": num(f.p10),
            "p90": num(f.p90),
            "p_late": num(f.p_late, 3),
            "expected_abs_error_s": num(f.expected_abs_error),
            "cause": cause,
            "segment": self._segment(plan, f),
            "issued_at": iso(f.issued_at),
            "time_to_event_s": num(f.planned + f.pred - t),
            "vehicle": self._vehicle(f.tr_id, f),
            "prediction_id": f.id,
            "source": f.source,
        }

    def _emit_incident(self, incident: Incident, action: str, t: float) -> None:
        incident.updated_at = t
        self.sink.write("incidents", incident.row())
        if self.publisher is not None:
            self.publisher.publish(
                CHANNEL_INCIDENTS,
                {"type": "incident", "stream_time": iso(t), "action": action, "incident": incident.out()},
            )

    def _close_incident(self, incident: Incident, t: float) -> None:
        if incident.status != "open":
            return
        incident.status = "closed"
        incident.closed_at = t
        incident.set_level("green", t)
        for alert_id in incident.alert_ids:
            alert = self.alert_log.get(alert_id)
            if alert is not None and alert.status == "open":
                alert.status = "closed"
                alert.closed_at = t
                self.sink.write("alerts", alert.row())
        self._emit_incident(incident, "close", t)

    def _confirm(self, tr_id: int, current: str, held: str) -> str:
        """An escalation (or opening) needs ``alert_confirm_ticks`` ticks in a row above ``current``."""
        if RISK_RANK[held] <= RISK_RANK.get(current, 0):
            self._rising.pop(tr_id, None)
            return held
        prev = self._rising.get(tr_id)
        count = prev[1] + 1 if prev is not None else 1
        if count >= self.s.alert_confirm_ticks:
            self._rising.pop(tr_id, None)
            return held
        self._rising[tr_id] = (held, count)
        return current

    def _incidents(self, t: float, active: Mapping[int, Track]) -> None:
        """Delay incidents per vehicle and their alerts (see the module docstring, item 5)."""
        window: dict[int, list[Forecast]] = {}
        for f in self.forecasts.values():
            if f.is_open and f.updated_at == t:
                window.setdefault(f.tr_id, []).append(f)
        for tr_id, incident in list(self.delay_incidents.items()):
            # the vehicle went silent, or has had no stop in the window for a whole horizon (end of the day)
            if tr_id not in active or (tr_id not in window and t - incident.seen_at > self.s.horizon_max_s):
                self._close_incident(incident, t)
                del self.delay_incidents[tr_id]
        for tr_id in sorted(set(window) | set(self.delay_incidents)):
            fs = window.get(tr_id)
            if not fs:
                continue  # no stop in the window (a layover): the incident stays as it is
            fs.sort(key=lambda x: (x.planned, x.stop_idx))
            rep = fs[0]  # the nearest target: the forecast closest to its event
            incident = self.delay_incidents.get(tr_id)
            current = incident.level if incident is not None else "green"
            level = self._confirm(tr_id, current, self.thresholds.hold(current, rep.pred, rep.p_late))
            if incident is None:
                if RISK_RANK.get(level, 0) < 1:
                    continue
                incident = Incident(id=self.ids(), kind="delay", tr_id=tr_id, opened_at=t, seen_at=t)
                incident.set_level(level, t)
                incident.body = self._incident_body(rep, t, level, rep.cause)
                incident.signature = self._signature(incident, rep)
                self.delay_incidents[tr_id] = incident
                self._emit_incident(incident, "open", t)
                self._raise(incident, rep, level, t)
            else:
                incident.seen_at = t
                if RISK_RANK.get(level, 0) < 1:
                    incident.clear += 1
                    if incident.clear >= self.s.incident_clear_ticks:
                        self._close_incident(incident, t)
                        del self.delay_incidents[tr_id]
                        continue
                    level = "yellow"  # below the exit threshold, not long enough yet
                else:
                    incident.clear = 0
                incident.set_level(level, t)
                incident.body = self._incident_body(rep, t, level, rep.cause)
                if RISK_RANK[level] > RISK_RANK[incident.alerted]:
                    self._raise(incident, rep, level, t)
                signature = self._signature(incident, rep)
                if signature != incident.signature:
                    incident.signature = signature
                    self._emit_incident(incident, "update", t)
            for f in fs:
                f.warned = higher(f.warned, incident.level)

    @staticmethod
    def _signature(incident: Incident, rep: Forecast) -> tuple[Any, ...]:
        return (incident.level, rep.stop_id, round(rep.pred / 15.0), rep.cause.get("code"))

    def _raise(self, incident: Incident, f: Forecast, level: str, t: float) -> None:
        """Alert of a delay incident at ``level`` about its nearest target ``f`` (if the settings alert)."""
        if not self.thresholds.alerting(level, f.p_late):
            return
        escalated_from = incident.alert_ids[-1] if incident.alert_ids else None
        alert = Alert(
            id=self.ids(),
            kind="delay",
            level=level,
            incident_id=incident.id,
            forecast=f,
            issued_at=t,
            cause=f.cause,
            stop_id=f.stop_id,
            stop_name=f.stop_name,
            planned=f.planned,
            pred=f.pred,
            p10=f.p10,
            p90=f.p90,
            p_late=f.p_late,
            escalated_from=escalated_from,
        )
        incident.alerted = level
        self._record_alert(incident, alert, CAUSE_METRIC.get(f.cause.get("code", ""), "other"), t)
        f.alert_ids.append(alert.id)

    def _record_alert(self, incident: Incident, alert: Alert, cause_label: str, t: float) -> None:
        self.alert_log[alert.id] = alert
        incident.alert_ids.append(alert.id)
        self.alerts += 1
        self.m_alerts.labels(level=LEVELS[alert.level], cause=cause_label).inc()
        self.sink.write("alerts", alert.row())
        if self.publisher is not None:
            message = {"type": "alert", "stream_time": iso(t), "alert": alert.out()}
            self.publisher.publish(CHANNEL_ALERTS, message)

    def _bunching(self, t: float, active: Mapping[int, Track]) -> None:
        """Pairs of vehicles of a route forecast too close (or too far) at a common stop ahead."""
        estimate: dict[int, Forecast] = {}
        for f in self.forecasts.values():
            if f.is_open and f.tr_id in active and math.isfinite(f.pred):
                best = estimate.get(f.tr_id)
                if best is None or f.planned < best.planned:
                    estimate[f.tr_id] = f
        groups: dict[tuple[str, str], list[tuple[float, float, int, int]]] = {}
        for tr_id, f in estimate.items():
            plan = self.schedule.vehicles[tr_id]
            if plan.route_id is None:
                continue
            det = self.detectors[tr_id]
            for idx in plan.window(t, t + self.s.bunching_horizon_s):
                if plan.valid[idx] and not det.decided[idx]:
                    tb = float(plan.tb[idx])
                    group = groups.setdefault((plan.route_id, plan.keys[idx]), [])
                    group.append((tb, tb + f.pred, tr_id, int(idx)))
        found: dict[tuple[int, int], tuple[str, float, float, float, int, int, int]] = {}
        for items in groups.values():
            if len(items) < 2:
                continue
            items.sort()
            for a, b in zip(items[:-1], items[1:], strict=True):
                if a[2] == b[2]:
                    continue
                hp, h = b[0] - a[0], b[1] - a[1]
                if hp >= 2 * self.s.bunching_s and h < self.s.bunching_s:
                    kind = "bunching"
                elif h - hp > self.s.gap_s and h > self.s.gap_factor * hp:
                    kind = "gap"
                else:
                    continue
                pair = (a[2], b[2])
                if pair not in found or a[1] < found[pair][2]:
                    found[pair] = (kind, h, a[1], hp, b[2], b[3], a[2])
        for pair, incident in list(self.bunching.items()):
            if pair in found:
                incident.missed = 0
                continue
            incident.missed += 1
            if incident.missed >= 2 or pair[0] not in active or pair[1] not in active:
                self._close_incident(incident, t)
                del self.bunching[pair]
        for pair, (kind, h, _arr, hp, follower, idx, leader) in found.items():
            f = estimate[follower]
            plan = self.schedule.vehicles[follower]
            text, recommendation = CAUSES["bunching"] if kind == "bunching" else GAP_CAUSE
            cause = {
                "code": "bunching",
                "text": text,
                "recommendation": recommendation,
                "factors": [
                    {"feature": "headway_s", "label": "Прогнозный интервал", "contribution_s": num(h)},
                    {"feature": "planned_headway_s", "label": "Плановый интервал", "contribution_s": num(hp)},
                ],
            }
            risk = "red" if kind == "bunching" and h < 0 else "yellow"
            body = self._incident_body(f, t, risk, cause)
            body["target_stop"] = {
                "stop_id": int(plan.stop_ids[idx]),
                "name": plan.names[idx],
                "lat": num(float(plan.slat[idx]), 6),
                "lon": num(float(plan.slon[idx]), 6),
                "planned_at": iso(float(plan.tb[idx])),
                "_planned_s": float(plan.tb[idx]),
            }
            body["time_to_event_s"] = num(float(plan.tb[idx]) + f.pred - t)
            body["bunching"] = {"kind": kind, "headway_s": num(h), "planned_headway_s": num(hp)}
            signature = (kind, int(plan.stop_ids[idx]), round(h / 15.0))
            incident = self.bunching.get(pair)
            if incident is None:
                incident = Incident(
                    id=self.ids(),
                    kind="bunching",
                    tr_id=follower,
                    related_tr_id=leader,
                    opened_at=t,
                    body=body,
                )
                incident.set_level(risk, t)
                incident.signature = signature
                self.bunching[pair] = incident
                self._emit_incident(incident, "open", t)
                if risk in LEVELS:
                    target = body["target_stop"]
                    alert = Alert(
                        id=self.ids(),
                        kind="bunching",
                        level=risk,
                        incident_id=incident.id,
                        forecast=f,
                        issued_at=t,
                        cause=cause,
                        stop_id=target["stop_id"],
                        stop_name=target["name"],
                        planned=target["_planned_s"],
                        pred=f.pred,
                        p_late=f.p_late,
                        related_tr_id=leader,
                    )
                    incident.alerted = risk
                    self._record_alert(incident, alert, "bunching", t)
            else:
                incident.body = body
                incident.set_level(risk, t)
                if signature != incident.signature:
                    incident.signature = signature
                    self._emit_incident(incident, "update", t)

    # ---- outputs -----------------------------------------------------------------------------------

    def vehicle_forecast(self, tr_id: int, t: float) -> dict[str, Any]:
        """Hot forecast state of a vehicle for the api (``VehicleOut`` additions of the API contract §2).

        ``risk`` is the level of the vehicle's open incident (held with the hysteresis), else the risk of its
        forecast; the forecast is the one the incident follows — the nearest target of the latest tick (the
        10–15 min window), not an older still-open one whose stop is now closer: its value is stale and
        would disagree with the level. A forecast not updated for a whole horizon (the vehicle left its
        stops: end of the day, off the route) is dropped, as the incident is. The derived features
        (criterion 3) come flat and in ``features``.
        """
        plan = self.schedule.vehicles.get(tr_id)
        own = [f for f in self.forecasts.values() if f.tr_id == tr_id and f.is_open]
        ticks = [f.updated_at for f in own if math.isfinite(f.updated_at)]
        if ticks:
            last = max(ticks)
            own = [f for f in own if f.updated_at == last] if t - last <= self.s.horizon_max_s else []
        nearest = min(own, key=lambda f: (f.planned, f.stop_idx)) if own else None
        incident = self.delay_incidents.get(tr_id)
        if incident is None:
            incident = next((i for i in self.bunching.values() if i.tr_id == tr_id), None)
        metrics = self.vehicle_metrics(tr_id, t, nearest)
        risk = nearest.risk if nearest is not None else "unknown"
        delay_incident = self.delay_incidents.get(tr_id)
        if delay_incident is not None:
            risk = delay_incident.level
        out: dict[str, Any] = {
            "tr_id": tr_id,
            "route_id": plan.route_id if plan else None,
            "scheduled": plan is not None,
            "risk": risk,
            "current_delay_s": metrics["current_delay_s"],
            "pred_delay_s": num(nearest.pred) if nearest is not None else None,
            "p10": num(nearest.p10) if nearest is not None else None,
            "p90": num(nearest.p90) if nearest is not None else None,
            "p_late": num(nearest.p_late, 3) if nearest is not None else None,
            "next_stop": None,
            "incident_id": incident.id if incident is not None else None,
            "incident_ids": [
                i.id
                for i in (delay_incident, *self.bunching.values())
                if i is not None and i.status == "open" and i.tr_id == tr_id
            ],
            "prediction_id": nearest.id if nearest is not None else None,
            "source": nearest.source if nearest is not None else None,
            "segment_speed_kmh": metrics["segment_speed_kmh"],
            "dwell_s": metrics["dwell_s"],
            "idle_s": metrics["idle_s"],
            "gps_age_s": metrics["gps_age_s"],
            "features": metrics,
            "stream_time": iso(t),
        }
        track = self.tracks.get(tr_id)
        if track is not None:
            out["position"] = {"lat": round(track.last_lat, 6), "lon": round(track.last_lon, 6)}
        if nearest is not None and plan is not None:
            # следующая остановка ТС — не цель прогноза (она в 10–15 мин), а конец текущего отрезка плана:
            # ТС между base и base + 1, где base = цель − n_to_target (привязка GPS к плану, иначе детектор)
            idx = nearest.stop_idx
            n_to = nearest.snapshot.get("n_to_target")
            if n_to is not None and math.isfinite(n_to) and n_to >= 0:
                idx = min(nearest.stop_idx, max(0, nearest.stop_idx - int(n_to) + 1))
            out["next_stop"] = {
                "stop_id": int(plan.stop_ids[idx]),
                "stop_key": plan.keys[idx],
                "name": plan.names[idx],
                "planned_at": iso(float(plan.tb[idx])),
            }
        return out

    def _publish_snapshot(self, t: float) -> None:
        if self.publisher is None:
            return
        vehicles = {tr_id: self.vehicle_forecast(tr_id, t) for tr_id in self.tracks}
        status = {
            "stream_time": iso(t),
            "epoch": self.epoch,
            "degraded": self.last_source == "fallback",
            "source": self.last_source,
            "model_version": self.model.model_version,
            "ml": self.model.status.state,
            "predictions_open": len(self.open_forecasts),
            "incidents_open": len(self.delay_incidents) + len(self.bunching),
            "scheduled_tr_ids": sorted(self.schedule.vehicles),
        }
        self.publisher.set_snapshot(vehicles, status)

    # ---- metrics and stats -------------------------------------------------------------------------

    def online_mae(self) -> tuple[float | None, float | None, int]:
        """Online MAE and baseline MAE of the forecasts closed in the last ``online_window_s``."""
        n = len(self.window)
        if n == 0:
            return None, None, 0
        err = float(np.mean([w[1] for w in self.window]))
        base = float(np.mean([w[2] for w in self.window]))
        return err, base, n

    def metrics(self) -> Iterator[Metric]:
        """Prometheus metric families of the engine (docs/observability.md §3.2)."""
        yield from self.m_predictions.collect()
        yield from self.m_lead.collect()
        yield from self.m_abs.collect()
        yield from self.m_alerts.collect()
        yield from self.m_retro.collect()
        mae, base, n = self.online_mae()
        if mae is not None and n >= self.s.online_min_closed:
            yield GaugeMetricFamily(
                "foresight_online_mae_seconds", "Online MAE of the forecasts closed in the last hour.", mae
            )
            yield GaugeMetricFamily(
                "foresight_online_baseline_mae_seconds",
                "MAE of the baseline «forecast = online cur_dev_s at issue» on the same forecasts.",
                base,
            )
        yield GaugeMetricFamily(
            "foresight_predictions_open",
            "Forecasts waiting for the pass of their stop.",
            len(self.open_forecasts),
        )

    def stats(self) -> dict[str, Any]:
        """Counters for ``/api/predictor/stats``."""
        mae, base, n = self.online_mae()
        return {
            "enabled": True,
            "vehicles_planned": len(self.schedule.vehicles),
            "routes": len(self.schedule.routes.routes),
            "ml": self.model.status.state,
            "model_version": self.model.model_version,
            "last_source": self.last_source,
            "active_vehicles": self.last_active,
            "targets": self.last_targets,
            "predictions_open": len(self.open_forecasts),
            "predictions_issued": self.issued,
            "forecasts_model": self.forecasts_by_source["model"],
            "forecasts_fallback": self.forecasts_by_source["fallback"],
            "closed": self.closed,
            "missed": self.missed,
            "expired": self.expired,
            "retroactive": self.retroactive,
            "skipped_near_stop": self.skipped_near,
            "alerts": self.alerts,
            "alerts_retroactive": self.alerts_retroactive,
            "alert_quality": self.quality.out(),
            "incidents_open": len(self.delay_incidents),
            "bunching_open": len(self.bunching),
            "online_mae_s": num(mae),
            "online_baseline_mae_s": num(base),
            "online_closed": n,
            "passages": self.passages,
            "passages_matched": self.passages_matched,
            "sequences_sent": self.sequences_sent,
            "tick_ms": {
                "total": round(self.timings.total * 1000, 1),
                "detector": round(self.timings.detector * 1000, 1),
                "features": round(self.timings.features * 1000, 1),
                "ml": round(self.timings.ml * 1000, 1),
            },
        }


def records_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Rows collected by a sink → DataFrame (used by the validation and tests)."""
    return pd.DataFrame(list(rows))
