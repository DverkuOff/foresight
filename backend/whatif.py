"""What-if of a dispatcher action on a route (``POST /api/whatif``): headways at the stops before and after.

The vehicles of the route arrive at their planned stops with the deviation the forecasts expect (the current
deviation now, the forecast of the stop 10–15 min ahead, held after it — as on the stringline). Actions:

* ``add_vehicle`` — a reserve vehicle leaves ``from_stop_key`` at ``depart_at`` and runs the route with the
  median planned run times between consecutive stops;
* ``hold`` — vehicle ``tr_id`` stands ``hold_s`` longer at its next stop: all its later arrivals shift.

Metrics of the horizon ``[at, at + horizon]``: the mean wait of a passenger who comes to a stop at a random
moment of the horizon until the next service there (also one after the horizon — no edge effects: a held
vehicle cannot «shorten» the waits by leaving the window), averaged over the stops; the largest headway
around the horizon (from the stop's last service before ``at``: a route run by one vehicle has the headway
of its loop); late stops (plan in the horizon, > 120 s behind) and bunching pairs (two vehicles with a
headway below 30 % of the planned one, 60 s…5 min). A simpler version of the model runs in the
dashboard's mock mode (``dashboard/src/mocks/whatifEngine.ts``).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from backend.schedule import VehiclePlan
from backend.stringline import interpolate, latest_tick, plan_seqs

LATE_S = 120.0
RESERVE_DEPART_S = 300.0
"""Default departure of a reserve vehicle: 5 min after ``at``."""
LOOKBACK_S = 3 * 3600.0
"""The first headway at a stop counts from its last service before ``at`` (a route run by one vehicle: its
previous loop), up to this far back; the next service after the horizon is looked for as far ahead."""
BUNCHING_MAX_S = 300.0
"""Bunching: two vehicles closer than 30 % of the planned headway, but at most 5 min (a route run by one
vehicle has its whole loop as the planned headway, and 30 % of it is not «back to back»)."""
SAME_SERVICE_S = 600.0
"""Two arrivals of one vehicle at one place this close are one service (a terminal: arrival and departure)."""


@dataclass(frozen=True)
class Arrival:
    stop_key: str
    seq: int
    plan: float | None
    """Plan, Unix seconds (``None``: a reserve vehicle has no plan)."""
    est: float
    """Expected arrival, Unix seconds."""


@dataclass
class Vehicle:
    tr_id: int | None
    arrivals: list[Arrival] = field(default_factory=list)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def mean_wait(gaps: Sequence[float]) -> float:
    """Mean wait of a passenger arriving at random: ``Σg² / 2Σg``."""
    total = sum(gaps)
    return sum(g * g for g in gaps) / (2 * total) if total > 0 else 0.0


def window_wait(services: Sequence[float], at: float, until: float, cap: float) -> float:
    """Mean wait of a passenger coming at a uniformly random moment of ``[at, until]`` for the next of the
    (sorted) ``services``; without one it waits until ``cap``."""
    total, t = 0.0, at
    for s in services:
        if s <= t:
            continue
        end = min(s, until)
        total += (end - t) * ((s - t) + (s - end)) / 2
        t = end
        if t >= until:
            break
    if t < until:
        total += (until - t) * ((cap - t) + (cap - until)) / 2
    return total / (until - at) if until > at else 0.0


def run_times(route: Mapping[str, Any], plans: Mapping[int, VehiclePlan]) -> tuple[list[str], list[float]]:
    """Stop keys of the route by ``seq`` and the median planned run time ``seq → seq + 1`` (60 s unknown)."""
    stops = sorted(route["stops"], key=lambda s: int(s["seq"]))
    keys = [s["stop_key"] for s in stops]
    n = len(keys)
    seq_of: dict[str, list[int]] = {}
    for s in stops:
        seq_of.setdefault(s["stop_key"], []).append(int(s["seq"]))
    runs: dict[int, list[float]] = {}
    for tr_id in route.get("tr_ids") or []:
        plan = plans.get(int(tr_id))
        if plan is None:
            continue
        seqs = plan_seqs(plan.keys, seq_of, n)
        for i in range(len(plan) - 1):
            a, b = seqs[i], seqs[i + 1]
            dt = float(plan.tb[i + 1] - plan.tb[i])
            if a is not None and b is not None and b == (a + 1) % n and 0 < dt < 1800:
                runs.setdefault(a, []).append(dt)
    return keys, [statistics.median(runs[s]) if s in runs else 60.0 for s in range(n)]


def route_vehicles(
    route: Mapping[str, Any],
    plans: Mapping[int, VehiclePlan],
    at: float,
    horizon_s: float,
    *,
    forecasts: Sequence[Mapping[str, Any]] = (),
    current: Mapping[int, float | None] | None = None,
) -> list[Vehicle]:
    """The planned arrivals of the route's vehicles around ``at`` with the expected deviation (the ones
    before ``at`` are the services passengers already had: the start of the first headways)."""
    stops = route["stops"]
    seq_of: dict[str, list[int]] = {}
    for s in stops:
        seq_of.setdefault(s["stop_key"], []).append(int(s["seq"]))
    n = len(stops)
    fc_by_tr: dict[int, list[Mapping[str, Any]]] = {}
    for f in forecasts:
        fc_by_tr.setdefault(int(f["tr_id"]), []).append(f)
    out = []
    for tr_id in route.get("tr_ids") or []:
        plan = plans.get(int(tr_id))
        if plan is None:
            continue
        seqs = plan_seqs(plan.keys, seq_of, n)
        cur = (current or {}).get(int(tr_id))
        cur = float(cur) if cur is not None and math.isfinite(float(cur)) else None
        anchors = []
        for f in latest_tick(fc_by_tr.get(int(tr_id), [])):
            i = plan.index.get(int(f.get("target_stop_id") or 0))
            if i is not None and f.get("pred_delay_s") is not None:
                anchors.append((float(plan.tb[i]), float(f["pred_delay_s"]), None, None))
        anchors.sort()
        start = (at, cur if cur is not None else (anchors[0][1] if anchors else 0.0), None, None)
        arrivals = []
        for i in range(len(plan)):
            tb = float(plan.tb[i])
            s = seqs[i]
            if s is None or tb < at - LOOKBACK_S - 1800 or tb > at + horizon_s + LOOKBACK_S:
                continue
            delay = interpolate(start, anchors, tb)[0] if anchors else start[1]
            arrivals.append(Arrival(plan.keys[i], s, tb, tb + delay))
        out.append(Vehicle(int(tr_id), arrivals))
    return out


def apply_action(
    vehicles: list[Vehicle],
    request: Mapping[str, Any],
    keys: Sequence[str],
    run_s: Sequence[float],
    at: float,
) -> list[Vehicle]:
    """The vehicles after the dispatcher's action."""
    params = request.get("params") or {}
    horizon_s = float(request.get("horizon_min") or 60) * 60
    if request.get("action") == "hold":
        hold = float(params.get("hold_s") or 120)
        tr_id = params.get("tr_id")
        out = []
        for v in vehicles:
            if tr_id is None or v.tr_id != int(tr_id):
                out.append(v)
                continue
            shifted, held = [], False
            for a in v.arrivals:  # the vehicle stands longer at its next stop: every later arrival shifts
                if a.est < at:
                    shifted.append(a)
                    continue
                shifted.append(replace(a, est=a.est + (hold if held else 0.0)))
                held = True
            out.append(Vehicle(v.tr_id, shifted))
        return out
    n = len(keys)
    if n == 0:
        return list(vehicles)
    from_key = params.get("from_stop_key") or keys[0]
    seq = keys.index(from_key) if from_key in keys else 0
    depart = _ts(params.get("depart_at")) or at + RESERVE_DEPART_S
    reserve, t = [], depart
    for _ in range(n * 20):  # the reserve keeps running after the horizon (the next services of the stops)
        if t > at + horizon_s + LOOKBACK_S:
            break
        reserve.append(Arrival(keys[seq], seq, None, t))
        t += max(20.0, run_s[seq] if seq < len(run_s) else 60.0)
        seq = (seq + 1) % n
    return [*vehicles, Vehicle(None, reserve)]


def _ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()


def evaluate(vehicles: Sequence[Vehicle], at: float, horizon_s: float) -> dict[str, Any]:
    """``Scenario`` of the contract: headways, mean wait, largest headway, late stops, bunching pairs."""
    until = at + horizon_s
    by_stop: dict[str, list[tuple[float, int, float | None]]] = {}
    late = 0
    for idx, v in enumerate(vehicles):
        for a in v.arrivals:
            if a.plan is not None and at <= a.plan <= until and a.est - a.plan > LATE_S:
                late += 1  # by the plan in the horizon: a held vehicle cannot «lose» its late stops past it
            if at - LOOKBACK_S <= a.est <= until + LOOKBACK_S:
                by_stop.setdefault(a.stop_key, []).append((a.est, idx, a.plan))
    headways: list[dict[str, Any]] = []
    waits: list[float] = []
    pairs: set[tuple[int, int]] = set()
    max_gap = 0.0
    for key, items in by_stop.items():
        items.sort()
        services: list[tuple[float, int, float | None]] = []
        for item in items:
            if services and services[-1][1] == item[1] and item[0] - services[-1][0] < SAME_SERVICE_S:
                continue
            services.append(item)
        items = services
        plans = sorted(p for _, _, p in items if p is not None)
        planned = [b - a for a, b in zip(plans, plans[1:], strict=False) if b - a > 30]
        threshold = min(max(60.0, (statistics.median(planned) if planned else 300.0) * 0.3), BUNCHING_MAX_S)
        for (t0, i0, _), (t1, i1, _) in zip(items, items[1:], strict=False):
            if t1 < at or t0 > until:
                continue  # the headway is not around the horizon
            gap = t1 - t0
            max_gap = max(max_gap, gap)
            if t1 <= until:
                headways.append({"stop_key": key, "t": _iso(t1), "gap_s": round(gap)})
                if i0 != i1 and gap < threshold:
                    pairs.add((min(i0, i1), max(i0, i1)))
        if any(at <= t <= until + LOOKBACK_S for t, _, _ in items):
            waits.append(window_wait([t for t, _, _ in items], at, until, until + LOOKBACK_S))
    headways.sort(key=lambda h: h["t"])
    return {
        "headways": headways,
        "mean_wait_s": round(sum(waits) / len(waits)) if waits else 0,
        "max_gap_s": round(max_gap),
        "late_stops": late,
        "bunching_pairs": len(pairs),
        "vehicles": [
            {
                "tr_id": v.tr_id,
                "arrivals": [
                    {"stop_key": a.stop_key, "seq": a.seq, "t": _iso(a.est)}
                    for a in v.arrivals
                    if at <= a.est <= until
                ],
            }
            for v in vehicles
        ],
    }


def request_error(route: Mapping[str, Any], request: Mapping[str, Any]) -> str | None:
    """Why the action cannot be simulated on the route (``None``: it can): a held vehicle must run the route,
    a reserve vehicle must leave from one of its stops — otherwise the scenario would silently equal the
    baseline."""
    params = request.get("params") or {}
    action = request.get("action")
    tr_id = params.get("tr_id")
    tr_ids = {int(t) for t in route.get("tr_ids") or []}
    if action == "hold" and tr_id is not None and int(tr_id) not in tr_ids:
        return f"vehicle {tr_id} does not run route {route.get('route_id')}"
    key = params.get("from_stop_key")
    stop_keys = {s.get("stop_key") for s in route.get("stops") or []}
    if action == "add_vehicle" and key is not None and key not in stop_keys:
        return f"stop {key} is not on route {route.get('route_id')}"
    return None


def run_whatif(
    route: Mapping[str, Any],
    plans: Mapping[int, VehiclePlan],
    request: Mapping[str, Any],
    at: float,
    *,
    forecasts: Sequence[Mapping[str, Any]] = (),
    current: Mapping[int, float | None] | None = None,
) -> dict[str, Any]:
    """``{baseline, scenario, delta}`` of the contract."""
    horizon_s = float(request.get("horizon_min") or 60) * 60
    keys, run_s = run_times(route, plans)
    vehicles = route_vehicles(route, plans, at, horizon_s, forecasts=forecasts, current=current)
    baseline = evaluate(vehicles, at, horizon_s)
    scenario = evaluate(apply_action(vehicles, request, keys, run_s, at), at, horizon_s)
    return {
        "baseline": baseline,
        "scenario": scenario,
        "delta": {
            k: scenario[k] - baseline[k] for k in ("mean_wait_s", "max_gap_s", "late_stops", "bunching_pairs")
        },
    }
