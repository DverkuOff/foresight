"""«Stringline» of a route (``GET /api/stringline``): time × stops for the plan, the fact and the forecast.

The stop axis is the route's full trip (forward, then back: ``RouteOut.stops`` with their ``seq``). A vehicle
runs the route in circles, so the same place can appear twice on the axis (a terminal, a stop served both
ways); each planned stop of a vehicle takes the ``seq`` that continues its previous one.

* ``planned`` — the plan schedule of the vehicles of the route;
* ``actual`` — the passes restored by the stop detector on the stream (``stop_passages``);
* ``forecast`` — from the vehicle's next stop and its current deviation now to the forecast of the target
  stop 10–15 min ahead (linear in between, the forecast delay held after it), with the P10–P90 band of the
  arrival time. A late vehicle still has to pass stops planned in the past: they are forecast too. Only the
  forecasts of the vehicle's latest tick count: an open forecast stops being updated once its stop leaves
  the 10–15 min window, and its value (made minutes ago) is older than the current deviation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from backend.schedule import VehiclePlan


def _iso(ts: float | None) -> str | None:
    if ts is None or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def _ts(value: Any) -> float | None:
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()
    if isinstance(value, str) and value:
        try:
            return _ts(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def plan_seqs(keys: Sequence[str], seq_of: Mapping[str, Sequence[int]], n: int) -> list[int | None]:
    """``seq`` of every planned stop of a vehicle: among the places of the stop on the axis the one that
    continues the previous stop (fewest steps forward, round the circle)."""
    out: list[int | None] = []
    prev: int | None = None
    for key in keys:
        cands = seq_of.get(key)
        if not cands:
            out.append(None)
            continue
        if prev is None or len(cands) == 1 or n <= 0:
            s = cands[0]
        else:
            p = prev
            s = min(cands, key=lambda c: (c - p - 1) % n)
        out.append(s)
        prev = s
    return out


def monotonic(points: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Points (in time order) without steps back along the route: the detector may confirm two close stops
    out of order, and the line would jump back and forth. A step back by more than half of the axis is the
    start of a new loop and stays."""
    out: list[dict[str, Any]] = []
    for p in points:
        if out and p["seq"] < out[-1]["seq"] and out[-1]["seq"] - p["seq"] <= n / 2:
            continue
        out.append(p)
    return out


def build_stringline(
    route: Mapping[str, Any],
    plans: Mapping[int, VehiclePlan],
    lo: float,
    hi: float,
    *,
    passages: Sequence[Mapping[str, Any]] = (),
    forecasts: Sequence[Mapping[str, Any]] = (),
    current: Mapping[int, float | None] | None = None,
    positions: Mapping[int, int] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The stringline of a route between ``lo`` and ``hi`` (Unix seconds of stream time).

    Args:
        route: ``RouteOut`` (dict) of the route.
        plans: Plan of the vehicles by ``tr_id``.
        lo: From.
        hi: To.
        passages: Stop passes (``tr_id``, ``stop_id``, ``pass_time``).
        forecasts: Open forecasts (``PredictionOut``) of the vehicles.
        current: Current deviation from the plan by ``tr_id`` (the start of the forecast line).
        positions: Next planned stop (``stop_id``) by ``tr_id``: the vehicle is drawn «now» half a stop
            before it (also when the detector missed the last stops and the actual line has a gap).
        now: Stream time.

    Returns:
        ``{route_id, stops, trips, stream_time}`` of the API contract.
    """
    stops = [
        {"stop_key": s["stop_key"], "name": s.get("name") or "", "seq": int(s["seq"])} for s in route["stops"]
    ]
    seq_of: dict[str, list[int]] = {}
    for s in stops:
        seq_of.setdefault(s["stop_key"], []).append(s["seq"])
    n = len(stops)
    by_tr: dict[int, list[Mapping[str, Any]]] = {}
    for p in passages:
        by_tr.setdefault(int(p["tr_id"]), []).append(p)
    fc_by_tr: dict[int, list[Mapping[str, Any]]] = {}
    for f in forecasts:
        fc_by_tr.setdefault(int(f["tr_id"]), []).append(f)
    trips = []
    for tr_id in route.get("tr_ids") or []:
        plan = plans.get(int(tr_id))
        if plan is None:
            continue
        seqs = plan_seqs(plan.keys, seq_of, n)
        planned = [
            {"t": _iso(float(plan.tb[i])), "seq": seqs[i]}
            for i in range(len(plan))
            if seqs[i] is not None and lo <= plan.tb[i] <= hi
        ]
        actual = []
        for p in by_tr.get(int(tr_id), []):
            i = plan.index.get(int(p["stop_id"]))
            t = _ts(p.get("pass_time"))
            if i is None or seqs[i] is None or t is None:
                continue
            actual.append({"t": _iso(t), "seq": seqs[i]})
        actual.sort(key=lambda x: x["t"] or "")
        actual = monotonic(actual, n)
        nxt = plan.index.get(int((positions or {}).get(int(tr_id)) or 0))
        forecast = _forecast_line(
            plan,
            seqs,
            latest_tick(fc_by_tr.get(int(tr_id), [])),
            (current or {}).get(int(tr_id)),
            now,
            hi,
            nxt,
        )
        position = None
        if now is not None and nxt is not None and seqs[nxt] is not None:
            delay = (current or {}).get(int(tr_id))
            position = {"t": _iso(now), "seq": max(seqs[nxt] - 0.5, 0.0), "delay_s": delay}
        if planned or actual or forecast or position:
            trip = {"tr_id": int(tr_id), "planned": planned, "actual": actual, "forecast": forecast}
            trips.append({**trip, "now": position})
    return {"route_id": route["route_id"], "stops": stops, "trips": trips, "stream_time": _iso(now)}


def latest_tick(forecasts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The forecasts of one vehicle updated on its latest tick (the 10–15 min window). The ones updated
    earlier are frozen: their stops left the window, and their values are older than the current deviation.
    Without update times all of them are kept."""
    stamps = [_ts(f.get("updated_at") or f.get("issued_at")) for f in forecasts]
    known = [t for t in stamps if t is not None]
    if not known:
        return list(forecasts)
    last = max(known)
    return [f for f, t in zip(forecasts, stamps, strict=True) if t is None or t >= last]


def _forecast_line(
    plan: VehiclePlan,
    seqs: Sequence[int | None],
    forecasts: Sequence[Mapping[str, Any]],
    current: float | None,
    now: float | None,
    hi: float,
    start_i: int | None = None,
) -> list[dict[str, Any]]:
    """Expected arrivals at the stops ahead (see the module docstring); empty without a forecast.

    ``start_i`` is the plan position of the vehicle's next stop: the line starts there, and the forecasts of
    the stops behind it (open only because the detector missed the pass) are ignored."""
    if now is None:
        return []
    anchors: list[tuple[float, float, float | None, float | None]] = []  # plan time, delay, p10, p90
    for f in forecasts:
        i = plan.index.get(int(f.get("target_stop_id") or 0))
        pred = f.get("pred_delay_s")
        if i is None or pred is None or (start_i is not None and i < start_i):
            continue
        anchors.append((float(plan.tb[i]), float(pred), f.get("p10"), f.get("p90")))
    if not anchors:
        return []
    anchors.sort()
    d0 = float(current) if current is not None and math.isfinite(current) else anchors[0][1]
    # on the plan time axis the vehicle is where the plan was ``d0`` seconds ago
    origin = now - d0
    anchors = [a for a in anchors if a[0] >= origin - 60]
    if not anchors:
        return []
    start = (origin, d0, None, None)
    points = []
    for i in range(start_i or 0, len(plan)):
        tb = float(plan.tb[i])
        if seqs[i] is None or tb > hi or (start_i is None and tb <= origin - 120):
            continue
        delay, p10, p90 = interpolate(start, anchors, tb)
        t = tb + delay
        if t < now:
            continue
        points.append(
            {
                "t": _iso(t),
                "seq": seqs[i],
                "p10": _iso(tb + p10) if p10 is not None else None,
                "p90": _iso(tb + p90) if p90 is not None else None,
            }
        )
    return points


def interpolate(
    start: tuple[float, float, float | None, float | None],
    anchors: Sequence[tuple[float, float, float | None, float | None]],
    tb: float,
) -> tuple[float, float | None, float | None]:
    """Delay (and its P10/P90) at plan time ``tb``: linear from the current deviation to the first forecast,
    between forecasts, the last one held after it. The band widens from zero now to the forecast's."""
    prev = start
    for a in anchors:
        if tb <= a[0]:
            span = a[0] - prev[0]
            w = 1.0 if span <= 0 else min(max((tb - prev[0]) / span, 0.0), 1.0)
            delay = prev[1] + w * (a[1] - prev[1])
            lo = _mix(prev[2], prev[1], a[2], a[1], w)
            up = _mix(prev[3], prev[1], a[3], a[1], w)
            return delay, lo, up
        prev = a
    return prev[1], prev[2], prev[3]


def _mix(b0: float | None, d0: float, b1: float | None, d1: float, w: float) -> float | None:
    """Band bound between two points; a point without a band is its own delay."""
    if b0 is None and b1 is None:
        return None
    x0 = d0 if b0 is None else float(b0)
    x1 = d1 if b1 is None else float(b1)
    return x0 + w * (x1 - x0)
