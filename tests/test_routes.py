"""Routes derived from the plan schedule (``shared/routes.py``, API contract §1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.routes import (
    ROUTE_COLORS,
    SEGMENTS_FORMAT,
    assign_seq,
    build_routes,
    build_segments,
    choose_pass,
    douglas_peucker,
    load_segments,
    path_length_m,
    proximity_passes,
    route_directions,
    route_sequence,
    segment_key,
    segment_passes,
    split_trips,
    stop_key,
    track_offset_m,
)

DATASET = Path(__file__).resolve().parent.parent / "dataset"

# a line of 5 stops, 1 km apart; the route goes there and back with a layover at both ends
STOPS = [(37.60 + 0.016 * i, 55.75, f"Остановка {chr(65 + i)}") for i in range(5)]


def _trips(tr_id: int, start: str, n_trips: int, first_id: int) -> list[dict]:
    """Plan of one vehicle: alternating trips A→E and E→A, 2 min between stops, 10 min layover."""
    rows = []
    t = pd.Timestamp(start)
    item = first_id
    for k in range(n_trips):
        order = range(5) if k % 2 == 0 else range(4, -1, -1)
        for i in order:
            lon, lat, name = STOPS[i]
            rows.append(
                {
                    "tr_id": tr_id,
                    "tt_action_item_id": item,
                    "time_begin": t,
                    "stop_lon": lon,
                    "stop_lat": lat,
                    "building_address": name,
                }
            )
            item += 1
            t += pd.Timedelta(minutes=2)
        t += pd.Timedelta(minutes=10)
    return rows


def _schedule() -> pd.DataFrame:
    rows = _trips(300, "2026-01-06 07:00", 4, 1000) + _trips(200, "2026-01-06 07:05", 3, 2000)
    # another route: three stops elsewhere
    other = [(37.40 + 0.01 * i, 55.80, f"Улица {i}") for i in range(3)]
    t = pd.Timestamp("2026-01-06 08:00")
    for j, (lon, lat, name) in enumerate(other * 2):
        rows.append(
            {
                "tr_id": 100,
                "tt_action_item_id": 3000 + j,
                "time_begin": t + pd.Timedelta(minutes=3 * j),
                "stop_lon": lon,
                "stop_lat": lat,
                "building_address": name,
            }
        )
    return pd.DataFrame(rows)


def test_stop_key_rounds_to_about_a_metre() -> None:
    assert stop_key(37.430707051, 55.8040083) == "37.43071,55.80401"
    assert stop_key(37.4307061, 55.80401) == stop_key(37.4307149, 55.8040149) == "37.43071,55.80401"
    assert stop_key(-0.5, 0.0) == "-0.50000,0.00000"


def test_trips_split_on_layovers() -> None:
    plan = np.array([0, 120, 240, 900, 1020, 1140, 1260], dtype=float)
    assert [(s.start, s.stop) for s in split_trips(plan)] == [(0, 3), (3, 7)]
    assert split_trips(np.array([])) == []


def test_route_sequence_glues_the_back_trip() -> None:
    keys = ["a", "b", "c", "d", "d", "c", "b", "a", "a", "b", "c", "d"]
    plan = np.array([0, 60, 120, 180, 900, 960, 1020, 1080, 1800, 1860, 1920, 1980], dtype=float)
    seq = route_sequence(keys, plan)
    assert seq == ["a", "b", "c", "d", "c", "b", "a"]  # the frequent trip a→d, then the back trip d→a
    # the second visit of a place is the next one along the route
    assert assign_seq(keys, plan, seq).tolist() == [0, 1, 2, 3, 3, 4, 5, 6, 0, 1, 2, 3]
    # short trips (service runs) do not define the sequence
    short = route_sequence(["a", "b", "c", "d", "x", "y"], np.array([0, 60, 120, 180, 900, 960.0]))
    assert short == list("abcd")


def test_vehicles_with_the_same_stops_share_a_route() -> None:
    routes = build_routes(_schedule(), {})
    assert [r.route_id for r in routes.routes] == ["R1", "R2"]
    r1, r2 = routes.routes
    # numbered by the smallest tr_id of the group; 200 and 300 run the same stops in another order
    assert r1.tr_ids == (100,) and r2.tr_ids == (200, 300)
    assert routes.by_tr == {100: "R1", 200: "R2", 300: "R2"}
    assert routes.of(300) is r2 and routes.of(999) is None and routes.route("R9") is None
    # the sequence of R2: the most frequent trip of tr 200 (A→E, twice) and the back trip E→A
    assert [s.name for s in r2.stops] == [f"Остановка {c}" for c in "ABCDEDCBA"]
    assert [s.seq for s in r2.stops] == list(range(9))
    assert r2.name == "Маршрут R2: Остановка A — Остановка E"  # a loop: the far terminal is the other end
    # each direction is its own line (no back-and-forth on one polyline); `line` is the forward one
    forward, back = r2.directions
    assert [s.name for s in forward.stops] == [f"Остановка {c}" for c in "ABCDE"]
    assert [s.name for s in back.stops] == [f"Остановка {c}" for c in "EDCBA"]
    assert [s.seq for s in back.stops] == [4, 5, 6, 7, 8]  # the seq of the stringline axis
    assert forward.name == "Остановка A → Остановка E" and (forward.direction, back.direction) == (0, 1)
    assert r2.line == [[s.lon, s.lat] for s in forward.stops]  # no geometry: straight segments
    assert [list(p) for p in back.line] == [[s.lon, s.lat] for s in back.stops]
    # R1: one trip U0 U1 U2 U0 U1 U2 without a layover, not circular, no back trip: one direction
    assert forward.gps_segments == 0 and routes.geometry_source == {
        "segments": 4 + 4 + 5,
        "gps_segments": 0,
        "straight_segments": 13,
    }
    assert r1.name == "Маршрут R1: Улица 0 — Улица 2" and len(r1.directions) == 1
    assert (r1.color, r2.color) == ROUTE_COLORS[:2]
    out = r2.to_dict()
    assert set(out) == {"route_id", "name", "tr_ids", "stops", "line", "directions", "color"}
    assert set(out["stops"][0]) == {"stop_key", "name", "lat", "lon", "seq"}
    assert set(out["directions"][1]) == {"direction", "name", "stops", "line", "segments", "gps_segments"}


def test_a_circular_trip_is_one_direction() -> None:
    keys = ["a", "b", "c", "d", "a", "a", "b", "c", "d", "a"]
    plan = np.array([0, 60, 120, 180, 240, 900, 960, 1020, 1080, 1140], dtype=float)
    assert route_directions(keys, plan) == [["a", "b", "c", "d", "a"]]
    assert route_directions([], np.array([])) == []


def _k(x: float, y: float) -> str:
    return stop_key(37.6 + 0.01 * x, 55.75 + 0.01 * y)  # 0.01° ≈ 627 m east, 1113 m north


def test_directions_of_real_trip_shapes() -> None:
    # there and back without a layover at the far end (one planned trip): the way back runs on the other side
    # of the street, ~55 m away — split at the far terminal into two directions
    there = [_k(i, 0) for i in range(6)]
    back = [_k(i, 0.05) for i in range(4, 0, -1)]
    loop = [*there, *back, there[0]]
    plan = np.arange(len(loop)) * 120.0
    assert route_directions(loop, plan) == [there, [there[-1], *back, there[0]]]
    assert route_sequence(loop, plan) == loop
    # a ring: the second half does not pass the stops of the first — one direction
    ring = [_k(0, 0), _k(1, 0), _k(2, 0), _k(2, 1), _k(2, 2), _k(1, 2), _k(0, 2), _k(0, 1), _k(0, 0)]
    assert route_directions(ring, np.arange(len(ring)) * 120.0) == [ring]
    # the back trip starts at another platform of the terminal, ~60 m from where the forward trip ends
    fwd = [_k(i, 0) for i in range(5)]
    rev = [_k(4.1, 0), *[_k(i, 0) for i in range(3, -1, -1)]]
    keys = fwd + rev + fwd
    plan = np.r_[np.arange(5) * 120.0, 1200 + np.arange(5) * 120.0, 2400 + np.arange(5) * 120.0]
    assert route_directions(keys, plan) == [fwd, rev]
    assert route_sequence(keys, plan) == fwd + rev  # no shared terminal: both ends kept
    t0 = pd.Timestamp("2026-01-06 07:00")
    rows = [
        {
            "tr_id": 1,
            "tt_action_item_id": i,
            "time_begin": t0 + pd.Timedelta(seconds=float(s)),
            "stop_lon": float(key.split(",")[0]),
            "stop_lat": float(key.split(",")[1]),
        }
        for i, (key, s) in enumerate(zip(keys, plan, strict=True))
    ]
    route = build_routes(pd.DataFrame(rows), {}).routes[0]
    forward, backward = route.directions
    assert [s.stop_key for s in forward.stops] == fwd and [s.stop_key for s in backward.stops] == rev
    assert [s.seq for s in backward.stops] == [5, 6, 7, 8, 9] and len(route.stops) == 10


def _curve(n: int = 41) -> list[list[float]]:
    """A quarter circle of radius ~500 m (a road bending between two stops)."""
    ang = np.linspace(0.0, math.pi / 2, n)
    k = 500.0 / 111_320.0
    return [
        [37.6 + k * math.sin(a) / math.cos(math.radians(55.75)), 55.75 + k * (1 - math.cos(a))] for a in ang
    ]


def test_douglas_peucker_keeps_the_shape_within_the_tolerance() -> None:
    curve = _curve()
    simple = douglas_peucker(curve, 5.0)
    assert simple[0] == curve[0] and simple[-1] == curve[-1] and 3 <= len(simple) < len(curve)
    # every dropped point is within 5 m of the simplified line; the length is kept
    assert path_length_m(simple) == pytest.approx(path_length_m(curve), rel=0.01)
    straight = [[37.6 + 0.001 * i, 55.75] for i in range(10)]
    assert douglas_peucker(straight, 5.0) == [straight[0], straight[-1]]
    assert douglas_peucker(straight[:2]) == straight[:2]


def _gps_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two stops at the ends of the curve (planned 4 min apart, passed 400 s apart); two trips an hour apart:
    one with a clean track, one with a GPS gap."""
    curve = _curve()
    a, b = curve[0], curve[-1]
    t0 = pd.Timestamp("2026-01-06 07:00")
    rows, plan, passes = [], [], []
    for trip, (start, gap) in enumerate([(0, False), (3600, True)]):
        for i, (lon, lat) in enumerate(curve):
            if gap and 10 <= i < 30:
                continue  # 200 s without GPS: not a "good" pass
            at = t0 + pd.Timedelta(seconds=start + 10 * i)
            rows.append({"tr_id": 7, "event_time": at, "lon": lon, "lat": lat})
        for j, (lon, lat) in enumerate((a, b)):
            item = 100 + 2 * trip + j
            planned = t0 + pd.Timedelta(seconds=start + 240 * j)
            plan.append(
                {
                    "tr_id": 7,
                    "tt_action_item_id": item,
                    "time_begin": planned,
                    "stop_lon": lon,
                    "stop_lat": lat,
                }
            )
            passed = t0 + pd.Timedelta(seconds=start + 400 * j)
            passes.append({"tr_id": 7, "tt_action_item_id": item, "pass_time": passed})
    return pd.DataFrame(rows), pd.DataFrame(plan), pd.DataFrame(passes)


def test_segment_geometry_follows_the_gps_track() -> None:
    traffic, plan, passages = _gps_case()
    found = segment_passes(traffic, plan, passages)
    key = segment_key(stop_key(*_curve()[0]), stop_key(*_curve()[-1]))
    assert list(found) == [key] and len(found[key]) == 2
    clean, gappy = found[key]
    assert clean.good and not gappy.good and gappy.max_step_s == pytest.approx(210.0)
    assert choose_pass(found[key]) is clean  # the good pass wins
    geometry = build_segments(traffic, plan, passages)
    line = geometry[key]["line"]
    assert geometry[key]["passes"] == 2 and geometry[key]["good"] == 1
    assert 3 <= len(line) < len(_curve())  # simplified, but a curve, not a straight segment
    assert path_length_m(line) == pytest.approx(math.pi / 2 * 500.0, rel=0.02)
    # the route line uses it; without the geometry it is the straight segment
    sched = plan.assign(building_address=["A", "B"] * 2)
    with_gps = build_routes(sched, {key: line})
    assert with_gps.routes[0].directions[0].gps_segments == 1
    assert len(with_gps.routes[0].line) == len(line)
    assert len(build_routes(sched, {}).routes[0].line) == 2


def test_a_stop_the_vehicles_do_not_pass_is_bypassed_along_the_track() -> None:
    traffic, plan, passages = _gps_case()
    curve = _curve()
    # the first trip gets a planned stop 1 km off the road between the ends: the detector never matches it
    off = {"tr_id": 7, "tt_action_item_id": 99, "stop_lon": curve[20][0] + 0.02, "stop_lat": curve[20][1]}
    off["time_begin"] = pd.Timestamp("2026-01-06 07:02")
    plan = pd.concat([plan, pd.DataFrame([off])], ignore_index=True)
    a, b, x = stop_key(*curve[0]), stop_key(*curve[-1]), stop_key(off["stop_lon"], off["stop_lat"])
    found = segment_passes(traffic, plan, passages)
    assert segment_key(a, b) in found and segment_key(a, x) not in found  # a → b around the unvisited x
    geometry = {k: v["line"] for k, v in build_segments(traffic, plan, passages).items()}
    route = build_routes(plan[plan["tt_action_item_id"] < 102], geometry).routes[0]
    forward = route.directions[0]
    assert [s.stop_key for s in forward.stops] == [a, x, b]
    assert forward.gps_segments == 2  # both segments a → x → b are covered by the track a → b
    assert [list(p) for p in forward.line] == [list(p) for p in geometry[segment_key(a, b)]]


def test_the_track_between_stop_visits_when_the_detector_has_no_pass() -> None:
    traffic, plan, passages = _gps_case()
    none = passages.iloc[0:0]
    key = segment_key(stop_key(*_curve()[0]), stop_key(*_curve()[-1]))
    assert segment_passes(traffic, plan, none) == {}
    found = proximity_passes(traffic, plan)
    assert list(found) == [key] and len(found[key]) == 2  # both trips visit a, then b within the limit
    assert proximity_passes(traffic, plan, skip={key}) == {}
    geometry = build_segments(traffic, plan, none)
    assert geometry[key]["source"] == "proximity" and geometry[key]["good"] == 1
    assert path_length_m(geometry[key]["line"]) == pytest.approx(math.pi / 2 * 500.0, rel=0.02)
    # with the detector's passes the detector's path wins; without the fallback there is nothing
    assert build_segments(traffic, plan, passages)[key]["source"] == "detector"
    assert build_segments(traffic, plan, none, proximity=False) == {}
    # a plan that goes b → a on a street the vehicles drive only a → b: their path, reversed
    swapped = plan.assign(stop_lon=plan["stop_lon"].to_numpy()[[1, 0, 3, 2]])
    swapped = swapped.assign(stop_lat=plan["stop_lat"].to_numpy()[[1, 0, 3, 2]])
    back = segment_key(stop_key(*_curve()[-1]), stop_key(*_curve()[0]))
    reversed_passes = proximity_passes(traffic, swapped)[back]
    assert len(reversed_passes) == 2
    line = reversed_passes[0].line
    assert list(line[0]) == pytest.approx(_curve()[-1], abs=1e-6)
    assert list(line[-1]) == pytest.approx(_curve()[0], abs=1e-6)


def test_segments_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    assert load_segments(path) == {}
    payload = {"format": SEGMENTS_FORMAT, "segments": {"a|b": {"line": [[1, 2], [3, 4]], "passes": 1}}}
    path.write_text(json.dumps(payload))
    assert load_segments(path) == {"a|b": [[1.0, 2.0], [3.0, 4.0]]}
    path.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        load_segments(path)


def test_routes_do_not_depend_on_row_order() -> None:
    schedule = _schedule()
    shuffled = schedule.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert build_routes(schedule) == build_routes(shuffled)


def test_routes_read_only_the_plan() -> None:
    schedule = _schedule()
    with_fact = schedule.assign(time_fact_begin=schedule["time_begin"] + pd.Timedelta(minutes=9))
    assert build_routes(with_fact) == build_routes(schedule)


@pytest.mark.skipif(not (DATASET / "test" / "schedule.csv").is_file(), reason="dataset is not available")
def test_routes_of_the_test_split() -> None:
    from shared.data import load_schedule

    schedule = load_schedule("test")
    routes = build_routes(schedule)
    assert set(routes.by_tr) == set(schedule["tr_id"].unique())
    assert [r.route_id for r in routes.routes] == [f"R{i + 1}" for i in range(len(routes.routes))]
    for route in routes.routes:
        assert route.name.startswith(f"Маршрут {route.route_id}: ") and len(route.stops) >= 2
        assert all(s.name for s in route.stops)
        assert sorted(route.tr_ids) == list(route.tr_ids)
    assert build_routes(load_schedule("test")) == routes  # deterministic


def test_track_offset_measures_how_far_a_road_path_leaves_the_track() -> None:
    track = [[37.60, 55.75], [37.61, 55.75]]
    same = [[37.60, 55.75], [37.605, 55.75], [37.61, 55.75]]
    assert track_offset_m(same, track) == pytest.approx((0.0, 0.0), abs=0.5)
    # a detour through a courtyard ~55 m off the street in the middle
    mean_m, max_m = track_offset_m([[37.60, 55.75], [37.605, 55.7505], [37.61, 55.75]], track)
    assert 50.0 < max_m < 60.0 and 0.0 < mean_m < max_m
