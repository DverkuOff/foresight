"""Детектор прохождения остановок: синтетические треки и каузальность на реальных данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.data import dataset_dir, load_schedule, load_traffic
from shared.stops import RESULT_COLUMNS, detect_all, detect_passages

LON0, LAT0 = 37.6, 55.75
M_PER_DEG = np.pi / 180.0 * 6_371_008.8
T0 = pd.Timestamp("2026-01-06 08:00:00")


def to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = LON0 + np.asarray(x, dtype=float) / (np.cos(np.deg2rad(LAT0)) * M_PER_DEG)
    lat = LAT0 + np.asarray(y, dtype=float) / M_PER_DEG
    return lon, lat


def make_track(t_s, x, y) -> pd.DataFrame:
    lon, lat = to_lonlat(x, y)
    return pd.DataFrame(
        {"event_time": T0 + pd.to_timedelta(np.asarray(t_s, dtype=float), unit="s"), "lon": lon, "lat": lat}
    )


def make_stops(plan_s, x, y) -> pd.DataFrame:
    lon, lat = to_lonlat(x, y)
    n = len(plan_s)
    return pd.DataFrame(
        {
            "tt_action_item_id": np.arange(1000, 1000 + n, dtype=np.int64),
            "time_begin": T0 + pd.to_timedelta(np.asarray(plan_s, dtype=float), unit="s"),
            "stop_lon": lon,
            "stop_lat": lat,
        }
    )


def pass_seconds(res: pd.DataFrame) -> np.ndarray:
    return ((res["pass_time"] - T0).dt.total_seconds()).to_numpy()


def straight_track(speed=10.0, step=10.0, x0=-1000.0, x1=1000.0):
    t = np.arange(0.0, (x1 - x0) / speed + step, step)
    return t, x0 + speed * t, np.zeros_like(t)


def test_straight_pass() -> None:
    t, x, y = straight_track()
    stops = make_stops([60, 120, 180], [-480, 20, 530], [5, 5, 5])
    res = detect_passages(make_track(t, x, y), stops)
    assert list(res.columns) == RESULT_COLUMNS
    # ТС на 10 м/с проходит x за (x + 1000) / 10 секунд
    np.testing.assert_allclose(pass_seconds(res), [52, 102, 153], atol=0.5)
    np.testing.assert_allclose(res["dist_m"], 5, atol=0.5)
    assert (res["confirmed_at"] >= res["pass_time"]).all()
    # подтверждение — после удаления от остановки дальше радиуса визита
    assert ((res["confirmed_at"] - res["pass_time"]).dt.total_seconds() > 10).all()


def dwell_track(dwell_from: float, dwell_to: float):
    """Подъезд к x=0 со скоростью 10 м/с, стоянка с GPS-шумом, отъезд."""
    t_in = np.arange(0.0, dwell_from + 1, 10.0)
    x_in = -dwell_from * 10 + 10 * t_in
    t_dw = np.arange(dwell_from + 10, dwell_to, 10.0)
    jitter = np.where(np.arange(len(t_dw)) % 2 == 0, 1.5, -1.5)
    x_dw, y_dw = jitter, -jitter
    t_out = np.arange(dwell_to, dwell_to + 200, 10.0)
    x_out = 10 * (t_out - dwell_to)
    t = np.concatenate([t_in, t_dw, t_out])
    x = np.concatenate([x_in, x_dw, x_out])
    y = np.concatenate([np.zeros_like(t_in), y_dw, np.zeros_like(t_out)])
    return t, x, y


def test_long_dwell_mid_stop() -> None:
    t, x, y = dwell_track(300, 480)
    stops = make_stops([200, 320, 440], [-500, 0, 500], [0, 0, 0])
    res = detect_passages(make_track(t, x, y), stops)
    p = pass_seconds(res)
    assert abs(p[0] - 250) < 1
    assert 299 <= p[1] <= 481
    assert abs(p[2] - 530) < 1
    assert res["confirmed_at"].iloc[1] > T0 + pd.Timedelta(seconds=480)


def test_terminal_arrival_and_departure() -> None:
    # конечная: прибытие в 300 с, отстой до 900 с; план прибытия 5:00, отправления 15:00 (разрыв 10 мин)
    t, x, y = dwell_track(300, 900)
    stops = make_stops([240, 300, 900, 960], [-500, 0, 0, 500], [0, 0, 0, 0])
    res = detect_passages(make_track(t, x, y), stops)
    p = pass_seconds(res)
    assert abs(p[0] - 250) < 1
    assert abs(p[1] - 300) <= 10  # прибытие, а не середина отстоя
    assert abs(p[2] - 900) <= 10  # отправление
    assert abs(p[3] - 950) < 1


def test_circular_route_with_repeated_stops() -> None:
    # круг радиусом 1500 м, 10 м/с — круг за ~942 с; 3 круга, 4 остановки на круге.
    radius, speed, step = 1500.0, 10.0, 10.0
    period = 2 * np.pi * radius / speed
    t = np.arange(0.0, 3 * period + 60, step)
    ang = speed * t / radius
    x, y = radius * np.cos(ang), radius * np.sin(ang)
    stop_ang = np.array([0.3, 0.3 + np.pi / 2, 0.3 + np.pi, 0.3 + 3 * np.pi / 2])
    true_pass = np.concatenate([(stop_ang + 2 * np.pi * lap) * radius / speed for lap in range(3)])
    angs = np.tile(stop_ang, 3)
    # ТС опаздывает на 7 минут: окно [plan − 10, plan + 15] захватывает и прошлый круг
    plan = true_pass - 420
    stops = make_stops(plan, radius * np.cos(angs), radius * np.sin(angs))
    res = detect_passages(make_track(t, x, y), stops)
    np.testing.assert_allclose(pass_seconds(res), true_pass, atol=1.0)


def test_missed_stop_does_not_break_sequence() -> None:
    # трек длиннее окна пропущенной остановки, чтобы её пропуск был подтверждён
    t, x, y = straight_track(x1=15000)
    # вторая остановка в 400 м от маршрута
    stops = make_stops([60, 120, 180], [-480, 20, 530], [5, 400, 5])
    res = detect_passages(make_track(t, x, y), stops)
    p = pass_seconds(res)
    assert np.isnan(p[1]) and np.isnan(res["dist_m"].iloc[1])
    np.testing.assert_allclose(p[[0, 2]], [52, 153], atol=0.5)
    assert res["confirmed_at"].notna().all()


def test_gps_gap_interpolation() -> None:
    t, x, y = straight_track(x0=-3000, x1=3000)
    stops = make_stops([300], [20], [5])
    keep = (t < 250) | (t > 350)  # дыра 110 с < max_gap_s: интерполируем по отрезку
    res = detect_passages(make_track(t[keep], x[keep], y[keep]), stops)
    assert abs(pass_seconds(res)[0] - 302) < 1
    keep = (t < 100) | (t > 500)  # дыра 420 с > max_gap_s: момент прохода не восстанавливаем
    res = detect_passages(make_track(t[keep], x[keep], y[keep]), stops)
    assert res["pass_time"].isna().all()


def test_empty_and_invalid_inputs() -> None:
    stops = make_stops([60, 120], [0, 100], [0, 0])
    empty_track = make_track([], [], [])
    res = detect_passages(empty_track, stops)
    assert len(res) == 2 and res["pass_time"].isna().all() and res["confirmed_at"].isna().all()

    t, x, y = straight_track()
    track = make_track(t, x, y)
    track.loc[::3, "lon"] = np.nan  # невалидные точки отбрасываются
    track = pd.concat([track, track.iloc[:5]])  # дубликаты времени
    res = detect_passages(track, stops)
    assert res["pass_time"].notna().all()

    res = detect_passages(track, stops.iloc[:0])
    assert list(res.columns) == RESULT_COLUMNS and res.empty

    one_point = make_track([0], [0], [0])
    assert detect_passages(one_point, stops)["pass_time"].isna().all()

    bad_stop = stops.copy()
    bad_stop.loc[0, "stop_lon"] = np.nan
    res = detect_passages(track, bad_stop)
    assert res["pass_time"].isna().iloc[0] and res["pass_time"].notna().iloc[1]


def test_detect_all_handles_vehicle_without_gps() -> None:
    t, x, y = straight_track()
    traffic = make_track(t, x, y).assign(tr_id=np.int64(1))
    stops = make_stops([60, 120, 180], [-480, 20, 530], [5, 5, 5])
    schedule = pd.concat([stops.assign(tr_id=np.int64(1)), stops.assign(tr_id=np.int64(2))])
    res = detect_all(traffic, schedule)
    assert list(res.columns) == ["tr_id", *RESULT_COLUMNS]
    assert res.loc[res["tr_id"] == 1, "pass_time"].notna().all()
    assert res.loc[res["tr_id"] == 2, "pass_time"].isna().all()


def assert_causal(track: pd.DataFrame, stops: pd.DataFrame, cutoffs) -> int:
    """Прогон на точках ≤ T совпадает с полным для остановок с confirmed_at ≤ T.

    Возвращает число сравнённых остановок.
    """
    full = detect_passages(track, stops)
    checked = 0
    for cut in cutoffs:
        part = detect_passages(track[track["event_time"] <= cut], stops)
        known = (full["confirmed_at"] <= cut).to_numpy()
        a, b = full[known], part[known]
        pd.testing.assert_series_equal(
            a["pass_time"].reset_index(drop=True), b["pass_time"].reset_index(drop=True), check_names=False
        )
        pd.testing.assert_series_equal(
            a["confirmed_at"].reset_index(drop=True),
            b["confirmed_at"].reset_index(drop=True),
            check_names=False,
        )
        # остановки, ещё не подтверждённые к T, не должны получить подтверждение позже T
        assert not (part.loc[~known, "confirmed_at"] <= cut).any()
        checked += int(known.sum())
    return checked


def test_causality_synthetic() -> None:
    t, x, y = dwell_track(300, 900)
    track = make_track(t, x, y)
    stops = make_stops([240, 300, 900, 960], [-500, 0, 0, 500], [0, 0, 0, 0])
    cutoffs = [T0 + pd.Timedelta(seconds=s) for s in range(0, 1200, 7)]
    assert assert_causal(track, stops, cutoffs) > 0


def _polyline_track(waypoints, step: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Трек по точкам (t, x, y) с линейной интерполяцией и шагом `step` секунд."""
    wt, wx, wy = (np.asarray(v, dtype=float) for v in zip(*waypoints, strict=True))
    t = np.arange(wt[0], wt[-1] + step / 2, step)
    return t, np.interp(t, wt, wx), np.interp(t, wt, wy)


def test_causality_depart_terminal_turnaround_loop() -> None:
    # отстой на конечной (0,0), отправление, проход следующей (100,0) и разворот обратно мимо конечной
    t, x, y = _polyline_track(
        [
            (0, 0, 0),
            (890, 0, 0),
            (920, 100, 0),
            (940, 100, 100),
            (960, 0, 100),
            (980, -100, 100),
            (1300, -1700, 100),
        ]
    )
    stops = make_stops([600, 660, 720], [0, 100, -600], [0, 0, 100])
    cutoffs = [T0 + pd.Timedelta(seconds=s) for s in range(850, 1300)]
    assert assert_causal(make_track(t, x, y), stops, cutoffs) > 0


def test_causality_depart_terminal_gps_gap_over_next_stop() -> None:
    # отправление вовремя, провал GPS над следующей остановкой, опоздание к третьей
    t, x, y = _polyline_track([(0, 0, 0), (600, 0, 0), (620, 200, 0)])
    t2, x2, y2 = _polyline_track([(1500, 1000, 0), (1600, 1500, 0), (2000, 3500, 0)])
    track = make_track(np.r_[t, t2], np.r_[x, x2], np.r_[y, y2])
    stops = make_stops([600, 660, 720], [0, 500, 1500], [0, 0, 0])
    cutoffs = [T0 + pd.Timedelta(seconds=s) for s in range(1500, 2000)]
    assert assert_causal(track, stops, cutoffs) > 0


@pytest.mark.skipif(not (dataset_dir() / "test" / "traffic.csv").exists(), reason="нет датасета")
def test_causality_real_data() -> None:
    traffic = load_traffic("test")
    schedule = load_schedule("test")
    counts = traffic["tr_id"].value_counts()
    rng = np.random.default_rng(0)
    checked = 0
    for tr_id in counts.index[:4]:
        track = traffic[traffic["tr_id"] == tr_id]
        stops = schedule[schedule["tr_id"] == tr_id]
        t_min, t_max = track["event_time"].min(), track["event_time"].max()
        span = (t_max - t_min).total_seconds()
        cutoffs = [t_min + pd.Timedelta(seconds=float(s)) for s in np.sort(rng.uniform(0, span, 12))]
        # точно в моменты подтверждений и рядом с ними — самые чувствительные места
        full = detect_passages(track, stops)
        conf = full["confirmed_at"].dropna()
        for c in conf.sample(min(6, len(conf)), random_state=0):
            cutoffs += [c, c - pd.Timedelta(microseconds=1)]
        checked += assert_causal(track, stops, cutoffs)
    assert checked > 100
