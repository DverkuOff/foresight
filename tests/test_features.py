"""Признаки прогнозной точки: честность (нет данных после T) и базовые значения на синтетике."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.data import dataset_dir, load_points, load_schedule, load_traffic
from shared.features import FEATURE_NAMES, FeatureContext, build_features
from shared.stops import detect_all

LON0, LAT0 = 37.6, 55.75
M_PER_DEG = np.pi / 180.0 * 6_371_008.8
T0 = pd.Timestamp("2026-01-06 08:00:00")
SPEED = 500.0 / 60.0  # м/с: 500 м между остановками за минуту плана
LAYOVER = 900.0

HAS_DATA = (dataset_dir() / "test" / "traffic.csv").exists()


def to_lonlat(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = LON0 + np.asarray(x, dtype=float) / (np.cos(np.deg2rad(LAT0)) * M_PER_DEG)
    return lon, np.full(len(lon), LAT0)


def route_plan() -> tuple[np.ndarray, np.ndarray]:
    """Прямой маршрут: 10 остановок через 500 м / 60 с, конечная с отстоем 900 с, ещё 6 остановок."""
    x = np.r_[500.0 * np.arange(10), 4500.0, 4500.0 + 500.0 * np.arange(1, 7)]
    plan = np.r_[60.0 * np.arange(10), 540.0 + LAYOVER, 540.0 + LAYOVER + 60.0 * np.arange(1, 7)]
    return x, plan


def vehicle_position(t: np.ndarray, late: float) -> np.ndarray:
    """ТС опаздывает на `late` c, ждёт на конечной до планового отправления и уходит по плану."""
    depart = 540.0 + LAYOVER
    before = np.minimum(SPEED * (t - late), 4500.0)
    after = 4500.0 + SPEED * (t - max(depart, 540.0 + late))
    x = np.where(t < max(depart, 540.0 + late), before, after)
    return np.clip(x, 0.0, None)


def make_vehicle(tr_id: int, shift: float, late: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, plan = route_plan()
    lon, lat = to_lonlat(x)
    schedule = pd.DataFrame(
        {
            "tt_action_item_id": tr_id * 1000 + np.arange(len(x), dtype=np.int64),
            "tr_id": np.int64(tr_id),
            "time_begin": T0 + pd.to_timedelta(plan + shift, unit="s"),
            "time_fact_begin": T0 + pd.to_timedelta(plan + shift + late, unit="s"),
            "manual_fill": False,
            "stop_lon": lon,
            "stop_lat": lat,
        }
    )
    t = np.arange(-300.0, 2100.0, 10.0)
    pos = vehicle_position(t, late)
    speed = np.r_[np.abs(np.diff(pos)) / 10.0 * 3.6, 0.0]
    tlon, tlat = to_lonlat(pos)
    traffic = pd.DataFrame(
        {
            "tr_id": np.int64(tr_id),
            "event_time": T0 + pd.to_timedelta(t + shift, unit="s"),
            "lon": tlon,
            "lat": tlat,
            "speed": speed,
            "heading": 90.0,
        }
    )
    return traffic, schedule


def synthetic_world() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [make_vehicle(1, 0.0, 90.0), make_vehicle(2, -600.0, 150.0)]
    traffic = pd.concat([p[0] for p in parts], ignore_index=True)
    schedule = pd.concat([p[1] for p in parts], ignore_index=True)
    return traffic, schedule


def make_points(schedule: pd.DataFrame, tr_id: int, t_s: list[float], stop_idx: list[int]) -> pd.DataFrame:
    st = schedule[schedule["tr_id"] == tr_id].reset_index(drop=True)
    return pd.DataFrame(
        {
            "tr_id": np.int64(tr_id),
            "T": T0 + pd.to_timedelta(np.asarray(t_s, dtype=float), unit="s"),
            "target_stop_id": st["tt_action_item_id"].to_numpy()[stop_idx],
            "target_time_begin": st["time_begin"].to_numpy()[stop_idx],
            "cur_dev_s": 60.0,
        }
    )


def perturb_after(traffic: pd.DataFrame, t: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Испортить всю телеметрию после t (координаты, скорость) и добавить лишние точки после t."""
    rng = np.random.default_rng(seed)
    out = traffic.copy()
    late = out["event_time"] > t
    out.loc[late, "lon"] += rng.normal(0, 0.01, late.sum())
    out.loc[late, "lat"] += rng.normal(0, 0.01, late.sum())
    out.loc[late, "speed"] = rng.uniform(0, 80, late.sum())
    extra = out[late].copy()
    extra["event_time"] += pd.Timedelta(seconds=3)
    return pd.concat([out, extra], ignore_index=True).sort_values(["tr_id", "event_time"], kind="stable")


def assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-6)


# --- честность ------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_s", [250.0, 700.0, 1000.0, 1500.0, 1800.0])
def test_features_ignore_data_after_t_synthetic(t_s: float) -> None:
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [t_s], [12])
    full = build_features(points, FeatureContext(traffic, schedule))
    t = T0 + pd.Timedelta(seconds=t_s)
    # 1) телеметрия только до T, детектор заново
    cut = FeatureContext(traffic[traffic["event_time"] <= t], schedule)
    assert_same(full, build_features(points, cut))
    # 2) телеметрия после T испорчена, детектор заново
    bad = FeatureContext(perturb_after(traffic, t), schedule)
    assert_same(full, build_features(points, bad))


def test_features_ignore_passages_after_t_and_facts() -> None:
    traffic, schedule = synthetic_world()
    passages = detect_all(traffic, schedule)
    t_s = 700.0
    t = T0 + pd.Timedelta(seconds=t_s)
    points = make_points(schedule, 1, [t_s], [12])
    full = build_features(points, FeatureContext(traffic, schedule, passages))
    rng = np.random.default_rng(1)
    bad = passages.copy()
    future = ~(bad["confirmed_at"] <= t)
    bad.loc[future, "pass_time"] = t + pd.to_timedelta(rng.uniform(1, 900, future.sum()), unit="s")
    bad.loc[future, "confirmed_at"] = t + pd.to_timedelta(rng.uniform(1, 900, future.sum()), unit="s")
    bad_schedule = schedule.copy()
    bad_schedule["time_fact_begin"] = bad_schedule["time_begin"] + pd.Timedelta(seconds=999)
    bad_schedule["manual_fill"] = True
    ctx = FeatureContext(perturb_after(traffic, t), bad_schedule, bad)
    assert_same(full, build_features(points, ctx))
    # без проходов после T — то же самое
    known = passages[passages["confirmed_at"] <= t]
    ctx = FeatureContext(traffic[traffic["event_time"] <= t], schedule, known)
    assert_same(full, build_features(points, ctx))


@pytest.mark.skipif(not HAS_DATA, reason="нет датасета")
def test_features_ignore_data_after_t_real_data() -> None:
    points = load_points("test")
    vehicles = sorted(points["tr_id"].unique())[:4]
    traffic = load_traffic("test")
    traffic = traffic[traffic["tr_id"].isin(vehicles)]
    schedule = load_schedule("test")
    schedule = schedule[schedule["tr_id"].isin(vehicles)]
    pts = points[points["tr_id"].isin(vehicles)].iloc[::7]
    full = build_features(pts, FeatureContext(traffic, schedule))
    assert full["pos_delay"].notna().mean() > 0.5
    for i, (idx, row) in enumerate(pts.iterrows()):
        t = row["T"]
        sched_plan = schedule.drop(columns=["time_fact_begin", "manual_fill"])
        ctx = FeatureContext(perturb_after(traffic, t, seed=i), sched_plan)
        assert_same(full.loc[[idx]], build_features(pts.loc[[idx]], ctx))


# --- значения на синтетике ------------------------------------------------------------------------


def test_basic_values_on_synthetic_route() -> None:
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [300.0, 1000.0], [12, 12])
    f = build_features(points, FeatureContext(traffic, schedule))
    assert list(f.columns) == FEATURE_NAMES
    moving, waiting = f.iloc[0], f.iloc[1]
    # в пути: опоздание 90 с по позиции и по детектору, впереди отстой 900 с, который его поглотит
    assert moving["pos_delay"] == pytest.approx(90.0, abs=5.0)
    assert moving["dev_1"] == pytest.approx(90.0, abs=5.0)
    assert moving["pos_on_layover"] == 0.0
    assert moving["has_layover"] == 1.0
    assert moving["layover_ahead"] == pytest.approx(LAYOVER)
    assert moving["phys"] == 0.0
    assert moving["spd_60"] == pytest.approx(SPEED * 3.6, rel=0.01)
    assert moving["stopfrac_60"] == 0.0
    assert moving["n_to_target"] > 0
    assert moving["lead_s"] == pytest.approx(540.0 + LAYOVER + 120.0 - 300.0)
    # на конечной в окне отстоя: не опаздывает, стоит с момента прибытия (540 + 90)
    assert waiting["pos_on_layover"] == 1.0
    assert waiting["pos_delay"] == pytest.approx(0.0, abs=1e-6)
    assert waiting["stop_dur"] == pytest.approx(1000.0 - 630.0, abs=15.0)
    assert waiting["stopfrac_300"] == 1.0
    assert waiting["has_layover"] == 1.0


def test_network_features_use_other_vehicles() -> None:
    traffic, schedule = synthetic_world()
    # ТС 2 идёт на 10 мин раньше по плану с опозданием 150 с: к T = 400 оно уже прошло цель ТС 1
    points = make_points(schedule, 1, [400.0], [5])
    f = build_features(points, FeatureContext(traffic, schedule)).iloc[0]
    assert f["tgt_loc_n"] >= 1
    assert f["tgt_loc_dev"] == pytest.approx(150.0, abs=5.0)
    assert np.isfinite(f["net_dev_30m"])


def test_unknown_vehicle_and_stop() -> None:
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [300.0], [3])
    points.loc[:, "tr_id"] = 999
    f = build_features(points, FeatureContext(traffic, schedule)).iloc[0]
    assert f["cur_best"] == 60.0
    assert np.isnan(f["pos_delay"])
    points = make_points(schedule, 1, [300.0], [3])
    points.loc[:, "target_stop_id"] = -1
    f = build_features(points, FeatureContext(traffic, schedule)).iloc[0]
    assert f["pos_delay"] == pytest.approx(90.0, abs=5.0)
    assert np.isnan(f["n_to_target"])
