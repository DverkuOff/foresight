"""Последовательности телеметрии (``shared/sequences.py``): честность (нет данных после T) и значения."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_features import (
    HAS_DATA,
    LAYOVER,
    SPEED,
    T0,
    make_points,
    perturb_after,
    synthetic_world,
)

from shared.data import load_points, load_schedule, load_traffic
from shared.features import FeatureContext
from shared.sequences import (
    N_CHANNELS,
    SEQ_CHANNELS,
    SEQ_LEN,
    SEQ_STEP_S,
    SEQ_WINDOW_S,
    SequenceBuilder,
    build_sequences,
    sequence_version,
    step_times,
)
from shared.stops import detect_all

CH = {name: i for i, name in enumerate(SEQ_CHANNELS)}


def test_grid_and_shape() -> None:
    assert SEQ_LEN * SEQ_STEP_S == SEQ_WINDOW_S == 1200.0
    tk = step_times(1000.0)
    assert len(tk) == SEQ_LEN and tk[-1] == 1000.0 and tk[0] == 1000.0 - (SEQ_LEN - 1) * SEQ_STEP_S
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [300.0, 1000.0], [12, 12])
    seqs = build_sequences(points, FeatureContext(traffic, schedule))
    assert seqs.shape == (2, SEQ_LEN, N_CHANNELS) and seqs.dtype == np.float32
    assert np.isfinite(seqs).all()
    assert len(sequence_version()) == 10


# --- честность ------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_s", [250.0, 700.0, 1000.0, 1500.0, 1800.0])
def test_sequences_ignore_data_after_t_synthetic(t_s: float) -> None:
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [t_s], [12])
    full = build_sequences(points, FeatureContext(traffic, schedule))
    t = T0 + pd.Timedelta(seconds=t_s)
    # 1) телеметрия только до T, детектор заново
    cut = FeatureContext(traffic[traffic["event_time"] <= t], schedule)
    np.testing.assert_array_equal(full, build_sequences(points, cut))
    # 2) телеметрия после T испорчена и дополнена лишними точками, детектор заново
    bad = FeatureContext(perturb_after(traffic, t), schedule)
    np.testing.assert_array_equal(full, build_sequences(points, bad))


def test_sequences_ignore_passages_after_t_and_facts() -> None:
    traffic, schedule = synthetic_world()
    passages = detect_all(traffic, schedule)
    t_s = 700.0
    t = T0 + pd.Timedelta(seconds=t_s)
    points = make_points(schedule, 1, [t_s], [12])
    full = build_sequences(points, FeatureContext(traffic, schedule, passages))
    rng = np.random.default_rng(1)
    bad = passages.copy()
    future = ~(bad["confirmed_at"] <= t)
    assert future.any()
    bad.loc[future, "pass_time"] = t + pd.to_timedelta(rng.uniform(1, 900, future.sum()), unit="s")
    bad.loc[future, "confirmed_at"] = t + pd.to_timedelta(rng.uniform(1, 900, future.sum()), unit="s")
    bad_schedule = schedule.copy()
    bad_schedule["time_fact_begin"] = bad_schedule["time_begin"] + pd.Timedelta(seconds=999)
    bad_schedule["manual_fill"] = True
    ctx = FeatureContext(perturb_after(traffic, t), bad_schedule, bad)
    np.testing.assert_array_equal(full, build_sequences(points, ctx))
    # без проходов после T — то же самое
    known = passages[passages["confirmed_at"] <= t]
    ctx = FeatureContext(traffic[traffic["event_time"] <= t], schedule, known)
    np.testing.assert_array_equal(full, build_sequences(points, ctx))


@pytest.mark.skipif(not HAS_DATA, reason="нет датасета")
def test_sequences_ignore_data_after_t_real_data() -> None:
    points = load_points("test")
    vehicles = sorted(points["tr_id"].unique())[:4]
    traffic = load_traffic("test")
    traffic = traffic[traffic["tr_id"].isin(vehicles)]
    schedule = load_schedule("test")
    schedule = schedule[schedule["tr_id"].isin(vehicles)]
    pts = points[points["tr_id"].isin(vehicles)].iloc[::7]
    full = build_sequences(pts, FeatureContext(traffic, schedule))
    assert full[:, :, CH["obs"]].mean() > 0.5
    assert full[:, :, CH["has_dev"]].mean() > 0.5
    sched_plan = schedule.drop(columns=["time_fact_begin", "manual_fill"])
    for i, (pos, (_, row)) in enumerate(zip(range(len(pts)), pts.iterrows(), strict=True)):
        t = row["T"]
        ctx = FeatureContext(perturb_after(traffic, t, seed=i), sched_plan)
        np.testing.assert_array_equal(full[pos], build_sequences(pts.iloc[[pos]], ctx)[0])


# --- значения на синтетике ------------------------------------------------------------------------


def test_values_on_synthetic_route() -> None:
    traffic, schedule = synthetic_world()
    points = make_points(schedule, 1, [300.0, 1000.0], [12, 12])
    seqs = build_sequences(points, FeatureContext(traffic, schedule))
    moving, waiting = seqs[0], seqs[1]
    last = moving[-1]
    # в пути: GPS каждые 10 с, скорость постоянная, опоздание 90 с по детектору
    assert last[CH["obs"]] == 1.0
    assert last[CH["speed"]] == pytest.approx(SPEED * 3.6 / 50.0, rel=0.01)
    assert last[CH["stopped"]] == 0.0
    assert last[CH["move"]] * 200.0 == pytest.approx(SPEED * SEQ_STEP_S, rel=0.4)
    assert last[CH["has_dev"]] == 1.0
    assert last[CH["dev"]] * 300.0 == pytest.approx(90.0, abs=5.0)
    assert last[CH["has_next"]] == 1.0
    assert last[CH["next_layover"]] == 0.0
    # начало окна (T − 20 мин) раньше трека: пропуск GPS помечен маской
    assert moving[0, CH["obs"]] == 0.0 and moving[0, CH["speed"]] == 0.0
    # на конечной в окне отстоя (прибыло в 630 с): стоит; прибытие детектор ещё не подтвердил (ТС в радиусе),
    # следующая остановка — отправление после отстоя, план которого впереди
    w = waiting[-1]
    assert w[CH["stopped"]] == 1.0 and w[CH["speed"]] == 0.0 and w[CH["move"]] == 0.0
    assert w[CH["next_layover"]] == 1.0
    assert w[CH["next_late"]] * 300.0 == pytest.approx(1000.0 - (540.0 + LAYOVER), abs=1e-3)
    assert w[CH["next_dist"]] < 0.05
    # в пути к конечной (T = 300 с) следующая — обычная остановка,
    # план которой не прошёл больше чем на опоздание
    assert last[CH["next_late"]] * 300.0 < 90.0 + 1e-3


def test_unknown_vehicle_and_duplicates() -> None:
    traffic, schedule = synthetic_world()
    ctx = FeatureContext(traffic, schedule)
    points = make_points(schedule, 1, [700.0, 700.0, 700.0], [12, 13, 14])
    seqs = SequenceBuilder(ctx).build(points)
    np.testing.assert_array_equal(seqs[0], seqs[1])
    np.testing.assert_array_equal(seqs[0], seqs[2])
    points.loc[:, "tr_id"] = 999
    assert not build_sequences(points, ctx).any()
    assert build_sequences(points.iloc[:0], ctx).shape == (0, SEQ_LEN, N_CHANNELS)
