"""Формат сабмита и вспомогательные функции ML-пайплайна."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.submit import check_submission, make_submission, write_submission
from shared.data import dataset_dir, load_points

HAS_DATA = (dataset_dir() / "validate" / "points.csv").exists()


def fake_points(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": [f"{100 + i}_{1767670500 + 300 * i}" for i in range(n)]})


def test_submission_roundtrip(tmp_path) -> None:
    points = fake_points()
    sub = make_submission(points, np.array([1.5, -2.0, 0.0, 100.25, -372.0]))
    path = tmp_path / "submission.csv"
    write_submission(sub, points, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "sample_id;prediction"
    assert len(lines) == len(points) + 1
    assert lines[1] == "100_1767670500;1.500"
    assert all(line.count(";") == 1 for line in lines)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.assign(prediction=[1.0, np.nan, 2.0, 3.0, 4.0]),
        lambda s: s.assign(prediction=[1.0, np.inf, 2.0, 3.0, 4.0]),
        lambda s: s.iloc[:-1],
        lambda s: s.iloc[::-1].reset_index(drop=True),
        lambda s: pd.concat([s.iloc[:-1], s.iloc[:1]], ignore_index=True),
        lambda s: s.rename(columns={"prediction": "pred"}),
        lambda s: s.assign(prediction=["a", "1", "2", "3", "4"]),
    ],
)
def test_check_submission_rejects_bad(mutate) -> None:
    points = fake_points()
    sub = make_submission(points, np.arange(5, dtype=float))
    check_submission(sub, points)
    with pytest.raises(ValueError):
        check_submission(mutate(sub), points)


@pytest.mark.skipif(not HAS_DATA, reason="нет датасета")
def test_submission_covers_validate_points(tmp_path) -> None:
    points = load_points("validate")
    assert len(points) == 151
    sub = make_submission(points, np.zeros(len(points)))
    path = tmp_path / "s.csv"
    write_submission(sub, points, path)
    back = pd.read_csv(path, sep=";", dtype={"sample_id": str})
    sample = pd.read_csv(dataset_dir() / "sample_submission.csv", sep=";", dtype={"sample_id": str})
    assert list(back["sample_id"]) == list(sample["sample_id"])


def test_vehicle_families_and_leak_mask() -> None:
    from ml.dataset import leak_mask, vehicle_families

    t0 = pd.Timestamp("2026-01-06 08:00:00")
    plan = t0 + pd.to_timedelta(np.arange(0, 3600, 120), unit="s")
    lon = 37.6 + 0.001 * np.arange(len(plan))
    real = pd.DataFrame({"tr_id": 1, "time_begin": plan, "stop_lon": lon, "stop_lat": 55.75})
    copy = real.assign(tr_id=9, time_begin=plan + pd.Timedelta(seconds=600))
    fam = vehicle_families(pd.concat([real, copy], ignore_index=True), {1}).set_index("tr_id")
    assert fam.loc[9, "family"] == 1 and fam.loc[9, "shift_s"] == 600.0
    assert fam.loc[1, "family"] == 1 and fam.loc[1, "shift_s"] == 0.0

    train = pd.DataFrame(
        {
            "tr_id": [1, 9, 9, 9],
            "T": t0 + pd.to_timedelta([0, 600, 3000, 4200], unit="s"),
            "is_real": [True, False, False, False],
            "family": 1,
            "shift_s": [0.0, 600.0, 600.0, 600.0],
        }
    )
    eval_points = pd.DataFrame({"tr_id": [1], "T": [t0]})
    # эквивалентные моменты копии: 08:00 (совпадает с точкой оценки), 08:40, 09:00
    np.testing.assert_array_equal(leak_mask(train, [eval_points]), [False, True, False, False])
