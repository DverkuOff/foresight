"""Загрузчики датасета."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.data import clean_traffic, dataset_dir, load_points, load_schedule, load_traffic, parse_point

HAS_DATA = (dataset_dir() / "test" / "traffic.csv").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="нет датасета")


def test_dataset_dir_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MT_DATASET_DIR", str(tmp_path))
    assert dataset_dir() == tmp_path


def test_parse_point() -> None:
    lon, lat = parse_point(pd.Series(["POINT (37.43070705 55.8040083)", "POINT(1 -2.5)"]))
    np.testing.assert_allclose(lon, [37.43070705, 1.0])
    np.testing.assert_allclose(lat, [55.8040083, -2.5])


def test_clean_traffic_filters() -> None:
    raw = pd.DataFrame(
        {
            "tr_id": [1, 1, 1, 1, 1, 2],
            "event_time": [
                "2026-01-06 10:00:10.5",
                "2026-01-06 10:00:00",
                "2026-01-06 10:00:20",
                "2026-01-06 10:00:30",
                "2026-01-06 10:00:00",
                "2026-01-06 10:00:00.123456",
            ],
            "location_valid": ["True", "True", "False", "True", "True", True],
            "lon": [37.0, 37.1, 37.2, np.nan, 37.1, 37.5],
            "lat": [55.0, 55.1, 55.2, 55.3, 55.1, 55.5],
            "speed": [10.0, 200.0, 5.0, 5.0, 20.0, 0.0],
            "heading": [0.0] * 6,
        }
    )
    out = clean_traffic(raw)
    assert out["event_time"].dtype == "datetime64[ns]"
    assert out["tr_id"].dtype == np.int64
    # выброс скорости, невалидная точка и NaN координата отброшены; порядок по (tr_id, time)
    assert list(zip(out["tr_id"], out["event_time"].dt.second, strict=True)) == [(1, 0), (1, 10), (2, 0)]
    assert out.loc[0, "speed"] == 20.0


@needs_data
@pytest.mark.parametrize("split", ["train", "test", "validate"])
def test_loaders_types(split: str) -> None:
    tr = load_traffic(split)
    assert list(tr.columns) == ["tr_id", "event_time", "lon", "lat", "speed", "heading"]
    assert tr["event_time"].dtype == "datetime64[ns]" and tr["tr_id"].dtype == np.int64
    assert not tr[["lon", "lat"]].isna().any().any() and (tr["speed"] <= 120).all()
    assert not tr.duplicated(["tr_id", "event_time"]).any()
    assert tr.sort_values(["tr_id", "event_time"]).index.equals(tr.index)

    sc = load_schedule(split)
    assert sc["time_begin"].dtype == "datetime64[ns]" and sc["tt_action_item_id"].dtype == np.int64
    assert sc[["stop_lon", "stop_lat"]].notna().all().all()
    assert ("time_fact_begin" in sc.columns) == (split != "validate")

    pts = load_points(split)
    assert pts["T"].dtype == "datetime64[ns]" and pts["tr_id"].dtype == np.int64
    assert ("target_delay_s" in pts.columns) == (split != "validate")
