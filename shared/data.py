"""Загрузка данных хакатона с нормализованными типами.

Время — datetime64[ns], идентификаторы — int64. Путь к датасету: `<repo>/dataset`,
переопределяется переменной окружения `MT_DATASET_DIR`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "test", "validate")
MAX_SPEED_KMH = 120.0

TRAFFIC_COLUMNS = ["tr_id", "event_time", "lon", "lat", "speed", "heading"]


def dataset_dir() -> Path:
    env = os.environ.get("MT_DATASET_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "dataset"


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")


def to_ns(values: pd.Series) -> pd.Series:
    """Строки/даты → datetime64[ns] (формат смешанный: с микросекундами и без)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.astype("datetime64[ns]")
    return pd.to_datetime(values, format="mixed").astype("datetime64[ns]")


def _to_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def clean_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрация сырой телеметрии: валидность, NaN координаты, выбросы скорости, дубликаты."""
    df = df.copy()
    if "location_valid" in df.columns:
        df = df[_to_bool(df["location_valid"]).to_numpy()]
    df = df.dropna(subset=["tr_id", "event_time", "lon", "lat"])
    df["event_time"] = to_ns(df["event_time"])
    df["tr_id"] = df["tr_id"].astype(np.int64)
    for col in ("lon", "lat", "speed", "heading"):
        if col in df.columns:
            df[col] = df[col].astype(np.float64)
    if "speed" in df.columns:
        df = df[~(df["speed"] > MAX_SPEED_KMH)]
    df = df.sort_values(["tr_id", "event_time"], kind="stable")
    df = df.drop_duplicates(subset=["tr_id", "event_time"], keep="first")
    return df.reset_index(drop=True)


def load_traffic(split: str) -> pd.DataFrame:
    """Телеметрия: tr_id, event_time, lon, lat, speed, heading; очищена и отсортирована."""
    _check_split(split)
    raw = pd.read_csv(
        dataset_dir() / split / "traffic.csv",
        usecols=[*TRAFFIC_COLUMNS, "location_valid"],
    )
    return clean_traffic(raw)[TRAFFIC_COLUMNS]


def parse_point(geom: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """'POINT (lon lat)' → (lon, lat)."""
    parts = geom.str.extract(r"POINT\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")
    return parts[0].astype(np.float64).to_numpy(), parts[1].astype(np.float64).to_numpy()


def load_schedule(split: str) -> pd.DataFrame:
    """Плановое расписание (+ time_fact_begin, если он есть) с координатами остановок."""
    _check_split(split)
    name = "schedule_plan.csv" if split == "validate" else "schedule.csv"
    df = pd.read_csv(dataset_dir() / split / name)
    df["tt_action_item_id"] = df["tt_action_item_id"].astype(np.int64)
    df["tr_id"] = df["tr_id"].astype(np.int64)
    df["time_begin"] = to_ns(df["time_begin"])
    if "time_fact_begin" in df.columns:
        df["time_fact_begin"] = to_ns(df["time_fact_begin"])
    df["stop_lon"], df["stop_lat"] = parse_point(df["geom"])
    return df.sort_values(["tr_id", "time_begin"], kind="stable").reset_index(drop=True)


def load_points(split: str) -> pd.DataFrame:
    """Прогнозные точки: labels_train / labels_test / validate/points."""
    _check_split(split)
    root = dataset_dir()
    if split == "validate":
        path = root / "validate" / "points.csv"
    else:
        path = root / "labels" / f"labels_{split}.csv"
    df = pd.read_csv(path)
    df["tr_id"] = df["tr_id"].astype(np.int64)
    df["target_stop_id"] = df["target_stop_id"].astype(np.int64)
    df["T"] = to_ns(df["T"])
    df["target_time_begin"] = to_ns(df["target_time_begin"])
    return df
