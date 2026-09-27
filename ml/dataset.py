"""Сборка таблиц признаков по сплитам с кэшем в ``artifacts/``.

Для каждого сплита контекст строится из его собственной телеметрии, планового расписания (факт
отбрасывается) и проходов детектора — так же, как для validate и на потоке. Разметка используется только
как таргет.

Синтетические ТС train — копии реальных со сдвигом расписания по времени (проверено: одинаковая
последовательность мест остановок, постоянный сдвиг плана, корреляция задержек ≈ 0.99). Копия покрывает
весь день, в том числе периоды точек test/validate, т.е. её разметка там — почти ответ для этих точек.
Поэтому :func:`vehicle_families` связывает копию с оригиналом (только по плану), а :func:`leak_mask`
выкидывает из обучения синтетические точки, чей «эквивалентный» момент близок к точкам оценки.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from shared import features as features_module
from shared import stops as stops_module
from shared.data import load_points, load_schedule, load_traffic
from shared.features import FEATURE_NAMES, NETWORK_FEATURES, FeatureContext, build_features, to_seconds
from shared.stops import detect_all

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
TARGET = "target_delay_s"
MODEL_FEATURES = [f for f in FEATURE_NAMES if f not in NETWORK_FEATURES]
LEAK_MARGIN_S = 1800.0


def code_version() -> str:
    """Хэш исходников признаков, детектора и сборки: кэш инвалидируется при их изменении."""
    parts = (features_module, stops_module, build_split)
    src = "".join(inspect.getsource(p) for p in parts)
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def real_vehicle_ids() -> set[int]:
    """Реальные ТС — те, что есть в разметке test (test/validate полностью реальные)."""
    return {int(x) for x in load_points("test")["tr_id"].unique()}


def vehicle_families(schedule: pd.DataFrame, real_ids: set[int]) -> pd.DataFrame:
    """Сопоставить каждое ТС с реальным «оригиналом» по плановому расписанию.

    Для синтетического ТС берётся реальное ТС с наибольшим числом совпадений (место остановки, сдвиг плана,
    округлённый до 10 с); используется только ``time_begin`` и координаты.

    Returns:
        DataFrame: tr_id, family (tr_id оригинала), shift_s (план копии − план оригинала).
    """
    sc = schedule[["tr_id", "time_begin", "stop_lon", "stop_lat"]].copy()
    sc["lon5"] = sc["stop_lon"].round(5)
    sc["lat5"] = sc["stop_lat"].round(5)
    groups = {int(k): g for k, g in sc.groupby("tr_id")}
    rows = []
    for tr_id, g in groups.items():
        if tr_id in real_ids:
            rows.append((tr_id, tr_id, 0.0))
            continue
        best = (tr_id, 0.0, 0)
        for r in real_ids:
            if r not in groups:
                continue
            m = g.merge(groups[r], on=["lon5", "lat5"], suffixes=("_s", "_r"))
            if m.empty:
                continue
            shift = ((m["time_begin_s"] - m["time_begin_r"]).dt.total_seconds().round(-1)).value_counts()
            if shift.iloc[0] > best[2]:
                best = (r, float(shift.index[0]), int(shift.iloc[0]))
        rows.append((tr_id, best[0], best[1]))
    return pd.DataFrame(rows, columns=["tr_id", "family", "shift_s"])


def leak_mask(
    train: pd.DataFrame, eval_points: list[pd.DataFrame], margin_s: float = LEAK_MARGIN_S
) -> np.ndarray:
    """Маска синтетических строк train, близких к точкам оценки.

    Строка помечается, если её эквивалентный момент ``T − shift_s`` ближе ``margin_s`` к точке оценки
    (test/validate) того же оригинала: такие строки — зашумлённые копии точек оценки.
    """
    pts = pd.concat([p[["tr_id", "T"]] for p in eval_points], ignore_index=True)
    t_eval = {int(k): np.sort(to_seconds(g["T"])) for k, g in pts.groupby("tr_id")}
    t_eq = to_seconds(train["T"]) - train["shift_s"].to_numpy(dtype=np.float64)
    out = np.zeros(len(train), dtype=bool)
    synth = ~train["is_real"].to_numpy()
    fam = train["family"].to_numpy()
    for i in np.flatnonzero(synth):
        ts = t_eval.get(int(fam[i]))
        if ts is None:
            continue
        j = np.searchsorted(ts, t_eq[i])
        near = [abs(ts[k] - t_eq[i]) for k in (j - 1, j) if 0 <= k < len(ts)]
        out[i] = bool(near) and min(near) < margin_s
    return out


def build_context(split: str) -> FeatureContext:
    """Контекст признаков сплита: телеметрия + плановое расписание + проходы детектора."""
    traffic = load_traffic(split)
    schedule = load_schedule(split).drop(columns=["time_fact_begin", "manual_fill"], errors="ignore")
    passages = detect_all(traffic, schedule)
    # в train сетевые статистики — только по реальным ТС (как в test/validate), см. FeatureContext
    network_ids = real_vehicle_ids() if split == "train" else None
    return FeatureContext(traffic, schedule, passages, network_ids=network_ids)


def build_split(split: str, use_cache: bool = True) -> pd.DataFrame:
    """Таблица сплита: мета-колонки, признаки :data:`FEATURE_NAMES` и таргет (кроме validate).

    Мета: sample_id, tr_id, T, target_stop_id, target_time_begin, split, is_real, family, shift_s.

    Args:
        split: ``train``, ``test`` или ``validate``.
        use_cache: читать/писать parquet-кэш в ``artifacts/``.

    Returns:
        DataFrame в порядке точек сплита.
    """
    path = ARTIFACTS / f"features_{split}_{code_version()}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    points = load_points(split)
    ctx = build_context(split)
    feats = build_features(points, ctx)
    out = points[["sample_id", "tr_id", "T", "target_stop_id", "target_time_begin"]].copy()
    out["split"] = split
    real = real_vehicle_ids()
    out["is_real"] = out["tr_id"].isin(real)
    fam = vehicle_families(load_schedule(split), real).set_index("tr_id")
    out["family"] = out["tr_id"].map(fam["family"]).fillna(out["tr_id"]).astype(np.int64)
    out["shift_s"] = out["tr_id"].map(fam["shift_s"]).fillna(0.0).astype(np.float64)
    out = pd.concat([out, feats], axis=1)
    if TARGET in points.columns:
        out[TARGET] = points[TARGET].to_numpy(dtype=np.float64)
    if use_cache:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def feature_matrix(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Матрица признаков модели в фиксированном порядке."""
    return df[names or MODEL_FEATURES].astype(np.float64)


def build_sequences_split(split: str, use_cache: bool = True) -> np.ndarray:
    """Последовательности телеметрии точек сплита (:mod:`shared.sequences`) в порядке :func:`build_split`.

    Контекст тот же, что у признаков (:func:`build_context`): телеметрия сплита, плановое расписание,
    проходы детектора. Кэш — ``artifacts/sequences_<split>_<code_version>_<sequence_version>.npy``.

    Returns:
        float32 ``(число точек, SEQ_LEN, N_CHANNELS)``.
    """
    from shared.sequences import build_sequences, sequence_version

    path = ARTIFACTS / f"sequences_{split}_{code_version()}_{sequence_version()}.npy"
    if use_cache and path.exists():
        return np.load(path)
    seqs = build_sequences(load_points(split), build_context(split))
    if use_cache:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        np.save(path, seqs)
    return seqs
