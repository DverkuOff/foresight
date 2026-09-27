"""ML v2: квантили P10/P50/P90, вероятность опоздания, последовательная модель (GRU / Transformer), ансамбль,
экспорт в ONNX + INT8 и упаковка версии ``v2``.

Запуск (на сервере, GPU)::

    uv run python -m ml.train_v2 --tag "$(git rev-parse --short HEAD)"       # все стадии
    uv run python -m ml.train_v2 --stages catboost                          # CatBoost: OOF + test
    uv run python -m ml.train_v2 --stages seq --configs gru_h64,transformer  # сети: OOF + test
    uv run python -m ml.train_v2 --stages final                             # выбор, финал, ONNX, пакет v2

Протокол (тот же, что у v1, ``ml/train.py``):

* из train выкинуты синтетические копии периодов test/validate (:func:`ml.dataset.leak_mask`);
* CV — GroupKFold(5) по семьям ТС (одни и те же фолды у всех компонент), метрики — по реальным ТС;
* test — holdout: модели обучены только на train, для выбора ничего не используется;
* по OOF train (реальные ТС) выбираются вариант сети (GRU / Transformer; абляции ``mlp`` / ``gru_seq`` в
  выбор не входят) и число эпох (ранняя остановка по OOF MAE), вес ансамбля, центр и поправка интервала
  (CQR), калибровка ``p_late`` (в т.ч. стекинг с прогнозом ансамбля) и способ ожидаемой ошибки
  (``k · (p90 − p10)`` или CatBoost по OOF-ошибкам); для веса ансамбля есть и вложенная оценка (вес
  подбирается по OOF остальных фолдов) — она пишется в манифест как ``cv_mae``;
* финальные модели — на train + test.

Промежуточные результаты — ``artifacts/v2/*.npz`` (OOF и test по стадиям), отчёт —
``artifacts/v2/report.json``, версия — ``<models_dir>/v2``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from ml.dataset import (
    ARTIFACTS,
    MODEL_FEATURES,
    TARGET,
    build_sequences_split,
    build_split,
    code_version,
    feature_matrix,
    leak_mask,
)
from ml.inference import LATE_THRESHOLD_S, models_dir
from ml.train import ENSEMBLE, PARAMS, SEEDS, fit_ensemble, layover_mask, mae, predict_ensemble
from shared.data import load_points
from shared.sequences import SEQ_CHANNELS, SEQ_LEN, SEQ_STEP_S, sequence_version

V2_DIR = ARTIFACTS / "v2"
N_FOLDS = 5
SYNTH_WEIGHT = 0.5
ALPHAS = (0.1, 0.5, 0.9)
COVERAGE = 0.8
QUANTILE_BASE = "cur_dev_s"
QUANTILE_PARAMS = {**PARAMS, "loss_function": "MultiQuantile:alpha=" + ",".join(str(a) for a in ALPHAS)}
LATE_PARAMS = {**PARAMS, "loss_function": "Logloss"}
ECE_BINS = 10
CALIBRATION_METHODS = ("none", "platt", "isotonic", "stack")
STACK_PRED_SCALE = 100.0
# ожидаемая |ошибка|: неглубокие деревья, мало итераций — цель шумная (одна |ошибка| на точку)
ERR_PARAMS = {
    **PARAMS,
    "loss_function": "RMSE",
    "iterations": 200,
    "depth": 4,
    "l2_leaf_reg": 10.0,
    "random_seed": 1,
}
BENCH_SIZES = (1, 100, 1000)


def seq_configs():
    """Варианты последовательной модели (импорт torch — только здесь и в стадиях сетей)."""
    from ml.seq_model import SeqConfig

    # регуляризованные варианты: данных мало (2449 строк, 1141 реальных) — без регуляризации сеть
    # переобучается за 2–3 эпохи (OOF MAE растёт с 78 до 89 к 80-й эпохе)
    reg = {"dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64, "max_epochs": 30}
    configs = [
        SeqConfig("gru_h64", kind="gru", hidden=64),
        SeqConfig("gru_h32", kind="gru", hidden=32),
        SeqConfig("transformer", kind="transformer", hidden=64, lr=1e-3),
        SeqConfig("gru_reg", kind="gru", hidden=32, tab_hidden=32, **reg),
        SeqConfig("gru_reg_h64", kind="gru", hidden=64, tab_hidden=32, **reg),
        SeqConfig("transformer_reg", kind="transformer", hidden=32, layers=1, tab_hidden=32, **reg),
        SeqConfig("gru_reg_slow", kind="gru", hidden=32, tab_hidden=32, **{**reg, "lr": 3e-4}),
        SeqConfig("gru_reg_d5", kind="gru", hidden=32, tab_hidden=32, **{**reg, "dropout": 0.5, "lr": 5e-4}),
        # абляции: только табличные признаки / только последовательность
        SeqConfig("mlp", kind="mlp"),
        SeqConfig("gru_seq", kind="gru_seq", hidden=64),
        SeqConfig("mlp_reg", kind="mlp", tab_hidden=32, **reg),
        SeqConfig("gru_seq_reg", kind="gru_seq", hidden=32, **reg),
    ]
    return {c.name: c for c in configs}


ABLATION_KINDS = ("mlp", "gru_seq")
COMPONENT = {"gru": "gru", "transformer": "transformer"}  # вид сети → компонент ансамбля в манифесте


# --- данные -------------------------------------------------------------------------------------------------
@dataclass
class Data:
    """Train (без синтетических копий периодов оценки) и test: признаки, последовательности, фолды."""

    train: pd.DataFrame
    test: pd.DataFrame
    seq_train: np.ndarray
    seq_test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    dropped: int

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=np.float64)

    @property
    def y_test(self) -> np.ndarray:
        return self.test[TARGET].to_numpy(dtype=np.float64)

    @property
    def real(self) -> np.ndarray:
        return self.train["is_real"].to_numpy()


def load_data(use_cache: bool = True) -> Data:
    """Таблицы и последовательности train/test; фолды GroupKFold по семьям ТС (как ``ml.train``)."""
    train_all = build_split("train", use_cache=use_cache)
    test = build_split("test", use_cache=use_cache)
    seq_all = build_sequences_split("train", use_cache=use_cache)
    seq_test = build_sequences_split("test", use_cache=use_cache)
    if len(seq_all) != len(train_all) or len(seq_test) != len(test):
        raise SystemExit("sequences are not aligned with feature tables")
    leak = leak_mask(train_all, [load_points("test"), load_points("validate")])
    train = train_all[~leak].reset_index(drop=True)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, groups=train["family"].to_numpy()))
    return Data(train, test, seq_all[~leak], seq_test, folds, int(leak.sum()))


def weights(df: pd.DataFrame, synth_weight: float = SYNTH_WEIGHT) -> np.ndarray:
    """Вес строки: реальные ТС — 1, синтетические — ``synth_weight``."""
    return np.where(df["is_real"].to_numpy(), 1.0, synth_weight)


def base_values(df: pd.DataFrame, col: str | None) -> np.ndarray:
    """База остатка (``NaN`` → 0), как ``ml.train._base`` и ``ml.inference``."""
    if col is None:
        return np.zeros(len(df))
    return df[col].astype(np.float64).fillna(0.0).to_numpy(dtype=np.float64)


# --- CatBoost: квантили и вероятность опоздания -------------------------------------------------------------
def fit_quantiles(df: pd.DataFrame) -> CatBoostRegressor:
    """MultiQuantile (P10/P50/P90) остатка к ``cur_dev_s``; синтетика с весом 0.5."""
    y = df[TARGET].to_numpy(dtype=np.float64) - base_values(df, QUANTILE_BASE)
    model = CatBoostRegressor(**QUANTILE_PARAMS)
    model.fit(Pool(feature_matrix(df), y, weight=weights(df)))
    return model


def predict_quantiles(model: CatBoostRegressor, df: pd.DataFrame) -> np.ndarray:
    """Квантили ``(n, 3)``: база + прогноз, упорядочены по строке (без пересечений)."""
    q = np.asarray(model.predict(feature_matrix(df)), dtype=np.float64).reshape(len(df), -1)
    return np.sort(q + base_values(df, QUANTILE_BASE)[:, None], axis=1)


def fit_late(df: pd.DataFrame) -> CatBoostClassifier:
    """Классификатор ``delay > 120 с`` (Logloss); синтетика с весом 0.5."""
    y = (df[TARGET].to_numpy(dtype=np.float64) > LATE_THRESHOLD_S).astype(int)
    model = CatBoostClassifier(**LATE_PARAMS)
    model.fit(Pool(feature_matrix(df), y, weight=weights(df)))
    return model


def predict_late(model: CatBoostClassifier, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(feature_matrix(df))[:, 1], dtype=np.float64)


def stage_catboost(data: Data) -> dict[str, np.ndarray]:
    """OOF и test для ансамбля задержки v1 (4 модели), квантилей и классификатора опоздания."""
    tr, te = data.train, data.test
    n = len(tr)
    out = {
        "oof_delay": np.full(n, np.nan),
        "oof_q": np.full((n, 3), np.nan),
        "oof_late": np.full(n, np.nan),
    }
    for i, (tr_idx, va_idx) in enumerate(data.folds):
        t0 = time.time()
        fit_df, va_df = tr.iloc[tr_idx], tr.iloc[va_idx]
        out["oof_delay"][va_idx] = predict_ensemble(fit_ensemble(fit_df, PARAMS), va_df)
        out["oof_q"][va_idx] = predict_quantiles(fit_quantiles(fit_df), va_df)
        out["oof_late"][va_idx] = predict_late(fit_late(fit_df), va_df)
        print(f"  catboost fold {i}: {time.time() - t0:.0f} s", flush=True)
    out["test_delay"] = predict_ensemble(fit_ensemble(tr, PARAMS), te)
    out["test_q"] = predict_quantiles(fit_quantiles(tr), te)
    out["test_late"] = predict_late(fit_late(tr), te)
    real = data.real
    print(
        f"[catboost] cv_real={mae(data.y[real], out['oof_delay'][real]):.2f} "
        f"test={mae(data.y_test, out['test_delay']):.2f} "
        f"q50 cv_real={mae(data.y[real], out['oof_q'][real, 1]):.2f} "
        f"test={mae(data.y_test, out['test_q'][:, 1]):.2f}",
        flush=True,
    )
    return out


# --- сети ---------------------------------------------------------------------------------------------------
def tab_matrix(df: pd.DataFrame) -> np.ndarray:
    return feature_matrix(df).to_numpy(dtype=np.float32)


def cv_seq(cfg, data: Data) -> dict[str, Any]:
    """OOF по эпохам (среднее по сидам) → число эпох по OOF MAE реальных ТС; test — сети на всём train."""
    from ml.seq_model import train_net

    tr, te = data.train, data.test
    tab, tab_te = tab_matrix(tr), tab_matrix(te)
    base, base_te = base_values(tr, cfg.base), base_values(te, cfg.base)
    target = data.y - base
    w = weights(tr, cfg.synth_weight)
    n_ep, n_seeds = cfg.max_epochs, len(cfg.seeds)
    oof_ep = np.zeros((n_ep, len(tr)))
    t0 = time.time()
    for seed in cfg.seeds:
        for tr_idx, va_idx in data.folds:

            def keep(epoch: int, preds: dict[str, np.ndarray], va_idx=va_idx) -> None:
                oof_ep[epoch - 1, va_idx] += preds["va"] / n_seeds

            train_net(
                cfg,
                seed,
                data.seq_train[tr_idx],
                tab[tr_idx],
                target[tr_idx],
                w[tr_idx],
                eval_sets={"va": (data.seq_train[va_idx], tab[va_idx])},
                on_epoch=keep,
            )
    real = data.real
    curve = np.array([mae(data.y[real], oof_ep[e, real] + base[real]) for e in range(n_ep)])
    best = int(np.argmin(curve)) + 1
    test_ep = np.zeros((n_ep, len(te)))
    for seed in cfg.seeds:

        def keep_te(epoch: int, preds: dict[str, np.ndarray]) -> None:
            test_ep[epoch - 1] += preds["te"] / n_seeds

        train_net(
            cfg,
            seed,
            data.seq_train,
            tab,
            target,
            w,
            eval_sets={"te": (data.seq_test, tab_te)},
            on_epoch=keep_te,
        )
    curve_te = np.array([mae(data.y_test, test_ep[e] + base_te) for e in range(n_ep)])
    res = {
        "oof": oof_ep[best - 1] + base,
        "test": test_ep[best - 1] + base_te,
        "curve_oof": curve,
        "curve_test": curve_te,
        "best_epoch": np.array(best),
        "cv_synth": np.array(mae(data.y[~real], oof_ep[best - 1, ~real] + base[~real])),
    }
    print(
        f"[{cfg.name}] best_epoch={best} cv_real={curve[best - 1]:.2f} cv_synth={float(res['cv_synth']):.2f} "
        f"test={curve_te[best - 1]:.2f} (last epoch: cv {curve[-1]:.2f}, test {curve_te[-1]:.2f}) "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    return res


def stage_seq(data: Data, names: list[str]) -> None:
    configs = seq_configs()
    for name in names:
        res = cv_seq(configs[name], data)
        np.savez(V2_DIR / f"seq_{name}.npz", **res)


# --- постобработка по OOF: вес ансамбля, интервалы, калибровка --------------------------------------------
W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


def best_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Вес ``w`` второго члена, минимизирующий MAE ``(1 − w)·a + w·b`` (сетка 0.01)."""
    scores = [mae(y, (1.0 - w) * a + w * b) for w in W_GRID]
    return float(W_GRID[int(np.argmin(scores))])


def interval(q: np.ndarray, pred: np.ndarray, center: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Интервал из квантилей: центр — P50 модели квантилей (``q50``) или прогноз ансамбля (``pred``)."""
    c = q[:, 1] if center == "q50" else pred
    return c + (q[:, 0] - q[:, 1]), c, c + (q[:, 2] - q[:, 1])


def cqr_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, coverage: float = COVERAGE) -> float:
    """Поправка conformalized quantile regression: интервал ``[lo − Q, hi + Q]`` покрывает ``coverage``."""
    score = np.maximum(lo - y, y - hi)
    level = min(1.0, coverage * (1.0 + 1.0 / len(y)))
    return float(np.quantile(score, level, method="higher"))


def apply_interval(lo: np.ndarray, c: np.ndarray, hi: np.ndarray, offset: float):
    lo2, hi2 = lo - offset, hi + offset
    return np.minimum(lo2, c), c, np.maximum(hi2, c)


def coverage_stats(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "below": float(np.mean(y < lo)),
        "above": float(np.mean(y > hi)),
        "mean_width_s": float(np.mean(hi - lo)),
    }


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(
    method: str, p: np.ndarray, y: np.ndarray, pred: np.ndarray | None = None
) -> dict[str, Any]:
    """Калибратор вероятности: ``none`` | ``platt`` (логистическая по logit p) | ``isotonic`` | ``stack``.

    ``stack`` — логистическая регрессия по logit p классификатора и прогнозу задержки ансамбля
    ``(pred − порог) / STACK_PRED_SCALE`` (``pred`` обязателен), см. :func:`ml.inference.calibrate`.
    """
    from sklearn.linear_model import LogisticRegression

    if method == "none":
        return {"method": "none"}
    if method == "platt":
        lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
        return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    if method == "stack":
        x = np.c_[logit(p), (pred - LATE_THRESHOLD_S) / STACK_PRED_SCALE]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
        return {
            "method": "stack",
            "a": float(lr.coef_[0, 0]),
            "c": float(lr.coef_[0, 1]),
            "b": float(lr.intercept_[0]),
            "pred_scale": STACK_PRED_SCALE,
        }
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return {
        "method": "isotonic",
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
    }


def apply_calibrator(cal: dict[str, Any], p: np.ndarray, pred: np.ndarray | None = None) -> np.ndarray:
    """То же, что ``ml.inference.calibrate``: без sklearn (np.interp / сигмоида)."""
    from ml.inference import calibrate

    return calibrate(cal, p, pred, LATE_THRESHOLD_S)


def fit_abs_error(x: pd.DataFrame, err: np.ndarray) -> CatBoostRegressor:
    """Ожидаемая абсолютная ошибка ансамбля: CatBoost RMSE по модулям out-of-sample ошибок."""
    model = CatBoostRegressor(**ERR_PARAMS)
    model.fit(Pool(x, err))
    return model


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> tuple[float, list[dict[str, float]]]:
    """ECE (равные по ширине корзины) и таблица калибровочной кривой."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - freq)
        table.append(
            {"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "p_mean": conf, "freq": freq}
        )
    return float(ece), table


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# --- ONNX: экспорт, точность, латентность -------------------------------------------------------------------
def ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_predict(sess, seq: np.ndarray, tab: np.ndarray) -> np.ndarray:
    from ml.seq_model import SEQ_INPUT, TAB_INPUT

    out = sess.run(None, {SEQ_INPUT: seq.astype(np.float32), TAB_INPUT: tab.astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float64)


def ort_latency(
    sess, seq: np.ndarray, tab: np.ndarray, sizes=BENCH_SIZES, repeat: int = 30
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    out = {}
    for n in sizes:
        idx = rng.integers(0, len(seq), size=n)
        s, t = seq[idx], tab[idx]
        ort_predict(sess, s, t)
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            ort_predict(sess, s, t)
            times.append((time.perf_counter() - t0) * 1000.0)
        out[str(n)] = {
            "p50_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
        }
    return out


def onnx_weight_bytes(path: Path) -> int:
    """Байты весов (инициализаторов) ONNX-модели: у INT8 веса MatMul — int8 + масштабы."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    return int(sum(numpy_helper.to_array(init).nbytes for init in model.graph.initializer))


def export_variants(nets: list, cfg, out_dir: Path, stem: str, n_tab: int) -> dict[str, Path]:
    """ONNX-варианты ансамбля сидов: fp32 (у GRU — оператор ``GRU``), fp32 развёрнутый, INT8 из него."""
    from ml.seq_model import SeedEnsemble, export_onnx, quantize_int8

    n_ch = len(SEQ_CHANNELS)
    paths: dict[str, Path] = {}
    if cfg.kind == "gru":
        import copy

        fused = SeedEnsemble([copy.deepcopy(n).cpu().eval() for n in nets])
        paths["fp32"] = export_onnx(fused, out_dir / f"{stem}.fp32.onnx", SEQ_LEN, n_ch, n_tab)
    unrolled = SeedEnsemble([n.for_export() for n in nets])
    paths["fp32_unrolled"] = export_onnx(
        unrolled, out_dir / f"{stem}.fp32_unrolled.onnx", SEQ_LEN, n_ch, n_tab
    )
    if "fp32" not in paths:
        paths["fp32"] = paths["fp32_unrolled"]
    paths["int8"] = quantize_int8(paths["fp32_unrolled"], out_dir / f"{stem}.int8.onnx")
    return paths


def torch_cpu_predict(nets: list, seq: np.ndarray, tab: np.ndarray) -> np.ndarray:
    import copy

    import torch

    from ml.seq_model import predict_net

    s = torch.as_tensor(seq, dtype=torch.float32)
    t = torch.as_tensor(tab, dtype=torch.float32)
    preds = [predict_net(copy.deepcopy(n).cpu().eval(), s, t) for n in nets]
    return np.mean(preds, axis=0)


# --- финал -----------------------------------------------------------------------------------------------
def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stage_final(data: Data, version: str, tag: str, root: Path, force: bool) -> dict[str, Any]:
    """Выбор по OOF, метрики, ONNX + INT8, финальные модели на train + test и упаковка версии."""
    from ml.registry import git_head, utc_iso, write_bundle
    from ml.seq_model import train_net

    cb = load_npz(V2_DIR / "catboost.npz")
    seq_res = {p.stem[4:]: load_npz(p) for p in sorted(V2_DIR.glob("seq_*.npz"))}
    configs = seq_configs()
    y, yt, real = data.y, data.y_test, data.real
    tr, te = data.train, data.test
    report: dict[str, Any] = {"created_at": utc_iso(), "git_commit": tag, "folds": N_FOLDS}

    # --- компоненты: CV / test ---
    comp: dict[str, dict[str, float]] = {
        "zero": {"cv": mae(y[real], 0 * y[real]), "test": mae(yt, 0 * yt)},
        "cur_dev_s": {"cv": mae(y[real], tr["cur_dev_s"][real]), "test": mae(yt, te["cur_dev_s"])},
        "catboost_v1": {
            "cv": mae(y[real], cb["oof_delay"][real]),
            "cv_synth": mae(y[~real], cb["oof_delay"][~real]),
            "test": mae(yt, cb["test_delay"]),
        },
        "catboost_q50": {"cv": mae(y[real], cb["oof_q"][real, 1]), "test": mae(yt, cb["test_q"][:, 1])},
    }
    for name, r in seq_res.items():
        comp[name] = {
            "cv": mae(y[real], r["oof"][real]),
            "cv_synth": float(r["cv_synth"]),
            "test": mae(yt, r["test"]),
            "best_epoch": int(r["best_epoch"]),
            "cv_last_epoch": float(r["curve_oof"][-1]),
        }
    cands = [c for c in seq_res if c in configs and configs[c].kind not in ABLATION_KINDS]
    if not cands:
        raise SystemExit("no sequence model results in artifacts/v2 (run --stages seq)")
    best = min(cands, key=lambda c: comp[c]["cv"])
    cfg = configs[best]
    sr = seq_res[best]
    epochs = int(sr["best_epoch"])
    print(f"sequence model: {best} (by CV among {cands}), epochs={epochs}", flush=True)

    # --- вес ансамбля по OOF (реальные ТС) + вложенная оценка ---
    w = best_weight(y[real], cb["oof_delay"][real], sr["oof"][real])
    oof = (1 - w) * cb["oof_delay"] + w * sr["oof"]
    nested = np.empty(len(tr))
    fold_w = []
    for _, va_idx in data.folds:
        m = real.copy()
        m[va_idx] = False
        wf = best_weight(y[m], cb["oof_delay"][m], sr["oof"][m])
        fold_w.append(wf)
        nested[va_idx] = (1 - wf) * cb["oof_delay"][va_idx] + wf * sr["oof"][va_idx]
    test_pred_torch = (1 - w) * cb["test_delay"] + w * sr["test"]
    comp["ensemble"] = {
        "cv": mae(y[real], oof[real]),
        "cv_nested": mae(y[real], nested[real]),
        "cv_synth": mae(y[~real], oof[~real]),
        "test": mae(yt, test_pred_torch),
    }
    lay = layover_mask(te)
    report["test_breakdown"] = {
        k: {
            "n": int(m.sum()),
            "catboost_v1": mae(yt[m], cb["test_delay"][m]),
            best: mae(yt[m], sr["test"][m]),
            "ensemble": mae(yt[m], test_pred_torch[m]),
            "cur_dev_s": mae(yt[m], te["cur_dev_s"].to_numpy()[m]),
        }
        for k, m in (("layover", lay), ("no_layover", ~lay))
    }
    report["ensemble"] = {"sequence_model": best, "weight_seq": w, "fold_weights": fold_w, "epochs": epochs}
    print(
        f"ensemble weight seq={w} (folds {fold_w}); cv={comp['ensemble']['cv']:.2f} "
        f"nested={comp['ensemble']['cv_nested']:.2f} test={comp['ensemble']['test']:.2f}",
        flush=True,
    )

    # --- интервалы P10–P90: CQR по OOF (перекрёстно по фолдам для оценки) ---
    yr = y[real]
    variants = {}
    for center in ("q50", "pred"):
        lo, c, hi = interval(cb["oof_q"], oof, center)
        raw = coverage_stats(yr, lo[real], hi[real])
        lo_cf, hi_cf = np.empty(len(tr)), np.empty(len(tr))
        for _, va_idx in data.folds:
            m = real.copy()
            m[va_idx] = False
            off = cqr_offset(lo[m], hi[m], y[m])
            lo_cf[va_idx], _, hi_cf[va_idx] = apply_interval(lo[va_idx], c[va_idx], hi[va_idx], off)
        variants[center] = {"raw": raw, "cqr_cross_fitted": coverage_stats(yr, lo_cf[real], hi_cf[real])}
    center = min(variants, key=lambda k: variants[k]["cqr_cross_fitted"]["mean_width_s"])
    lo, c, hi = interval(cb["oof_q"], oof, center)
    offset = cqr_offset(lo[real], hi[real], yr)
    lo_o, c_o, hi_o = apply_interval(lo, c, hi, offset)
    lo_t, c_t, hi_t = apply_interval(*interval(cb["test_q"], test_pred_torch, center), offset)
    report["intervals"] = {
        "variants_oof": variants,
        "center": center,
        "cqr_offset_s": offset,
        "oof": coverage_stats(yr, lo_o[real], hi_o[real]),
        "test_raw": coverage_stats(yt, *interval(cb["test_q"], test_pred_torch, center)[::2]),
        "test": coverage_stats(yt, lo_t, hi_t),
    }
    print(
        f"intervals: center={center} offset={offset:.1f} s oof={report['intervals']['oof']} "
        f"test={report['intervals']['test']}",
        flush=True,
    )

    # --- ожидаемая |ошибка|: k · (p90 − p10) или модель по OOF-ошибкам (выбор по OOF Spearman) ---
    abs_err, abs_err_t = np.abs(y - oof), np.abs(yt - test_pred_torch)
    k_err = float(abs_err[real].mean() / (hi_o - lo_o)[real].mean())
    x_tr, x_te = feature_matrix(tr), feature_matrix(te)
    err_cf = np.empty(len(tr))
    for _, va_idx in data.folds:
        m = real.copy()
        m[va_idx] = False
        err_cf[va_idx] = fit_abs_error(x_tr[m], abs_err[m]).predict(x_tr.iloc[va_idx])
    width_oof = k_err * (hi_o - lo_o)
    err_eval = {
        "width": {
            "spearman_oof": spearman(width_oof[real], abs_err[real]),
            "mean_oof_s": float(width_oof[real].mean()),
        },
        "model": {
            "spearman_oof": spearman(err_cf[real], abs_err[real]),
            "mean_oof_s": float(err_cf[real].mean()),
        },
    }
    err_method = max(err_eval, key=lambda k: err_eval[k]["spearman_oof"])
    if err_method == "model":
        exp_err_t = np.maximum(fit_abs_error(x_tr[real], abs_err[real]).predict(x_te), 0.0)
    else:
        exp_err_t = k_err * (hi_t - lo_t)
    terc = np.quantile(exp_err_t, [1 / 3, 2 / 3])
    groups = np.digitize(exp_err_t, terc)
    report["expected_abs_error"] = {
        "method": err_method,
        "oof_cross_fitted": err_eval,
        "mean_actual_oof_s": float(abs_err[real].mean()),
        "abs_error_k": k_err,
        "test": {
            "mean_expected_s": float(exp_err_t.mean()),
            "mean_actual_s": float(abs_err_t.mean()),
            "spearman": spearman(exp_err_t, abs_err_t),
            "spearman_width": spearman(k_err * (hi_t - lo_t), abs_err_t),
            "by_tercile": [
                {
                    "n": int((groups == g).sum()),
                    "expected_s": float(exp_err_t[groups == g].mean()),
                    "actual_s": float(abs_err_t[groups == g].mean()),
                }
                for g in range(3)
            ],
        },
    }
    print(
        f"expected_abs_error: {err_method} {json.dumps(report['expected_abs_error'], default=float)}",
        flush=True,
    )

    # --- p_late: калибровка по OOF (перекрёстно по фолдам для оценки) ---
    yl, ylt = (y > LATE_THRESHOLD_S).astype(float), (yt > LATE_THRESHOLD_S).astype(float)
    p_oof, p_te = cb["oof_late"], cb["test_late"]
    cal_eval = {}
    for method in CALIBRATION_METHODS:
        p_cf = np.empty(len(tr))
        for _, va_idx in data.folds:
            m = real.copy()
            m[va_idx] = False
            cal_f = fit_calibrator(method, p_oof[m], yl[m], oof[m])
            p_cf[va_idx] = apply_calibrator(cal_f, p_oof[va_idx], oof[va_idx])
        ece, _ = reliability(yl[real], p_cf[real])
        cal_eval[method] = {"brier": brier(yl[real], p_cf[real]), "ece": ece, "p_cf": p_cf}
    method = min(cal_eval, key=lambda k: cal_eval[k]["brier"])
    cal = fit_calibrator(method, p_oof[real], yl[real], oof[real])
    p_te_cal = apply_calibrator(cal, p_te, test_pred_torch)
    ece_te, table_te = reliability(ylt, p_te_cal)
    ece_te_raw, _ = reliability(ylt, p_te)
    ece_oof, table_oof = reliability(yl[real], cal_eval[method]["p_cf"][real])
    base_rate = float(yl[real].mean())
    report["p_late"] = {
        "threshold_s": LATE_THRESHOLD_S,
        "rate_oof": base_rate,
        "rate_test": float(ylt.mean()),
        "oof_cross_fitted": {k: {"brier": v["brier"], "ece": v["ece"]} for k, v in cal_eval.items()},
        "brier_climatology_oof": base_rate * (1 - base_rate),
        "calibration": method,
        "calibrator": cal,
        "test": {
            "brier": brier(ylt, p_te_cal),
            "ece": ece_te,
            "brier_raw": brier(ylt, p_te),
            "ece_raw": ece_te_raw,
            "brier_climatology": float(ylt.mean() * (1 - ylt.mean())),
        },
        "reliability_oof": table_oof,
        "reliability_test": table_te,
    }
    print(
        f"p_late: calibration={method} oof={report['p_late']['oof_cross_fitted']} "
        f"test={report['p_late']['test']}",
        flush=True,
    )

    # --- сети train-only с выбранным числом эпох: воспроизводимость, ONNX fp32 / INT8, MAE и латентность ---
    out_dir = V2_DIR / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab, tab_te = tab_matrix(tr), tab_matrix(te)
    base, base_te = base_values(tr, cfg.base), base_values(te, cfg.base)
    wts = weights(tr, cfg.synth_weight)
    t0 = time.time()
    nets = [train_net(cfg, s, data.seq_train, tab, y - base, wts, epochs=epochs) for s in cfg.seeds]
    ref_cpu = torch_cpu_predict(nets, data.seq_test, tab_te)
    repro = float(np.max(np.abs(ref_cpu + base_te - sr["test"])))
    paths = export_variants(nets, cfg, out_dir, f"{best}_train", tab.shape[1])
    onnx_rep: dict[str, Any] = {"retrain_s": round(time.time() - t0, 1), "reproduce_max_abs_diff_s": repro}
    val_feats = build_split("validate")
    val_seq = build_sequences_split("validate")
    pool_seq = np.concatenate([data.seq_test, val_seq])
    pool_tab = np.concatenate([tab_te, tab_matrix(val_feats)])
    for variant, path in paths.items():
        sess = ort_session(path)
        resid = ort_predict(sess, data.seq_test, tab_te)
        seq_pred = resid + base_te
        ens = (1 - w) * cb["test_delay"] + w * seq_pred
        onnx_rep[variant] = {
            "file_bytes": path.stat().st_size,
            "weight_bytes": onnx_weight_bytes(path),
            "max_abs_diff_vs_torch_s": float(np.max(np.abs(resid - ref_cpu))),
            "test_mae_component": mae(yt, seq_pred),
            "test_mae_ensemble": mae(yt, ens),
            "latency_cpu": ort_latency(sess, pool_seq, pool_tab),
        }
        print(f"onnx {variant}: {json.dumps(onnx_rep[variant])}", flush=True)
    onnx_rep["torch_test_mae_component"] = mae(yt, ref_cpu + base_te)
    report["onnx"] = onnx_rep
    int8_test = ort_predict(ort_session(paths["int8"]), data.seq_test, tab_te) + base_te
    test_pred = (1 - w) * cb["test_delay"] + w * int8_test
    comp["ensemble"]["test_int8"] = mae(yt, test_pred)
    comp[best]["test_int8"] = mae(yt, int8_test)
    report["components"] = comp

    # --- финальные модели на train + test ---
    full = pd.concat([tr, te], ignore_index=True)
    seq_full = np.concatenate([data.seq_train, data.seq_test])
    tab_full = tab_matrix(full)
    stage = V2_DIR / "final"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files: dict[str, Path] = {}
    members: list[dict[str, Any]] = []
    t0 = time.time()
    cb_members = fit_ensemble(full, PARAMS)
    seeds = [s for _ in ENSEMBLE for s in SEEDS]
    for i, ((model, mcfg), seed) in enumerate(zip(cb_members, seeds, strict=True)):
        name = f"cb_delay_{i}.cbm"
        model.save_model(str(stage / name))
        files[name] = stage / name
        members.append(
            {
                "name": f"{mcfg.name}_s{seed}",
                "component": "catboost",
                "target": "delay",
                "file": name,
                "base": mcfg.base,
                "synth_weight": mcfg.synth_weight,
                "seed": seed,
            }
        )
    qm = fit_quantiles(full)
    qm.save_model(str(stage / "cb_quantiles.cbm"))
    files["cb_quantiles.cbm"] = stage / "cb_quantiles.cbm"
    members.append(
        {
            "name": "quantiles",
            "component": "catboost",
            "target": "quantiles",
            "file": "cb_quantiles.cbm",
            "base": QUANTILE_BASE,
            "alphas": list(ALPHAS),
            "synth_weight": SYNTH_WEIGHT,
        }
    )
    lm = fit_late(full)
    lm.save_model(str(stage / "cb_late.cbm"))
    files["cb_late.cbm"] = stage / "cb_late.cbm"
    members.append(
        {
            "name": "p_late",
            "component": "catboost",
            "target": "p_late",
            "file": "cb_late.cbm",
            "threshold_s": LATE_THRESHOLD_S,
            "calibration": cal,
            "synth_weight": SYNTH_WEIGHT,
        }
    )
    if err_method == "model":
        # out-of-sample ошибки: OOF на train (реальные ТС) + test (модели, обученные только на train)
        x_err = pd.concat([x_tr[real], x_te], ignore_index=True)
        em = fit_abs_error(x_err, np.r_[abs_err[real], abs_err_t])
        em.save_model(str(stage / "cb_abs_error.cbm"))
        files["cb_abs_error.cbm"] = stage / "cb_abs_error.cbm"
        members.append(
            {
                "name": "abs_error",
                "component": "catboost",
                "target": "abs_error",
                "file": "cb_abs_error.cbm",
                "base": None,
                "rows": int(len(x_err)),
            }
        )
    base_full = base_values(full, cfg.base)
    y_full = full[TARGET].to_numpy(dtype=np.float64)
    nets_full = [
        train_net(
            cfg, s, seq_full, tab_full, y_full - base_full, weights(full, cfg.synth_weight), epochs=epochs
        )
        for s in cfg.seeds
    ]
    fpaths = export_variants(nets_full, cfg, stage, "seq", tab_full.shape[1])
    fpaths["fp32_unrolled"].unlink()
    files["seq.int8.onnx"] = fpaths["int8"]
    files["seq.fp32.onnx"] = fpaths["fp32"]
    component = COMPONENT[cfg.kind]
    members.append(
        {
            "name": best,
            "component": component,
            "target": "delay",
            "file": "seq.int8.onnx",
            "fp32_file": "seq.fp32.onnx",
            "precision": "int8",
            "base": cfg.base,
            "seeds": list(cfg.seeds),
            "epochs": epochs,
            "config": cfg.to_dict(),
            "sequence": {
                "len": SEQ_LEN,
                "step_s": SEQ_STEP_S,
                "channels": list(SEQ_CHANNELS),
                "version": sequence_version(),
            },
        }
    )
    print(f"final models trained on train+test ({len(full)} rows): {time.time() - t0:.0f} s", flush=True)

    ens_metrics = comp["ensemble"]
    manifest = {
        "version": version,
        "created_at": utc_iso(),
        "git_commit": tag or git_head(),
        "packed_commit": git_head(),
        "model": "catboost_gru_ensemble" if component == "gru" else f"catboost_{component}_ensemble",
        "description": (
            f"CatBoost MAE (4 модели v1) + {best} по последовательности 20 мин (ONNX INT8), "
            f"вес {w:.2f} по OOF; MultiQuantile P10/P50/P90 + CQR; p_late — CatBoost Logloss + {method}; "
            f"ожидаемая ошибка — {'CatBoost по OOF-ошибкам' if err_method == 'model' else 'k · (p90 − p10)'}"
        ),
        "features": list(MODEL_FEATURES),
        "features_version": code_version(),
        "sequence_version": sequence_version(),
        "components": ["catboost", component],
        "precision": "int8",
        "capabilities": ["quantiles", "p_late", "factors", "sequence"],
        "late_threshold_s": LATE_THRESHOLD_S,
        "metrics": {
            "cv_mae": round(ens_metrics["cv_nested"], 6),
            "cv_mae_global_weight": round(ens_metrics["cv"], 6),
            "cv_mae_synth": round(ens_metrics["cv_synth"], 6),
            "test_mae": round(ens_metrics["test_int8"], 6),
            "test_mae_fp32": round(ens_metrics["test"], 6),
            "cv_mae_catboost": round(comp["catboost_v1"]["cv"], 6),
            "test_mae_catboost": round(comp["catboost_v1"]["test"], 6),
            "cv_mae_seq": round(comp[best]["cv"], 6),
            "test_mae_seq": round(comp[best]["test_int8"], 6),
            "baseline_cur_dev_cv_mae": round(comp["cur_dev_s"]["cv"], 6),
            "baseline_cur_dev_test_mae": round(comp["cur_dev_s"]["test"], 6),
            "baseline_zero_test_mae": round(comp["zero"]["test"], 6),
            "interval_coverage_oof": round(report["intervals"]["oof"]["coverage"], 6),
            "interval_coverage_test": round(report["intervals"]["test"]["coverage"], 6),
            "p_late_brier_oof": round(cal_eval[method]["brier"], 6),
            "p_late_brier_test": round(report["p_late"]["test"]["brier"], 6),
            "p_late_ece_oof": round(cal_eval[method]["ece"], 6),
            "p_late_ece_test": round(report["p_late"]["test"]["ece"], 6),
            "p_late_brier_climatology_oof": round(report["p_late"]["brier_climatology_oof"], 6),
            "expected_abs_error_spearman_oof": round(err_eval[err_method]["spearman_oof"], 6),
            "expected_abs_error_spearman_test": round(report["expected_abs_error"]["test"]["spearman"], 6),
        },
        "ensemble": {"method": "weighted", "weights": {"catboost": round(1 - w, 4), component: round(w, 4)}},
        "intervals": {
            "center": center,
            "coverage_target": COVERAGE,
            "cqr_offset_s": round(offset, 4),
            "abs_error_k": round(k_err, 6),
        },
        "members": members,
        "train": {
            "rows": int(len(full)),
            "dropped_synthetic_rows": data.dropped,
            "folds": N_FOLDS,
            "catboost_params": PARAMS,
            "sequence_model": best,
            "sequence_epochs": epochs,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    path = write_bundle(root / version, manifest, files, overwrite=force)
    report["bundle"] = str(path.parent)
    print(f"packed {path}", flush=True)
    return report


def check_bundle(data: Data, version: str, root: Path, report: dict[str, Any]) -> None:
    """Загрузить упакованную версию и проверить интерфейс на test (модели train+test — не метрика)."""
    from ml.inference import load_bundle

    b = load_bundle(version, root=root)
    out = b.predict(data.test, sequences=data.seq_test)
    no_seq = b.predict(data.test)
    report["bundle_check"] = {
        "repr": repr(b),
        "finite": bool(np.isfinite(out.to_numpy()).all()),
        "p10_le_p50_le_p90": bool(((out["p10"] <= out["p50"]) & (out["p50"] <= out["p90"])).all()),
        "mean_abs_diff_without_sequences_s": float(
            np.mean(np.abs(out["pred_delay_s"] - no_seq["pred_delay_s"]))
        ),
    }
    print(f"bundle check: {report['bundle_check']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="версия кода (git rev) для манифеста")
    ap.add_argument("--stages", default="catboost,seq,final", help="catboost,seq,final")
    ap.add_argument("--configs", default=",".join(seq_configs_names()), help="варианты сетей для стадии seq")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--root", default="", help="каталог моделей (по умолчанию $FORESIGHT_MODELS_DIR)")
    ap.add_argument("--force", action="store_true", help="перезаписать существующую версию")
    ap.add_argument("--no-cache", action="store_true", help="пересобрать признаки и последовательности")
    args = ap.parse_args(argv)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    V2_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(use_cache=not args.no_cache)
    print(
        f"train={len(data.train)} (dropped {data.dropped} synthetic copies of test/validate periods; "
        f"real {int(data.real.sum())}) test={len(data.test)} seq={data.seq_train.shape} "
        f"features_version={code_version()} sequence_version={sequence_version()}",
        flush=True,
    )
    if "catboost" in stages:
        np.savez(V2_DIR / "catboost.npz", **stage_catboost(data))
    if "seq" in stages:
        stage_seq(data, [c.strip() for c in args.configs.split(",") if c.strip()])
    if "final" in stages:
        root = Path(args.root) if args.root else models_dir()
        report = stage_final(data, args.version, args.tag, root, args.force)
        check_bundle(data, args.version, root, report)
        report["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        (V2_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
        print(f"report: {V2_DIR / 'report.json'}")


def seq_configs_names() -> list[str]:
    return list(seq_configs())


if __name__ == "__main__":
    main()
