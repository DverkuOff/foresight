"""Обучение CatBoost (MAE): CV по семействам ТС, оценка на test, финальная модель на train + test.

Запуск (на сервере)::

    uv run python -m ml.train --tag <git-rev>

Протокол:

* из train выкидываются синтетические точки, чей эквивалентный момент ближе 30 мин к точкам test/validate
  того же оригинала (см. :mod:`ml.dataset`) — иначе модель учится на зашумлённых копиях ответов;
* CV — GroupKFold по «семейству» (реальное ТС + его синтетические копии), OOF MAE отдельно для реальных и
  синтетических ТС;
* test — модели обучены только на train;
* итоговая модель — среднее ансамбля :data:`ENSEMBLE` × :data:`SEEDS` (реальные ТС напрямую + остаток к
  ``cur_dev_s`` с синтетикой веса 0.5), финальные модели обучены на train + test.

Модели и отчёт: ``artifacts/model_v1_*.cbm``, ``artifacts/model_v1.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold

from ml.dataset import ARTIFACTS, MODEL_FEATURES, TARGET, build_split, code_version, feature_matrix, leak_mask
from shared.data import load_points
from shared.features import LAYOVER_S

MODEL_NAME = "model_v1"


@dataclass(frozen=True)
class Config:
    """Вариант обучения.

    Attributes:
        name: имя варианта.
        synth_weight: вес синтетических ТС (0 — не использовать).
        base: колонка-база, от которой учится остаток (``None`` — учим задержку напрямую).
    """

    name: str
    synth_weight: float = 1.0
    base: str | None = None


# варианты для отчёта (влияние синтетики и остатка)
CONFIGS = [
    Config("synth_w1"),
    Config("synth_w0.5", synth_weight=0.5),
    Config("real_only", synth_weight=0.0),
    Config("synth_w1_resid_cur", base="cur_dev_s"),
    Config("synth_w0.5_resid_cur", synth_weight=0.5, base="cur_dev_s"),
]
ENSEMBLE = [
    Config("real_only", synth_weight=0.0),
    Config("synth_w0.5_resid_cur", synth_weight=0.5, base="cur_dev_s"),
]
SEEDS = (1, 2)

PARAMS = {
    "loss_function": "MAE",
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
}


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64))))


def _base(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    if cfg.base is None:
        return np.zeros(len(df))
    return df[cfg.base].fillna(0.0).to_numpy(dtype=np.float64)


def fit(df: pd.DataFrame, cfg: Config, params: dict) -> CatBoostRegressor:
    """Обучить модель варианта ``cfg`` на таблице ``df`` (с таргетом)."""
    w = np.where(df["is_real"].to_numpy(), 1.0, cfg.synth_weight)
    keep = w > 0
    d = df[keep]
    y = d[TARGET].to_numpy(dtype=np.float64) - _base(d, cfg)
    model = CatBoostRegressor(**params)
    model.fit(Pool(feature_matrix(d), y, weight=w[keep]))
    return model


def predict(model: CatBoostRegressor, df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Прогноз задержки (база варианта + предсказанный остаток)."""
    names = list(model.feature_names_) if model.feature_names_ else MODEL_FEATURES
    return model.predict(feature_matrix(df, names)) + _base(df, cfg)


def fit_ensemble(df: pd.DataFrame, params: dict) -> list[tuple[CatBoostRegressor, Config]]:
    return [(fit(df, cfg, {**params, "random_seed": s}), cfg) for cfg in ENSEMBLE for s in SEEDS]


def predict_ensemble(members: list[tuple[CatBoostRegressor, Config]], df: pd.DataFrame) -> np.ndarray:
    """Среднее прогнозов членов ансамбля."""
    return np.mean([predict(m, df, cfg) for m, cfg in members], axis=0)


def load_ensemble(meta_path: Path) -> tuple[list[tuple[CatBoostRegressor, Config]], dict]:
    """Загрузить модели ансамбля по файлу метаданных ``model_v1.json``."""
    meta = json.loads(meta_path.read_text())
    members = []
    for item in meta["members"]:
        model = CatBoostRegressor()
        model.load_model(str(meta_path.parent / item["file"]))
        members.append((model, Config(**item["config"])))
    return members, meta


def cross_validate(train: pd.DataFrame, fitter, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold прогноз на train (GroupKFold по семейству ТС); ``fitter(df) -> predict(df)``."""
    oof = np.full(len(train), np.nan)
    groups = train["family"].to_numpy()
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(train, groups=groups):
        oof[va_idx] = fitter(train.iloc[tr_idx])(train.iloc[va_idx])
    return oof


def layover_mask(df: pd.DataFrame) -> np.ndarray:
    """Точки, у которых между текущей позицией (или последней остановкой) и целью есть плановый отстой."""
    gap = df["maxgap_ahead"].fillna(df["maxgap_anchor"]).fillna(0.0)
    return ((df["has_layover"].fillna(0.0) > 0) | (gap > LAYOVER_S)).to_numpy()


def report_block(name: str, y: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, float]:
    out = {k: mae(y, p) for k, p in preds.items()}
    print(f"  {name:12s} n={len(y):5d} " + " ".join(f"{k}={v:6.1f}" for k, v in out.items()))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="версия кода (git rev) для метаданных")
    ap.add_argument("--no-cache", action="store_true", help="пересобрать признаки")
    ap.add_argument("--skip-grid", action="store_true", help="не считать отчётные варианты CONFIGS")
    args = ap.parse_args(argv)
    params = dict(PARAMS)

    train_all = build_split("train", use_cache=not args.no_cache)
    test = build_split("test", use_cache=not args.no_cache)
    leak = leak_mask(train_all, [load_points("test"), load_points("validate")])
    train = train_all[~leak].reset_index(drop=True)
    y_tr, y_te = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    real = train["is_real"].to_numpy()
    print(
        f"features={len(MODEL_FEATURES)} train={len(train_all)} → {len(train)} after dropping {leak.sum()} "
        f"synthetic copies of test/validate periods (real {real.sum()}, synthetic {(~real).sum()}) "
        f"test={len(test)}"
    )

    def scores(oof: np.ndarray, pred_te: np.ndarray) -> dict[str, float]:
        return {
            "cv_real": mae(y_tr[real], oof[real]),
            "cv_synth": mae(y_tr[~real], oof[~real]) if (~real).any() else float("nan"),
            "test": mae(y_te, pred_te),
        }

    results: dict[str, dict] = {}
    for cfg in [] if args.skip_grid else CONFIGS:

        def single(df: pd.DataFrame, cfg: Config = cfg):
            model = fit(df, cfg, params)
            return lambda x: predict(model, x, cfg)

        results[cfg.name] = scores(cross_validate(train, single), single(train)(test))
        print(f"[{cfg.name}] " + " ".join(f"{k}={v:.2f}" for k, v in results[cfg.name].items()), flush=True)

    def ensemble(df: pd.DataFrame):
        members = fit_ensemble(df, params)
        return lambda x: predict_ensemble(members, x)

    members = fit_ensemble(train, params)
    pred_te = predict_ensemble(members, test)
    results["ensemble"] = scores(cross_validate(train, ensemble), pred_te)
    print("[ensemble] " + " ".join(f"{k}={v:.2f}" for k, v in results["ensemble"].items()), flush=True)

    base = {
        "cur_dev_s_cv_real": mae(y_tr[real], train["cur_dev_s"][real]),
        "cur_dev_s_cv_synth": mae(y_tr[~real], train["cur_dev_s"][~real]),
        "zero_cv_real": mae(y_tr[real], 0.0 * y_tr[real]),
        "cur_dev_s_test": mae(y_te, test["cur_dev_s"]),
        "zero_test": mae(y_te, 0.0 * y_te),
    }
    print("baselines: " + " ".join(f"{k}={v:.2f}" for k, v in base.items()))

    # диагностика (не для выбора): обучение со всеми синтетическими точками, включая копии периодов test
    leaky = Config("leaky")
    diag = mae(y_te, predict(fit(train_all, leaky, params), test, leaky))
    print(f"diagnostic: test MAE with synthetic copies of test periods kept = {diag:.2f} (leak, not used)")

    lay = layover_mask(test)
    print("test breakdown (ensemble trained on train only):")
    preds = {"model": pred_te, "cur_dev_s": test["cur_dev_s"].to_numpy(), "zero": 0.0 * y_te}
    breakdown: dict[str, object] = {
        "all": report_block("all", y_te, preds),
        "layover": report_block("layover", y_te[lay], {k: v[lay] for k, v in preds.items()}),
        "no_layover": report_block("no layover", y_te[~lay], {k: v[~lay] for k, v in preds.items()}),
        "n_layover": int(lay.sum()),
    }
    imp = pd.Series(
        np.mean([m.get_feature_importance() for m, _ in members], axis=0), index=MODEL_FEATURES
    ).sort_values(ascending=False)
    print("feature importance (mean over ensemble trained on train):")
    print(imp.head(20).round(2).to_string())

    full = pd.concat([train, test], ignore_index=True)
    final = fit_ensemble(full, params)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (model, cfg) in enumerate(final):
        name = f"{MODEL_NAME}_{i}.cbm"
        model.save_model(str(ARTIFACTS / name))
        items.append({"file": name, "config": asdict(cfg)})
    meta = {
        "model": MODEL_NAME,
        "created": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "features_version": code_version(),
        "features": MODEL_FEATURES,
        "members": items,
        "seeds": list(SEEDS),
        "params": params,
        "results": results,
        "baselines": base,
        "diagnostic_leaky_test": diag,
        "test_breakdown": breakdown,
        "importance": imp.round(3).to_dict(),
        "train_rows": int(len(full)),
        "dropped_synthetic_rows": int(leak.sum()),
    }
    (ARTIFACTS / f"{MODEL_NAME}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {ARTIFACTS / (MODEL_NAME + '.json')} and {len(items)} models")


if __name__ == "__main__":
    main()
