"""Прогноз по validate и запись ``submission.csv`` (``sample_id;prediction``) с проверкой формата.

Запуск (на сервере)::

    uv run python -m ml.submit [--model artifacts/model_v1.json]
    uv run python -m ml.submit --bundle v1          # версия из реестра моделей (ml/inference.py)
    uv run python -m ml.submit --bundle v2          # v2: + последовательности validate (shared.sequences)

Файл пишется в ``artifacts/submission_<YYYYmmdd_HHMM>.csv``. На платформу не отправляется.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.dataset import ARTIFACTS, build_sequences_split, build_split, code_version
from ml.train import MODEL_NAME, load_ensemble, predict_ensemble
from shared.data import load_points
from shared.features import FEATURE_NAMES

SUBMISSION_COLUMNS = ["sample_id", "prediction"]


def check_submission(sub: pd.DataFrame, points: pd.DataFrame) -> None:
    """Проверить сабмит: колонки, все ``sample_id`` точек в том же порядке, без дублей и NaN.

    Raises:
        ValueError: при любом нарушении формата.
    """
    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns {list(sub.columns)} != {SUBMISSION_COLUMNS}")
    if sub["sample_id"].duplicated().any():
        raise ValueError("duplicated sample_id")
    if len(sub) != len(points) or list(sub["sample_id"].astype(str)) != list(points["sample_id"].astype(str)):
        raise ValueError("sample_id do not match points (set or order)")
    pred = pd.to_numeric(sub["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("prediction has NaN / inf / non-numeric values")


def write_submission(sub: pd.DataFrame, points: pd.DataFrame, path: Path) -> None:
    """Проверить и записать сабмит (разделитель ``;``, заголовок), затем перечитать и проверить ещё раз."""
    check_submission(sub, points)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, sep=";", index=False, float_format="%.3f")
    back = pd.read_csv(path, sep=";", dtype={"sample_id": str})
    check_submission(back, points)


def make_submission(points: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Таблица сабмита из точек validate и прогнозов в их порядке."""
    return pd.DataFrame({"sample_id": points["sample_id"].astype(str).to_numpy(), "prediction": pred})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ARTIFACTS / f"{MODEL_NAME}.json"), help="метаданные ансамбля")
    ap.add_argument("--bundle", default="", help="версия или каталог модели реестра (вместо --model)")
    ap.add_argument("--force", action="store_true", help="разрешить несовпадение версии кода признаков")
    ap.add_argument("--out", default="", help="путь сабмита (по умолчанию artifacts/submission_<время>.csv)")
    args = ap.parse_args(argv)

    if args.bundle:
        from ml.inference import BundleError, load_bundle

        try:
            bundle = load_bundle(args.bundle, strict=not args.force)
        except BundleError as e:
            raise SystemExit(str(e)) from e
        features, tag, name = bundle.features, bundle.manifest.get("git_commit"), bundle.version
        # последовательности validate тем же модулем, что на потоке (shared.sequences), в порядке points.csv
        seqs = build_sequences_split("validate") if bundle.sequence_shape else None

        def predict(df: pd.DataFrame) -> np.ndarray:
            return bundle.predict(df, sequences=seqs)["pred_delay_s"].to_numpy()
    else:
        model_path = Path(args.model)
        members, meta = load_ensemble(model_path)
        if meta.get("features_version") != code_version() and not args.force:
            raise SystemExit(
                f"features code changed since training ({meta.get('features_version')} != {code_version()}); "
                "retrain or pass --force"
            )
        features, tag, name = meta["features"], meta.get("tag"), model_path.name

        def predict(df: pd.DataFrame) -> np.ndarray:
            return predict_ensemble(members, df)

    if not set(features) <= set(FEATURE_NAMES):
        raise SystemExit("model uses features unknown to shared.features")
    points = load_points("validate")
    feats = build_split("validate")
    if list(feats["sample_id"]) != list(points["sample_id"]):
        raise SystemExit("validate features are not aligned with points.csv")
    pred = predict(feats)
    sub = make_submission(points, pred)
    out = Path(args.out) if args.out else ARTIFACTS / f"submission_{datetime.now():%Y%m%d_%H%M}.csv"
    write_submission(sub, points, out)
    print(f"model={name} tag={tag} rows={len(sub)}")
    print(f"prediction: mean={pred.mean():.1f} median={np.median(pred):.1f}", end=" ")
    print(f"range=[{pred.min():.1f}, {pred.max():.1f}]")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
