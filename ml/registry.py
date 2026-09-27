"""Реестр версий моделей: упаковка, список, проверка и замер латентности.

Версия — каталог ``<models_dir>/<version>/`` с ``manifest.json`` и файлами моделей (формат —
:mod:`ml.inference`). Запуск (на сервере)::

    uv run python -m ml.registry pack --meta artifacts/model_v1.json --version v1 \\
        --metric platform_score=0.87636          # ансамбль v1 (ml/train.py) → artifacts/models/v1
    uv run python -m ml.registry list
    uv run python -m ml.registry verify v1
    uv run python -m ml.registry bench v1 --sizes 1 100 1000 --write

Каталог моделей — ``--root`` или ``$FORESIGHT_MODELS_DIR`` (по умолчанию ``artifacts/models``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.inference import (
    CAP_FACTORS,
    FORMAT,
    LATE_THRESHOLD_S,
    MANIFEST_NAME,
    BundleError,
    file_sha256,
    list_bundles,
    load_bundle,
    models_dir,
)


def utc_iso(value: str | None = None) -> str:
    """Момент в ISO 8601 UTC (``...Z``); наивная метка считается локальным временем машины."""
    ts = datetime.fromisoformat(value) if value else datetime.now(UTC)
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    """Короткий хэш HEAD репозитория (пусто, если git недоступен)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def write_bundle(
    bundle_dir: Path, manifest: dict[str, Any], files: dict[str, Path], overwrite: bool = False
) -> Path:
    """Записать версию: скопировать файлы моделей, посчитать хэши и сохранить ``manifest.json``.

    Args:
        bundle_dir: каталог версии (создаётся).
        manifest: манифест без ``files`` и ``format`` (заполняются здесь).
        files: ``{имя в каталоге версии: исходный файл}``.
        overwrite: разрешить перезапись существующей версии.

    Returns:
        Путь к ``manifest.json``.

    Raises:
        FileExistsError: версия уже есть, а ``overwrite`` не задан.
    """
    if (bundle_dir / MANIFEST_NAME).exists() and not overwrite:
        raise FileExistsError(f"{bundle_dir} already exists; pass --force to overwrite")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    listed = {}
    for name, src in files.items():
        dst = bundle_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        listed[name] = {"sha256": file_sha256(dst), "bytes": dst.stat().st_size}
    full = {"format": FORMAT, **manifest, "files": listed}
    path = bundle_dir / MANIFEST_NAME
    path.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def manifest_from_v1_meta(meta: dict[str, Any], version: str, extra_metrics: dict[str, float]) -> dict:
    """Манифест для ансамбля формата ``ml/train.py`` (``model_v1.json``): CatBoost MAE, среднее членов."""
    seeds = list(meta.get("seeds", []))
    members = []
    for i, item in enumerate(meta["members"]):
        cfg = item["config"]
        seed = seeds[i % len(seeds)] if seeds else None  # ml.train.fit_ensemble: варианты × сиды
        members.append(
            {
                "name": f"{cfg['name']}_s{seed}" if seed is not None else cfg["name"],
                "component": "catboost",
                "target": "delay",
                "file": item["file"],
                "base": cfg.get("base"),
                "synth_weight": cfg.get("synth_weight"),
                "seed": seed,
            }
        )
    ens = meta.get("results", {}).get("ensemble", {})
    base = meta.get("baselines", {})
    metrics = {
        "cv_mae": ens.get("cv_real"),
        "cv_mae_synth": ens.get("cv_synth"),
        "test_mae": ens.get("test"),
        "baseline_cur_dev_cv_mae": base.get("cur_dev_s_cv_real"),
        "baseline_cur_dev_test_mae": base.get("cur_dev_s_test"),
        "baseline_zero_test_mae": base.get("zero_test"),
        **extra_metrics,
    }
    return {
        "version": version,
        "created_at": utc_iso(meta.get("created")),
        "packed_at": utc_iso(),
        "git_commit": meta.get("tag", ""),
        "packed_commit": git_head(),
        "model": "catboost_mae_ensemble",
        "description": "CatBoost MAE: 2 варианта × 2 сида (реальные ТС; остаток к cur_dev_s), среднее",
        "features": list(meta["features"]),
        "features_version": meta["features_version"],
        "components": ["catboost"],
        "precision": "fp32",
        "capabilities": [CAP_FACTORS],
        "late_threshold_s": LATE_THRESHOLD_S,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items() if v is not None},
        "ensemble": {"method": "mean"},
        "members": members,
        "train": {
            "rows": meta.get("train_rows"),
            "dropped_synthetic_rows": meta.get("dropped_synthetic_rows"),
            "params": meta.get("params"),
            "source_meta": meta.get("model"),
        },
    }


def parse_metrics(items: list[str]) -> dict[str, float]:
    """``["name=1.5", ...]`` → ``{"name": 1.5}``."""
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"bad --metric {item!r}, expected name=value")
        out[key] = float(value)
    return out


def cmd_pack(args: argparse.Namespace) -> None:
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = manifest_from_v1_meta(meta, args.version, parse_metrics(args.metric))
    files = {m["file"]: meta_path.parent / m["file"] for m in manifest["members"]}
    root = Path(args.root) if args.root else models_dir()
    path = write_bundle(root / args.version, manifest, files, overwrite=args.force)
    bundle = load_bundle(path.parent, strict=not args.no_strict)
    print(f"packed {bundle!r} → {path}")


def cmd_list(args: argparse.Namespace) -> None:
    root = Path(args.root) if args.root else models_dir()
    bundles = list_bundles(root)
    if not bundles:
        print(f"no model versions in {root}")
        return
    head = ("version", "created_at", "commit", "precision", "cv_mae", "test_mae")
    print("{:10s} {:21s} {:9s} {:9s} {:>7s} {:>8s}  caps".format(*head))
    for m in bundles:
        met = m.get("metrics", {})
        print(
            f"{m['version']:10s} {m.get('created_at', ''):21s} {m.get('git_commit', ''):9s} "
            f"{m.get('precision', ''):9s} {met.get('cv_mae', float('nan')):7.2f} "
            f"{met.get('test_mae', float('nan')):8.2f}  {','.join(m.get('capabilities', []))}"
        )


def cmd_verify(args: argparse.Namespace) -> None:
    root = Path(args.root) if args.root else None
    bundle = load_bundle(args.version, root=root, strict=not args.no_strict)
    probe = pd.DataFrame([dict.fromkeys(bundle.features, np.nan)])
    seq = np.zeros((1, *bundle.sequence_shape), dtype=np.float32) if bundle.sequence_shape else None
    out = bundle.predict(probe, sequences=seq)
    if not np.isfinite(out["pred_delay_s"]).all():
        raise SystemExit("prediction on an all-NaN row is not finite")
    produced = [c for c in out.columns if np.isfinite(out[c]).all()]
    print(
        f"ok {bundle!r} features_version={bundle.features_version} ok={bundle.features_version_ok} "
        f"outputs={produced}"
    )


def bench_rows(n: int, pool: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """``n`` строк признаков: выборка с возвращением из ``pool``."""
    return pool.iloc[bench_index(n, len(pool), seed)].reset_index(drop=True)


def bench_index(n: int, size: int, seed: int = 0) -> np.ndarray:
    """Индексы выборки с возвращением (одни и те же для признаков и последовательностей)."""
    return np.random.default_rng(seed).integers(0, size, size=n)


def measure(fn, repeat: int) -> dict[str, float]:
    """Латентность вызова ``fn()``: p50/p95/max, мс (первый вызов — прогрев, не учитывается)."""
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def benchmark(
    bundle, sizes: list[int], repeat: int, pool: pd.DataFrame, seq_pool: np.ndarray | None = None
) -> dict[str, Any]:
    """Замер ``predict`` и ``explain`` на CPU для пакетов размера ``sizes``.

    ``seq_pool`` — последовательности строк ``pool`` (для версий с ``sequence``); у таких версий ``predict``
    меряется с последовательностями и без них (``predict_no_seq``).
    """
    out: dict[str, Any] = {}
    for n in sizes:
        idx = bench_index(n, len(pool))
        rows = pool.iloc[idx].reset_index(drop=True)
        seqs = seq_pool[idx] if seq_pool is not None and bundle.sequence_shape else None
        res = {
            "predict": measure(lambda rows=rows, seqs=seqs: bundle.predict(rows, sequences=seqs), repeat),
            "explain": measure(lambda rows=rows: bundle.explain(rows, top=3), max(3, repeat // 4)),
        }
        if seqs is not None:
            res["predict_no_seq"] = measure(lambda rows=rows: bundle.predict(rows), repeat)
        out[str(n)] = res
        print(f"batch {n:5d}: " + "  ".join(f"{k} {v}" for k, v in res.items()), flush=True)
    return out


def cmd_bench(args: argparse.Namespace) -> None:
    from ml.dataset import build_sequences_split, build_split

    root = Path(args.root) if args.root else None
    bundle = load_bundle(args.version, root=root, strict=not args.no_strict, precision=args.precision)
    pool = build_split("validate")[bundle.features]
    seq_pool = build_sequences_split("validate") if bundle.sequence_shape else None
    result = {
        "measured_at": utc_iso(),
        "host": os.uname().nodename,
        "cpus": os.cpu_count(),
        "repeat": args.repeat,
        "precision": bundle.precision,
        "batches": benchmark(bundle, args.sizes, args.repeat, pool, seq_pool),
    }
    if args.write:
        path = bundle.path / MANIFEST_NAME
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["latency_cpu"] = result
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"written to {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="", help="каталог моделей (по умолчанию $FORESIGHT_MODELS_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="упаковать ансамбль ml/train.py (model_v1.json) в версию")
    p.add_argument("--meta", required=True, help="метаданные ансамбля (model_v1.json)")
    p.add_argument("--version", required=True)
    p.add_argument("--metric", action="append", default=[], help="доп. метрика name=value (повторяемый)")
    p.add_argument("--force", action="store_true", help="перезаписать существующую версию")
    p.add_argument("--no-strict", action="store_true", help="не проверять версию кода признаков")
    p.set_defaults(fn=cmd_pack)

    p = sub.add_parser("list", help="версии в каталоге моделей")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("verify", help="проверить хэши, версию признаков и прогноз")
    p.add_argument("version")
    p.add_argument("--no-strict", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("bench", help="латентность predict / explain на CPU")
    p.add_argument("version")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 100, 1000])
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--write", action="store_true", help="записать результат в manifest.json (latency_cpu)")
    p.add_argument(
        "--precision", default=None, choices=["fp32", "int8"], help="точность последовательной модели"
    )
    p.add_argument("--no-strict", action="store_true")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except (BundleError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
