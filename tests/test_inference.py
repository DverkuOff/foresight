"""Библиотека инференса ``ml/inference.py`` (контракт ``docs/api-contract.md`` §4) и реестр версий."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from ml.feature_labels import FEATURE_INFO, FEATURE_LABELS, cause_hint, label
from ml.inference import (
    CAPABILITIES,
    EXPECTED_COLUMN,
    MANIFEST_NAME,
    OUTPUT_COLUMNS,
    BundleError,
    FeaturesVersionError,
    current_features_version,
    features_frame,
    list_bundles,
    load_bundle,
    models_dir,
)
from ml.registry import manifest_from_v1_meta, utc_iso, write_bundle
from shared.features import FEATURE_NAMES

FEATS = ["cur_dev_s", "hour", "stop_dur"]
CAUSES = {"dwell_long", "slow_segment", "layover", "accumulated_delay", "bunching", "gps_lost", "unknown"}


def _train_frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cur_dev_s": rng.normal(30.0, 60.0, n),
            "hour": rng.uniform(5.0, 23.0, n),
            "stop_dur": rng.exponential(60.0, n),
        }
    )
    y = 0.8 * df["cur_dev_s"].to_numpy() + 0.5 * df["stop_dur"].to_numpy() + rng.normal(0.0, 10.0, n)
    return df, y


def _tiny_manifest(members: list[dict], **extra) -> dict:
    return {
        "version": "tiny",
        "created_at": "2026-09-25T11:00:00Z",
        "git_commit": "test",
        "features": FEATS,
        "features_version": current_features_version(),
        "components": ["catboost"],
        "precision": "fp32",
        "capabilities": ["factors"],
        "metrics": {"cv_mae": 1.0, "test_mae": 2.0},
        "ensemble": {"method": "mean"},
        "members": members,
        **extra,
    }


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory) -> Path:
    """Каталог моделей с версией ``tiny``: два члена CatBoost (напрямую и остаток к ``cur_dev_s``)."""
    root = tmp_path_factory.mktemp("models")
    src = tmp_path_factory.mktemp("src")
    df, y = _train_frame()
    files, members = {}, []
    for i, base in enumerate([None, "cur_dev_s"]):
        target = y - (df[base].to_numpy() if base else 0.0)
        params = {"iterations": 40, "depth": 3, "loss_function": "MAE", "random_seed": i}
        model = CatBoostRegressor(**params, verbose=False, allow_writing_files=False)
        model.fit(df[FEATS], target)
        name = f"m{i}.cbm"
        model.save_model(str(src / name))
        files[name] = src / name
        members.append(
            {"name": f"m{i}", "component": "catboost", "target": "delay", "file": name, "base": base}
        )
    write_bundle(root / "tiny", _tiny_manifest(members), files)
    return root


def _copy(tiny_root: Path, tmp_path: Path, edit) -> Path:
    """Копия версии ``tiny`` с правкой манифеста ``edit(manifest)``."""
    dst = tmp_path / "copy"
    shutil.copytree(tiny_root / "tiny", dst)
    path = dst / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    return dst


# --- подписи признаков -----------------------------------------------------------------------------------
def test_feature_labels_cover_all_features() -> None:
    assert set(FEATURE_NAMES) <= set(FEATURE_INFO)
    assert set(FEATURE_LABELS) == set(FEATURE_INFO)
    for name, info in FEATURE_INFO.items():
        assert info.label and info.label != name
        assert info.cause is None or info.cause in CAUSES
    assert label("stop_dur") == "Длительность текущей стоянки"
    assert label("no_such_feature") == "no_such_feature"
    assert cause_hint("stop_dur") == "dwell_long"
    assert cause_hint("no_such_feature") is None


# --- загрузка и интерфейс --------------------------------------------------------------------------------
def test_load_by_version_path_manifest_latest_env(tiny_root: Path, monkeypatch) -> None:
    for ref in ("tiny", "latest"):
        assert load_bundle(ref, root=tiny_root).version == "tiny"
        assert load_bundle(ref, root=str(tiny_root)).version == "tiny"  # каталог моделей строкой
    assert [m["version"] for m in list_bundles(str(tiny_root))] == ["tiny"]
    assert load_bundle(tiny_root / "tiny").version == "tiny"
    assert load_bundle(str(tiny_root / "tiny" / MANIFEST_NAME)).version == "tiny"
    monkeypatch.setenv("FORESIGHT_MODELS_DIR", str(tiny_root))
    assert models_dir() == tiny_root
    assert load_bundle("tiny").version == "tiny"
    assert [m["version"] for m in list_bundles()] == ["tiny"]


def test_missing_version(tiny_root: Path) -> None:
    with pytest.raises(BundleError):
        load_bundle("nope", root=tiny_root)


def test_bundle_attributes_and_info(tiny_root: Path) -> None:
    b = load_bundle("tiny", root=tiny_root)
    assert b.features == FEATS
    assert b.precision == "fp32"
    assert b.capabilities == frozenset({"factors"}) and b.capabilities <= CAPABILITIES
    assert b.components == ["catboost"]
    assert b.features_version_ok
    info = b.info()
    # поля GET /model/info (docs/api-contract.md §3)
    for key in ("version", "created_at", "features", "metrics", "precision", "components"):
        assert key in info
    assert {"cv_mae", "test_mae"} <= set(info["metrics"])


def test_predict_contract(tiny_root: Path) -> None:
    b = load_bundle("tiny", root=tiny_root)
    df, _ = _train_frame(50, seed=1)
    df.index = pd.RangeIndex(100, 150)
    out = b.predict(df)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.index.equals(df.index)
    assert (out.dtypes == np.float64).all()
    assert np.isfinite(out["pred_delay_s"]).all()
    # v1-подобная версия без квантилей и p_late: остальные колонки — NaN (null в ml-service)
    assert out[OUTPUT_COLUMNS[1:]].isna().all().all()
    # прогноз = среднее членов (второй — остаток к cur_dev_s)
    m0, m1 = (m.model for m in b._members)
    manual = (m0.predict(df[FEATS]) + m1.predict(df[FEATS]) + df["cur_dev_s"].to_numpy()) / 2.0
    np.testing.assert_allclose(out["pred_delay_s"].to_numpy(), manual, rtol=0, atol=1e-9)
    # порядок колонок и лишние колонки не важны; sequences у версии без sequence игнорируются
    shuffled = df[FEATS[::-1]].assign(extra=1.0)
    again = b.predict(shuffled, sequences=[None] * 50)
    np.testing.assert_array_equal(again["pred_delay_s"], out["pred_delay_s"])


def test_predict_missing_features_and_rows(tiny_root: Path) -> None:
    b = load_bundle("tiny", root=tiny_root)
    rows = [{"cur_dev_s": 60.0, "hour": 8.5, "stop_dur": None}, {"hour": 9.0}, {}]
    frame = features_frame(rows, b.features)
    assert list(frame.columns) == FEATS and frame.isna().sum().sum() == 6
    out = b.predict(frame)
    assert np.isfinite(out["pred_delay_s"]).all()
    # отсутствующая колонка = NaN, отсутствующая база остатка = 0
    no_col = b.predict(pd.DataFrame({"hour": [9.0]}))
    np.testing.assert_array_equal(no_col["pred_delay_s"], out["pred_delay_s"].iloc[[1]])
    empty = b.predict(features_frame([], b.features))
    assert list(empty.columns) == OUTPUT_COLUMNS and len(empty) == 0
    assert b.explain(features_frame([], b.features)) == []


def test_explain_top_factors_and_additivity(tiny_root: Path) -> None:
    b = load_bundle("tiny", root=tiny_root)
    df, _ = _train_frame(20, seed=2)
    pred = b.predict(df)["pred_delay_s"].to_numpy()
    contrib = b.contributions(df)
    assert list(contrib.columns) == [*FEATS, EXPECTED_COLUMN]
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, rtol=0, atol=1e-6)
    exact = b.contributions(df, shap_calc_type="Regular")
    np.testing.assert_allclose(exact.sum(axis=1).to_numpy(), pred, rtol=0, atol=1e-6)
    factors = b.explain(df, top=3)
    assert len(factors) == len(df)
    for row, c in zip(factors, contrib[FEATS].to_numpy(), strict=True):
        assert [f["feature"] for f in row] == [FEATS[j] for j in np.argsort(-np.abs(c), kind="stable")]
        assert all(set(f) == {"feature", "label", "contribution_s"} for f in row)
        assert all(f["label"] == label(f["feature"]) for f in row)
        abs_c = [abs(f["contribution_s"]) for f in row]
        assert abs_c == sorted(abs_c, reverse=True)
    assert all(len(r) == 1 for r in b.explain(df, top=1))
    assert all(len(r) == len(FEATS) for r in b.explain(df, top=10))


# --- отказы -----------------------------------------------------------------------------------------------
def test_refuses_features_version_mismatch(tiny_root: Path, tmp_path: Path) -> None:
    def edit(m: dict) -> None:
        m["features_version"] = "0000000000"

    d = _copy(tiny_root, tmp_path, edit)
    with pytest.raises(FeaturesVersionError, match="features code changed"):
        load_bundle(d)
    b = load_bundle(d, strict=False)
    assert not b.features_version_ok
    assert np.isfinite(b.predict(_train_frame(5)[0])["pred_delay_s"]).all()


def test_refuses_corrupted_file(tiny_root: Path, tmp_path: Path) -> None:
    def edit(m: dict) -> None:
        m["files"]["m0.cbm"]["sha256"] = "0" * 64

    d = _copy(tiny_root, tmp_path, edit)
    with pytest.raises(BundleError, match="sha256"):
        load_bundle(d)
    assert load_bundle(d, verify_hashes=False).version == "tiny"


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["members"][0].update(component="gru"),
        lambda m: m["members"][0].update(target="quantiles"),
        lambda m: m.update(capabilities=["factors", "teleport"]),
        lambda m: m.update(precision="fp8"),
        lambda m: m.update(ensemble={"method": "stacking"}),
        lambda m: m.update(format="other/1"),
        lambda m: m.update(features=["hour", "cur_dev_s", "stop_dur"]),
        lambda m: m["members"][1].update(base="dev_1"),
    ],
)
def test_refuses_unsupported_manifest(tiny_root: Path, tmp_path: Path, edit) -> None:
    with pytest.raises(BundleError):
        load_bundle(_copy(tiny_root, tmp_path, edit))


def test_write_bundle_refuses_overwrite(tiny_root: Path) -> None:
    with pytest.raises(FileExistsError):
        write_bundle(tiny_root / "tiny", _tiny_manifest([]), {})


# --- упаковка v1 -------------------------------------------------------------------------------------------
def test_manifest_from_v1_meta() -> None:
    meta = {
        "model": "model_v1",
        "created": "2026-09-25T14:04:07+03:00",
        "tag": "1bb1499",
        "features_version": "06699d49a9",
        "features": FEATS,
        "seeds": [1, 2],
        "members": [
            {"file": f"model_v1_{i}.cbm", "config": {"name": n, "synth_weight": w, "base": b}}
            for i, (n, w, b) in enumerate(
                [("real_only", 0.0, None)] * 2 + [("synth_w0.5_resid_cur", 0.5, "cur_dev_s")] * 2
            )
        ],
        "results": {"ensemble": {"cv_real": 76.956, "cv_synth": 79.374, "test": 75.79}},
        "baselines": {"cur_dev_s_cv_real": 86.8, "cur_dev_s_test": 93.4, "zero_test": 103.3},
        "train_rows": 2802,
    }
    m = manifest_from_v1_meta(meta, "v1", {"platform_score": 0.87636})
    assert m["version"] == "v1" and m["created_at"] == "2026-09-25T11:04:07Z"
    assert m["capabilities"] == ["factors"] and m["precision"] == "fp32" and m["components"] == ["catboost"]
    assert m["metrics"]["cv_mae"] == 76.956 and m["metrics"]["test_mae"] == 75.79
    assert m["metrics"]["platform_score"] == 0.87636
    assert [x["seed"] for x in m["members"]] == [1, 2, 1, 2]
    assert [x["base"] for x in m["members"]] == [None, None, "cur_dev_s", "cur_dev_s"]
    assert m["git_commit"] == "1bb1499" and m["features_version"] == "06699d49a9"
    assert utc_iso("2026-01-06T08:00:00+00:00") == "2026-01-06T08:00:00Z"


# --- v1 на реальных данных (на сервере: artifacts/models/v1 и artifacts/model_v1.json) ---------------------
V1_DIR = models_dir() / "v1"
HAS_V1 = (V1_DIR / MANIFEST_NAME).exists()
needs_v1 = pytest.mark.skipif(not HAS_V1, reason="нет версии v1 в каталоге моделей")


def _legacy_path(env: str, name: str) -> Path:
    from ml.dataset import ARTIFACTS

    return Path(os.environ.get(env) or ARTIFACTS / name)


@pytest.fixture(scope="module")
def v1():
    return load_bundle("v1")


@pytest.fixture(scope="module")
def validate_features() -> pd.DataFrame:
    from ml.dataset import build_split

    return build_split("validate")


@needs_v1
def test_v1_matches_submit_bitwise(v1, validate_features: pd.DataFrame) -> None:
    """Прогноз библиотеки на validate совпадает с ``ml/submit.py`` (ансамбль ml.train) бит в бит."""
    from ml.train import load_ensemble, predict_ensemble

    meta = _legacy_path("FORESIGHT_V1_META", "model_v1.json")
    if not meta.exists():
        pytest.skip(f"нет {meta}")
    members, _ = load_ensemble(meta)
    expected = predict_ensemble(members, validate_features)
    got = v1.predict(validate_features)["pred_delay_s"].to_numpy()
    assert len(got) == 151
    assert got.tobytes() == expected.tobytes()


@needs_v1
def test_v1_reproduces_submitted_file(v1, validate_features: pd.DataFrame) -> None:
    """Прогноз совпадает с загруженным на платформу сабмитом (score 0.87636) до 3 знаков файла."""
    path = _legacy_path("FORESIGHT_V1_SUBMISSION", "submission_20260925_1404.csv")
    if not path.exists():
        pytest.skip(f"нет {path}")
    sub = pd.read_csv(path, sep=";", dtype={"sample_id": str})
    assert list(sub["sample_id"]) == list(validate_features["sample_id"].astype(str))
    got = v1.predict(validate_features)["pred_delay_s"].to_numpy()
    assert np.abs(got - sub["prediction"].to_numpy()).max() <= 0.0005 + 1e-9


@needs_v1
def test_v1_capabilities_and_factors(v1, validate_features: pd.DataFrame) -> None:
    assert v1.version == "v1" and v1.capabilities == frozenset({"factors"})
    assert v1.features_version_ok and len(v1.features) == 54
    out = v1.predict(validate_features)
    assert out[["p10", "p50", "p90", "p_late", "expected_abs_error_s"]].isna().all().all()
    factors = v1.explain(validate_features.head(5), top=3)
    assert all(len(r) == 3 for r in factors)
    assert all(f["label"] != f["feature"] for r in factors for f in r)
    contrib = v1.contributions(validate_features.head(20))
    np.testing.assert_allclose(contrib.sum(axis=1), out["pred_delay_s"].head(20), rtol=0, atol=1e-6)


def _p50_ms(fn, repeat: int = 15) -> float:
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@needs_v1
@pytest.mark.parametrize(("n", "predict_ms", "explain_ms"), [(100, 100.0, 250.0), (1000, 250.0, 1000.0)])
def test_v1_latency_cpu(v1, validate_features: pd.DataFrame, n: int, predict_ms: float, explain_ms: float):
    """Латентность на CPU (замер печатается; пороги — с запасом на параллельную нагрузку сервера)."""
    rows = validate_features.sample(n=n, replace=True, random_state=0).reset_index(drop=True)
    t_pred = _p50_ms(lambda: v1.predict(rows))
    t_expl = _p50_ms(lambda: v1.explain(rows, top=3), repeat=3)
    print(f"\nv1 CPU batch={n}: predict p50 {t_pred:.1f} ms, explain p50 {t_expl:.1f} ms")
    assert t_pred < predict_ms
    assert t_expl < explain_ms
