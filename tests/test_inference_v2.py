"""ML v2: интерфейс версии с квантилями, ``p_late`` и последовательной моделью (ONNX fp32 / INT8).

Маленькая версия ``tiny2`` собирается в тесте (CatBoost + GRU → ONNX) — нужен extra ``train`` (torch, onnx);
без него эти тесты пропускаются. Тесты настоящей ``v2`` (каталог моделей) пропускаются, если её нет.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from ml.inference import (
    CAPABILITIES,
    EXPECTED_COLUMN,
    MANIFEST_NAME,
    ONNX_OUTPUT,
    ONNX_SEQ_INPUT,
    ONNX_TAB_INPUT,
    OUTPUT_COLUMNS,
    SEQUENCE_COLUMN,
    BundleError,
    FeaturesVersionError,
    calibrate,
    current_features_version,
    load_bundle,
    models_dir,
)
from ml.registry import write_bundle
from shared.sequences import sequence_version

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from ml.seq_model import (  # noqa: E402  (после importorskip: модуль импортирует torch)
    SeedEnsemble,
    SeqConfig,
    export_onnx,
    quantize_int8,
    train_net,
)

FEATS = ["cur_dev_s", "hour", "stop_dur"]
SEQ_L, SEQ_C = 6, 3
W_SEQ = 0.3
CQR_OFFSET, ABS_K = 5.0, 0.4
STACK = {"method": "stack", "a": 0.8, "c": 1.2, "b": -0.3, "pred_scale": 100.0}
CPU = torch.device("cpu")
TINY_CFG = SeqConfig(
    "tiny_gru", kind="gru", hidden=8, tab_hidden=8, max_epochs=3, batch_size=64, seeds=(1, 2)
)


def _data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Признаки, последовательности и задержка: задержка зависит и от признаков, и от последовательности."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cur_dev_s": rng.normal(60.0, 80.0, n),
            "hour": rng.uniform(5.0, 23.0, n),
            "stop_dur": rng.exponential(60.0, n),
        }
    )
    seq = rng.normal(0.0, 1.0, (n, SEQ_L, SEQ_C)).astype(np.float32)
    y = (
        0.8 * df["cur_dev_s"].to_numpy()
        + 0.5 * df["stop_dur"].to_numpy()
        + 30.0 * seq[:, -1, 0]
        + rng.normal(0.0, 10.0, n)
    )
    return df, seq, y


def _cb(params: dict, cls=CatBoostRegressor):
    return cls(**params, iterations=60, depth=3, verbose=False, allow_writing_files=False)


@pytest.fixture(scope="module")
def tiny2(tmp_path_factory) -> dict:
    """``tiny2``: 2 CatBoost задержки, MultiQuantile, Logloss, |ошибка|, GRU 2 сида (ONNX INT8, fp32)."""
    root = tmp_path_factory.mktemp("models")
    src = tmp_path_factory.mktemp("src")
    df, seq, y = _data()
    files, members = {}, []
    base = df["cur_dev_s"].to_numpy()
    for i, b in enumerate([None, "cur_dev_s"]):
        model = _cb({"loss_function": "MAE", "random_seed": i})
        model.fit(df[FEATS], y - (base if b else 0.0))
        files[f"cb{i}.cbm"] = src / f"cb{i}.cbm"
        model.save_model(str(files[f"cb{i}.cbm"]))
        members.append(
            {"name": f"cb{i}", "component": "catboost", "target": "delay", "file": f"cb{i}.cbm", "base": b}
        )
    q = _cb({"loss_function": "MultiQuantile:alpha=0.1,0.5,0.9"})
    q.fit(df[FEATS], y - base)
    q.save_model(str(src / "q.cbm"))
    files["q.cbm"] = src / "q.cbm"
    members.append(
        {
            "name": "quantiles",
            "component": "catboost",
            "target": "quantiles",
            "file": "q.cbm",
            "base": "cur_dev_s",
            "alphas": [0.1, 0.5, 0.9],
        }
    )
    late = _cb({"loss_function": "Logloss"}, CatBoostClassifier)
    late.fit(df[FEATS], (y > 120.0).astype(int))
    late.save_model(str(src / "late.cbm"))
    files["late.cbm"] = src / "late.cbm"
    members.append(
        {
            "name": "p_late",
            "component": "catboost",
            "target": "p_late",
            "file": "late.cbm",
            "threshold_s": 120.0,
            "calibration": STACK,
        }
    )
    err = _cb({"loss_function": "RMSE"})
    err.fit(df[FEATS], np.abs(y - base))
    err.save_model(str(src / "err.cbm"))
    files["err.cbm"] = src / "err.cbm"
    members.append(
        {"name": "abs_error", "component": "catboost", "target": "abs_error", "file": "err.cbm", "base": None}
    )
    tab = df[FEATS].to_numpy(dtype=np.float32)
    nets = [train_net(TINY_CFG, s, seq, tab, y - base, np.ones(len(y)), dev=CPU) for s in TINY_CFG.seeds]
    fused = SeedEnsemble([copy.deepcopy(n).eval() for n in nets])
    export_onnx(fused, src / "seq.fp32.onnx", SEQ_L, SEQ_C, len(FEATS))
    unrolled = SeedEnsemble([n.for_export() for n in nets])
    export_onnx(unrolled, src / "seq.unrolled.onnx", SEQ_L, SEQ_C, len(FEATS))
    quantize_int8(src / "seq.unrolled.onnx", src / "seq.int8.onnx")
    files["seq.fp32.onnx"] = src / "seq.fp32.onnx"
    files["seq.int8.onnx"] = src / "seq.int8.onnx"
    members.append(
        {
            "name": "tiny_gru",
            "component": "gru",
            "target": "delay",
            "file": "seq.int8.onnx",
            "fp32_file": "seq.fp32.onnx",
            "precision": "int8",
            "base": "cur_dev_s",
            "sequence": {
                "len": SEQ_L,
                "step_s": 15.0,
                "channels": ["a", "b", "c"],
                "version": sequence_version(),
            },
        }
    )
    manifest = {
        "version": "tiny2",
        "created_at": "2026-09-26T10:00:00Z",
        "git_commit": "test",
        "features": FEATS,
        "features_version": current_features_version(),
        "sequence_version": sequence_version(),
        "components": ["catboost", "gru"],
        "precision": "int8",
        "capabilities": sorted(CAPABILITIES),
        "late_threshold_s": 120.0,
        "metrics": {"cv_mae": 1.0, "test_mae": 2.0},
        "ensemble": {"method": "weighted", "weights": {"catboost": 1 - W_SEQ, "gru": W_SEQ}},
        "intervals": {"center": "pred", "cqr_offset_s": CQR_OFFSET, "abs_error_k": ABS_K},
        "members": members,
    }
    write_bundle(root / "tiny2", manifest, files)
    probe_df, probe_seq, _ = _data(50, seed=1)
    return {
        "root": root,
        "src": src,
        "nets": nets,
        "cb": [m for m in members if m["target"] == "delay" and m["component"] == "catboost"],
        "df": probe_df,
        "seq": probe_seq,
    }


def _copy(tiny2: dict, tmp_path: Path, edit) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(tiny2["root"] / "tiny2", dst)
    path = dst / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    return dst


def _torch_seq(nets, df: pd.DataFrame, seq: np.ndarray) -> np.ndarray:
    tab = torch.as_tensor(df[FEATS].to_numpy(dtype=np.float32))
    with torch.no_grad():
        out = [n.eval()(torch.as_tensor(seq), tab).numpy().astype(np.float64) for n in nets]
    return np.mean(out, axis=0)


# --- интерфейс ----------------------------------------------------------------------------------------------
def test_v2_attributes_and_info(tiny2: dict) -> None:
    b = load_bundle("tiny2", root=tiny2["root"])
    assert b.capabilities == CAPABILITIES
    assert b.precision == "int8" and b.sequence_shape == (SEQ_L, SEQ_C)
    assert b.components == ["catboost", "gru"] and b.features_version_ok
    info = b.info()
    assert info["precision"] == "int8" and info["capabilities"] == sorted(CAPABILITIES)
    assert info["sequence"] == {"len": SEQ_L, "channels": SEQ_C, "version": sequence_version()}
    fp32 = load_bundle("tiny2", root=tiny2["root"], precision="fp32")
    assert fp32.precision == "fp32"


def test_v2_predict_contract(tiny2: dict) -> None:
    b = load_bundle("tiny2", root=tiny2["root"])
    df, seq = tiny2["df"].set_index(pd.RangeIndex(100, 150)), tiny2["seq"]
    out = b.predict(df, sequences=seq)
    assert list(out.columns) == OUTPUT_COLUMNS and out.index.equals(df.index)
    assert (out.dtypes == np.float64).all() and np.isfinite(out.to_numpy()).all()
    assert ((out["p10"] <= out["p50"]) & (out["p50"] <= out["p90"])).all()
    assert out["p_late"].between(0.0, 1.0).all()
    np.testing.assert_allclose(out["p50"], out["pred_delay_s"])  # center = pred
    assert (out["p90"] - out["p10"] >= 2 * CQR_OFFSET - 1e-9).all()
    err = CatBoostRegressor()
    err.load_model(str(tiny2["root"] / "tiny2" / "err.cbm"))
    np.testing.assert_allclose(out["expected_abs_error_s"], np.maximum(err.predict(df[FEATS]), 0.0))
    assert (out["expected_abs_error_s"] >= 0).all()


def test_v2_expected_error_falls_back_to_interval_width(tiny2: dict, tmp_path: Path) -> None:
    """Без члена ``abs_error`` ожидаемая ошибка — ``k · (p90 − p10)`` из ``intervals.abs_error_k``."""
    path = _copy(
        tiny2, tmp_path, lambda m: m.update(members=[x for x in m["members"] if x["name"] != "abs_error"])
    )
    out = load_bundle(path).predict(tiny2["df"], sequences=tiny2["seq"])
    np.testing.assert_allclose(out["expected_abs_error_s"], ABS_K * (out["p90"] - out["p10"]))


def test_v2_weighted_ensemble_and_fallback_without_sequences(tiny2: dict) -> None:
    """С последовательностями — (1 − w)·CatBoost + w·GRU; без них — только CatBoost (вес нормируется)."""
    b = load_bundle("tiny2", root=tiny2["root"], precision="fp32")
    df, seq = tiny2["df"], tiny2["seq"]
    x = df[FEATS]
    cb = []
    for m in tiny2["cb"]:
        model = CatBoostRegressor()
        model.load_model(str(tiny2["root"] / "tiny2" / m["file"]))
        cb.append(model.predict(x) + (df[m["base"]].to_numpy() if m["base"] else 0.0))
    cb_mean = np.mean(cb, axis=0)
    gru = _torch_seq(tiny2["nets"], df, seq) + df["cur_dev_s"].to_numpy()
    with_seq = b.predict(df, sequences=seq)["pred_delay_s"].to_numpy()
    np.testing.assert_allclose(with_seq, (1 - W_SEQ) * cb_mean + W_SEQ * gru, atol=1e-3)
    np.testing.assert_allclose(b.predict(df)["pred_delay_s"].to_numpy(), cb_mean, atol=1e-9)
    with pytest.raises(ValueError, match="sequences shape"):
        b.predict(df, sequences=seq[:, :-1])


def test_v2_p_late_calibration(tiny2: dict) -> None:
    b = load_bundle("tiny2", root=tiny2["root"])
    late = CatBoostClassifier()
    late.load_model(str(tiny2["root"] / "tiny2" / "late.cbm"))
    raw = late.predict_proba(tiny2["df"][FEATS])[:, 1]
    for seq in (tiny2["seq"], None):
        out = b.predict(tiny2["df"], sequences=seq)
        np.testing.assert_allclose(out["p_late"], calibrate(STACK, raw, out["pred_delay_s"], 120.0))
    platt = calibrate({"method": "platt", "a": 1.0, "b": 0.0}, raw)
    np.testing.assert_allclose(platt, raw, rtol=1e-6)
    stacked = calibrate(STACK, np.full(3, 0.5), np.array([20.0, 120.0, 220.0]), 120.0)
    np.testing.assert_allclose(stacked, 1 / (1 + np.exp(-(np.array([-1.2, 0.0, 1.2]) - 0.3))))
    with pytest.raises(ValueError, match="stack"):
        calibrate(STACK, raw)
    iso = {"method": "isotonic", "x": [0.2, 0.8], "y": [0.1, 0.9]}
    np.testing.assert_allclose(calibrate(iso, np.array([0.0, 0.5, 1.0])), [0.1, 0.5, 0.9])


def test_v2_onnx_matches_torch_and_int8_close(tiny2: dict) -> None:
    """ONNX fp32 (оператор GRU) и развёрнутый граф совпадают с PyTorch; INT8 близок к fp32."""
    import onnxruntime as ort

    df, seq = tiny2["df"], tiny2["seq"]
    feeds = {ONNX_SEQ_INPUT: seq, ONNX_TAB_INPUT: df[FEATS].to_numpy(dtype=np.float32)}
    ref = _torch_seq(tiny2["nets"], df, seq)
    out = {}
    for name in ("seq.fp32.onnx", "seq.unrolled.onnx", "seq.int8.onnx"):
        sess = ort.InferenceSession(str(tiny2["src"] / name), providers=["CPUExecutionProvider"])
        out[name] = sess.run([ONNX_OUTPUT], feeds)[0].astype(np.float64)
    assert np.abs(out["seq.fp32.onnx"] - ref).max() < 1e-3
    assert np.abs(out["seq.unrolled.onnx"] - ref).max() < 1e-3
    assert np.abs(out["seq.int8.onnx"] - ref).max() < 0.1 * np.abs(ref).max() + 1.0
    int8 = load_bundle("tiny2", root=tiny2["root"]).predict(df, sequences=seq)["pred_delay_s"]
    b32 = load_bundle("tiny2", root=tiny2["root"], precision="fp32")
    fp32 = b32.predict(df, sequences=seq)["pred_delay_s"]
    assert 0 < np.abs(int8 - fp32).max() < W_SEQ * (0.1 * np.abs(ref).max() + 1.0)


def test_v2_contributions_include_sequence_and_sum_to_prediction(tiny2: dict) -> None:
    b = load_bundle("tiny2", root=tiny2["root"])
    df, seq = tiny2["df"].head(10), tiny2["seq"][:10]
    for s in (seq, None):
        c = b.contributions(df, sequences=s)
        assert list(c.columns) == [*FEATS, SEQUENCE_COLUMN, EXPECTED_COLUMN]
        pred = b.predict(df, sequences=s)["pred_delay_s"]
        np.testing.assert_allclose(c.sum(axis=1), pred, atol=1e-3)
        assert (c[SEQUENCE_COLUMN] == 0).all() == (s is None)
    factors = b.explain(df, top=2, sequences=seq)
    assert all(len(r) == 2 and {f["feature"] for f in r} <= set(FEATS) for r in factors)


def test_v2_empty_and_missing_features(tiny2: dict) -> None:
    b = load_bundle("tiny2", root=tiny2["root"])
    empty = b.predict(tiny2["df"].iloc[:0], sequences=tiny2["seq"][:0])
    assert empty.empty and list(empty.columns) == OUTPUT_COLUMNS
    out = b.predict(pd.DataFrame([{"hour": 8.0}]), sequences=np.zeros((1, SEQ_L, SEQ_C), np.float32))
    assert np.isfinite(out.to_numpy()).all()


@pytest.mark.parametrize(
    ("edit", "error"),
    [
        (lambda m: m.update(sequence_version="0000000000"), FeaturesVersionError),
        (lambda m: m["ensemble"].update(weights={"catboost": 1.0}), BundleError),
        (lambda m: m["ensemble"].update(weights={"catboost": 0.0, "gru": 1.0}), BundleError),
        (lambda m: m["members"][-1]["sequence"].update(len=SEQ_L + 1), BundleError),
        (lambda m: m["members"][-1].pop("sequence"), BundleError),
        (lambda m: m["members"][2].update(alphas=[0.1, 0.9]), BundleError),
        (lambda m: m["members"][3].update(calibration={"method": "beta"}), BundleError),
        (lambda m: m["members"][3].update(calibration={"method": "stack", "a": 1.0, "b": 0.0}), BundleError),
        (lambda m: m["members"][4].update(file="late.cbm"), BundleError),
        (lambda m: m["members"][-1].update(component="lstm"), BundleError),
    ],
)
def test_v2_refuses_inconsistent_manifest(tiny2: dict, tmp_path: Path, edit, error) -> None:
    with pytest.raises(error):
        load_bundle(_copy(tiny2, tmp_path, edit))


def test_v2_sequence_version_mismatch_loads_non_strict(tiny2: dict, tmp_path: Path) -> None:
    path = _copy(tiny2, tmp_path, lambda m: m.update(sequence_version="0000000000"))
    b = load_bundle(path, strict=False)
    assert not b.features_version_ok


# --- обучение: воспроизводимость и эквивалентность экспорта ---------------------------------------------
def test_train_net_reproducible_by_seed() -> None:
    df, seq, y = _data(200, seed=3)
    tab = df[FEATS].to_numpy(dtype=np.float32)
    args = (seq, tab, y - df["cur_dev_s"].to_numpy(), np.ones(len(y)))
    a = train_net(TINY_CFG, 7, *args, dev=CPU)
    b = train_net(TINY_CFG, 7, *args, dev=CPU)
    c = train_net(TINY_CFG, 8, *args, dev=CPU)
    pa, pb, pc = (_torch_seq([n], df, seq) for n in (a, b, c))
    assert pa.tobytes() == pb.tobytes()
    assert np.abs(pa - pc).max() > 1e-3


def test_unrolled_gru_equals_torch_gru() -> None:
    df, seq, y = _data(64, seed=4)
    net = train_net(TINY_CFG, 1, seq, df[FEATS].to_numpy(np.float32), y, np.ones(len(y)), dev=CPU)
    np.testing.assert_allclose(_torch_seq([net.for_export()], df, seq), _torch_seq([net], df, seq), atol=1e-4)


# --- настоящая v2 (на сервере: artifacts/models/v2) -----------------------------------------------------
V2_DIR = models_dir() / "v2"
needs_v2 = pytest.mark.skipif(not (V2_DIR / MANIFEST_NAME).exists(), reason="нет v2 в каталоге моделей")


@pytest.fixture(scope="module")
def v2():
    return load_bundle("v2")


@pytest.fixture(scope="module")
def validate_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    from ml.dataset import build_sequences_split, build_split

    return build_split("validate"), build_sequences_split("validate")


@needs_v2
def test_real_v2_interface(v2, validate_inputs) -> None:
    feats, seqs = validate_inputs
    assert v2.capabilities == CAPABILITIES and v2.precision == "int8" and v2.features_version_ok
    assert v2.sequence_shape == seqs.shape[1:]
    out = v2.predict(feats, sequences=seqs)
    assert len(out) == 151 and np.isfinite(out.to_numpy()).all()
    assert ((out["p10"] <= out["p50"]) & (out["p50"] <= out["p90"])).all()
    assert out["p_late"].between(0.0, 1.0).all()
    fp32 = load_bundle("v2", precision="fp32").predict(feats, sequences=seqs)
    diff = np.abs(out["pred_delay_s"] - fp32["pred_delay_s"])
    print(f"\nv2 validate: int8 vs fp32 pred_delay_s max |diff| {diff.max():.3f} s, mean {diff.mean():.3f} s")
    assert diff.max() < 5.0
    factors = v2.explain(feats.head(5), top=3, sequences=seqs[:5])
    assert all(len(r) == 3 for r in factors)


def _p50_ms(fn, repeat: int = 15) -> float:
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


@needs_v2
@pytest.mark.parametrize(("n", "predict_ms"), [(1, 100.0), (100, 100.0), (1000, 400.0)])
def test_real_v2_latency_cpu(v2, validate_inputs, n: int, predict_ms: float) -> None:
    """``predict`` (все выходы, GRU INT8 на ONNX Runtime CPU): пакет 100 — быстрее 100 мс.

    Замер имеет смысл только на свободном CPU: если порог превышен, а сервер занят другими процессами
    (load average выше половины ядер), тест пропускается с причиной, а не падает.
    """
    feats, seqs = validate_inputs
    idx = np.random.default_rng(0).integers(0, len(feats), size=n)
    rows, s = feats.iloc[idx].reset_index(drop=True), seqs[idx]
    t = _p50_ms(lambda: v2.predict(rows, sequences=s))
    load = os.getloadavg()[0]
    print(f"\nv2 CPU batch={n}: predict p50 {t:.1f} ms (load average {load:.1f})")
    if t >= predict_ms and load > 0.5 * (os.cpu_count() or 1):
        pytest.skip(f"CPU занят другими процессами (load {load:.1f}): p50 {t:.1f} мс не показателен")
    assert t < predict_ms
