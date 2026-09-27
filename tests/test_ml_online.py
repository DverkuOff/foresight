"""Retraining on the stream (``ml/online.py``) and ``POST /model/retrain`` of ml-service: the correction of
the active version by its closed forecasts becomes a new version that serves ``a + b · forecast``."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("catboost")

from ml.online import MIN_ROWS, RetrainError, fit_l1, fit_online, next_version  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
ROW = {"hour": 8.5, "lead_s": 700.0, "cur_dev_s": 120.0, "dev_1": 90.0, "cur_best": 100.0, "stop_dur": 30.0}


def test_fit_l1_finds_the_correction_despite_outliers() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 300, 2000)
    y = 12.0 + 1.1 * x + rng.laplace(0, 20, len(x))
    y[:100] += 2000  # gross outliers: the median regression ignores them
    a, b = fit_l1(x, y)
    assert a == pytest.approx(12.0, abs=4.0) and b == pytest.approx(1.1, abs=0.03)


def test_fit_online_checks_on_the_later_part_and_needs_enough_rows() -> None:
    rng = np.random.default_rng(1)
    pred = rng.uniform(-60, 240, 1000)
    actual = 20.0 + 1.2 * pred + rng.normal(0, 15, len(pred))
    fit = fit_online(pred, actual)
    assert (fit.n_fit, fit.n_eval) == (700, 300)
    assert fit.mae_after_s < fit.mae_before_s
    with pytest.raises(RetrainError, match="мало закрытых прогнозов"):
        fit_online(pred[: MIN_ROWS - 1], actual[: MIN_ROWS - 1])


def test_next_version_counts_up(tmp_path: Path) -> None:
    assert next_version(tmp_path, "v1") == "v1-online1"
    (tmp_path / "v1-online1").mkdir()
    (tmp_path / "v1-online3").mkdir()
    assert next_version(tmp_path, "v1") == "v1-online4"


def test_retrain_endpoint_registers_a_corrected_version(tmp_path: Path) -> None:
    if not (MODELS / "v1" / "manifest.json").is_file():
        pytest.skip("models/v1 is not in the working tree")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ml.inference import features_frame, load_bundle
    from ml.service import ServiceSettings, create_app

    shutil.copytree(MODELS / "v1", tmp_path / "v1")
    settings = ServiceSettings(models_dir=str(tmp_path), version="v1", database_url="")
    rng = np.random.default_rng(2)
    pred = rng.uniform(-60, 240, 1000)
    closed = np.column_stack([pred, 30.0 + 1.1 * pred + rng.normal(0, 10, len(pred))])

    async def fetch_closed(version: str) -> np.ndarray:
        assert version == "v1"
        return closed

    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/model/retrain").json()["state"] == "idle"
        app.state.fetch_closed = fetch_closed
        started = client.post("/model/retrain")
        assert started.status_code == 202 and started.json()["base_version"] == "v1"
        deadline = time.monotonic() + 30
        run = started.json()
        while run["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
            run = client.get("/model/retrain").json()
        assert run["state"] == "done", run
        assert run["version"] == "v1-online1" and run["mae_after_s"] < run["mae_before_s"]
        versions = {v["version"]: v for v in client.get("/model/versions").json()}
        assert versions["v1-online1"]["online"]["base"] == "v1" and versions["v1"]["active"]
        assert versions["v1-online1"]["cv_mae"] is None  # CV / test measured the base, not the correction
        assert client.get("/health").json()["model_version"] == "v1"  # not activated by itself

        # a correction learnt on the earlier forecasts that hurts on the later ones is not offered
        worse = np.column_stack([pred, pred + np.where(np.arange(len(pred)) < 700, 60.0, 0.0)])

        async def fetch_worse(version: str) -> np.ndarray:
            return worse

        app.state.fetch_closed = fetch_worse
        run = client.post("/model/retrain").json()
        while run["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
            run = client.get("/model/retrain").json()
        assert (run["state"], run["improved"], run["version"]) == ("done", False, None), run
        assert run["mae_after_s"] >= run["mae_before_s"]
        assert "v1-online2" not in {v["version"] for v in client.get("/model/versions").json()}
        base = client.post("/predict", json={"rows": [{"row_id": "a", "features": ROW}]}).json()
        assert client.post("/model/reload", json={"version": "v1-online1"}).status_code == 200
        new = client.post("/predict", json={"rows": [{"row_id": "a", "features": ROW}]}).json()
    shift = load_bundle(tmp_path / "v1-online1").online_shift
    assert shift is not None
    a, b = shift
    assert new["predictions"][0]["pred_delay_s"] == pytest.approx(
        a + b * base["predictions"][0]["pred_delay_s"]
    )
    features = load_bundle(tmp_path / "v1").features
    corrected = load_bundle(tmp_path / "v1-online1").predict(features_frame([ROW], features))
    assert corrected["pred_delay_s"].iloc[0] == pytest.approx(new["predictions"][0]["pred_delay_s"])
