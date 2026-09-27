"""ml-service (``ml/service.py``, API contract §3) with the model of the repository, and its client in the
predictor (``backend/mlclient.py``): timeout, breaker, recovery."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.mlclient import HttpError, MLClient, decode_result, encode_rows  # noqa: E402
from backend.runtime import DependencyStatus  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
REAL_DB_URL = os.environ.get("FORESIGHT_TEST_DATABASE_URL")

ROW = {"hour": 8.5, "lead_s": 700.0, "cur_dev_s": 120.0, "dev_1": 90.0, "cur_best": 100.0, "stop_dur": 30.0}


def _needs_model() -> None:
    pytest.importorskip("catboost")
    if not (MODELS / "v1" / "manifest.json").is_file():
        pytest.skip("models/v1 is not in the working tree")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    _needs_model()
    from ml.service import ServiceSettings, create_app

    settings = ServiceSettings(models_dir=str(MODELS), version="v1", database_url="")
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_model_info(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok" and body["model_version"] == "v1" and body["database"] == "disabled"
    info = client.get("/model/info").json()
    assert info["version"] == "v1" and len(info["features"]) == 54 and info["precision"] == "fp32"
    assert info["components"] == ["catboost"]
    assert info["metrics"]["test_mae"] == pytest.approx(75.79, abs=0.01)
    assert info["registered"] is False
    assert client.get("/openapi.json").json()["info"]["title"] == "Foresight · ML"
    # the predictor learns from /health whether to build telemetry sequences (v1: no sequence input)
    assert body["capabilities"] == ["factors"] and body["sequence"] is None and info["sequence"] is None


def test_predict_matches_the_inference_library(client: TestClient) -> None:
    from ml.inference import features_frame

    rows = [{"row_id": "a", "features": ROW}, {"row_id": 7, "features": {"hour": None}}]
    resp = client.post("/predict", json={"rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v1" and body["precision"] == "fp32" and body["latency_ms"] > 0
    preds = body["predictions"]
    assert [p["row_id"] for p in preds] == ["a", 7]
    bundle = client.app.state.model.bundle  # type: ignore[attr-defined]
    expected = bundle.predict(features_frame([ROW, {}], bundle.features))["pred_delay_s"]
    for p, e in zip(preds, expected, strict=True):
        assert p["pred_delay_s"] == pytest.approx(e) and math.isfinite(p["pred_delay_s"])
        # v1 has no quantiles and no p_late: null, the predictor then uses pred_delay_s only
        assert p["p10"] is None and p["p50"] is None and p["p90"] is None and p["p_late"] is None
        assert 1 <= len(p["factors"]) <= 3 and all(f["label"] for f in p["factors"])
    # the request of the predictor's client and its reading of the answer
    body = encode_rows([ROW, {"hour": math.nan}], explain=True)
    sent = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    result = decode_result(sent.content, 2)
    assert result.model_version == "v1" and result.predictions[0].factors
    assert result.predictions[1].pred_delay_s == pytest.approx(expected.iloc[1])
    no_factors = client.post("/predict", json={"rows": rows[:1], "explain": False}).json()
    assert no_factors["predictions"][0]["factors"] == []
    assert client.post("/predict", json={"rows": []}).json()["predictions"] == []


def test_predict_limits(client: TestClient) -> None:
    rows = [{"row_id": i, "features": {}} for i in range(1001)]
    assert client.post("/predict", json={"rows": rows}).status_code == 413
    bad = client.post("/predict", json={"rows": [{"row_id": 1, "features": {"hour": "noon"}}]})
    assert bad.status_code == 422


def test_metrics_of_the_contract(client: TestClient) -> None:
    client.post("/predict", json={"rows": [{"row_id": 1, "features": ROW}] * 5})
    client.get("/model/info")
    text = client.get("/metrics").text
    assert 'foresight_ml_request_duration_seconds_bucket{endpoint="/predict",le="0.001"}' in text
    for endpoint in ("/predict", "/model/info", "/model/reload", "/health"):  # every series from the start
        assert f'foresight_ml_request_duration_seconds_count{{endpoint="{endpoint}"}}' in text
    assert 'foresight_ml_batch_size_bucket{le="5.0"}' in text
    assert 'foresight_ml_model_info{model="catboost_mae_ensemble",precision="fp32",version="v1"} 1.0' in text
    assert text.count("foresight_ml_model_info{") == 1
    assert "/metrics" not in {line.split('"')[1] for line in text.splitlines() if 'endpoint="' in line}


def test_reload(client: TestClient) -> None:
    missing = client.post("/model/reload", json={"version": "v999"})
    assert missing.status_code == 404
    assert client.get("/health").json()["model_version"] == "v1"  # the loaded version keeps serving
    again = client.post("/model/reload", json={"version": "v1"})
    assert again.status_code == 200 and again.json()["version"] == "v1"
    assert client.post("/model/reload").status_code == 200


def test_health_is_503_until_a_model_is_loaded() -> None:
    _needs_model()
    from ml.service import ServiceSettings, create_app

    settings = ServiceSettings(models_dir=str(MODELS / "nowhere"), version="latest", database_url="")
    with TestClient(create_app(settings)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "error"
        assert c.post("/predict", json={"rows": []}).status_code == 503
    with TestClient(create_app(settings, load=False)) as c:
        assert c.get("/health").json()["status"] == "loading"


def test_registry_row() -> None:
    _needs_model()
    from ml.inference import load_bundle
    from ml.service import registry_row

    bundle = load_bundle(MODELS / "v1")
    row = registry_row(bundle)
    assert row[0] == "v1" and row[1] == "catboost"
    assert row[2] == pytest.approx(76.956, abs=0.01) and row[3] == pytest.approx(75.79, abs=0.01)
    assert json.loads(row[6])["platform_score"] == pytest.approx(0.87636)
    assert row[8] is not None and row[8].tzinfo is not None


@pytest.mark.skipif(not REAL_DB_URL, reason="FORESIGHT_TEST_DATABASE_URL is not set")
def test_registration_in_postgres() -> None:
    _needs_model()
    asyncpg = pytest.importorskip("asyncpg")
    from backend.db import Database
    from ml.inference import load_bundle
    from ml.service import register

    async def scenario() -> None:
        db = Database(REAL_DB_URL)  # type: ignore[arg-type]
        await db.connect()  # the schema (model_versions) comes from the backend
        await db.close()
        await register(REAL_DB_URL, load_bundle(MODELS / "v1"))  # type: ignore[arg-type]
        con = await asyncpg.connect(REAL_DB_URL)
        try:
            rows = await con.fetch("SELECT version, active, test_mae FROM model_versions WHERE active")
        finally:
            await con.close()
        assert [(r["version"], r["active"]) for r in rows] == [("v1", True)]

    asyncio.run(scenario())


# ---- the client of the predictor ---------------------------------------------------------------------


class FakeMLService:
    """A tiny HTTP server: ``ok`` answers like ml-service, ``hang`` never answers, ``error`` is HTTP 500."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.requests = 0
        self.server: Any = None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with contextlib.suppress(Exception):
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.requests += 1
                length = 0
                for line in head.decode().split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        length = int(line.split(":")[1])
                body = await reader.readexactly(length) if length else b""
                if self.mode == "hang":
                    await asyncio.sleep(3600)
                if self.mode == "error":
                    out, status = b'{"detail": "boom"}', "500 Internal Server Error"
                elif head.startswith(b"GET /health"):
                    out, status = b'{"status": "ok", "model_version": "vX"}', "200 OK"
                else:
                    rows = json.loads(body)["rows"]
                    preds = [{"row_id": r["row_id"], "pred_delay_s": 42.0, "p_late": None} for r in rows]
                    out, status = json.dumps({"model_version": "vX", "predictions": preds}).encode(), "200 OK"
                head_out = f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: "
                writer.write(f"{head_out}{len(out)}\r\n\r\n".encode() + out)
                await writer.drain()

    async def start(self) -> int:
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]


def test_client_timeout_breaker_and_recovery() -> None:
    async def scenario() -> dict[str, Any]:
        fake = FakeMLService()
        port = await fake.start()
        status = DependencyStatus("ml-service")
        ml = MLClient(f"http://127.0.0.1:{port}", status, timeout_s=0.2, retry_s=60.0)
        rows = [ROW, {"hour": math.nan}]
        ok = await ml.predict(rows)
        assert ok is not None and [p.pred_delay_s for p in ok.predictions] == [42.0, 42.0]
        assert status.ok is True and ml.model_version == "vX"
        fake.mode = "hang"
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        assert await ml.predict(rows) is None  # no answer in 200 ms: the fallback
        assert loop.time() - t0 < 1.0 and status.ok is False and status.outages == 1
        requests = fake.requests
        assert await ml.predict(rows) is None and fake.requests == requests  # the breaker is open
        assert ml.skipped == 1
        fake.mode = "ok"
        assert await ml.probe() is True and status.ok is True  # the probe closes the breaker at once
        assert (await ml.predict(rows)) is not None
        fake.mode = "error"
        assert await ml.predict(rows) is None and "HTTP 500" in (ml.last_error or "")
        ml.close()
        fake.server.close()  # type: ignore[union-attr]
        return {"failures": ml.failures, "calls": ml.calls}

    out = asyncio.run(scenario())
    assert out == {"failures": 2, "calls": 2}


def test_client_without_a_service() -> None:
    async def scenario() -> None:
        disabled = MLClient("")
        assert await disabled.predict([ROW]) is None and disabled.status.state == "disabled"
        refused = MLClient("http://127.0.0.1:9", timeout_s=0.5)  # nothing listens on the discard port
        assert await refused.predict([ROW]) is None and refused.status.state == "down"
        assert await refused.probe() is False

    asyncio.run(scenario())


def test_payload_and_answer_format() -> None:
    body = json.loads(encode_rows([{"a": 1.0, "b": math.nan, "c": None}], explain=False))
    assert body == {"rows": [{"row_id": "0", "features": {"a": 1.0, "b": None, "c": None}}], "explain": False}
    with pytest.raises(HttpError):
        decode_result(b'{"model_version": "v", "predictions": []}', 1)  # a row is missing
    with pytest.raises(HttpError):
        decode_result(b'{"predictions": [{"row_id": "0", "pred_delay_s": null}]}', 1)
    with pytest.raises(ValueError):
        MLClient("https://ml:8003")


def test_payload_with_sequences_sends_one_array_per_vehicle() -> None:
    np = pytest.importorskip("numpy")
    a = np.arange(80 * 12, dtype=np.float32).reshape(80, 12) / 7.0
    b = np.ones((80, 12), dtype=np.float32)
    body = json.loads(encode_rows([ROW, ROW, ROW], explain=True, sequences=[a, a, b]))
    assert [r["sequence_id"] for r in body["rows"]] == ["s0", "s0", "s1"]
    assert set(body["sequences"]) == {"s0", "s1"}
    decoded = np.frombuffer(base64.b64decode(body["sequences"]["s0"]), dtype="<f4").reshape(80, 12)
    assert np.array_equal(decoded, a)  # exact: float32 bytes, not decimal text
    none = json.loads(encode_rows([ROW], explain=True, sequences=[None]))
    assert "sequence_id" not in none["rows"][0] and none["sequences"] == {}
    assert "sequences" not in json.loads(encode_rows([ROW], explain=True))


def test_client_reads_the_sequence_input_from_health() -> None:
    ml = MLClient("")
    ml.read_health(
        {
            "model_version": "v2",
            "capabilities": ["factors", "sequence"],
            "sequence": {"len": 80, "channels": 12},
        }
    )
    assert ml.model_version == "v2" and ml.sequence_shape == (80, 12) and "sequence" in ml.capabilities
    ml.read_health({"model_version": "v1", "capabilities": ["factors"], "sequence": None})
    assert ml.model_version == "v1" and ml.sequence_shape is None
    ml.read_health({"model_version": "v0"})  # an older ml-service without capabilities: keep what is known
    assert ml.model_version == "v0" and ml.capabilities == frozenset({"factors"})


def _needs_v2() -> None:
    _needs_model()
    pytest.importorskip("onnxruntime")
    if not (MODELS / "v2" / "manifest.json").is_file():
        pytest.skip("models/v2 is not in the working tree")


def test_v2_forecasts_with_the_telemetry_sequences() -> None:
    _needs_v2()
    np = pytest.importorskip("numpy")
    from ml.service import ServiceSettings, create_app

    settings = ServiceSettings(models_dir=str(MODELS), version="v2", database_url="")
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        assert health["model_version"] == "v2" and "sequence" in health["capabilities"]
        steps, channels = health["sequence"]["len"], health["sequence"]["channels"]
        info = c.get("/model/info").json()
        assert info["components"] == ["catboost", "gru"] and info["precision"] == "int8"
        rng = np.random.default_rng(0)
        seq = (rng.random((steps, channels)) * 0.5).astype(np.float32)
        rows = [{"row_id": i, "features": ROW, "sequence_id": "s0"} for i in range(2)]
        rows.append({"row_id": 2, "features": ROW})  # no sequence: CatBoost part only
        b64 = base64.b64encode(seq.astype("<f4").tobytes()).decode()
        out = c.post("/predict", json={"rows": rows, "sequences": {"s0": b64}}).json()
        assert out["sequences_used"] == 2 and out["model_version"] == "v2"
        p = out["predictions"]
        assert p[0]["pred_delay_s"] == pytest.approx(p[1]["pred_delay_s"])
        assert p[0]["pred_delay_s"] != pytest.approx(p[2]["pred_delay_s"], abs=1e-6)  # the GRU part counts
        for pred in p:  # v2: interval, probability of a delay, expected error
            assert pred["p10"] <= pred["pred_delay_s"] <= pred["p90"] and 0.0 <= pred["p_late"] <= 1.0
            assert pred["expected_abs_error_s"] is not None and pred["factors"]
        # the same through the predictor's client encoding (nested lists are accepted too)
        listed = c.post("/predict", json={"rows": rows[:1], "sequences": {"s0": seq.tolist()}}).json()
        assert listed["predictions"][0]["pred_delay_s"] == pytest.approx(p[0]["pred_delay_s"], abs=1e-3)
        body = encode_rows([ROW, ROW], explain=False, sequences=[seq, None])
        result = decode_result(
            c.post("/predict", content=body, headers={"Content-Type": "application/json"}).content, 2
        )
        assert result.predictions[0].pred_delay_s == pytest.approx(p[0]["pred_delay_s"], abs=1e-3)
        assert result.predictions[0].p_late is not None and result.predictions[1].p10 is not None
        # a sequence of another shape is ignored (forecast without it), garbage is an error
        wrong = c.post("/predict", json={"rows": rows[:1], "sequences": {"s0": [[0.0] * 3] * 4}}).json()
        assert wrong["sequences_used"] == 0
        assert c.post("/predict", json={"rows": rows[:1], "sequences": {"s0": "!!"}}).status_code == 422
        assert 'foresight_ml_model_info{model="catboost_gru_ensemble",precision="int8",version="v2"} 1.0' in (
            c.get("/metrics").text
        )
