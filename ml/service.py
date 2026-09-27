"""ml-service: stateless forecast model over HTTP (``docs/api-contract.md`` §3), Swagger «Foresight · ML».

Run with ``python -m ml.service`` (port 8003). The model is a version of the registry
(:mod:`ml.inference`, ``<models_dir>/<version>/manifest.json``) loaded with :func:`ml.inference.load_bundle`:

* ``POST /predict`` — a batch of up to 1000 rows ``{row_id, features: {name: number | null}}`` →
  ``pred_delay_s``, ``p10`` / ``p50`` / ``p90``, ``p_late``, ``expected_abs_error_s`` (``null`` when the
  version cannot give them, e.g. ``v1``) and the top-3 feature contributions (CatBoost SHAP, seconds).
  Extension for models with a sequence component (``v2``: CatBoost + GRU): ``sequences`` — telemetry sequences
  by key (``shared.sequences``, nested lists or base64 float32) and ``sequence_id`` of a row; rows without a
  (fitting) sequence are forecast by the other components, as before;
* ``GET /model/info`` — the active version: features, metrics, precision, components;
* ``POST /model/reload`` — load the active (or the given) version and switch to it without stopping; the
  previous version keeps serving until the new one is loaded;
* ``POST /model/retrain`` — retraining on the stream in the background (:mod:`ml.online`): a correction of the
  active version by its forecasts closed on the stream (``predictions`` in PostgreSQL) becomes a new version
  ``<base>-online<N>`` (not activated) if it lowers the MAE on the later ones; ``GET /model/retrain`` — the
  state of the last run;
* ``GET /health`` — 200 with the version, capabilities and sequence input once a model is loaded, 503 before
  (or if it failed to load);
* ``GET /metrics`` — the contract of docs/observability.md §7.1: ``foresight_ml_request_duration_seconds``
  (``endpoint`` = route template), ``foresight_ml_batch_size``, ``foresight_ml_model_info`` (one series, the
  active version).

Settings (environment): ``FORESIGHT_MODELS_DIR`` (models of the repository, ``/app/models`` in the image),
``FORESIGHT_MODEL_VERSION`` (default ``latest`` — the newest ``created_at``), ``FORESIGHT_ML_PORT`` (8003),
``FORESIGHT_DATABASE_URL`` (registry table ``model_versions``; empty — none), ``FORESIGHT_LOG_LEVEL``.

The version in service is registered in PostgreSQL (``model_versions``: metrics, artifact, ``active``) at
start and after every reload. PostgreSQL down or the table not created yet (the backend services create the
schema) is not an error: the registration is retried in the background until it succeeds.

The image runs on CPU (CatBoost + ONNX Runtime CPU) and has no PyTorch.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from pydantic import BaseModel, Field

from ml.inference import (
    CAP_FACTORS,
    OUTPUT_COLUMNS,
    BundleError,
    ModelBundle,
    features_frame,
    load_bundle,
    models_dir,
)
from ml.online import fit_online, register_version

log = logging.getLogger("ml.service")

MAX_ROWS = 1000
SERVICE_VERSION = "0.1.0"
REQUEST_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5)
BATCH_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
ENDPOINTS = ("/predict", "/model/info", "/model/reload", "/health")
"""Routes measured by ``foresight_ml_request_duration_seconds`` (``/metrics`` is not)."""


@dataclass(frozen=True)
class ServiceSettings:
    """Environment of the service."""

    models_dir: str = field(default_factory=lambda: str(models_dir()))
    version: str = field(default_factory=lambda: os.environ.get("FORESIGHT_MODEL_VERSION") or "latest")
    host: str = field(default_factory=lambda: os.environ.get("FORESIGHT_ML_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("FORESIGHT_ML_PORT", "8003")))
    database_url: str = field(default_factory=lambda: os.environ.get("FORESIGHT_DATABASE_URL", ""))
    log_level: str = field(default_factory=lambda: os.environ.get("FORESIGHT_LOG_LEVEL", "info"))
    explain_top: int = field(default_factory=lambda: int(os.environ.get("FORESIGHT_ML_EXPLAIN_TOP", "3")))
    register_retry_s: float = 10.0


# ---- schemas ---------------------------------------------------------------------------------------------


class PredictRow(BaseModel):
    """One forecast point: its id (echoed back), features (missing or ``null`` — unknown) and, optionally, the
    key of its telemetry sequence in ``PredictIn.sequences``."""

    row_id: str | int = Field(description="Id of the row, returned with its prediction.")
    features: dict[str, float | None] = Field(description="Feature values by name (shared.features).")
    sequence_id: str | None = Field(
        None,
        description="Key of the row's telemetry sequence in `sequences` (rows of one vehicle share one). "
        "Only models with the `sequence` capability use it; without it they forecast with the other "
        "components.",
    )


class PredictIn(BaseModel):
    """Batch of forecast points."""

    rows: list[PredictRow] = Field(description=f"Up to {MAX_ROWS} rows.")
    explain: bool = Field(True, description="Add the top feature contributions (CatBoost SHAP, s).")
    sequences: dict[str, list[list[float]] | str] | None = Field(
        None,
        description="Telemetry sequences by key (shared.sequences: SEQ_LEN steps × N_CHANNELS, points <= T): "
        "a nested list or base64 of float32 little-endian values. Extension of the contract §3, optional.",
    )


class FactorOut(BaseModel):
    """Contribution of one feature to the forecast, seconds."""

    feature: str
    label: str | None = None
    contribution_s: float


class PredictionOut(BaseModel):
    """Forecast of one row; ``null`` — the model version does not give this output."""

    row_id: str | int
    pred_delay_s: float = Field(description="Forecast delay at the target stop, s (+ late, − early).")
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    p_late: float | None = Field(None, description="P(delay > late_threshold_s).")
    expected_abs_error_s: float | None = None
    factors: list[FactorOut] = Field(default_factory=list)


class PredictOut(BaseModel):
    """Forecasts of a batch."""

    model_version: str
    precision: Literal["fp32", "int8"]
    latency_ms: float = Field(description="Time of the model call (without HTTP), ms.")
    predictions: list[PredictionOut]
    sequences_used: int = Field(0, description="Rows forecast with their telemetry sequence (ML v2).")


class ModelInfoOut(BaseModel):
    """The active model version."""

    version: str
    created_at: str
    features: list[str]
    metrics: dict[str, Any]
    precision: Literal["fp32", "int8"]
    components: list[str]
    capabilities: list[str] = Field(default_factory=list)
    sequence: dict[str, Any] | None = Field(
        None, description="Sequence input of the model: {len, channels, version}; null — none."
    )
    features_version: str | None = None
    loaded_at: str | None = None
    path: str | None = None
    registered: bool = Field(False, description="Registered in PostgreSQL model_versions.")


class VersionOut(BaseModel):
    """A version of the model registry (``<models_dir>/<version>/manifest.json``)."""

    version: str
    created_at: str | None = None
    model: str | None = None
    description: str | None = None
    precision: str | None = None
    components: list[str] = Field(default_factory=list)
    cv_mae: float | None = None
    test_mae: float | None = None
    baseline_test_mae: float | None = Field(None, description="«Forecast = cur_dev_s» on test.")
    online: dict[str, Any] | None = Field(
        None, description="Retrained on the stream: the correction a + b·forecast of the base and its check."
    )
    active: bool = False


def list_versions(root: Path, active: str | None) -> list[VersionOut]:
    """Versions of the registry by their manifests (unreadable ones are skipped)."""
    out = []
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metrics = m.get("metrics") or {}
        version = str(m.get("version") or manifest.parent.name)
        out.append(
            VersionOut(
                version=version,
                created_at=m.get("created_at"),
                model=m.get("model"),
                description=m.get("description"),
                precision=m.get("precision"),
                components=list(m.get("components") or []),
                cv_mae=_clean(metrics.get("cv_mae")),
                test_mae=_clean(metrics.get("test_mae")),
                baseline_test_mae=_clean(metrics.get("baseline_cur_dev_test_mae")),
                online=m.get("online_calibration"),
                active=version == active,
            )
        )
    return out


class ReloadIn(BaseModel):
    """Which version to load (``null`` — the configured one, re-read from disk)."""

    version: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=64)


class RetrainOut(BaseModel):
    """The last retraining on the stream (``POST /model/retrain``)."""

    state: Literal["idle", "running", "done", "error"] = "idle"
    base_version: str | None = Field(None, description="The corrected version (active at the start).")
    version: str | None = Field(None, description="The new version (not activated; null: not better).")
    improved: bool | None = Field(None, description="The correction lowered the MAE (else no version).")
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_fit: int | None = None
    n_eval: int | None = None
    mae_before_s: float | None = Field(None, description="MAE of the base on the later closed forecasts.")
    mae_after_s: float | None = Field(None, description="MAE of the new version on them.")
    error: str | None = None


class HealthOut(BaseModel):
    """Liveness and the loaded model."""

    status: Literal["ok", "loading", "error"]
    service: str = "ml-service"
    version: str = SERVICE_VERSION
    model_version: str | None = None
    capabilities: list[str] = Field(
        default_factory=list, description="Of the loaded model: quantiles, p_late, factors, sequence."
    )
    sequence: dict[str, int] | None = Field(
        None, description="Sequence input of the loaded model {len, channels} (the predictor builds them)."
    )
    uptime_s: float
    error: str | None = None
    database: Literal["up", "down", "disabled", "unknown"] = "unknown"


# ---- state -----------------------------------------------------------------------------------------------


def _clean(x: Any) -> float | None:
    if x is None:
        return None
    value = float(x)
    return value if math.isfinite(value) else None


class ModelState:
    """The loaded bundle, swapped atomically by :meth:`load`."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.bundle: ModelBundle | None = None
        self.loaded_at: datetime | None = None
        self.error: str | None = None
        self.registered_version: str | None = None
        self.db_state: Literal["up", "down", "disabled", "unknown"] = (
            "unknown" if settings.database_url else "disabled"
        )
        self._lock = threading.Lock()
        self.retrain = RetrainOut()

    def load(self, version: str | None = None) -> ModelBundle:
        """Load a version (default: the configured one) and make it active.

        Raises:
            BundleError: The version is missing, damaged or was trained on other feature code.
        """
        with self._lock:
            target = version or self.settings.version
            bundle = load_bundle(target, root=_root(self.settings))
            probe = features_frame([{}], bundle.features)
            bundle.predict(probe)  # warm-up: the first CatBoost call builds its caches
            if bundle.sequence_shape is not None:  # ... and the first ONNX Runtime call its kernels
                bundle.predict(probe, sequences=np.zeros((1, *bundle.sequence_shape), dtype=np.float32))
            self.bundle = bundle
            self.loaded_at = datetime.now(UTC)
            self.error = None
            log.info("model %r loaded from %s", bundle.version, bundle.path)
            return bundle


def _root(settings: ServiceSettings) -> Path:
    return Path(settings.models_dir)


def decode_sequences(
    bundle: ModelBundle, sequences: dict[str, list[list[float]] | str] | None
) -> dict[str, np.ndarray]:
    """Sequences of the request that fit the model's input (others are ignored: forecast without them).

    Raises:
        ValueError: A value is neither a nested list nor base64 of float32.
    """
    shape = bundle.sequence_shape
    if not sequences or shape is None:
        return {}
    out = {}
    for key, value in sequences.items():
        if isinstance(value, str):
            arr = np.frombuffer(base64.b64decode(value, validate=True), dtype="<f4")
            if arr.size != shape[0] * shape[1]:
                continue
            arr = arr.reshape(shape)
        else:
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != shape:
                continue
        if np.isfinite(arr).all():
            out[key] = arr.astype(np.float32)
    return out


def predict_rows(
    bundle: ModelBundle,
    rows: list[PredictRow],
    explain: bool,
    top: int,
    sequences: dict[str, np.ndarray] | None = None,
) -> tuple[list[PredictionOut], int]:
    """Forecasts of the rows: one model call for the rows with a sequence and one for the rest (and the same
    for the SHAP contributions).

    Returns:
        The predictions in the order of the rows and the number of rows forecast with their sequence.
    """
    frame = features_frame([r.features for r in rows], bundle.features)
    seq = sequences or {}
    with_seq = [i for i, r in enumerate(rows) if r.sequence_id is not None and r.sequence_id in seq]
    parts: list[tuple[list[int], np.ndarray | None]] = []
    if with_seq:
        parts.append((with_seq, np.stack([seq[str(rows[i].sequence_id)] for i in with_seq])))
    rest = sorted(set(range(len(rows))) - set(with_seq))
    if rest:
        parts.append((rest, None))
    out = pd.DataFrame(index=frame.index, columns=OUTPUT_COLUMNS, dtype=np.float64)
    factors: list[list[dict]] = [[] for _ in rows]
    for idx, arr in parts:
        sub = frame.iloc[idx]
        out.iloc[idx] = bundle.predict(sub, sequences=arr).to_numpy()
        if explain and CAP_FACTORS in bundle.capabilities:
            for i, fs in zip(idx, bundle.explain(sub, top=top, sequences=arr), strict=True):
                factors[i] = fs
    cols = {c: out[c].to_numpy(dtype=np.float64) for c in out.columns}
    result = []
    for i, row in enumerate(rows):
        pred = _clean(cols["pred_delay_s"][i])
        result.append(
            PredictionOut(
                row_id=row.row_id,
                pred_delay_s=pred if pred is not None else 0.0,
                p10=_clean(cols["p10"][i]),
                p50=_clean(cols["p50"][i]),
                p90=_clean(cols["p90"][i]),
                p_late=_clean(cols["p_late"][i]),
                expected_abs_error_s=_clean(cols["expected_abs_error_s"][i]),
                factors=[FactorOut(**f) for f in factors[i] if _clean(f.get("contribution_s")) is not None],
            )
        )
    return result, len(with_seq)


# ---- registry in PostgreSQL ------------------------------------------------------------------------------


DEACTIVATE_SQL = "UPDATE model_versions SET active = FALSE WHERE active AND version <> $1"
REGISTER_SQL = """
INSERT INTO model_versions (version, kind, active, cv_mae, test_mae, artifact_uri, params, metrics, notes,
                            trained_at)
VALUES ($1, $2, TRUE, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9)
ON CONFLICT (version) DO UPDATE SET kind = EXCLUDED.kind, active = TRUE, cv_mae = EXCLUDED.cv_mae,
    test_mae = EXCLUDED.test_mae, artifact_uri = EXCLUDED.artifact_uri, params = EXCLUDED.params,
    metrics = EXCLUDED.metrics, notes = EXCLUDED.notes, trained_at = EXCLUDED.trained_at
"""


def registry_row(bundle: ModelBundle) -> tuple[Any, ...]:
    """Values of :data:`REGISTER_SQL` for a bundle."""
    metrics = bundle.metrics
    trained = None
    with contextlib.suppress(ValueError):
        trained = datetime.fromisoformat(bundle.created_at.replace("Z", "+00:00"))
    params = {
        "components": bundle.components,
        "precision": bundle.precision,
        "capabilities": sorted(bundle.capabilities),
        "features": len(bundle.features),
        "features_version": bundle.features_version,
        "git_commit": bundle.manifest.get("git_commit"),
        "train": bundle.manifest.get("train"),
    }
    return (
        bundle.version,
        "+".join(bundle.components) or "catboost",
        _clean(metrics.get("cv_mae")),
        _clean(metrics.get("test_mae")),
        str(bundle.path),
        json.dumps(params, ensure_ascii=False, default=str),
        json.dumps(metrics, ensure_ascii=False, default=str),
        bundle.manifest.get("description"),
        trained,
    )


CLOSED_SQL = """
SELECT pred_delay_s, actual_delay_s FROM (
    SELECT pred_delay_s, actual_delay_s, closed_at FROM predictions
    WHERE status = 'closed' AND source = 'model' AND model_version = $1 AND actual_delay_s IS NOT NULL
    ORDER BY closed_at DESC LIMIT $2
) recent ORDER BY closed_at
"""
RETRAIN_ROWS = 20_000


async def closed_forecasts(database_url: str, version: str, timeout_s: float = 30.0) -> np.ndarray:
    """The latest :data:`RETRAIN_ROWS` forecasts of ``version`` closed on the stream, in closing order:
    ``(n, 2)`` — forecast, fact."""
    import asyncpg

    con = await asyncpg.connect(database_url, timeout=timeout_s)
    try:
        rows = await con.fetch(CLOSED_SQL, version, RETRAIN_ROWS, timeout=timeout_s)
    finally:
        await con.close(timeout=timeout_s)
    return np.array([(r[0], r[1]) for r in rows], dtype=np.float64).reshape(-1, 2)


async def register(database_url: str, bundle: ModelBundle, timeout_s: float = 5.0) -> None:
    """Mark the version active in ``model_versions`` (others inactive) in one transaction.

    Raises:
        Exception: PostgreSQL unreachable or the table not created yet.
    """
    import asyncpg

    con = await asyncpg.connect(database_url, timeout=timeout_s)
    try:
        async with con.transaction():
            await con.execute(DEACTIVATE_SQL, bundle.version, timeout=timeout_s)
            await con.execute(REGISTER_SQL, *registry_row(bundle), timeout=timeout_s)
    finally:
        await con.close(timeout=timeout_s)


# ---- metrics ---------------------------------------------------------------------------------------------


class ModelInfoCollector(Collector):
    """``foresight_ml_model_info{version, model, precision} 1`` — exactly one series, the active version."""

    def __init__(self, state: ModelState) -> None:
        self.state = state

    def collect(self) -> list[Metric]:
        family = GaugeMetricFamily(
            "foresight_ml_model_info", "Active model version (1).", labels=["version", "model", "precision"]
        )
        bundle = self.state.bundle
        if bundle is not None:
            model = str(bundle.manifest.get("model") or "+".join(bundle.components))
            family.add_metric([bundle.version, model, bundle.precision], 1)
        return [family]


# ---- the app ---------------------------------------------------------------------------------------------

_DESCRIPTION = """
Foresight ML service: the delay forecast model behind the predictor (stateless, CPU).

* `POST /predict` — batch of forecast points (up to 1000) → delay forecast, P10/P50/P90, p_late (null if the
  model version does not give them) and the top feature contributions;
* `GET /model/info` — active model version, its features and metrics;
* `POST /model/reload` — switch to another version (or re-read the active one) without stopping;
* `GET /health` — 200 once the model is loaded; `GET /metrics` — Prometheus.
"""


def create_app(settings: ServiceSettings | None = None, *, load: bool = True) -> FastAPI:
    """Build the ml-service application.

    Args:
        settings: Environment (default: :class:`ServiceSettings` from the environment).
        load: Load the model at start (tests may load it themselves through ``app.state.model``).

    Returns:
        The application; ``app.state.model`` is the :class:`ModelState`.
    """
    settings = settings or ServiceSettings()
    state = ModelState(settings)
    started = time.monotonic()
    registry = CollectorRegistry()
    request_seconds = Histogram(
        "foresight_ml_request_duration_seconds",
        "Time to handle a request, s (for /predict: the whole batch).",
        ["endpoint"],
        buckets=REQUEST_BUCKETS,
        registry=registry,
    )
    batch_size = Histogram(
        "foresight_ml_batch_size",
        "Forecast points in one POST /predict.",
        buckets=BATCH_BUCKETS,
        registry=registry,
    )
    for endpoint in ENDPOINTS:
        request_seconds.labels(endpoint=endpoint)
    registry.register(ModelInfoCollector(state))
    tasks: set[asyncio.Task[Any]] = set()

    async def register_loop(bundle: ModelBundle) -> None:
        if not settings.database_url:
            return
        while True:
            try:
                await register(settings.database_url, bundle)
            except Exception as exc:  # PostgreSQL down or the schema not created yet: try again later
                state.db_state = "down"
                log.info("model registry: %s; retry in %.0f s", exc, settings.register_retry_s)
                await asyncio.sleep(settings.register_retry_s)
                continue
            state.db_state = "up"
            state.registered_version = bundle.version
            log.info("model %r registered as active in model_versions", bundle.version)
            return

    def schedule_register(bundle: ModelBundle) -> None:
        for task in list(tasks):
            task.cancel()
        task = asyncio.get_running_loop().create_task(register_loop(bundle))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def load_model(version: str | None = None) -> ModelBundle:
        bundle = await asyncio.to_thread(state.load, version)
        schedule_register(bundle)
        return bundle

    async def fetch_closed(version: str) -> np.ndarray:
        return await closed_forecasts(settings.database_url, version)

    async def retrain_job(bundle: ModelBundle) -> None:
        run = state.retrain
        try:
            closed = await app.state.fetch_closed(bundle.version)
            fit = await asyncio.to_thread(fit_online, closed[:, 0], closed[:, 1])
            improved = fit.mae_after_s < fit.mae_before_s
            # a correction that does not help on the later forecasts is not offered as a version
            version = None
            if improved:
                version = await asyncio.to_thread(register_version, bundle, fit, _root(settings))
        except Exception as exc:  # the run fails, the service keeps serving
            run.state, run.error = "error", str(exc) or type(exc).__name__
            log.warning("retraining on the stream failed: %s", run.error)
        else:
            run.state, run.version, run.improved = "done", version, improved
            run.n_fit, run.n_eval = fit.n_fit, fit.n_eval
            run.mae_before_s, run.mae_after_s = round(fit.mae_before_s, 2), round(fit.mae_after_s, 2)
            log.info("retrained: %s (MAE %.1f → %.1f s)", version, fit.mae_before_s, fit.mae_after_s)
        run.finished_at = datetime.now(UTC)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load:
            try:
                await load_model()
            except (BundleError, OSError, ValueError) as exc:
                state.error = f"{type(exc).__name__}: {exc}"
                log.error("model not loaded: %s", state.error)
        try:
            yield
        finally:
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(
        title="Foresight · ML", version=SERVICE_VERSION, description=_DESCRIPTION, lifespan=lifespan
    )
    app.state.model = state
    app.state.registry = registry
    app.state.load_model = load_model
    app.state.fetch_closed = fetch_closed  # tests replace the source of the closed forecasts

    @app.middleware("http")
    async def measure(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        path = request.url.path  # the measured routes have no path parameters: the path is the template
        if path in ENDPOINTS:
            request_seconds.labels(endpoint=path).observe(time.perf_counter() - t0)
        return response

    def active() -> ModelBundle:
        bundle = state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail=f"model is not loaded: {state.error or 'loading'}")
        return bundle

    @app.post("/predict", response_model=PredictOut, tags=["model"], summary="Forecast a batch of points")
    def predict(body: PredictIn) -> PredictOut:
        bundle = active()
        if len(body.rows) > MAX_ROWS:
            raise HTTPException(status_code=413, detail=f"at most {MAX_ROWS} rows per request")
        batch_size.observe(len(body.rows))
        t0 = time.perf_counter()
        try:
            sequences = decode_sequences(bundle, body.sequences)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=422, detail=f"bad sequences: {exc}") from exc
        try:
            preds, used = (
                predict_rows(bundle, body.rows, body.explain, settings.explain_top, sequences)
                if body.rows
                else ([], 0)
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"bad features: {exc}") from exc
        return PredictOut(
            model_version=bundle.version,
            precision=bundle.precision,  # type: ignore[arg-type]
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            predictions=preds,
            sequences_used=used,
        )

    @app.get("/model/info", response_model=ModelInfoOut, tags=["model"], summary="Active model version")
    def model_info() -> ModelInfoOut:
        bundle = active()
        info = bundle.info()
        return ModelInfoOut(
            version=info["version"],
            created_at=info["created_at"],
            features=info["features"],
            metrics=info["metrics"],
            precision=info["precision"],
            components=info["components"],
            capabilities=info["capabilities"],
            sequence=info.get("sequence"),
            features_version=info["features_version"],
            loaded_at=state.loaded_at.isoformat() if state.loaded_at else None,
            path=str(bundle.path),
            registered=state.registered_version == bundle.version,
        )

    @app.get(
        "/model/versions", response_model=list[VersionOut], tags=["model"], summary="Versions of the registry"
    )
    def model_versions() -> list[VersionOut]:
        """All versions in ``FORESIGHT_MODELS_DIR`` with their metrics; ``active`` — the one in service
        (switch with ``POST /model/reload``)."""
        return list_versions(_root(settings), state.bundle.version if state.bundle is not None else None)

    @app.post(
        "/model/reload",
        response_model=ModelInfoOut,
        tags=["model"],
        summary="Load a version and switch to it",
        responses={404: {"description": "No such version or it cannot be loaded"}},
    )
    async def reload(body: ReloadIn | None = None) -> ModelInfoOut:
        version = body.version if body is not None else None
        try:
            await load_model(version)
        except (BundleError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=f"{type(exc).__name__}: {exc}") from exc
        return model_info()

    @app.post(
        "/model/retrain",
        response_model=RetrainOut,
        status_code=202,
        tags=["model"],
        summary="Retrain the active version on the stream (in the background)",
        responses={409: {"description": "A run is going"}, 503: {"description": "No model or no journal"}},
    )
    async def retrain() -> RetrainOut:
        """Start a correction of the active version by its forecasts closed on the stream; the result is a new
        version (see ``GET /model/retrain``), activated separately with ``POST /model/reload``."""
        bundle = active()
        if state.retrain.state == "running":
            raise HTTPException(status_code=409, detail="retraining is already running")
        if not settings.database_url and app.state.fetch_closed is fetch_closed:
            raise HTTPException(status_code=503, detail="no journal of closed forecasts (database URL)")
        state.retrain = RetrainOut(state="running", base_version=bundle.version, started_at=datetime.now(UTC))
        task = asyncio.get_running_loop().create_task(retrain_job(bundle))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return state.retrain

    @app.get(
        "/model/retrain", response_model=RetrainOut, tags=["model"], summary="State of the last retraining"
    )
    def retrain_status() -> RetrainOut:
        return state.retrain

    @app.get("/health", response_model=HealthOut, tags=["ops"], summary="Liveness and the loaded model")
    def health() -> JSONResponse:
        bundle = state.bundle
        shape = bundle.sequence_shape if bundle is not None else None
        body = HealthOut(
            status="ok" if bundle is not None else ("error" if state.error else "loading"),
            model_version=bundle.version if bundle is not None else None,
            capabilities=sorted(bundle.capabilities) if bundle is not None else [],
            sequence={"len": shape[0], "channels": shape[1]} if shape is not None else None,
            uptime_s=round(time.monotonic() - started, 3),
            error=state.error,
            database=state.db_state,
        )
        return JSONResponse(body.model_dump(), status_code=200 if bundle is not None else 503)

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics",
        response_class=Response,
        responses={200: {"content": {CONTENT_TYPE_LATEST: {}}, "description": "Prometheus text format"}},
    )
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    """Entry point of ``python -m ml.service``."""
    import uvicorn

    settings = ServiceSettings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    np.seterr(all="ignore")
    uvicorn.run(
        "ml.service:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # predictor keeps one connection and probes /health every few seconds: a 5 s default keep-alive made
        # the server close it right between requests
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
