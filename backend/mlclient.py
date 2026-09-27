"""Client of ml-service (``docs/api-contract.md`` §3) with a timeout, a circuit breaker and health probes.

The predictor asks ``POST /predict`` once per tick with all forecast rows. Any failure — connection
refused, HTTP error, malformed answer or no answer within ``timeout_s`` (500 ms by default) — returns
``None``: the caller falls back to its own formula (:mod:`backend.forecast`). After a failure the service is
not asked again for ``retry_s``; a background probe of ``GET /health`` closes the breaker as soon as
ml-service answers, so the model is back on the next tick after a recovery. The state is reported as the
``ml-service`` dependency.

The probe also reads the active model's capabilities: when it has a sequence component (ML v2), the predictor
sends the telemetry sequence of every vehicle with the rows (``sequences`` map of base64 float32 arrays and a
``sequence_id`` per row — an extension of ``POST /predict``, backward compatible: without it ml-service
forecasts with the other components).

The HTTP/1.1 client is a small asyncio one (keep-alive, ``Content-Length`` / chunked bodies): the backend
image has no HTTP client library, and a thread with ``urllib`` could not be cancelled on the timeout.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from backend.runtime import DependencyStatus

log = logging.getLogger(__name__)

ML_DEPENDENCY = "ml-service"


class HttpError(Exception):
    """A request to ml-service failed (status, protocol or connection)."""


def _stale_connection(exc: BaseException) -> bool:
    """Whether ``exc`` means the peer had already closed a reused keep-alive connection."""
    if isinstance(exc, HttpError):
        return str(exc).startswith("connection closed before the response")
    return isinstance(exc, ConnectionResetError | BrokenPipeError | asyncio.IncompleteReadError)


class HttpClient:
    """Minimal HTTP/1.1 client over one keep-alive connection (requests are serialised).

    A request that fails because the server has closed the idle keep-alive connection is retried once on a new
    connection (all requests of this client — ``/predict``, ``/health`` — are safe to repeat).

    Args:
        base_url: ``http://host:port``.
    """

    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"unsupported ml-service URL {base_url!r} (expected http://host:port)")
        self.host = parts.hostname
        self.port = parts.port or 80
        self.prefix = parts.path.rstrip("/")
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        """Send one request and read the whole response (the caller bounds the time).

        Raises:
            HttpError: Protocol violation.
            OSError: Connection failure.
        """
        async with self._lock:
            reused = self._writer is not None and not self._writer.is_closing()
            try:
                return await self._request(method, path, body)
            except (HttpError, ConnectionError, asyncio.IncompleteReadError) as exc:
                self.close()  # a failed exchange leaves the connection in an unknown state
                if not (reused and _stale_connection(exc)):
                    raise
                # The server closed our idle keep-alive connection between two requests (uvicorn's keep-alive
                # timeout): the request never reached it, so one retry on a fresh connection is safe.
            except BaseException:
                self.close()  # a cancelled exchange leaves the connection in an unknown state
                raise
            try:
                return await self._request(method, path, body)
            except BaseException:
                self.close()
                raise

    async def _request(self, method: str, path: str, body: bytes | None) -> tuple[int, bytes]:
        if self._writer is None or self._reader is None or self._writer.is_closing():
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        head = [
            f"{method} {self.prefix}{path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Connection: keep-alive",
            "Accept: application/json",
        ]
        if body is not None:
            head += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
        self._writer.write(("\r\n".join(head) + "\r\n\r\n").encode("ascii") + (body or b""))
        await self._writer.drain()
        reader = self._reader
        status_line = await reader.readline()
        if not status_line:
            raise HttpError("connection closed before the response")
        parts = status_line.decode("latin-1").split(" ", 2)
        if len(parts) < 2 or not parts[0].startswith("HTTP/1."):
            raise HttpError(f"bad status line {status_line!r}")
        status = int(parts[1])
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        if headers.get("transfer-encoding", "").lower() == "chunked":
            data = await self._read_chunked(reader)
        elif "content-length" in headers:
            data = await reader.readexactly(int(headers["content-length"]))
        else:
            data = await reader.read()
            self.close()
        if headers.get("connection", "").lower() == "close":
            self.close()
        return status, data

    @staticmethod
    async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
        chunks = []
        while True:
            size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                await reader.readline()
                return b"".join(chunks)
            chunks.append(await reader.readexactly(size))
            await reader.readline()

    def close(self) -> None:
        """Drop the connection (the next request opens a new one)."""
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
        self._reader = self._writer = None


@dataclass(frozen=True, slots=True)
class MLPrediction:
    """One row of ``POST /predict`` (``None`` — not supported by the model)."""

    pred_delay_s: float
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    p_late: float | None = None
    expected_abs_error_s: float | None = None
    factors: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class MLResult:
    """Answer of ``POST /predict``: model version, precision, latency and the rows in request order."""

    model_version: str
    precision: str | None
    latency_ms: float | None
    predictions: list[MLPrediction] = field(default_factory=list)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def encode_sequence(arr: np.ndarray) -> str:
    """A telemetry sequence as base64 of float32 little-endian (exact, ~5 KB for 80 × 12)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def encode_rows(
    rows: Sequence[Mapping[str, float]],
    explain: bool,
    sequences: Sequence[np.ndarray | None] | None = None,
) -> bytes:
    """Body of ``POST /predict``; NaN / inf features become ``null`` (missing).

    Args:
        rows: Feature rows.
        explain: Ask for the top feature contributions.
        sequences: Telemetry sequence of every row (``shared.sequences``) or ``None``; rows of one vehicle
            share one array object, which is sent once (``sequences`` map + ``sequence_id`` of the rows).
    """

    def clean(v: Any) -> float | None:
        return float(v) if v is not None and math.isfinite(float(v)) else None

    items: list[dict[str, Any]] = [
        {"row_id": str(i), "features": {k: clean(v) for k, v in row.items()}} for i, row in enumerate(rows)
    ]
    payload: dict[str, Any] = {"rows": items, "explain": explain}
    if sequences is not None:
        ids: dict[int, str] = {}
        encoded: dict[str, str] = {}
        for item, arr in zip(items, sequences, strict=True):
            if arr is None:
                continue
            key = ids.get(id(arr))
            if key is None:
                key = ids[id(arr)] = f"s{len(ids)}"
                encoded[key] = encode_sequence(arr)
            item["sequence_id"] = key
        payload["sequences"] = encoded
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()


def decode_result(data: bytes, n: int) -> MLResult:
    """Parse the answer of ``POST /predict`` for ``n`` rows.

    Raises:
        HttpError: Malformed answer or rows missing.
    """
    try:
        body = json.loads(data)
        by_id = {str(p["row_id"]): p for p in body["predictions"]}
        preds = []
        for i in range(n):
            p = by_id[str(i)]
            pred = _num(p["pred_delay_s"])
            if pred is None:
                raise HttpError(f"row {i}: pred_delay_s is not a number")
            preds.append(
                MLPrediction(
                    pred_delay_s=pred,
                    p10=_num(p.get("p10")),
                    p50=_num(p.get("p50")),
                    p90=_num(p.get("p90")),
                    p_late=_num(p.get("p_late")),
                    expected_abs_error_s=_num(p.get("expected_abs_error_s")),
                    factors=tuple(p.get("factors") or ()),
                )
            )
        return MLResult(
            model_version=str(body.get("model_version", "")),
            precision=body.get("precision"),
            latency_ms=_num(body.get("latency_ms")),
            predictions=preds,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HttpError(f"malformed /predict answer: {exc}") from exc


class MLClient:
    """``POST /predict`` with a timeout and a breaker; ``None`` means «use the fallback».

    Args:
        url: ml-service base URL (empty: disabled, always ``None``).
        status: Dependency status to report to (``ml-service``).
        timeout_s: Timeout of one prediction call.
        retry_s: Pause after a failure before the next call (the health probe may end it earlier).
        explain: Ask for feature contributions.
        probe_s: Period of ``GET /health`` probes (:meth:`run`).
    """

    def __init__(
        self,
        url: str,
        status: DependencyStatus | None = None,
        *,
        timeout_s: float = 0.5,
        retry_s: float = 5.0,
        explain: bool = True,
        probe_s: float = 5.0,
    ) -> None:
        self.url = url
        self.status = status or DependencyStatus(ML_DEPENDENCY)
        self.status.enabled = bool(url)
        self.timeout_s = timeout_s
        self.retry_s = retry_s
        self.explain = explain
        self.probe_s = probe_s
        self.calls = 0
        self.failures = 0
        self.skipped = 0
        self.last_latency_s: float | None = None
        self.model_version: str | None = None
        self.capabilities: frozenset[str] = frozenset()
        self.sequence_shape: tuple[int, int] | None = None
        """``(steps, channels)`` of the active model's sequence input (from ``GET /health``), ``None`` — the
        model takes no sequences (the predictor then does not build them)."""
        self.last_error: str | None = None
        self._http = HttpClient(url) if url else None
        self._probe = HttpClient(url) if url else None
        self._retry_at = 0.0

    @property
    def available(self) -> bool:
        """Whether the next :meth:`predict` will ask ml-service (enabled and the breaker is closed)."""
        return self._http is not None and time.monotonic() >= self._retry_at

    def _fail(self, exc: BaseException | str) -> None:
        self.failures += 1
        self.last_error = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        self.status.mark_down(self.last_error)
        self._retry_at = time.monotonic() + self.retry_s

    async def predict(
        self, rows: Sequence[Mapping[str, float]], sequences: Sequence[np.ndarray | None] | None = None
    ) -> MLResult | None:
        """Forecast the rows (with their telemetry sequences, if given); ``None`` on any failure or while the
        breaker is open."""
        if not rows:
            return None
        if self._http is None or not self.available:
            self.skipped += 1
            return None
        body = encode_rows(rows, self.explain, sequences)
        started = time.perf_counter()
        try:
            call = self._http.request("POST", "/predict", body)
            status, data = await asyncio.wait_for(call, self.timeout_s)
            if status != 200:
                raise HttpError(f"POST /predict: HTTP {status}: {data[:200]!r}")
            result = decode_result(data, len(rows))
        except Exception as exc:  # any failure of the call means «use the fallback»
            if isinstance(exc, TimeoutError):
                exc = HttpError(f"no answer in {self.timeout_s:.2f} s")
            self._fail(exc)
            log.warning("ml-service: %s; fallback for %.0f s", self.last_error, self.retry_s)
            return None
        self.calls += 1
        self.last_latency_s = time.perf_counter() - started
        self.model_version = result.model_version or self.model_version
        self.status.mark_ok()
        return result

    async def probe(self) -> bool:
        """``GET /health``: 200 means the model is loaded; closes the breaker on success."""
        if self._probe is None:
            return False
        try:
            status, data = await asyncio.wait_for(self._probe.request("GET", "/health"), 2.0)
        except Exception as exc:
            self._fail(exc if not isinstance(exc, TimeoutError) else "health probe timed out")
            return False
        if status != 200:
            self._fail(f"GET /health: HTTP {status}")
            return False
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError):
            self.read_health(json.loads(data))
        self.status.mark_ok()
        self._retry_at = 0.0
        return True

    def read_health(self, body: Mapping[str, Any]) -> None:
        """Take the active model's version, capabilities and sequence input from a ``GET /health`` body."""
        self.model_version = body.get("model_version") or self.model_version
        caps = body.get("capabilities")
        if caps is None:
            return  # an older ml-service: keep what is known
        self.capabilities = frozenset(str(c) for c in caps)
        seq = body.get("sequence") or {}
        if "sequence" in self.capabilities and seq.get("len") and seq.get("channels"):
            self.sequence_shape = (int(seq["len"]), int(seq["channels"]))
        else:
            self.sequence_shape = None

    async def run(self) -> None:
        """Probe loop (runs until cancelled)."""
        if self._probe is None:
            return
        while True:
            try:
                await self.probe()
            except Exception:  # a probe must never stop the loop
                log.exception("ml-service probe failed")
            await asyncio.sleep(self.probe_s)

    def close(self) -> None:
        """Drop the connections."""
        for client in (self._http, self._probe):
            if client is not None:
                client.close()
