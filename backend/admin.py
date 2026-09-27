"""Admin API of the api service (``/api/admin/*``): thresholds, model versions, journal export, the health of
all services and the vehicle directory.

* thresholds live in the ``settings`` table; the predictor re-reads them every ``settings_refresh_s`` (30 s),
  so a change applies on the fly, without a restart;
* model versions come from ml-service (``GET /model/versions``); activation is ``POST /model/reload`` — the
  service switches without stopping (until its restart: then the version is ``FORESIGHT_MODEL_VERSION``);
  retraining (``POST /model/retrain``) corrects the active version by its forecasts closed on the stream and
  registers the result as a new version (:mod:`ml.online`); the fallback forecast of the predictor is shown
  with its share among the forecasts now;
* the journal (alerts, forecasts of the current stream timeline) is exported as JSON or CSV.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

from backend.journal import epoch_start
from backend.mlclient import HttpClient

if TYPE_CHECKING:
    from backend.api import ApiService

log = logging.getLogger(__name__)

SETTING_KEYS = ("risk_thresholds", "alert_thresholds")
VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
"""A model version is a directory name in the models directory: no path separators, no ``..``."""


class RiskSettings(BaseModel):
    """Map colours (contract §1): red above either red threshold, green below both green ones."""

    red_delay_s: float = Field(120, ge=0, le=3600)
    red_p_late: float = Field(0.6, ge=0, le=1)
    green_delay_s: float = Field(60, ge=-600, le=3600)
    green_p_late: float = Field(0.3, ge=0, le=1)

    @model_validator(mode="after")
    def _order(self) -> RiskSettings:
        if self.green_delay_s > self.red_delay_s or self.green_p_late > self.red_p_late:
            raise ValueError("green thresholds must not exceed the red ones")
        return self


class AlertSettings(BaseModel):
    """An incident opens (and alerts) at ``min_level`` or higher with ``p_late >= min_p_late``."""

    min_level: Literal["yellow", "red"] = "yellow"
    min_p_late: float = Field(0.0, ge=0, le=1)


class SettingsOut(BaseModel):
    """``GET/PUT /api/admin/settings``."""

    risk: RiskSettings = Field(default_factory=RiskSettings)
    alert: AlertSettings = Field(default_factory=AlertSettings)
    updated_at: datetime | None = None
    applies_within_s: float = Field(30, description="The predictor re-reads the thresholds this often.")


class ModelVersionOut(BaseModel):
    version: str
    created_at: str | None = None
    model: str | None = None
    description: str | None = None
    precision: str | None = None
    components: list[str] = Field(default_factory=list)
    cv_mae: float | None = None
    test_mae: float | None = None
    baseline_test_mae: float | None = None
    online_mae_s: float | None = Field(None, description="On the stream now (the active version only).")
    online_closed: int | None = None
    online: dict[str, Any] | None = Field(
        None, description="Retrained on the stream: the correction of the base version and its check."
    )
    active: bool = False


class RetrainOut(BaseModel):
    """The last retraining on the stream (ml-service ``GET /model/retrain``)."""

    state: Literal["idle", "running", "done", "error"] = "idle"
    base_version: str | None = None
    version: str | None = Field(None, description="The new version (activated separately; null: not better).")
    improved: bool | None = Field(None, description="The correction lowered the MAE (else no new version).")
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_fit: int | None = None
    n_eval: int | None = None
    mae_before_s: float | None = Field(None, description="MAE of the base on the later closed forecasts.")
    mae_after_s: float | None = Field(None, description="MAE of the new version on them.")
    error: str | None = None


class FallbackOut(BaseModel):
    """The forecast of the predictor while ml-service is down or slow: ``intercept_s + coef · feature``."""

    intercept_s: float
    coef: float
    feature: str
    forecasts: int = Field(0, description="Forecasts of the vehicles now.")
    share: float | None = Field(None, description="Share of them made by the fallback (null: none now).")


class ModelsOut(BaseModel):
    active: str | None = None
    ml: str = Field(description="State of ml-service: up / down / disabled.")
    versions: list[ModelVersionOut]
    retrain: RetrainOut | None = Field(None, description="The last retraining (null: ml-service down).")
    fallback: FallbackOut | None = None


class ServiceOut(BaseModel):
    name: str
    state: Literal["up", "down", "degraded", "unknown", "disabled"]
    detail: str | None = None
    latency_ms: float | None = None


class ServicesOut(BaseModel):
    checked_at: datetime
    services: list[ServiceOut]


class UnitOut(BaseModel):
    unit_id: int
    tr_id: int | None = None
    route_id: str | None = None
    scheduled: bool | None = None
    status: str
    risk: str | None = None
    last_packet_at: datetime | None = None


async def _call(
    client: HttpClient | None, method: str, path: str, body: Any = None, timeout: float = 5.0
) -> Any:
    """JSON of a call to a neighbour service; ``HTTPException`` 502/503 when it fails."""
    if client is None:
        raise HTTPException(status_code=503, detail="service is not configured")
    payload = None if body is None else json.dumps(body).encode()
    try:
        status, data = await asyncio.wait_for(client.request(method, path, payload), timeout)
    except Exception as exc:
        client.close()
        raise HTTPException(status_code=502, detail=f"unavailable: {exc or type(exc).__name__}") from exc
    try:
        content = json.loads(data) if data else None
    except ValueError:
        content = {"detail": data.decode("utf-8", "replace")}
    if status >= 400:
        detail = content.get("detail") if isinstance(content, dict) else content
        raise HTTPException(status_code=status, detail=detail)
    return content


def _flat(row: dict[str, Any]) -> dict[str, Any]:
    """A journal row for a spreadsheet: a nested object's scalar fields become ``key_field`` columns
    (``cause`` → ``cause_code``, ``cause_text``, …); lists and deeper objects stay JSON."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for sub, x in v.items():
                out[f"{k}_{sub}"] = json.dumps(x, ensure_ascii=False) if isinstance(x, dict | list) else x
        else:
            out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    return out


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    flat = [_flat(r) for r in rows]
    if flat:
        fields = list(dict.fromkeys(k for r in flat for k in r))
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)
    return buf.getvalue()


def admin_router(svc: ApiService) -> APIRouter:
    """The ``/api/admin`` routes over the api service."""
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    @router.get("/settings", response_model=SettingsOut, summary="Risk and alert thresholds")
    async def get_settings() -> SettingsOut:
        try:
            rows = await svc.journal.fetch(
                "SELECT key, value, updated_at FROM settings WHERE key = ANY($1::text[])", list(SETTING_KEYS)
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"journal unavailable: {exc}") from exc
        values = {
            r["key"]: json.loads(r["value"]) if isinstance(r["value"], str) else r["value"] for r in rows
        }
        updated = max((r["updated_at"] for r in rows), default=None)
        risk, alert = values.get("risk_thresholds") or {}, values.get("alert_thresholds") or {}
        return SettingsOut(
            risk=RiskSettings.model_validate(
                {k: v for k, v in risk.items() if k in RiskSettings.model_fields}
            ),
            alert=AlertSettings.model_validate(
                {k: v for k, v in alert.items() if k in AlertSettings.model_fields}
            ),
            updated_at=updated,
            applies_within_s=svc.settings.settings_refresh_s,
        )

    @router.put("/settings", response_model=SettingsOut, summary="Change the thresholds (applied on the fly)")
    async def put_settings(body: SettingsOut) -> SettingsOut:
        """Write the thresholds to ``settings``; the predictor takes them within ``applies_within_s``. Other
        keys of the rows (hysteresis and the like) are kept."""
        try:
            for key, part in (("risk_thresholds", body.risk), ("alert_thresholds", body.alert)):
                await svc.journal.fetch(
                    "INSERT INTO settings (key, value, updated_by, updated_at) "
                    "VALUES ($1, $2::jsonb, 'admin', now()) "
                    "ON CONFLICT (key) DO UPDATE SET value = settings.value || EXCLUDED.value, "
                    "updated_by = EXCLUDED.updated_by, updated_at = EXCLUDED.updated_at",
                    key,
                    json.dumps(part.model_dump()),
                )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"journal unavailable: {exc}") from exc
        svc.events.write("settings", "thresholds changed", details=body.model_dump(exclude={"updated_at"}))
        return await get_settings()

    def fallback() -> FallbackOut:
        s = svc.settings
        sources = [fc.get("source") for fc in svc.cache.forecasts.values()]
        n = len(sources)
        return FallbackOut(
            intercept_s=s.fallback_intercept_s,
            coef=s.fallback_coef,
            feature=s.fallback_feature,
            forecasts=n,
            share=sum(x == "fallback" for x in sources) / n if n else None,
        )

    @router.get("/models", response_model=ModelsOut, summary="Model versions")
    async def models() -> ModelsOut:
        """Versions of the ml-service registry with CV / test MAE and, for the active one, the online MAE of
        the forecasts closed on the stream (current timeline); the last retraining and the fallback."""
        try:
            versions = await _call(svc.ml, "GET", "/model/versions")
        except HTTPException:
            state = "disabled" if svc.ml is None else "down"
            return ModelsOut(active=None, ml=state, versions=[], fallback=fallback())
        retrain = None
        with contextlib.suppress(HTTPException, ValueError):
            retrain = RetrainOut.model_validate(await _call(svc.ml, "GET", "/model/retrain"))
        items = [ModelVersionOut.model_validate(v) for v in versions or []]
        active = next((v for v in items if v.active), None)
        if active is not None:
            try:
                total = (
                    await svc.journal.horizon(
                        epoch=svc.epoch, since=epoch_start(svc.epoch), closed_after=None, late_s=120.0
                    )
                )["total"]
                active.online_mae_s, active.online_closed = total["mae_s"], total["closed"]
            except Exception as exc:
                log.debug("online MAE for the models: %s", exc)
        return ModelsOut(
            active=active.version if active else None,
            ml="up",
            versions=items,
            retrain=retrain,
            fallback=fallback(),
        )

    @router.post(
        "/models/{version}/activate", response_model=ModelsOut, summary="Switch ml-service to a version"
    )
    async def activate(
        version: Annotated[str, Path(pattern=VERSION_PATTERN, max_length=64)],
    ) -> ModelsOut:
        """Load the version in ml-service and switch to it without stopping (the previous one serves until the
        new one is loaded); the next forecasts of the predictor use it."""
        await _call(svc.ml, "POST", "/model/reload", {"version": version}, timeout=60.0)
        svc.events.write("model", f"model {version} activated")
        return await models()

    @router.post(
        "/models/retrain",
        response_model=ModelsOut,
        status_code=202,
        summary="Retrain the active model on the stream (async; the state in GET /models)",
    )
    async def retrain() -> ModelsOut:
        """Start the correction of the active version by its forecasts closed on the stream in ml-service; the
        result is a new version (``retrain.version``), activated separately. 409 while a run is going."""
        run = await _call(svc.ml, "POST", "/model/retrain")
        svc.events.write("model", f"retraining of {run.get('base_version')} on the stream started")
        return await models()

    @router.get("/journal", summary="Journal of alerts or forecasts (JSON / CSV)")
    async def journal(
        kind: Annotated[Literal["alerts", "predictions"], Query()] = "alerts",
        from_: Annotated[datetime | None, Query(alias="from", description="Stream time from.")] = None,
        to: Annotated[datetime | None, Query(description="Stream time to.")] = None,
        tr_id: Annotated[int | None, Query()] = None,
        status: Annotated[Literal["open", "closed"] | None, Query(description="Forecasts only.")] = None,
        format: Annotated[Literal["json", "csv"], Query()] = "json",  # noqa: A002
        limit: Annotated[int, Query(ge=1, le=20000)] = 1000,
    ) -> Response:
        """Alerts (by issue time) or forecasts (by plan time of the target stop) of the current stream
        timeline, the latest ``limit`` first; ``format=csv`` downloads a file."""
        try:
            if kind == "alerts":
                rows = await svc.journal.alerts(
                    since=epoch_start(svc.epoch),
                    issued_after=from_,
                    issued_before=to,
                    tr_id=tr_id,
                    limit=limit,
                )
            else:
                rows = await svc.journal.predictions(
                    epoch=svc.epoch,
                    tr_id=tr_id,
                    status=status,
                    planned_from=from_,
                    planned_to=to,
                    limit=limit,
                    newest_first=True,
                )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"journal unavailable: {exc}") from exc
        if format == "csv":
            name = f"foresight-{kind}-{datetime.now(UTC):%Y%m%d-%H%M%S}.csv"
            return Response(
                "﻿" + _csv(rows),  # BOM: Excel opens the UTF-8 file with Cyrillic correctly
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{name}"'},
            )
        return Response(
            json.dumps({"count": len(rows), "items": rows}, ensure_ascii=False, default=str),
            media_type="application/json",
        )

    @router.get("/services", response_model=ServicesOut, summary="Health of all services")
    async def services() -> ServicesOut:
        """The api's own dependencies and a ``/health`` probe of the predictor, ml-service, the replayer."""

        async def probe(name: str, client: HttpClient | None) -> ServiceOut:
            if client is None:
                return ServiceOut(name=name, state="disabled")
            t0 = time.perf_counter()  # the loop clock of uvloop ticks in whole milliseconds
            try:
                status, data = await asyncio.wait_for(client.request("GET", "/health"), 3.0)
            except Exception as exc:
                client.close()  # a probe client of its own: no other request is on this connection
                return ServiceOut(name=name, state="down", detail=str(exc) or type(exc).__name__)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            try:
                body = json.loads(data)
            except ValueError:
                body = {}
            ok = status == 200 and body.get("status", "ok") == "ok"
            detail = body.get("model_version") or body.get("error")
            if detail is None and not ok:
                detail = body.get("status")
            return ServiceOut(name=name, state="up" if ok else "degraded", detail=detail, latency_ms=ms)

        probes = await asyncio.gather(*(probe(name, client) for name, client in svc.probes.items()))
        ingest = svc.check_ingest()
        own = [
            ServiceOut(name="api", state="up", detail=f"клиентов WebSocket: {svc.ws_clients}"),
            ServiceOut(
                name="ingest",
                state={True: "up", False: "down", None: "unknown"}[ingest],
                detail=svc.ingest_status.last_error if ingest is False else None,
            ),
            ServiceOut(name="redis", state=svc.redis_status.state, detail=svc.redis_status.last_error),  # type: ignore[arg-type]
            ServiceOut(name="postgres", state=svc.pg_status.state, detail=svc.pg_status.last_error),  # type: ignore[arg-type]
        ]
        return ServicesOut(checked_at=datetime.now(UTC), services=[*own, *probes])

    @router.get("/units", response_model=list[UnitOut], summary="Vehicle directory")
    async def units() -> list[UnitOut]:
        """Vehicles seen by the ingest: ``unit_id`` ↔ ``tr_id``, route (of the plan, also without a forecast
        now), link status, risk now."""
        now = datetime.now(UTC)
        link_down = svc.ingest_down()
        route_of = {
            int(tr): str(route["route_id"])
            for route in svc.cache.routes or []
            for tr in route.get("tr_ids") or []
            if route.get("route_id")
        }
        out = []
        for r in svc.cache.all():
            fc = svc.cache.forecast_of(r.tr_id) or {}
            scheduled = (
                None if svc.cache.scheduled is None or r.tr_id is None else r.tr_id in svc.cache.scheduled
            )
            out.append(
                UnitOut(
                    unit_id=r.unit_id,
                    tr_id=r.tr_id,
                    route_id=fc.get("route_id") or (route_of.get(r.tr_id) if r.tr_id is not None else None),
                    scheduled=scheduled,
                    status="offline" if link_down else str(svc.cache.status(r, now).value),
                    risk=fc.get("risk"),
                    last_packet_at=r.last_packet_at,
                )
            )
        return out

    return router
