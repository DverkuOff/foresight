"""Control API of the replayer (FastAPI, port 8010 by default).

Run with ``python -m replayer serve``; the OpenAPI UI is at ``/docs``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from replayer import __version__
from replayer.bridge import HttpClient
from replayer.clock import Clock
from replayer.controller import Loader, ReplayController
from replayer.engine import MAX_SPEED, MIN_SPEED, ReplayConfig
from replayer.metrics import build_registry
from replayer.source import load_replay_data

_DESCRIPTION = """
Foresight replayer: plays the historical `traffic.csv` of a split back as real NDTP over TCP (handshake +
realtime Nav00 with CRC, one connection per device) in the order and at the pace of `receive_time`, with
`timestamp = event_time` inside the packets. The `bridge` mode drives the official emulator instead.

* `POST /replay/start` — start (or restart) with optional overrides;
* `POST /replay/stop`, `POST /replay/speed` — control the running replay;
* `GET /replay/status` — progress, connections, lag, events; `epoch` grows on every (re)start of the data;
* `GET /metrics` — Prometheus metrics.
"""

State = Literal["idle", "loading", "created", "waiting", "running", "finished", "stopped", "failed"]


class StartIn(BaseModel):
    """Parameters of ``POST /replay/start``; omitted fields keep their current values."""

    split: Literal["train", "test", "validate"] | None = Field(None, description="Dataset split.")
    speed: float | None = Field(None, ge=MIN_SPEED, le=MAX_SPEED, description="Data seconds per wall second.")
    start: str | None = Field(
        None,
        description="Start of the receive_time window: `HH:MM[:SS]` of the data day or ISO datetime; "
        "empty string clears it.",
        examples=["07:00"],
    )
    until: str | None = Field(None, description="End of the window (exclusive); empty string clears it.")
    units: list[int] | None = Field(None, description="unit_id or tr_id values; empty list means all.")
    loop: bool | None = Field(None, description="Start over after the last packet.")
    mode: Literal["ndtp", "bridge"] | None = Field(None, description="Own NDTP or the official emulator.")


class SpeedIn(BaseModel):
    """Body of ``POST /replay/speed``."""

    speed: float = Field(..., ge=MIN_SPEED, le=MAX_SPEED, description="Data seconds per wall second.")


class ReplayEventOut(BaseModel):
    """A lifecycle event: start, restart, speed, finish, stop or fail."""

    at: datetime
    event: str
    epoch: int
    details: dict[str, Any] = {}


class ReplayStatusOut(BaseModel):
    """State of the replay."""

    state: State
    epoch: int = Field(description="Number of data (re)starts; consumers reset the stream clock on change.")
    mode: str
    split: str
    speed: float
    loop: bool
    start: str | None = None
    until: str | None = None
    units_filter: list[int] = []
    target: str
    cycle: int = 0
    units: int = 0
    connections_active: int = 0
    packets_total: int = 0
    packets_dispatched: int = 0
    progress: float = 0.0
    packets_sent: int = 0
    packets_dropped: int = 0
    reconnects: int = 0
    bridge_posts: int = 0
    backlog: int = 0
    lag_s: float = 0.0
    data_time: datetime | None = None
    data_first: datetime | None = None
    data_last: datetime | None = None
    started_at: datetime | None = None
    error: str | None = None
    events: list[ReplayEventOut] = []


class HealthOut(BaseModel):
    """Liveness of the service (the replay itself may be idle)."""

    status: Literal["ok"]
    state: State
    epoch: int


def create_app(
    config: ReplayConfig | None = None,
    *,
    autostart: bool = False,
    loader: Loader = load_replay_data,
    clock: Clock | None = None,
    http: HttpClient | None = None,
) -> FastAPI:
    """Build the control API.

    Args:
        config: Default replay parameters.
        autostart: Start replaying with ``config`` when the app starts.
        loader: Split loader (tests pass one bound to a temporary dataset).
        clock: Clock for sessions (tests).
        http: HTTP client for the bridge mode (tests).

    Returns:
        The application; ``app.state.controller`` is the :class:`ReplayController`.
    """
    controller = ReplayController(config, clock=clock, http=http, loader=loader)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if autostart:
            controller.start_in_background()
        try:
            yield
        finally:
            await controller.shutdown()

    app = FastAPI(
        title="Foresight · Replayer", version=__version__, description=_DESCRIPTION, lifespan=lifespan
    )
    app.state.controller = controller
    app.state.registry = build_registry(controller)

    @app.get("/health", tags=["ops"], summary="Liveness", response_model=HealthOut)
    async def health() -> dict[str, Any]:
        return {"status": "ok", "state": controller.state, "epoch": controller.epoch}

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics",
        response_class=Response,
        responses={200: {"content": {CONTENT_TYPE_LATEST: {}}, "description": "Prometheus text format"}},
    )
    async def metrics(request: Request) -> Response:
        return Response(generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/replay/status", tags=["replay"], summary="Replay status", response_model=ReplayStatusOut)
    async def replay_status() -> dict[str, Any]:
        return controller.status()

    @app.post(
        "/replay/start",
        tags=["replay"],
        summary="Start or restart the replay",
        response_model=ReplayStatusOut,
        responses={
            400: {"description": "Invalid parameters or empty selection"},
            404: {"description": "No data"},
        },
    )
    async def replay_start(body: StartIn | None = None) -> dict[str, Any]:
        params = body or StartIn()
        try:
            return await controller.start(**params.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/replay/stop", tags=["replay"], summary="Stop the replay", response_model=ReplayStatusOut)
    async def replay_stop() -> dict[str, Any]:
        return await controller.stop()

    @app.post(
        "/replay/speed", tags=["replay"], summary="Change the replay speed", response_model=ReplayStatusOut
    )
    async def replay_speed(body: SpeedIn) -> dict[str, Any]:
        return controller.set_speed(body.speed)

    return app
