"""Backend settings loaded from environment variables with the ``FORESIGHT_`` prefix.

One :class:`Settings` class serves all three services (``ingest``, ``predictor``, ``api``); each service reads
the fields it needs. Every field can be overridden by an environment variable, e.g.
``FORESIGHT_REDIS_URL=redis://redis:6379/0`` or ``FORESIGHT_NDTP_PORT=9301``.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.clock import JumpHook, StreamClock

_REPO_DATASET = Path(__file__).resolve().parent.parent / "dataset"


class Settings(BaseSettings):
    """Runtime configuration of the backend services."""

    model_config = SettingsConfigDict(env_prefix="FORESIGHT_", extra="ignore")

    # ---- common ----------------------------------------------------------------------------------
    log_level: str = Field("info", description="Log level for the service and uvicorn.")
    instance: str = Field(
        default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}",
        description="Instance name (service events, Redis consumer name by default).",
    )
    http_host: str = Field("0.0.0.0", description="Bind address of the HTTP servers.")
    http_port: int = Field(8000, ge=0, le=65535, description="HTTP port of the api service.")
    ingest_http_port: int = Field(8001, ge=0, le=65535, description="HTTP port of the ingest service.")
    predictor_http_port: int = Field(8002, ge=0, le=65535, description="HTTP port of the predictor service.")
    backoff_initial_s: float = Field(0.5, gt=0, description="First reconnect delay to Redis / PostgreSQL.")
    backoff_max_s: float = Field(5.0, gt=0, description="Largest reconnect delay to Redis / PostgreSQL.")

    # ---- Redis -----------------------------------------------------------------------------------
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL (bus, hot state, pub/sub).")
    redis_timeout_s: float = Field(5.0, gt=0, description="Redis socket connect/read timeout.")
    stream_maxlen: int = Field(200_000, ge=1000, description="Approximate MAXLEN of the telemetry stream.")
    vehicle_ttl_s: int = Field(600, ge=1, description="TTL of the per-vehicle hot-state hash.")

    # ---- PostgreSQL ------------------------------------------------------------------------------
    database_url: str = Field(
        "postgresql://foresight:foresight@localhost:5432/foresight",
        description="PostgreSQL DSN; an empty string disables the database (events are dropped).",
    )
    db_buffer_max: int = Field(50_000, ge=1, description="Rows kept in memory while PostgreSQL is down.")
    db_batch_size: int = Field(500, ge=1, description="Rows per INSERT batch.")
    db_flush_interval_s: float = Field(0.5, gt=0, description="Idle period of the database writer loop.")
    db_health_interval_s: float = Field(5.0, gt=0, description="Period of PostgreSQL health probes.")

    # ---- ingest ----------------------------------------------------------------------------------
    ndtp_enabled: bool = Field(True, description="Start the NDTP TCP server on startup.")
    ndtp_host: str = Field("0.0.0.0", description="Bind address of the NDTP TCP server.")
    ndtp_port: int = Field(9201, ge=0, le=65535, description="NDTP TCP port; 0 picks a free port (tests).")
    ndtp_max_data_size: int = Field(
        8192, ge=64, le=65535, description="Largest accepted NPL dataSize; larger means garbage."
    )
    ndtp_max_crc_errors: int = Field(
        32, ge=1, description="CRC errors since the last valid frame before disconnect (anti-DoS)."
    )
    ndtp_read_timeout_s: float = Field(
        300.0, gt=0, description="Close a connection that sent nothing for this long (half-open sockets)."
    )
    ndtp_first_frame_timeout_s: float = Field(
        10.0,
        gt=0,
        description="Close a connection that sent no valid frame (handshake or realtime) this long after "
        "accept (anti-DoS).",
    )
    ndtp_max_garbage_bytes: int = Field(
        65_536,
        ge=1024,
        description="Bytes discarded since the last valid frame (or accept) before disconnect (anti-DoS).",
    )
    ndtp_max_connections: int = Field(
        1024,
        ge=1,
        description="Open NDTP connections at most; further ones are closed at once (anti-DoS).",
    )
    stats_connections_max: int = Field(
        100,
        ge=0,
        description="Connections listed in the ingest stats (oldest first; connections_active is exact).",
    )
    unit_map_dir: Path = Field(_REPO_DATASET, description="Dataset directory with <split>/traffic.csv.")
    unit_map_splits: str = Field(
        "test,train", description="Comma-separated splits whose traffic.csv define unit_id -> tr_id."
    )
    bus_buffer_max: int = Field(100_000, ge=1, description="Events kept in memory while Redis is down.")
    bus_batch_size: int = Field(1000, ge=1, description="Events per Redis pipeline.")
    bus_flush_interval_s: float = Field(0.25, gt=0, description="Idle period of the Redis publisher loop.")
    stats_interval_s: float = Field(2.0, gt=0, description="Period of ingest stats snapshots in Redis.")

    # ---- stream clock (the ingest decides, the predictor follows; see backend.clock) ---------------
    clock_jump_s: float = Field(
        300.0,
        gt=0,
        description="Fixes further than this from the stream clock are off the timeline; a quorum of devices "
        "moving that far together is a source restart (the clock jumps).",
    )
    clock_confirm: int = Field(
        3, ge=1, description="Consecutive off-timeline fixes before a device counts as moved."
    )
    clock_quorum: float = Field(
        0.5, gt=0, le=1, description="Share of the active devices that must move together for a clock jump."
    )
    clock_active_s: float = Field(
        60.0, gt=0, description="A device takes part in the quorum if heard within this many seconds."
    )
    clock_min_time: datetime = Field(
        datetime(2020, 1, 1, tzinfo=UTC), description="Earliest plausible fix time; earlier ones are garbage."
    )
    realtime_plan_day: date | None = Field(
        date(2026, 1, 6),
        description="Day of the schedule: a realtime stream (current UTC fix times: the emulator, a live "
        "feed) goes onto it by the local time of day (backend.clock.to_plan_day); replays stay. None: off.",
    )
    realtime_tz_offset_s: float = Field(
        10_800.0, description="Local time of the schedule minus UTC, s (Moscow)."
    )

    # ---- predictor -------------------------------------------------------------------------------
    consumer_group: str = Field("predictors", description="Redis consumer group of the predictors.")
    consumer_name: str | None = Field(
        None, description="Consumer name; defaults to the host name (docker-compose sets predictor-1)."
    )
    consumer_batch: int = Field(500, ge=1, description="Entries per XREADGROUP call.")
    consumer_block_ms: int = Field(1000, ge=1, description="XREADGROUP BLOCK timeout.")
    consumer_claim_idle_ms: int = Field(
        30_000, ge=0, description="Pending entries idle this long are claimed from dead consumers."
    )
    consumer_gc_idle_ms: int = Field(
        600_000,
        ge=0,
        description="Consumers without pending entries idle this long are removed from the group (0: never).",
    )
    consumer_start_id: str = Field("0", description="Stream id the consumer group starts from on creation.")
    history_max_entries: int = Field(
        200_000,
        ge=0,
        description="Stream entries read back at start to refill the track windows (0: start empty).",
    )
    track_window_s: float = Field(
        5400.0,
        gt=0,
        description="Track history kept per tr_id (stream time): the features look an hour back, the stop "
        "detector resumes from the last matched stop.",
    )
    tick_period_s: float = Field(30.0, gt=0, description="Prediction tick period in stream time.")
    tick_lag_s: float = Field(
        1.0,
        gt=0,
        description="Behind real time by more than this (wall seconds since the ingest received the event), "
        "a tick the stream has already passed is skipped instead of queued.",
    )

    # ---- forecasts (predictor; see backend.forecast) -----------------------------------------------
    forecast_enabled: bool = Field(
        True, description="Run the forecast engine on every tick (off: the tick only logs the windows)."
    )
    schedule_dir: Path | None = Field(
        None, description="Dataset directory with the plan schedule (default: unit_map_dir)."
    )
    schedule_split: str = Field(
        "test",
        description="Split of the plan schedule: test / train -> <split>/schedule.csv, validate -> "
        "validate/schedule_plan.csv. Fact columns are dropped on load.",
    )
    ml_url: str = Field(
        "http://localhost:8003", description="ml-service base URL; empty: fallback forecasts only."
    )
    ml_timeout_s: float = Field(0.5, gt=0, description="Timeout of one POST /predict (then fallback).")
    ml_retry_s: float = Field(
        5.0, ge=0, description="After a failed call the ML service is not asked again for this long."
    )
    ml_explain: bool = Field(True, description="Ask ml-service for feature contributions (cause factors).")
    ml_sequences: bool = Field(
        False,
        description="Send the telemetry sequences (shared.sequences, points <= t) with POST /predict when "
        "the active model has a sequence component (ML v2: CatBoost + GRU); without them ml-service answers "
        "with the CatBoost part (and the quantiles, p_late, expected error). Off by default: the GRU of v2 "
        "forecasts a residual over the dataset hint cur_dev_s, which the stream only approximates, and on "
        "the test day stream it is 4 s worse in MAE than the CatBoost part (docs/online-validation.md).",
    )
    horizon_min_s: float = Field(600.0, gt=0, description="Forecast targets: plan in (t + min, t + max].")
    horizon_max_s: float = Field(900.0, gt=0, description="Forecast targets: plan in (t + min, t + max].")
    active_s: float = Field(
        900.0, gt=0, description="A vehicle is active (gets forecasts) if its last GPS fix is this recent."
    )
    near_stop_m: float = Field(
        150.0,
        ge=0,
        description="No forecast for a target the vehicle is standing at (closer than this): it may be "
        "passing it right now, the forecast would come after the fact. A terminal departure is forecast: the "
        "vehicle waits there, its pass (the departure) is ahead.",
    )
    cur_dev_mode: Literal["median3", "last_stop", "last_regular", "nan"] = Field(
        "median3",
        description="Online analogue of the dataset hint cur_dev_s from the stops confirmed by the detector: "
        "median delay of the last 3 (median3), delay at the last one (last_stop), at the last one that is "
        "not a terminal (last_regular), or none (nan). Chosen by the online validation "
        "(docs/online-validation.md): median3 gives the lowest MAE of the model on the stream.",
    )
    cur_dev_lag_s: float = Field(
        0.0, ge=0, description="last_stop: only passages confirmed this long before the tick count."
    )
    fallback_feature: str = Field(
        "dev_1",
        description="Deviation of the fallback forecast (a feature of shared.features): dev_1 is the delay "
        "at the last stop confirmed by the detector.",
    )
    fallback_coef: float = Field(
        0.23,
        description="Fallback forecast = intercept + coef * deviation; MAE-optimal on train (real vehicles), "
        "scripts/validate_online.py fallback-coef.",
    )
    fallback_intercept_s: float = Field(14.0, description="Fallback forecast intercept, s.")
    alert_hysteresis_s: float = Field(
        20.0,
        ge=0,
        description="Hysteresis of the incident level: a vehicle leaves red only below red_delay_s minus "
        "this (and green only below green_delay_s minus this), so a forecast wandering around a threshold "
        "does not flap the incident and re-raise alerts.",
    )
    alert_hysteresis_p: float = Field(
        0.1, ge=0, description="The same hysteresis for the p_late thresholds (ML v2)."
    )
    alert_confirm_ticks: int = Field(
        1,
        ge=1,
        description="Consecutive ticks at a higher level before an incident opens or escalates (and alerts).",
    )
    incident_clear_ticks: int = Field(
        4,
        ge=1,
        description="Consecutive ticks below the yellow exit threshold before a delay incident closes; a new "
        "alert for the vehicle comes only after that (or with an escalation).",
    )
    routes_segments: Path | None = Field(
        None,
        description="Cache of the route geometry by GPS (python -m shared.routes build); default: the file "
        "of the repository (backend/assets/route_segments.json); missing: straight lines between the stops.",
    )
    bunching_s: float = Field(
        60.0, gt=0, description="Bunching: two vehicles of a route forecast closer than this at a stop."
    )
    gap_s: float = Field(
        300.0, gt=0, description="Interval gap: forecast headway exceeds the planned one by this much..."
    )
    gap_factor: float = Field(2.0, gt=1, description="... and is at least this many times the planned one.")
    bunching_horizon_s: float = Field(
        1200.0, gt=0, description="Stops with plan in (t, t + this] are checked for bunching."
    )
    online_window_s: float = Field(3600.0, gt=0, description="Window of the online MAE, stream seconds.")
    online_min_closed: int = Field(
        30, ge=1, description="Closed forecasts before the online MAE is exported."
    )
    expire_after_s: float = Field(
        1800.0, gt=0, description="An open forecast without a detector decision this long after plan expires."
    )
    settings_refresh_s: float = Field(30.0, gt=0, description="Period of risk / alert threshold re-reads.")
    log_updates: bool = Field(True, description="Write every forecast of every tick to prediction_updates.")

    # ---- api -------------------------------------------------------------------------------------
    stale_after_s: float = Field(30.0, gt=0, description="Seconds of silence before a vehicle is stale.")
    offline_after_s: float = Field(
        120.0, gt=0, description="Vehicle becomes 'offline' after this many seconds of silence."
    )
    ws_interval_s: float = Field(1.0, gt=0, description="Maximum period between WebSocket delta messages.")
    ws_snapshot_every_s: float = Field(30.0, gt=0, description="Period of full snapshots on the WebSocket.")
    api_resync_s: float = Field(10.0, gt=0, description="Period of full hot-state re-reads from Redis.")
    ingest_stale_after_s: float = Field(
        10.0,
        gt=0,
        description="api: ingest stats older than this (they are written every stats_interval_s) mean the "
        "ingest is down: /health degraded, stats unavailable, every vehicle offline.",
    )
    predictor_url: str = Field(
        "http://localhost:8002",
        description="api: predictor base URL (its /metrics for the performance page); empty: not scraped.",
    )
    replayer_url: str = Field(
        "http://localhost:8010",
        description="api: replayer control API behind /api/replay/*; empty: no proxy.",
    )
    api_schedule: bool = Field(
        True,
        description="api: load the plan schedule (schedule_dir / schedule_split, as the predictor) for the "
        "stringline; without it the stringline has no planned lines.",
    )
    perf_scrape_s: float = Field(5.0, gt=0, description="api: period of the /metrics scrapes (performance).")

    @property
    def splits(self) -> list[str]:
        """Dataset splits used for the ``unit_id -> tr_id`` map."""
        return [s.strip() for s in self.unit_map_splits.split(",") if s.strip()]

    @property
    def consumer(self) -> str:
        """Redis consumer name (stable across restarts of the same container)."""
        return self.consumer_name or socket.gethostname()

    def stream_clock(self, on_jump: JumpHook | None = None) -> StreamClock:
        """A :class:`~backend.clock.StreamClock` with the ``clock_*`` settings.

        Args:
            on_jump: Hook called after every jump of the clock.
        """
        return StreamClock(
            jump_s=self.clock_jump_s,
            confirm=self.clock_confirm,
            quorum=self.clock_quorum,
            active_s=self.clock_active_s,
            min_ts=self.clock_min_time.timestamp(),
            on_jump=on_jump,
        )
