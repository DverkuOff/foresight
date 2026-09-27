"""Pydantic response models of the public APIs (they define the Swagger/OpenAPI schemas)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.runtime import DependencyStatus
from backend.state import LinkStatus


class DependencyOut(BaseModel):
    """Availability of one external dependency."""

    state: Literal["up", "down", "unknown", "disabled"] = Field(description="Current state.")
    since: datetime | None = Field(None, description="Time of the last transition (UTC).")
    error: str | None = Field(None, description="Last error, if any.")
    outages: int = Field(0, description="Number of outages since the service started.")

    @classmethod
    def of(cls, status: DependencyStatus) -> DependencyOut:
        """Build from a :class:`~backend.runtime.DependencyStatus`."""
        return cls(
            state=status.state,  # type: ignore[arg-type]
            since=status.since,
            error=status.last_error if status.ok is not True else None,
            outages=status.outages,
        )


class HealthOut(BaseModel):
    """Liveness of a service and the state of its dependencies."""

    status: Literal["ok", "degraded"] = Field(
        description="'degraded' if a dependency is down (or, for ingest, NDTP is not listening)."
    )
    service: str = Field(description="Service name.")
    version: str = Field(description="Backend version.")
    uptime_s: float = Field(description="Seconds since startup.")
    dependencies: dict[str, DependencyOut] = Field(
        description="Redis / PostgreSQL state; the api also lists the ingest (down: its stats went stale)."
    )
    ndtp_listening: bool | None = Field(None, description="Ingest: whether NDTP accepts connections.")
    ndtp_port: int | None = Field(None, description="Ingest: NDTP TCP port.")
    vehicles: int | None = Field(None, description="Number of known vehicles.")


class DoorsOut(BaseModel):
    """Door state from the IRMA cell (Irma04)."""

    zone: int
    odometer: int
    door_in: list[int] = Field(description="Passengers entered through doors 1..4.")
    door_out: list[int] = Field(description="Passengers exited through doors 1..4.")
    door_present: list[bool] = Field(description="Door sensors 1..4 present.")
    door_closed: list[bool] = Field(description="Doors 1..4 closed.")


class NextStopOut(BaseModel):
    """The vehicle's nearest stop with a forecast."""

    stop_id: int = Field(description="Planned arrival (tt_action_item_id).")
    stop_key: str = Field(description="Place of the stop (rounded coordinates, shared.routes).")
    name: str = Field(description="Stop name (address).")
    planned_at: datetime | None = Field(None, description="Planned arrival (UTC).")


class VehicleOut(BaseModel):
    """Latest state of one vehicle: last valid position, link status, forecast and derived features.

    The forecast fields (API contract §2) and the derived features (criterion 3 of the task: current deviation
    from the plan, speed on the segment, dwell / idle time) come from the predictor; they are null for a
    vehicle without a forecast, and a vehicle that is not in the plan schedule has ``scheduled: false``,
    ``route_id: null`` and ``risk: unknown``.
    """

    unit_id: int = Field(description="NDTP device id (unitId).")
    tr_id: int | None = Field(None, description="Vehicle id in the schedule; null for an unknown device.")
    status: LinkStatus = Field(description="Link status: online / stale / offline.")
    connected: bool = Field(description="Whether a TCP connection of this device is open.")
    lat: float | None = Field(
        None,
        description="Latitude of the last valid fix, degrees (WGS-84); null until the device sends one (an "
        "invalid fix never moves the vehicle to 0, 0).",
    )
    lon: float | None = Field(None, description="Longitude of the last valid fix, degrees (WGS-84).")
    valid: bool | None = Field(None, description="Whether the latest GPS fix is valid.")
    position_time: datetime | None = Field(None, description="Fix time of the position shown (UTC).")
    position_age_s: float | None = Field(
        None,
        description="How much older the position shown is than the latest fix, s (0: the latest fix is "
        "valid; > 0: GPS lost, the last known position).",
    )
    speed_kmh: int | None = Field(None, description="Average speed, km/h.")
    speed_max_kmh: int | None = Field(None, description="Maximum speed, km/h.")
    course_deg: int | None = Field(None, description="Course, degrees 0..360.")
    altitude_m: int | None = Field(None, description="Altitude, m.")
    satellites: int | None = Field(None, description="Number of satellites.")
    event_time: datetime | None = Field(None, description="Fix time from the packet (UTC).")
    received_at: datetime | None = Field(None, description="Server time of the last position (UTC).")
    last_packet_at: datetime | None = Field(None, description="Server time of the last frame (UTC).")
    age_s: float | None = Field(None, description="Seconds since the last frame.")
    packets: int = Field(0, description="Realtime packets received.")
    reconnects: int = Field(0, description="Number of reconnections (handshakes after the first one).")
    doors: DoorsOut | None = Field(None, description="Door state, if the device reports Irma04.")
    scheduled: bool | None = Field(
        None,
        description="The vehicle is in the plan schedule of the forecasts (false: no route, no forecasts, "
        "risk unknown; null: the predictor has not published its plan yet).",
    )
    route_id: str | None = Field(None, description="Route (R1, R2, …, derived from the plan).")
    risk: Literal["green", "yellow", "red", "unknown"] = Field(
        "unknown",
        description="Risk: the level of the vehicle's open incident (held with a hysteresis), else of its "
        "nearest forecast; unknown without a forecast.",
    )
    current_delay_s: float | None = Field(
        None, description="Current deviation from the plan, s (+ late): detector and position on the plan."
    )
    pred_delay_s: float | None = Field(None, description="Forecast delay at the nearest target stop, s.")
    p10: float | None = Field(None, description="P10 of the forecast delay, s (ML v2).")
    p90: float | None = Field(None, description="P90 of the forecast delay, s (ML v2).")
    p_late: float | None = Field(None, description="Probability of a delay over 120 s (ML v2).")
    next_stop: NextStopOut | None = Field(None, description="Nearest stop with a forecast.")
    incident_id: int | None = Field(None, description="Open incident of the vehicle.")
    segment_speed_kmh: float | None = Field(
        None, description="Mean speed on the current segment (from the last passed stop, ≤ 15 min), km/h."
    )
    dwell_s: float | None = Field(None, description="Standing time at the current place (speed < 2 km/h), s.")
    idle_s: float | None = Field(None, description="Seconds standing in the last 10 minutes.")
    forecast_source: Literal["model", "fallback"] | None = Field(
        None, description="Source of the forecast: ml-service or the fallback formula."
    )


class VehicleListOut(BaseModel):
    """All vehicles with summary counters."""

    count: int
    server_time: datetime = Field(description="Current server time (UTC).")
    stream_time: datetime | None = Field(None, description="Stream clock: max fix time seen (UTC).")
    status_counts: dict[LinkStatus, int] = Field(description="Number of vehicles per link status.")
    degraded: bool = Field(
        False,
        description="Redis is unavailable or the ingest is down (then every vehicle is offline): this is the "
        "last known state.",
    )
    synced_at: datetime | None = Field(None, description="Last successful update from Redis (UTC).")
    vehicles: list[VehicleOut]


class ConnectionOut(BaseModel):
    """One open NDTP connection."""

    conn_id: int
    peer: str
    unit_id: int | None
    connected_at: datetime
    frames: int
    last_frame_at: datetime | None


class IngestStatsOut(BaseModel):
    """NDTP ingest counters (and the state of its Redis publisher)."""

    available: bool = Field(
        True,
        description="False if no stats were received from the ingest yet, or (api) they are older than "
        "ingest_stale_after_s: the ingest is down, the live fields are zeroed, the counters are the last "
        "known.",
    )
    degraded: bool = Field(
        False, description="api: Redis is unavailable or the ingest is down, the stats are the last known."
    )
    updated_at: datetime | None = Field(None, description="When the ingest produced these stats (UTC).")
    age_s: float | None = Field(None, description="api: seconds since the ingest produced these stats.")
    stream_time: datetime | None = Field(None, description="Stream clock (UTC).")
    clock_epoch: int = Field(0, description="Epoch of the stream clock; changes when the clock jumps.")
    clock_resets: int = Field(0, description="Stream clock jumps (source restarts).")
    clock_garbage: int = Field(0, description="Fixes with an implausible time (ignored by the clock).")
    clock_off_timeline: int = Field(0, description="Fixes further than the jump threshold from the clock.")
    listening: bool = False
    port: int = 0
    started_at: datetime | None = None
    connections_active: int = 0
    connections_total: int = 0
    disconnects: int = 0
    read_timeouts: int = 0
    bytes_received: int = 0
    bytes_discarded: int = Field(0, description="Bytes skipped while resynchronising on 0x7E7E.")
    frames: int = Field(0, description="Frames with a valid CRC.")
    handshakes: int = 0
    realtime_packets: int = 0
    nav_records: int = 0
    other_frames: int = Field(0, description="Valid frames of other service/type (ignored).")
    crc_errors: int = Field(0, description="Frames dropped because of a CRC mismatch.")
    bad_headers: int = Field(0, description="Signatures rejected because of an implausible NPL header.")
    parse_errors: int = Field(0, description="Frames with a malformed body or truncated cells.")
    unknown_cells: int = Field(0, description="Packets where parsing stopped at an unknown cell type.")
    listener_errors: int = 0
    abusive_disconnects: int = Field(
        0, description="Connections closed for exceeding the CRC error or garbage budget."
    )
    first_frame_timeouts: int = Field(0, description="Connections closed without a valid frame in time.")
    rejected_connections: int = Field(0, description="Connections closed at once: the connection limit.")
    packets_per_s: float = Field(0.0, description="Realtime packets per second over the last 10 s.")
    redis: Literal["up", "down", "unknown", "disabled"] = Field(
        "unknown", description="Ingest -> Redis link."
    )
    bus_published: int = Field(0, description="Events written to the telemetry stream.")
    bus_buffered: int = Field(0, description="Events waiting in memory (Redis down or catching up).")
    bus_evicted: int = Field(0, description="Events dropped because the buffer overflowed.")
    unmapped_events: int = Field(0, description="Events from devices without a known tr_id.")
    connections: list[ConnectionOut] = Field(
        default_factory=list,
        description="Open connections, oldest first, at most stats_connections_max (connections_active is "
        "exact).",
    )


class TickTimingsOut(BaseModel):
    """Durations of the parts of the last tick, ms."""

    total: float = 0.0
    detector: float = Field(0.0, description="Stop detector on the tracks.")
    features: float = Field(0.0, description="Features of the forecast points (shared.features).")
    ml: float = Field(0.0, description="POST /predict to ml-service (or the fallback).")


class ForecastStatsOut(BaseModel):
    """Forecast engine of the predictor: what it forecasts, how the forecasts turn out."""

    enabled: bool = Field(description="False: no plan schedule, the ticks only log the windows.")
    schedule_split: str | None = Field(None, description="Split of the plan schedule.")
    vehicles_planned: int = Field(0, description="Vehicles in the plan schedule.")
    routes: int = Field(0, description="Routes derived from the plan (shared.routes).")
    ml: Literal["up", "down", "unknown", "disabled"] = Field("unknown", description="ml-service state.")
    model_version: str | None = Field(None, description="Model version of ml-service.")
    last_source: Literal["model", "fallback"] | None = Field(None, description="Source of the last tick.")
    active_vehicles: int = Field(0, description="Vehicles with a recent GPS fix at the last tick.")
    targets: int = Field(0, description="Forecast points (vehicle × stop 10–15 min ahead), last tick.")
    predictions_open: int = Field(0, description="Forecasts waiting for the pass of their stop.")
    predictions_issued: int = Field(0, description="Forecasts issued (each vehicle × stop once).")
    forecasts_model: int = Field(0, description="Forecast values from the model (all ticks).")
    forecasts_fallback: int = Field(0, description="Forecast values from the fallback formula (all ticks).")
    closed: int = Field(0, description="Forecasts closed with the detected fact.")
    missed: int = Field(0, description="Forecasts whose stop the detector marked as skipped.")
    expired: int = Field(0, description="Forecasts without a detector decision 30 min after the plan.")
    retroactive: int = Field(0, description="Forecasts issued at or after the pass (must stay 0).")
    skipped_near_stop: int = Field(0, description="Targets not forecast: the vehicle stands at the stop.")
    alerts: int = Field(0, description="Alerts raised (delay incidents opened / escalated, bunching).")
    alerts_retroactive: int = Field(0, description="Alerts issued at or after the pass (must stay 0).")
    alert_quality: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="Alerts checked against the detected fact: precision of red (fact > 120 s) and yellow "
        "(fact ≥ 60 s) alerts, share of late stops (> 120 s) whose vehicle was red / at least yellow while "
        "the stop was 10–15 min ahead.",
    )
    sequences_sent: int = Field(0, description="Telemetry sequences sent to ml-service (ML v2).")
    incidents_open: int = Field(0, description="Open delay incidents.")
    bunching_open: int = Field(0, description="Open bunching / headway gap incidents.")
    online_mae_s: float | None = Field(None, description="Online MAE over the last hour of stream time.")
    online_baseline_mae_s: float | None = Field(
        None, description="MAE of «forecast = online cur_dev_s at issue» on the same forecasts."
    )
    online_closed: int = Field(0, description="Closed forecasts in the online MAE window.")
    passages: int = Field(0, description="Stop decisions of the detector on the stream.")
    passages_matched: int = Field(0, description="... of them passes (the rest: skipped stops).")
    tick_ms: TickTimingsOut = Field(default_factory=TickTimingsOut, description="Last tick, ms.")


class PredictorStatsOut(BaseModel):
    """Predictor consumer and stream-clock state."""

    redis: Literal["up", "down", "unknown", "disabled"]
    postgres: Literal["up", "down", "unknown", "disabled"]
    consumer: str = Field(description="Consumer name in the group.")
    group: str = Field(description="Consumer group.")
    events: int = Field(description="Events processed.")
    events_per_s: float = Field(description="Events per second over the last 10 s.")
    unmapped_events: int = Field(description="Events without tr_id (not windowed).")
    malformed: int = Field(description="Malformed stream entries (ACKed and skipped).")
    recovered: int = Field(description="Own pending entries re-read after a restart or reconnect.")
    claimed: int = Field(description="Pending entries claimed from dead consumers.")
    history_entries: int = Field(0, description="Processed entries read back at start to refill the windows.")
    consumers_removed: int = Field(0, description="Idle consumers without pending entries removed.")
    lag: int | None = Field(None, description="Consumer group lag (entries not yet delivered).")
    pending: int | None = Field(None, description="Delivered but not acknowledged entries.")
    stream_length: int | None = Field(None, description="Length of the telemetry stream.")
    stream_time: datetime | None = Field(None, description="Stream clock (UTC), the ingest's.")
    clock_epoch: int = Field(0, description="Epoch of the stream clock; changes when the clock jumps.")
    clock_resets: int = Field(description="Stream clock jumps (replayer restarts, resumes after a pause).")
    clock_garbage: int = Field(0, description="Events with an implausible fix time (skipped).")
    off_timeline: int = Field(0, description="Events further than the jump threshold from the clock.")
    tracks: int = Field(description="Vehicles (tr_id) with a track window.")
    window_points: int = Field(description="Points in all track windows.")
    late_points: int = Field(
        0,
        description="On-timeline points inserted before newer ones (reordered packets); points far behind "
        "(black box) count as off_timeline.",
    )
    duplicate_points: int = Field(0, description="Exact repeats dropped (at-least-once delivery).")
    expired_points: int = Field(0, description="Points older than the window dropped.")
    ahead_points: int = Field(0, description="Points ahead of the stream clock (held, not shown).")
    held_points: int = Field(0, description="Off-timeline points held in case the clock jumps to them.")
    restored_points: int = Field(0, description="Points refilled from the stream at start.")
    ticks: int = Field(description="Prediction ticks run.")
    ticks_skipped: int = Field(
        description="Tick boundaries skipped: after a stream jump or while catching up (ticks_lagging)."
    )
    ticks_lagging: int = Field(
        0, description="Ticks skipped because the stream was already past them (the predictor caught up)."
    )
    last_tick: datetime | None = Field(None, description="Stream time of the last tick.")
    last_tick_duration_s: float | None = Field(None, description="Duration of the last tick.")
    db_buffered: int = Field(description="Rows waiting for PostgreSQL.")
    db_dropped: int = Field(description="Rows lost (buffer overflow).")
    forecast: ForecastStatsOut | None = Field(None, description="Forecast engine state.")


class WsMessage(BaseModel):
    """Message pushed on ``/ws``.

    The first message is a ``snapshot`` with all vehicles, then ``delta`` messages carry only vehicles that
    changed (new packet, connection or status change). A full ``snapshot`` is repeated periodically.
    """

    type: Literal["snapshot", "delta"]
    version: int = Field(description="Cache version the message is consistent with.")
    server_time: datetime
    stream_time: datetime | None
    degraded: bool = Field(
        False, description="Redis is unavailable or the ingest is down: the state is the last known."
    )
    vehicles: list[VehicleOut]


class RouteStopOut(BaseModel):
    """A stop of a route."""

    stop_key: str = Field(description="Place of the stop: rounded coordinates «lon,lat».")
    name: str = Field(description="Stop name (address).")
    lat: float
    lon: float
    seq: int = Field(description="Position in the route's sequence (forward, then back).")


class RouteDirectionOut(BaseModel):
    """One direction of a route: its stops in the order of a full planned trip and its line by GPS tracks."""

    direction: int = Field(description="0 — forward (the most frequent trip), 1 — back.")
    name: str = Field(description="«first stop → last stop».")
    stops: list[RouteStopOut]
    line: list[list[float]] = Field(
        description="[[lon, lat], …]: the real path of the vehicles between consecutive stops (GPS, the pass "
        "closest to the median length, Douglas–Peucker 5 m); a straight segment where there are no tracks."
    )
    segments: int = Field(description="Segments between consecutive stops.")
    gps_segments: int = Field(description="... of them drawn by GPS tracks (the rest: straight).")


class RouteOut(BaseModel):
    """A route derived from the plan (``shared.routes``, API contract §1)."""

    route_id: str
    name: str = Field(description="«Маршрут R1: <terminal> — <terminal>».")
    tr_ids: list[int]
    stops: list[RouteStopOut] = Field(description="Forward trip, then the back trip (the «stringline» axis).")
    line: list[list[float]] = Field(description="Line of the forward direction (kept for compatibility).")
    directions: list[RouteDirectionOut] = Field(
        default_factory=list, description="Each direction separately."
    )
    color: str


# ---- journal views of the dashboard (API contract §2) ------------------------------------------------------

Risk = Literal["green", "yellow", "red", "unknown"]


class CauseOut(BaseModel):
    """Cause of a delay (contract §1): code, dispatcher text, recommendation, top feature contributions."""

    model_config = ConfigDict(extra="allow")

    code: str = Field(
        description="dwell_long / slow_segment / layover / accumulated_delay / bunching / gps_lost / unknown."
    )
    text: str = ""
    recommendation: str = ""
    factors: list[dict[str, Any]] = Field(
        default_factory=list, description="[{feature, label, contribution_s}] — top-3 contributions, seconds."
    )


class AlertOut(BaseModel):
    """An alert: an incident opened or escalated (one per incident and level)."""

    model_config = ConfigDict(extra="allow")

    alert_id: int
    prediction_id: int | None = None
    incident_id: int | None = None
    kind: Literal["delay", "bunching"] = "delay"
    tr_id: int | None = None
    unit_id: int | None = None
    route_id: str | None = None
    level: Literal["yellow", "red"] | None = None
    status: str | None = Field(None, description="open / closed (the incident closed).")
    cause: CauseOut | None = None
    issued_at: datetime | None = Field(None, description="Stream time of the alert.")
    planned_at: datetime | None = Field(None, description="Plan time of the target stop.")
    target_stop_id: int | None = None
    target_stop_name: str | None = None
    pred_delay_s: float | None = None
    p10: float | None = None
    p90: float | None = None
    p_late: float | None = None
    escalated_from: int | None = Field(None, description="Alert this one escalates (yellow → red).")
    related_tr_id: int | None = Field(None, description="The other vehicle of a bunching pair.")
    acknowledged: bool = False
    retroactive: bool | None = Field(
        None, description="Issued at or after the pass of its stop (must be false)."
    )
    actual_delay_s: float | None = Field(None, description="Fact at the target stop (detector), when known.")


class AlertListOut(BaseModel):
    """``GET /api/alerts``."""

    stream_time: datetime | None = None
    degraded: bool = False
    count: int = 0
    items: list[AlertOut]


class PredictionOut(BaseModel):
    """A forecast «vehicle × target stop 10–15 min ahead» and its check against the fact."""

    model_config = ConfigDict(extra="allow")

    prediction_id: int
    tr_id: int
    unit_id: int | None = None
    route_id: str | None = None
    target_stop_id: int | None = None
    target_stop_name: str | None = None
    planned_at: datetime | None = None
    issued_at: datetime | None = Field(None, description="First issue (stream time).")
    lead_s: float | None = Field(None, description="Plan − first issue (600–900 s by construction).")
    pred_delay_s: float | None = Field(None, description="Last forecast value before the pass.")
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    p_late: float | None = None
    risk: Risk = "unknown"
    model_version: str | None = None
    source: Literal["model", "fallback"] = "model"
    status: Literal["open", "closed"] = "open"
    outcome: str | None = Field(None, description="open / closed / missed / expired / reset.")
    actual_delay_s: float | None = None
    abs_error_s: float | None = None
    actual_lead_s: float | None = Field(None, description="Pass − first issue: how early it really was.")
    retroactive: bool | None = None
    closed_at: datetime | None = None


class PredictionListOut(BaseModel):
    """``GET /api/predictions``."""

    stream_time: datetime | None = None
    degraded: bool = False
    count: int = 0
    items: list[PredictionOut]


class IncidentOut(BaseModel):
    """A problem vehicle for the dispatcher: forecast delay (or bunching), cause, segment, live position."""

    model_config = ConfigDict(extra="allow")

    incident_id: int
    kind: Literal["delay", "bunching"]
    tr_id: int
    unit_id: int | None = None
    route_id: str | None = None
    route_name: str | None = None
    risk: Risk = "unknown"
    target_stop: dict[str, Any] | None = Field(None, description="{stop_id, name, lat, lon, planned_at}.")
    pred_delay_s: float | None = None
    p10: float | None = None
    p90: float | None = None
    p_late: float | None = None
    cause: CauseOut | None = None
    segment: dict[str, Any] | None = Field(None, description="{from_stop, to_stop, line: [[lon, lat], …]}.")
    issued_at: datetime | None = None
    time_to_event_s: float | None = Field(None, description="Seconds of stream time to the expected arrival.")
    vehicle: dict[str, Any] | None = Field(
        None, description="{lat, lon, course_deg, speed_kmh, current_delay_s} — live, from the hot state."
    )
    related_tr_id: int | None = None
    status: str = "open"
    opened_at: datetime | None = None
    closed_at: datetime | None = None


class IncidentListOut(BaseModel):
    """``GET /api/incidents``: red first, then by p_late and the forecast delay."""

    stream_time: datetime | None = None
    degraded: bool = False
    count: int = 0
    items: list[IncidentOut]


class DelayPointOut(BaseModel):
    t: datetime
    delay_s: float


class ForecastPointOut(BaseModel):
    stop_id: int
    name: str | None = None
    planned_at: datetime | None = None
    pred_delay_s: float | None = None
    p10: float | None = None
    p90: float | None = None


class IncidentDetailOut(IncidentOut):
    """``GET /api/incidents/{id}``: the incident, the deviation of the vehicle over 30 min (stop passes by the
    detector and the current deviation), its open forecasts and the alerts of the incident."""

    history: list[DelayPointOut] = Field(default_factory=list)
    forecast: list[ForecastPointOut] = Field(default_factory=list)
    alerts: list[AlertOut] = Field(default_factory=list)


class StringlineStopOut(BaseModel):
    stop_key: str
    name: str
    seq: int


class StringlinePointOut(BaseModel):
    t: datetime
    seq: int


class StringlineForecastPointOut(StringlinePointOut):
    p10: datetime | None = Field(None, description="Early bound (P10) of the arrival time.")
    p90: datetime | None = Field(None, description="Late bound (P90) of the arrival time.")


class StringlineNowOut(BaseModel):
    t: datetime
    seq: float = Field(description="Half a stop before the next planned stop of the vehicle.")
    delay_s: float | None = Field(None, description="Current deviation from the plan.")


class StringlineTripOut(BaseModel):
    tr_id: int
    planned: list[StringlinePointOut]
    actual: list[StringlinePointOut]
    forecast: list[StringlineForecastPointOut]
    now: StringlineNowOut | None = Field(None, description="Where the vehicle is now (its next stop).")


class StringlineOut(BaseModel):
    """``GET /api/stringline``: time × stops of a route."""

    route_id: str
    stops: list[StringlineStopOut]
    trips: list[StringlineTripOut]
    stream_time: datetime | None = None
    degraded: bool = False


class LeadBucketOut(BaseModel):
    from_s: int
    to_s: int
    count: int


class MaeByHourOut(BaseModel):
    hour: int
    mae_s: float | None = None
    baseline_s: float | None = None
    n: int


class HorizonOut(BaseModel):
    """``GET /api/metrics/horizon``: forecasts checked against the fact on the stream."""

    model_config = ConfigDict(extra="allow")

    closed: int = Field(0, description="Forecasts closed with the fact in the window.")
    online_mae_s: float | None = None
    baseline_mae_s: float | None = Field(None, description="«Forecast = current deviation at issue».")
    warned_share: float | None = Field(
        None,
        description="Late stops (fact > 120 s) whose vehicle was yellow or red while the stop was ahead.",
    )
    retroactive: int = Field(0, description="Forecasts and alerts issued at or after the pass (must be 0).")
    lead_hist: list[LeadBucketOut] = Field(
        default_factory=list,
        description="Plan of the target stop − first issue, seconds (600–900 by construction: the horizon).",
    )
    actual_lead_hist: list[LeadBucketOut] = Field(
        default_factory=list,
        description="Pass of the stop (detector) − first issue: how early it really was.",
    )
    mae_by_hour: list[MaeByHourOut] = Field(default_factory=list)
    stream_time: datetime | None = None
    degraded: bool = False


class PerfOut(BaseModel):
    """``GET /api/metrics/perf``: p95 over the last 5 min from the services' own metrics."""

    model_config = ConfigDict(extra="allow")

    ingest_pps: float | None = None
    e2e_p95_s: float | None = Field(
        None, description="Packet received by ingest → processed by predictor, p95."
    )
    inference_p95_ms: float | None = Field(
        None, description="POST /predict of ml-service (whole batch), p95."
    )
    tick_p95_ms: float | None = Field(None, description="Forecast tick of the predictor, p95.")
    consumer_lag: float | None = None
    vehicles_online: int | None = None
    deps: dict[str, str] = Field(default_factory=dict)


class WhatIfParamsIn(BaseModel):
    from_stop_key: str | None = Field(None, description="add_vehicle: stop the reserve vehicle leaves from.")
    depart_at: datetime | None = Field(None, description="add_vehicle: departure (default: in 5 min).")
    hold_s: float | None = Field(
        None, ge=0, le=3600, description="hold: extra standing time, s (default 120)."
    )
    tr_id: int | None = Field(None, description="hold: the vehicle held.")


class WhatIfIn(BaseModel):
    """``POST /api/whatif``."""

    route_id: str
    at: datetime | None = Field(None, description="Moment of the action (default: the stream time now).")
    action: Literal["add_vehicle", "hold"]
    params: WhatIfParamsIn = Field(default_factory=WhatIfParamsIn)
    horizon_min: float = Field(60, gt=0, le=240)


class HeadwayOut(BaseModel):
    stop_key: str
    t: datetime
    gap_s: float


class ScenarioOut(BaseModel):
    headways: list[HeadwayOut]
    mean_wait_s: float = Field(description="Mean passenger wait at the stops, Σg² / 2Σg over the headways.")
    max_gap_s: float
    late_stops: int = Field(description="Arrivals more than 120 s behind the plan.")
    bunching_pairs: int = Field(description="Pairs of vehicles with a headway below 30 % of the planned one.")
    vehicles: list[dict[str, Any]] = Field(
        description="[{tr_id (null: the reserve), arrivals: [{stop_key, seq, t}]}]; ``seq`` is the place on "
        "the route's trip (a stop served twice — a terminal, both ways — has two).",
    )


class WhatIfOut(BaseModel):
    baseline: ScenarioOut
    scenario: ScenarioOut
    delta: dict[str, float] = Field(description="scenario − baseline of the four metrics.")
