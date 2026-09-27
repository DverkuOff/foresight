/**
 * Типы API Foresight строго по контракту `docs/api-contract.md` (утверждён 25.09.2026).
 * Все времена — строки ISO 8601 в UTC, задержки — секунды (+ опоздание, − опережение), координаты — WGS-84.
 */

/** ISO 8601 (UTC). */
export type IsoTime = string

/** Уровень риска (контракт §1). */
export type Risk = 'green' | 'yellow' | 'red' | 'unknown'

/** Статус связи ТС (`backend/state.py` LinkStatus). */
export type LinkStatus = 'online' | 'stale' | 'offline'

/** Коды причин (контракт §1). */
export type CauseCode =
  'dwell_long' | 'slow_segment' | 'layover' | 'accumulated_delay' | 'bunching' | 'gps_lost' | 'unknown'

export interface CauseFactor {
  feature: string
  label: string
  contribution_s: number
}

export interface Cause {
  code: CauseCode
  text: string
  recommendation: string
  /** Топ-3 вклада признаков. */
  factors: CauseFactor[]
}

/** Пороги риска (контракт §1, `GET/PUT /api/admin/settings`). */
export interface RiskThresholds {
  red_delay_s: number
  red_p_late: number
  green_delay_s: number
  green_p_late: number
}

// ---------------------------------------------------------------- ТС

export interface NextStop {
  stop_id: number
  stop_key: string
  name: string
  planned_at: IsoTime
}

export interface DoorsOut {
  zone: number
  odometer: number
  door_in: number[]
  door_out: number[]
  door_present: boolean[]
  door_closed: boolean[]
}

/** `VehicleOut` (существующие поля api + дополнения контракта §2). */
export interface VehicleOut {
  unit_id: number
  tr_id: number | null
  status: LinkStatus
  connected: boolean
  lat: number | null
  lon: number | null
  valid: boolean | null
  speed_kmh: number | null
  speed_max_kmh?: number | null
  course_deg: number | null
  altitude_m?: number | null
  satellites?: number | null
  event_time: IsoTime | null
  received_at: IsoTime | null
  last_packet_at?: IsoTime | null
  age_s: number | null
  packets: number
  reconnects: number
  doors?: DoorsOut | null
  // дополнения контракта (nullable, пока нет прогнозов)
  route_id?: string | null
  risk?: Risk | null
  current_delay_s?: number | null
  pred_delay_s?: number | null
  p_late?: number | null
  next_stop?: NextStop | null
  incident_id?: string | null
}

export interface VehicleListOut {
  count: number
  server_time: IsoTime
  stream_time: IsoTime | null
  status_counts: Partial<Record<LinkStatus, number>>
  degraded: boolean
  synced_at?: IsoTime | null
  vehicles: VehicleOut[]
}

// ---------------------------------------------------------------- маршруты

export interface RouteStop {
  stop_key: string
  name: string
  lat: number
  lon: number
  seq: number
}

/** [lon, lat] */
export type LngLat = [number, number]

export interface RouteDirection {
  direction: number
  name?: string
  line: LngLat[]
}

export interface RouteOut {
  route_id: string
  name: string
  tr_ids: number[]
  stops: RouteStop[]
  /** Линия прямого направления (контракт); на карте рисуются все `directions`. */
  line: LngLat[]
  directions?: RouteDirection[]
  color: string
}

// ---------------------------------------------------------------- инциденты

export type IncidentKind = 'delay' | 'bunching'

export interface IncidentTargetStop {
  stop_id: number
  name: string
  lat: number
  lon: number
  planned_at: IsoTime
}

export interface IncidentSegment {
  from_stop: string
  to_stop: string
  line: LngLat[]
}

export interface IncidentVehicle {
  lat: number | null
  lon: number | null
  course_deg: number | null
  speed_kmh: number | null
  current_delay_s: number | null
}

export interface IncidentOut {
  incident_id: string
  kind: IncidentKind
  tr_id: number
  unit_id: number
  route_id: string | null
  route_name: string | null
  risk: Risk
  target_stop: IncidentTargetStop
  pred_delay_s: number
  p10: number | null
  p90: number | null
  p_late: number | null
  cause: Cause
  segment: IncidentSegment | null
  issued_at: IsoTime
  time_to_event_s: number
  vehicle: IncidentVehicle
  /** Для `bunching` — второе ТС пары. */
  related_tr_id?: number | null
}

export interface DelayPoint {
  t: IsoTime
  delay_s: number
}

export interface ForecastPoint {
  stop_id: number
  name: string
  planned_at: IsoTime
  pred_delay_s: number
  p10: number | null
  p90: number | null
}

/** `GET /api/incidents/{id}`: инцидент + история отклонения за 30 мин + прогноз на 15 мин. */
export interface IncidentDetailOut extends IncidentOut {
  history: DelayPoint[]
  forecast: ForecastPoint[]
}

/** Общие поля ответа-списка (контракт §0). */
export interface ListMeta {
  stream_time?: IsoTime | null
  degraded?: boolean
}

export interface IncidentListOut extends ListMeta {
  items: IncidentOut[]
}

// ---------------------------------------------------------------- прогнозы и алерты

export interface PredictionOut {
  prediction_id: string
  tr_id: number
  unit_id: number | null
  route_id: string | null
  target_stop_id: number
  target_stop_name: string
  planned_at: IsoTime
  issued_at: IsoTime
  lead_s: number
  pred_delay_s: number
  p10: number | null
  p50: number | null
  p90: number | null
  p_late: number | null
  risk: Risk
  model_version: string
  source: 'model' | 'fallback'
  status: 'open' | 'closed'
  actual_delay_s: number | null
  abs_error_s: number | null
  closed_at: IsoTime | null
}

export interface PredictionListOut extends ListMeta {
  items: PredictionOut[]
}

export type AlertLevel = 'yellow' | 'red'

export interface AlertOut {
  alert_id: string
  prediction_id: string
  tr_id: number
  route_id: string | null
  level: AlertLevel
  cause: Cause
  issued_at: IsoTime
  planned_at: IsoTime
  pred_delay_s: number
  p_late: number | null
  acknowledged: boolean
}

export interface AlertListOut extends ListMeta {
  items: AlertOut[]
}

// ---------------------------------------------------------------- «нитка»

export interface StringlineStop {
  stop_key: string
  name: string
  seq: number
}

export interface StringlinePoint {
  t: IsoTime
  seq: number
}

/** Точка прогноза «нитки»: `t` — ожидаемое прибытие, `p10`/`p90` — границы интервала прибытия (ISO). */
export interface StringlineForecastPoint extends StringlinePoint {
  p10: IsoTime | null
  p90: IsoTime | null
}

/** Где ТС сейчас: `seq` — полостановки до его следующей плановой остановки. */
export interface StringlineNow {
  t: IsoTime
  seq: number
  delay_s?: number | null
}

export interface StringlineTrip {
  tr_id: number
  planned: StringlinePoint[]
  actual: StringlinePoint[]
  forecast: StringlineForecastPoint[]
  now?: StringlineNow | null
}

export interface StringlineOut {
  route_id: string
  stops: StringlineStop[]
  trips: StringlineTrip[]
  stream_time: IsoTime | null
}

// ---------------------------------------------------------------- метрики

export interface LeadBucket {
  from_s: number
  to_s: number
  count: number
}

export interface MaeByHour {
  hour: number
  mae_s: number | null
  baseline_s: number | null
  n: number
}

export interface HorizonOut {
  closed: number
  online_mae_s: number | null
  baseline_mae_s: number | null
  /** Доля опозданий, предупреждённых заранее (0…1). */
  warned_share: number | null
  /** Алерты «задним числом» (должно быть 0). */
  retroactive: number
  lead_hist: LeadBucket[]
  /** Выдача → фактический проход остановки (детектор): насколько заранее прогноз был на самом деле. */
  actual_lead_hist?: LeadBucket[]
  mae_by_hour: MaeByHour[]
}

export type DependencyName = 'redis' | 'postgres' | 'ml' | 'ingest' | 'predictor'
export type DependencyState = 'up' | 'down' | 'unknown' | 'disabled' | 'degraded'

export interface PerfOut {
  ingest_pps: number | null
  e2e_p95_s: number | null
  inference_p95_ms: number | null
  tick_p95_ms: number | null
  consumer_lag: number | null
  vehicles_online: number | null
  deps: Partial<Record<DependencyName, DependencyState>>
}

export interface DependencyOut {
  state: 'up' | 'down' | 'unknown' | 'disabled'
  since?: IsoTime | null
  error?: string | null
  outages?: number
}

export interface HealthOut {
  status: 'ok' | 'degraded'
  service: string
  version: string
  uptime_s: number
  dependencies: Record<string, DependencyOut>
}

// ---------------------------------------------------------------- What-if

export type WhatIfAction = 'add_vehicle' | 'hold'

export interface WhatIfParams {
  from_stop_key?: string
  depart_at?: IsoTime
  hold_s?: number
  tr_id?: number
}

export interface WhatIfRequest {
  route_id: string
  at?: IsoTime
  action: WhatIfAction
  params: WhatIfParams
  horizon_min: number
}

export interface Headway {
  stop_key: string
  t: IsoTime
  gap_s: number
}

export interface ScenarioVehicle {
  tr_id: number | null
  /** `seq` — место на рейсе маршрута: остановка, которую ТС проходит дважды, имеет два (старый API — без него). */
  arrivals: { stop_key: string; seq?: number; t: IsoTime }[]
}

export interface Scenario {
  headways: Headway[]
  mean_wait_s: number
  max_gap_s: number
  late_stops: number
  bunching_pairs: number
  vehicles: ScenarioVehicle[]
}

export interface WhatIfDelta {
  mean_wait_s: number
  max_gap_s: number
  late_stops: number
  bunching_pairs: number
}

export interface WhatIfOut {
  baseline: Scenario
  scenario: Scenario
  delta: WhatIfDelta
}

// ---------------------------------------------------------------- WebSocket /ws

export interface WsVehiclesMessage {
  type: 'snapshot' | 'delta'
  stream_time: IsoTime | null
  server_time?: IsoTime
  version?: number
  degraded?: boolean
  vehicles: VehicleOut[]
}

export interface WsAlertMessage {
  type: 'alert'
  stream_time: IsoTime | null
  alert: AlertOut
}

export interface WsIncidentMessage {
  type: 'incident'
  stream_time: IsoTime | null
  action: 'open' | 'update' | 'close'
  incident: IncidentOut
}

export interface WsPredictionClosedMessage {
  type: 'prediction_closed'
  stream_time: IsoTime | null
  prediction: PredictionOut
}

export interface WsClockMessage {
  type: 'clock'
  stream_time: IsoTime | null
  epoch: number
}

export type WsMessage =
  WsVehiclesMessage | WsAlertMessage | WsIncidentMessage | WsPredictionClosedMessage | WsClockMessage

// ---------------------------------------------------------------- администрирование (`/api/admin/*`)

export interface AdminSettings {
  risk: RiskThresholds
  alert: { min_level: 'yellow' | 'red'; min_p_late: number }
  updated_at?: IsoTime | null
  applies_within_s?: number
}

export interface ModelVersion {
  version: string
  created_at: string | null
  model: string | null
  description: string | null
  precision: string | null
  components: string[]
  cv_mae: number | null
  test_mae: number | null
  baseline_test_mae: number | null
  online_mae_s: number | null
  online_closed: number | null
  /** версия дообучена на потоке: поправка базовой версии и её проверка */
  online?: {
    base: string
    parent?: string
    n_fit: number
    n_eval: number
    mae_before_s: number
    mae_after_s: number
    trained_at?: string
  } | null
  active: boolean
}

export interface RetrainState {
  state: 'idle' | 'running' | 'done' | 'error'
  base_version: string | null
  version: string | null
  improved?: boolean | null
  started_at: IsoTime | null
  finished_at: IsoTime | null
  n_fit: number | null
  n_eval: number | null
  mae_before_s: number | null
  mae_after_s: number | null
  error: string | null
}

export interface FallbackInfo {
  intercept_s: number
  coef: number
  feature: string
  forecasts: number
  share: number | null
}

export interface ModelsOut {
  active: string | null
  ml: string
  versions: ModelVersion[]
  retrain?: RetrainState | null
  fallback?: FallbackInfo | null
}

export interface ServiceHealth {
  name: string
  state: 'up' | 'down' | 'degraded' | 'unknown' | 'disabled'
  detail: string | null
  latency_ms: number | null
}

export interface ServicesOut {
  checked_at: IsoTime
  services: ServiceHealth[]
}

export interface UnitOut {
  unit_id: number
  tr_id: number | null
  route_id: string | null
  scheduled: boolean | null
  status: LinkStatus
  risk: Risk | null
  last_packet_at: IsoTime | null
}

export interface ReplayStatus {
  state: string
  mode: string
  split: string
  speed: number
  loop: boolean
  start: string | null
  cycle: number
  units: number
  connections_active: number
  progress: number
  packets_sent: number
  reconnects?: number
  bridge_posts?: number
  target?: string
  data_time: IsoTime | null
  error: string | null
}

/** `GET /api/ingest/stats` — приёмник NDTP (поля, которые показывает админка). */
export interface IngestStats {
  available: boolean
  degraded: boolean
  age_s: number | null
  listening: boolean
  port: number
  connections_active: number
  connections_total: number
  disconnects: number
  frames: number
  crc_errors: number
  parse_errors: number
  packets_per_s: number
}

export interface JournalOut<T> {
  count: number
  items: T[]
}
