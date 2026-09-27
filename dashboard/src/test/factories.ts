/** Фабрики тестовых объектов по контракту API. */
import type { AlertOut, IncidentOut, PredictionOut, VehicleOut } from '../api/types'

export function makeIncident(patch: Partial<IncidentOut> = {}): IncidentOut {
  return {
    incident_id: 'inc-1',
    kind: 'delay',
    tr_id: 100,
    unit_id: 900,
    route_id: 'R1',
    route_name: 'Маршрут R1: A — B',
    risk: 'yellow',
    target_stop: {
      stop_id: 1,
      name: 'Каширское ш., д.49',
      lat: 55.6,
      lon: 37.6,
      planned_at: '2026-01-06T08:27:00Z',
    },
    pred_delay_s: 90,
    p10: 30,
    p90: 150,
    p_late: 0.4,
    cause: {
      code: 'slow_segment',
      text: 'Низкая скорость на участке',
      recommendation: 'Проверить участок',
      factors: [],
    },
    segment: null,
    issued_at: '2026-01-06T08:15:00Z',
    time_to_event_s: 720,
    vehicle: { lat: 55.6, lon: 37.6, course_deg: 90, speed_kmh: 12, current_delay_s: 40 },
    related_tr_id: null,
    ...patch,
  }
}

export function makeVehicle(patch: Partial<VehicleOut> = {}): VehicleOut {
  return {
    unit_id: 900,
    tr_id: 100,
    status: 'online',
    connected: true,
    lat: 55.6,
    lon: 37.6,
    valid: true,
    speed_kmh: 20,
    course_deg: 90,
    event_time: '2026-01-06T08:15:00Z',
    received_at: '2026-01-06T08:15:01Z',
    age_s: 1,
    packets: 10,
    reconnects: 0,
    route_id: 'R1',
    risk: null,
    current_delay_s: 30,
    pred_delay_s: 40,
    p_late: 0.1,
    next_stop: null,
    incident_id: null,
    ...patch,
  }
}

export function makeAlert(patch: Partial<AlertOut> = {}): AlertOut {
  return {
    alert_id: 'a-1',
    prediction_id: 'p-1',
    tr_id: 100,
    route_id: 'R1',
    level: 'yellow',
    cause: { code: 'unknown', text: 'Причина не определена', recommendation: '—', factors: [] },
    issued_at: '2026-01-06T08:10:00Z',
    planned_at: '2026-01-06T08:22:00Z',
    pred_delay_s: 80,
    p_late: 0.4,
    acknowledged: false,
    ...patch,
  }
}

export function makePrediction(patch: Partial<PredictionOut> = {}): PredictionOut {
  return {
    prediction_id: 'p-1',
    tr_id: 100,
    unit_id: 900,
    route_id: 'R1',
    target_stop_id: 1,
    target_stop_name: 'Каширское ш., д.49',
    planned_at: '2026-01-06T08:27:00Z',
    issued_at: '2026-01-06T08:15:00Z',
    lead_s: 720,
    pred_delay_s: 90,
    p10: 30,
    p50: 90,
    p90: 150,
    p_late: 0.4,
    risk: 'yellow',
    model_version: 'test',
    source: 'model',
    status: 'closed',
    actual_delay_s: 130,
    abs_error_s: 40,
    closed_at: '2026-01-06T08:29:10Z',
    ...patch,
  }
}
