/**
 * Разбор сообщений `/ws` (контракт §2). Терпим к двум формам `alert` / `prediction_closed`: объект во вложенном поле
 * (`{type, alert}`) или поля объекта прямо в сообщении (`{type, alert_id, ...}`).
 */
import type { AlertOut, IncidentOut, PredictionOut, VehicleOut, WsMessage } from './types'

type Obj = Record<string, unknown>

function isObj(value: unknown): value is Obj {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function streamTime(msg: Obj): string | null {
  return typeof msg.stream_time === 'string' ? msg.stream_time : null
}

/** Идентификаторы приводим к строке (backend может отдать число). */
export function normalizeIncident(incident: IncidentOut): IncidentOut {
  return typeof incident.incident_id === 'string'
    ? incident
    : { ...incident, incident_id: String(incident.incident_id) }
}

export function normalizeVehicle(vehicle: VehicleOut): VehicleOut {
  const id = vehicle.incident_id
  return id === null || id === undefined || typeof id === 'string'
    ? vehicle
    : { ...vehicle, incident_id: String(id) }
}

export function parseWsMessage(raw: unknown): WsMessage | null {
  if (!isObj(raw) || typeof raw.type !== 'string') return null
  const st = streamTime(raw)
  switch (raw.type) {
    case 'snapshot':
    case 'delta': {
      if (!Array.isArray(raw.vehicles)) return null
      return {
        type: raw.type,
        stream_time: st,
        server_time: typeof raw.server_time === 'string' ? raw.server_time : undefined,
        version: typeof raw.version === 'number' ? raw.version : undefined,
        degraded: raw.degraded === true,
        vehicles: (raw.vehicles as VehicleOut[]).map(normalizeVehicle),
      }
    }
    case 'alert': {
      const alert = isObj(raw.alert) ? raw.alert : 'alert_id' in raw ? raw : null
      if (!alert) return null
      return {
        type: 'alert',
        stream_time: st,
        alert: { ...(alert as unknown as AlertOut), alert_id: String(alert.alert_id) },
      }
    }
    case 'incident': {
      const incident = isObj(raw.incident) ? raw.incident : null
      const action = raw.action
      if (!incident || (action !== 'open' && action !== 'update' && action !== 'close')) return null
      return {
        type: 'incident',
        stream_time: st,
        action,
        incident: normalizeIncident(incident as unknown as IncidentOut),
      }
    }
    case 'prediction_closed': {
      const prediction = isObj(raw.prediction) ? raw.prediction : 'prediction_id' in raw ? raw : null
      if (!prediction) return null
      return {
        type: 'prediction_closed',
        stream_time: st,
        prediction: prediction as unknown as PredictionOut,
      }
    }
    case 'clock':
      return { type: 'clock', stream_time: st, epoch: typeof raw.epoch === 'number' ? raw.epoch : 0 }
    default:
      return null
  }
}
