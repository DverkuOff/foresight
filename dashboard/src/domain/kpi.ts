import type { AlertOut, VehicleOut } from '../api/types'
import { toMs } from './format'
import { vehicleRisk } from './risk'

export interface KpiValues {
  total: number
  inWork: number
  stale: number
  red: number
  yellow: number
  meanPredS: number | null
  maxPred: { value: number; vehicle: VehicleOut } | null
}

/** KPI-полоса: ТС в работе, в риске (жёлтые + красные), средний и максимальный прогноз опоздания. */
export function computeKpis(vehicles: Iterable<VehicleOut>): KpiValues {
  let total = 0
  let inWork = 0
  let stale = 0
  let red = 0
  let yellow = 0
  let sum = 0
  let n = 0
  let maxPred: KpiValues['maxPred'] = null
  for (const v of vehicles) {
    total += 1
    if (v.status === 'online') inWork += 1
    else if (v.status === 'stale') stale += 1
    if (v.status === 'offline') continue
    const risk = vehicleRisk(v)
    if (risk === 'red') red += 1
    else if (risk === 'yellow') yellow += 1
    if (v.pred_delay_s !== null && v.pred_delay_s !== undefined) {
      sum += v.pred_delay_s
      n += 1
      if (!maxPred || v.pred_delay_s > maxPred.value) maxPred = { value: v.pred_delay_s, vehicle: v }
    }
  }
  return { total, inWork, stale, red, yellow, meanPredS: n ? sum / n : null, maxPred }
}

/** Алерты, выданные за `windowMs` до времени потока. */
export function alertsInWindow(
  alerts: readonly AlertOut[],
  streamTime: string | null,
  windowMs = 15 * 60_000,
): { total: number; red: number } {
  const now = toMs(streamTime)
  if (now === null) return { total: 0, red: 0 }
  let total = 0
  let red = 0
  for (const a of alerts) {
    const t = toMs(a.issued_at)
    if (t === null || t < now - windowMs || t > now + 1000) continue
    total += 1
    if (a.level === 'red') red += 1
  }
  return { total, red }
}
