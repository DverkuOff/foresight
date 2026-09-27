import type { IncidentOut } from '../api/types'
import { RISK_ORDER } from './risk'

function desc(a: number | null | undefined, b: number | null | undefined): number {
  const av = a ?? Number.NEGATIVE_INFINITY
  const bv = b ?? Number.NEGATIVE_INFINITY
  if (av === bv) return 0
  return bv > av ? 1 : -1
}

/** Порядок контракта: `red` → `yellow` (→ остальные), затем `p_late` и `pred_delay_s` по убыванию. */
export function compareIncidents(a: IncidentOut, b: IncidentOut): number {
  return (
    RISK_ORDER[a.risk] - RISK_ORDER[b.risk] ||
    desc(a.p_late, b.p_late) ||
    desc(a.pred_delay_s, b.pred_delay_s) ||
    a.time_to_event_s - b.time_to_event_s ||
    a.incident_id.localeCompare(b.incident_id)
  )
}

export function sortIncidents(items: Iterable<IncidentOut>): IncidentOut[] {
  return [...items].sort(compareIncidents)
}

export type IncidentFilter = 'all' | 'red' | 'yellow' | 'bunching'

export function filterIncidents(items: IncidentOut[], filter: IncidentFilter, query = ''): IncidentOut[] {
  const q = query.trim().toLowerCase()
  return items.filter((inc) => {
    if (filter === 'red' && inc.risk !== 'red') return false
    if (filter === 'yellow' && inc.risk !== 'yellow') return false
    if (filter === 'bunching' && inc.kind !== 'bunching') return false
    if (q === '') return true
    const route = (inc.route_id ?? '').toLowerCase()
    // «r1» — это маршрут R1, а не R10 и R13
    if (/^r\d+$/.test(q)) return route === q
    return (
      String(inc.tr_id).includes(q) || route.includes(q) || inc.target_stop.name.toLowerCase().includes(q)
    )
  })
}
