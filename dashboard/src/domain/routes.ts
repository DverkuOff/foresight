import { shortStopName } from './stringline'
import type { RouteOut } from '../api/types'

/** «Ореховый бульв. — Большая Серпуховская ул.» из «Маршрут R1: Ореховый бульв. — Большая Серпуховская ул.». */
export function routeShortName(route: Pick<RouteOut, 'route_id' | 'name'>): string {
  const clean = route.name.replace(/^Маршрут\s+\S+:\s*/u, '').trim()
  return clean === '' ? route.route_id : clean
}

/** Откуда → куда коротко: `Ореховый бульв. 24к2 → Большая Серпуховская 6`. */
export function routeEnds(route: Pick<RouteOut, 'route_id' | 'name'>): string {
  const parts = routeShortName(route).split(/\s+—\s+/u)
  if (parts.length < 2) return routeShortName(route)
  return `${shortStopName(parts[0])} → ${shortStopName(parts[parts.length - 1])}`
}

/** Маршрут по умолчанию: больше всего активных проблем (красные весят вдвое), иначе первый. */
export function busiestRoute(
  routes: readonly Pick<RouteOut, 'route_id'>[],
  incidents: readonly { route_id: string | null; risk: string }[],
): string | null {
  if (!routes.length) return null
  const known = new Set(routes.map((r) => r.route_id))
  const score = new Map<string, number>()
  for (const inc of incidents) {
    if (!inc.route_id || !known.has(inc.route_id)) continue
    score.set(inc.route_id, (score.get(inc.route_id) ?? 0) + (inc.risk === 'red' ? 2 : 1))
  }
  let best: string | null = null
  let bestScore = 0
  for (const r of routes) {
    const s = score.get(r.route_id) ?? 0
    if (s > bestScore) {
      best = r.route_id
      bestScore = s
    }
  }
  return best ?? routes[0].route_id
}

/** Маршрут для «Нитки» и What-if: со сбивкой (её видно на графике), иначе с наибольшим числом проблем. */
export function problemRoute(
  routes: readonly Pick<RouteOut, 'route_id'>[],
  incidents: readonly { route_id: string | null; risk: string; kind?: string }[],
): string | null {
  const known = new Set(routes.map((r) => r.route_id))
  const bunched = incidents.find((i) => i.kind === 'bunching' && i.route_id !== null && known.has(i.route_id))
  return bunched?.route_id ?? busiestRoute(routes, incidents)
}

/** Ближайшая остановка маршрута к точке (для привязки инцидента к оси «нитки»). */
export function nearestStopSeq(
  stops: readonly { seq: number; lat: number; lon: number }[],
  lat: number,
  lon: number,
): number | null {
  let best: number | null = null
  let bestD = Number.POSITIVE_INFINITY
  const k = Math.cos((lat * Math.PI) / 180)
  for (const s of stops) {
    const d = ((s.lon - lon) * k) ** 2 + (s.lat - lat) ** 2
    if (d < bestD) {
      bestD = d
      best = s.seq
    }
  }
  return best
}
