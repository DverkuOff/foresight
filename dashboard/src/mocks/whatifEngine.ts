/**
 * Упрощённая модель What-if для mock-сервера: интервалы движения по остановкам маршрута до и после действия
 * диспетчера («выпустить резервное ТС» или «придержать ТС»). Задержка ТС распространяется вперёд постоянной.
 */
import type { Headway, Scenario, ScenarioVehicle, WhatIfDelta, WhatIfRequest } from '../api/types'
import { meanWait } from '../domain/whatif'

export interface EngineArrival {
  stopKey: string
  seq: number
  /** План, мс (null — у резервного ТС плана нет). */
  plan: number | null
  /** Ожидаемое прибытие, мс. */
  est: number
}

export interface EngineVehicle {
  trId: number | null
  arrivals: EngineArrival[]
}

export interface EngineRoute {
  /** stop_key по seq. */
  stopKeys: string[]
  /** Плановое время хода seq → seq+1, с (последний элемент — переход с конца круга на начало). */
  runS: number[]
}

export interface EngineInput {
  route: EngineRoute
  vehicles: EngineVehicle[]
  /** Момент расчёта, мс. */
  at: number
  request: WhatIfRequest
}

const LATE_S = 120

function iso(ms: number): string {
  return new Date(ms).toISOString().replace('.000Z', 'Z')
}

/** Прибытия резервного ТС: отправление с `fromSeq` в `depart`, далее по плановым временам хода. */
export function reserveArrivals(
  route: EngineRoute,
  fromSeq: number,
  depart: number,
  until: number,
): EngineArrival[] {
  const out: EngineArrival[] = []
  const n = route.stopKeys.length
  if (n === 0) return out
  let t = depart
  let seq = ((fromSeq % n) + n) % n
  for (let guard = 0; t <= until && guard < n * 3; guard += 1) {
    out.push({ stopKey: route.stopKeys[seq], seq, plan: null, est: t })
    t += Math.max(20, route.runS[seq] ?? 60) * 1000
    seq = (seq + 1) % n
  }
  return out
}

/** Применить действие к прибытиям. */
export function applyAction(input: EngineInput): EngineVehicle[] {
  const { request, route, at } = input
  const until = at + request.horizon_min * 60_000
  if (request.action === 'hold') {
    const holdMs = (request.params.hold_s ?? 120) * 1000
    const trId = request.params.tr_id
    return input.vehicles.map((v) => {
      if (v.trId !== trId) return v
      // ТС стоит на ближайшей остановке дольше на hold_s: все следующие прибытия сдвигаются
      let held = false
      return {
        ...v,
        arrivals: v.arrivals.map((a) => {
          if (a.est < at) return a
          const shifted = { ...a, est: a.est + (held ? holdMs : 0) }
          held = true
          return shifted
        }),
      }
    })
  }
  const fromKey = request.params.from_stop_key ?? route.stopKeys[0]
  const fromSeq = Math.max(0, route.stopKeys.indexOf(fromKey))
  const depart = request.params.depart_at ? Date.parse(request.params.depart_at) : at + 5 * 60_000
  return [...input.vehicles, { trId: null, arrivals: reserveArrivals(route, fromSeq, depart, until) }]
}

function median(values: number[]): number | null {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/** Метрики сценария в окне [at, at + horizon]. */
export function evaluate(vehicles: EngineVehicle[], at: number, horizonMin: number): Scenario {
  const until = at + horizonMin * 60_000
  const byStop = new Map<string, { idx: number; est: number; plan: number | null }[]>()
  let lateStops = 0
  vehicles.forEach((v, idx) => {
    for (const a of v.arrivals) {
      if (a.est < at || a.est > until) continue
      const list = byStop.get(a.stopKey) ?? []
      list.push({ idx, est: a.est, plan: a.plan })
      byStop.set(a.stopKey, list)
      if (a.plan !== null && (a.est - a.plan) / 1000 > LATE_S) lateStops += 1
    }
  })
  const headways: Headway[] = []
  const waits: number[] = []
  const pairs = new Set<string>()
  let maxGap = 0
  for (const [stopKey, list] of byStop) {
    list.sort((a, b) => a.est - b.est)
    const plans = list
      .map((x) => x.plan)
      .filter((p): p is number => p !== null)
      .sort((a, b) => a - b)
    const plannedHeadway = median(
      plans
        .slice(1)
        .map((p, i) => (p - plans[i]) / 1000)
        .filter((g) => g > 30),
    )
    // сбивка — ближе 30 % планового интервала, но не дальше 5 мин (как backend/whatif.py)
    const bunchThreshold = Math.min(Math.max(60, (plannedHeadway ?? 300) * 0.3), 300)
    const gaps: number[] = []
    for (let i = 1; i < list.length; i += 1) {
      if (list[i].idx === list[i - 1].idx) continue
      const gap = (list[i].est - list[i - 1].est) / 1000
      gaps.push(gap)
      headways.push({ stop_key: stopKey, t: iso(list[i].est), gap_s: Math.round(gap) })
      maxGap = Math.max(maxGap, gap)
      if (gap < bunchThreshold) {
        const [a, b] = [list[i - 1].idx, list[i].idx].sort((x, y) => x - y)
        pairs.add(`${a}-${b}`)
      }
    }
    if (gaps.length) waits.push(meanWait(gaps))
  }
  headways.sort((a, b) => Date.parse(a.t) - Date.parse(b.t))
  const out: ScenarioVehicle[] = vehicles.map((v) => ({
    tr_id: v.trId,
    arrivals: v.arrivals
      .filter((a) => a.est >= at && a.est <= until)
      .map((a) => ({ stop_key: a.stopKey, t: iso(a.est) })),
  }))
  return {
    headways,
    mean_wait_s: waits.length ? Math.round(waits.reduce((a, b) => a + b, 0) / waits.length) : 0,
    max_gap_s: Math.round(maxGap),
    late_stops: lateStops,
    bunching_pairs: pairs.size,
    vehicles: out,
  }
}

export function runWhatIf(input: EngineInput): {
  baseline: Scenario
  scenario: Scenario
  delta: WhatIfDelta
} {
  const baseline = evaluate(input.vehicles, input.at, input.request.horizon_min)
  const scenario = evaluate(applyAction(input), input.at, input.request.horizon_min)
  return {
    baseline,
    scenario,
    delta: {
      mean_wait_s: scenario.mean_wait_s - baseline.mean_wait_s,
      max_gap_s: scenario.max_gap_s - baseline.max_gap_s,
      late_stops: scenario.late_stops - baseline.late_stops,
      bunching_pairs: scenario.bunching_pairs - baseline.bunching_pairs,
    },
  }
}
