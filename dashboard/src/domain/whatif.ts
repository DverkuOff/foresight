import type { Scenario, WhatIfDelta, WhatIfRequest } from '../api/types'
import { formatMinSec } from './format'

export type WhatIfMetric = keyof WhatIfDelta

export const WHATIF_METRICS: { key: WhatIfMetric; label: string; unit: 's' | 'count'; hint: string }[] = [
  {
    key: 'mean_wait_s',
    label: 'Среднее ожидание автобуса',
    unit: 's',
    hint: 'сколько в среднем пассажир ждёт автобус на остановке',
  },
  {
    key: 'max_gap_s',
    label: 'Самый долгий перерыв между автобусами',
    unit: 's',
    hint: 'наибольший интервал между соседними автобусами на остановке',
  },
  {
    key: 'late_stops',
    label: 'Опоздания больше 2 минут',
    unit: 'count',
    hint: 'сколько раз автобусы приедут на остановку позже расписания больше чем на 2 мин',
  },
  {
    key: 'bunching_pairs',
    label: 'Автобусы идут вплотную (сбивка)',
    unit: 'count',
    hint: 'сколько пар автобусов приедут на остановку с интервалом меньше 5 мин',
  },
]

/** Дельта «после − до» по всем метрикам. */
export function computeDelta(baseline: Scenario, scenario: Scenario): WhatIfDelta {
  return {
    mean_wait_s: scenario.mean_wait_s - baseline.mean_wait_s,
    max_gap_s: scenario.max_gap_s - baseline.max_gap_s,
    late_stops: scenario.late_stops - baseline.late_stops,
    bunching_pairs: scenario.bunching_pairs - baseline.bunching_pairs,
  }
}

export type DeltaTone = 'better' | 'worse' | 'same'

/** Для всех метрик What-if меньше — лучше. Изменения в пределах `epsilon` — «без изменений». */
export function deltaTone(metric: WhatIfMetric, delta: number): DeltaTone {
  const epsilon = metric === 'mean_wait_s' || metric === 'max_gap_s' ? 5 : 0
  if (Math.abs(delta) <= epsilon) return 'same'
  return delta < 0 ? 'better' : 'worse'
}

export function relativeChange(before: number, after: number): number | null {
  if (!Number.isFinite(before) || before === 0) return null
  return (after - before) / before
}

export interface StopHeadway {
  stop_key: string
  mean_gap_s: number | null
  max_gap_s: number | null
  n: number
}

/** Средний и максимальный интервал по каждой остановке в порядке `stopKeys`. */
export function headwaysByStop(scenario: Scenario, stopKeys: string[]): StopHeadway[] {
  const byStop = new Map<string, number[]>()
  for (const h of scenario.headways) {
    const list = byStop.get(h.stop_key) ?? []
    list.push(h.gap_s)
    byStop.set(h.stop_key, list)
  }
  return stopKeys.map((key) => {
    const gaps = byStop.get(key) ?? []
    return {
      stop_key: key,
      mean_gap_s: gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : null,
      max_gap_s: gaps.length ? Math.max(...gaps) : null,
      n: gaps.length,
    }
  })
}

/** Ожидание пассажира при случайном подходе: Σh² / (2·Σh). */
export function meanWait(gaps: number[]): number {
  const total = gaps.reduce((a, b) => a + b, 0)
  if (total <= 0) return 0
  return gaps.reduce((a, b) => a + b * b, 0) / (2 * total)
}

/** От 10 минут секунды — шум: `176 мин` вместо `176 мин 43 с`. */
const WHOLE_MINUTES_S = 600

/** Значение метрики What-if для карточки: `4 мин 12 с`, `63 мин` или `3`. */
export function formatMetric(metric: WhatIfMetric, value: number): string {
  const unit = WHATIF_METRICS.find((m) => m.key === metric)?.unit ?? 'count'
  if (unit !== 's') return String(Math.round(value))
  return Math.abs(value) >= WHOLE_MINUTES_S ? `${Math.round(Math.abs(value) / 60)} мин` : formatMinSec(value)
}

/** Дельта со знаком: `−1 мин 05 с`, `+2`, `0`. */
export function formatDeltaValue(metric: WhatIfMetric, delta: number): string {
  const r = Math.round(delta)
  if (r === 0) return metric === 'mean_wait_s' || metric === 'max_gap_s' ? '0 с' : '0'
  const sign = r > 0 ? '+' : '−'
  return `${sign}${formatMetric(metric, Math.abs(r))}`
}

export type VerdictTone = 'better' | 'mixed' | 'worse' | 'same'

/** Общий итог действия: улучшает, компромисс, ухудшает или не влияет. */
export function verdictTone(delta: WhatIfDelta): VerdictTone {
  let better = 0
  let worse = 0
  for (const m of WHATIF_METRICS) {
    const tone = deltaTone(m.key, delta[m.key])
    if (tone === 'better') better += 1
    else if (tone === 'worse') worse += 1
  }
  if (better && worse) return 'mixed'
  if (better) return 'better'
  if (worse) return 'worse'
  return 'same'
}

/**
 * Подсказка для «выпустить резервное ТС»: остановка и время, при которых резерв придёт в середину
 * наибольшего разрыва базового сценария. Отправление не раньше `notBefore` (мс).
 */
export function suggestReserve(baseline: Scenario, notBefore: number): ReserveCandidate | null {
  return reserveCandidates(baseline, notBefore, 1, [0])[0] ?? null
}

export interface ReserveCandidate {
  from_stop_key: string
  depart_at: string
  gap_s: number
}

/** Разрыв, который кончается раньше, чем через столько после `notBefore`, резерв уже не закроет, мс. */
export const MIN_FUTURE_GAP_MS = 5 * 60_000

/**
 * Варианты «резерва» для перебора: середины будущей части (от `notBefore` до прихода ТС) `stops` наибольших
 * разрывов на разных остановках и сдвиги отправления на `shiftsS` секунд. Разрыв начинается и в прошлом (от
 * предыдущего прохода остановки), поэтому середина всего разрыва могла бы оказаться «сейчас» — и резерв поехал
 * бы следом за подходящим ТС. Первый вариант совпадает с {@link suggestReserve}.
 */
export function reserveCandidates(
  baseline: Scenario,
  notBefore: number,
  stops = 3,
  shiftsS: readonly number[] = [0, -120, 120],
): ReserveCandidate[] {
  const future = baseline.headways
    .map((h) => {
      const end = Date.parse(h.t)
      const start = Math.max(end - h.gap_s * 1000, notBefore)
      return { h, end, start, span: end - start }
    })
    .filter((x) => Number.isFinite(x.end) && x.span >= MIN_FUTURE_GAP_MS)
    .sort((a, b) => b.span - a.span)
  const usedStops = new Set<string>()
  const seen = new Set<string>()
  const out: ReserveCandidate[] = []
  for (const { h, start, span } of future) {
    if (usedStops.size >= stops) break
    if (usedStops.has(h.stop_key)) continue
    usedStops.add(h.stop_key)
    for (const shift of shiftsS) {
      const depart = Math.max(notBefore, start + span / 2 + shift * 1000)
      const departAt = new Date(Math.round(depart / 1000) * 1000).toISOString()
      const key = `${h.stop_key}@${departAt}`
      if (seen.has(key)) continue
      seen.add(key)
      out.push({ from_stop_key: h.stop_key, depart_at: departAt, gap_s: h.gap_s })
    }
  }
  return out
}

/** Сводная оценка дельты (меньше — лучше): ожидание пассажира, разрывы, сбивки и опоздания. */
export function deltaScore(delta: WhatIfDelta): number {
  return delta.mean_wait_s + 0.3 * delta.max_gap_s + 90 * delta.bunching_pairs + 20 * delta.late_stops
}

/** Варианты «придержать ТС», которые перебирает подбор действия, с. */
export const HOLD_OPTIONS_S = [120, 180, 300] as const

/**
 * Все варианты действий для подбора: выпуск резерва в середину наибольших разрывов (±2 мин) и придержка
 * каждого ТС маршрута на 2, 3 или 5 мин.
 */
export function candidateRequests(
  base: WhatIfRequest,
  baseline: Scenario,
  trIds: readonly number[],
  notBefore: number,
  maxVehicles = 8,
): WhatIfRequest[] {
  const reserve: WhatIfRequest[] = reserveCandidates(baseline, notBefore).map((c) => ({
    ...base,
    action: 'add_vehicle',
    params: { from_stop_key: c.from_stop_key, depart_at: c.depart_at },
  }))
  const holds: WhatIfRequest[] = trIds.slice(0, maxVehicles).flatMap((tr) =>
    HOLD_OPTIONS_S.map((holdS): WhatIfRequest => ({
      ...base,
      action: 'hold',
      params: { tr_id: tr, hold_s: holdS },
    })),
  )
  return [...reserve, ...holds]
}
