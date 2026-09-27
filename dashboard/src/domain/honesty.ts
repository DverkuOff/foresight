/** Расчёты страницы «Честность прогноза»: окно заблаговременности, итог закрытого прогноза, выигрыш у baseline. */
import type { LeadBucket, PredictionOut } from '../api/types'

/** Окно горизонта прогноза (архитектура §6): план строго в (t+10, t+15] мин. */
export const LEAD_WINDOW_S: [number, number] = [600, 900]
/** Опоздание — более 2 минут (как порог `red`). */
export const LATE_S = 120

export function bucketInWindow(b: LeadBucket, window: [number, number] = LEAD_WINDOW_S): boolean {
  return b.from_s >= window[0] && b.to_s <= window[1]
}

/** Доля прогнозов с заблаговременностью в окне 10–15 мин (null, если прогнозов нет). */
export function leadWindowShare(
  hist: readonly LeadBucket[],
  window: [number, number] = LEAD_WINDOW_S,
): number | null {
  let total = 0
  let inside = 0
  for (const b of hist) {
    total += b.count
    if (bucketInWindow(b, window)) inside += b.count
  }
  return total ? inside / total : null
}

/** Медиана заблаговременности по гистограмме (середина корзины), с. */
export function leadMedianS(hist: readonly LeadBucket[]): number | null {
  const total = hist.reduce((a, b) => a + b.count, 0)
  if (!total) return null
  const sorted = [...hist].sort((a, b) => a.from_s - b.from_s)
  let acc = 0
  for (const b of sorted) {
    acc += b.count
    if (acc >= total / 2) return (b.from_s + b.to_s) / 2
  }
  return null
}

/** На сколько онлайн-MAE лучше baseline (доля, > 0 — модель лучше). */
export function maeGain(online: number | null, baseline: number | null): number | null {
  if (online === null || baseline === null || baseline <= 0) return null
  return 1 - online / baseline
}

export type Outcome = 'hit' | 'miss' | 'false_alarm' | 'ok' | 'pending'

export const OUTCOME_LABEL: Record<Outcome, string> = {
  hit: 'предупредили',
  miss: 'пропуск',
  false_alarm: 'ложная тревога',
  ok: 'норма',
  pending: 'ожидает факта',
}

/**
 * Итог закрытого прогноза: алертом считается риск `yellow`/`red`, опозданием — факт > 2 мин.
 * Ложная тревога — алерт, а ТС пришло с отклонением меньше минуты.
 */
export function outcome(p: Pick<PredictionOut, 'risk' | 'actual_delay_s'>): Outcome {
  if (p.actual_delay_s === null) return 'pending'
  const alarm = p.risk === 'red' || p.risk === 'yellow'
  const late = p.actual_delay_s > LATE_S
  if (alarm && late) return 'hit'
  if (!alarm && late) return 'miss'
  if (alarm && p.actual_delay_s < 60) return 'false_alarm'
  return 'ok'
}

/** Факт попал в интервал P10–P90 прогноза. */
export function inInterval(p: Pick<PredictionOut, 'p10' | 'p90' | 'actual_delay_s'>): boolean | null {
  if (p.p10 === null || p.p90 === null || p.actual_delay_s === null) return null
  return p.actual_delay_s >= p.p10 && p.actual_delay_s <= p.p90
}
