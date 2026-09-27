/** Пороги и подписи страницы «Производительность» (цели из docs/observability.md §4). */
import type { DependencyName, DependencyState, PerfOut } from '../api/types'

export type PerfKey = Exclude<keyof PerfOut, 'deps'>
export type PerfStatus = 'ok' | 'warn' | 'bad' | 'none'

interface Thresholds {
  warn: number
  bad: number
}

/** Больше — хуже: жёлтый с `warn`, красный с `bad`. */
export const PERF_THRESHOLDS: Partial<Record<PerfKey, Thresholds>> = {
  e2e_p95_s: { warn: 0.5, bad: 1 },
  inference_p95_ms: { warn: 100, bad: 1000 },
  tick_p95_ms: { warn: 500, bad: 1000 },
  consumer_lag: { warn: 100, bad: 500 },
}

export function perfStatus(key: PerfKey, value: number | null | undefined): PerfStatus {
  const t = PERF_THRESHOLDS[key]
  if (!t || value === null || value === undefined || !Number.isFinite(value)) return 'none'
  if (value >= t.bad) return 'bad'
  if (value >= t.warn) return 'warn'
  return 'ok'
}

export const DEPENDENCIES: { key: DependencyName; label: string; hint: string }[] = [
  { key: 'ingest', label: 'Приём NDTP', hint: 'TCP-сервер телеметрии' },
  { key: 'redis', label: 'Redis', hint: 'шина событий и горячее состояние' },
  { key: 'predictor', label: 'Predictor', hint: 'признаки и тики прогнозов' },
  { key: 'ml', label: 'ML-сервис', hint: 'инференс модели' },
  { key: 'postgres', label: 'PostgreSQL', hint: 'журнал прогнозов и алертов' },
]

export const DEP_STATE_LABEL: Record<DependencyState, string> = {
  up: 'работает',
  degraded: 'деградация',
  down: 'недоступен',
  unknown: 'нет данных',
  disabled: 'отключён',
}

export function depStatus(state: DependencyState | undefined): PerfStatus {
  if (state === 'up') return 'ok'
  if (state === 'degraded') return 'warn'
  if (state === 'down') return 'bad'
  return 'none'
}
