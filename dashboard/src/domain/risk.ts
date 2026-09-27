import type { Risk, RiskThresholds, VehicleOut } from '../api/types'

/** Пороги по умолчанию (контракт §1). */
export const DEFAULT_THRESHOLDS: RiskThresholds = {
  red_delay_s: 120,
  red_p_late: 0.6,
  green_delay_s: 60,
  green_p_late: 0.3,
}

/**
 * Уровень риска по порогам контракта:
 * - `red`: `pred_delay_s > 120` или `p_late > 0.6`;
 * - `green`: `pred_delay_s < 60` и (`p_late` нет или `< 0.3`);
 * - `yellow`: остальное; `unknown`: прогноза нет.
 */
export function riskLevel(
  predDelayS: number | null | undefined,
  pLate: number | null | undefined,
  thresholds: RiskThresholds = DEFAULT_THRESHOLDS,
): Risk {
  const hasPred = predDelayS !== null && predDelayS !== undefined && Number.isFinite(predDelayS)
  const hasP = pLate !== null && pLate !== undefined && Number.isFinite(pLate)
  if (!hasPred) return hasP && pLate > thresholds.red_p_late ? 'red' : 'unknown'
  if (predDelayS > thresholds.red_delay_s || (hasP && pLate > thresholds.red_p_late)) return 'red'
  if (predDelayS < thresholds.green_delay_s && (!hasP || pLate < thresholds.green_p_late)) return 'green'
  return 'yellow'
}

/** Риск ТС: из ответа backend, иначе — вычисленный по прогнозу. */
export function vehicleRisk(vehicle: Pick<VehicleOut, 'risk' | 'pred_delay_s' | 'p_late'>): Risk {
  return vehicle.risk ?? riskLevel(vehicle.pred_delay_s, vehicle.p_late)
}

export const RISK_ORDER: Record<Risk, number> = { red: 0, yellow: 1, green: 2, unknown: 3 }

export const RISK_COLOR: Record<Risk, string> = {
  green: '#22c55e',
  yellow: '#f5b90b',
  red: '#ef4444',
  unknown: '#64748b',
}

/** Текст цвета риска на белом фоне — темнее самого цвета (жёлтый иначе не читается). */
export const RISK_TEXT_COLOR: Record<Risk, string> = {
  green: '#15803d',
  yellow: '#b45309',
  red: '#dc2626',
  unknown: '#475569',
}

export const RISK_LABEL: Record<Risk, string> = {
  green: 'Норма',
  yellow: 'Риск опоздания',
  red: 'Высокий риск',
  unknown: 'Нет прогноза',
}

export const RISK_SHORT: Record<Risk, string> = {
  green: 'норма',
  yellow: 'риск',
  red: 'высокий',
  unknown: 'нет прогноза',
}

export function isAtRisk(risk: Risk): boolean {
  return risk === 'red' || risk === 'yellow'
}
