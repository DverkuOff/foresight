import { describe, expect, it } from 'vitest'
import { DEFAULT_THRESHOLDS, isAtRisk, riskLevel, vehicleRisk } from './risk'

describe('riskLevel (пороги контракта §1)', () => {
  it('красный: прогноз > 120 с или p_late > 0.6', () => {
    expect(riskLevel(121, null)).toBe('red')
    expect(riskLevel(30, 0.61)).toBe('red')
    expect(riskLevel(500, 0.1)).toBe('red')
  })

  it('граница 120 с и 0.6 — ещё не красный', () => {
    expect(riskLevel(120, null)).toBe('yellow')
    expect(riskLevel(30, 0.6)).toBe('yellow')
  })

  it('зелёный: прогноз < 60 с и p_late нет или < 0.3', () => {
    expect(riskLevel(59, null)).toBe('green')
    expect(riskLevel(-200, 0.29)).toBe('green')
    expect(riskLevel(0, undefined)).toBe('green')
  })

  it('жёлтый: всё промежуточное', () => {
    expect(riskLevel(60, null)).toBe('yellow')
    expect(riskLevel(59, 0.3)).toBe('yellow')
    expect(riskLevel(90, 0.5)).toBe('yellow')
  })

  it('unknown без прогноза; одна вероятность > 0.6 даёт красный', () => {
    expect(riskLevel(null, null)).toBe('unknown')
    expect(riskLevel(undefined, 0.2)).toBe('unknown')
    expect(riskLevel(Number.NaN, null)).toBe('unknown')
    expect(riskLevel(null, 0.7)).toBe('red')
  })

  it('пороги из админки применяются', () => {
    const strict = { ...DEFAULT_THRESHOLDS, red_delay_s: 60, green_delay_s: 30 }
    expect(riskLevel(61, null, strict)).toBe('red')
    expect(riskLevel(45, null, strict)).toBe('yellow')
    expect(riskLevel(29, null, strict)).toBe('green')
  })
})

describe('vehicleRisk', () => {
  it('берёт риск backend, иначе считает по прогнозу', () => {
    expect(vehicleRisk({ risk: 'yellow', pred_delay_s: 500, p_late: 0.9 })).toBe('yellow')
    expect(vehicleRisk({ risk: null, pred_delay_s: 150, p_late: null })).toBe('red')
    expect(vehicleRisk({ pred_delay_s: null, p_late: null })).toBe('unknown')
  })

  it('isAtRisk — только жёлтые и красные', () => {
    expect(isAtRisk('red')).toBe(true)
    expect(isAtRisk('yellow')).toBe(true)
    expect(isAtRisk('green')).toBe(false)
    expect(isAtRisk('unknown')).toBe(false)
  })
})
