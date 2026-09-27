import { describe, expect, it } from 'vitest'
import { makePrediction } from '../test/factories'
import { inInterval, leadMedianS, leadWindowShare, maeGain, outcome } from './honesty'

const hist = [
  { from_s: 480, to_s: 540, count: 0 },
  { from_s: 600, to_s: 660, count: 10 },
  { from_s: 660, to_s: 720, count: 30 },
  { from_s: 840, to_s: 900, count: 10 },
  { from_s: 900, to_s: 960, count: 0 },
]

describe('заблаговременность', () => {
  it('доля прогнозов в окне 10–15 мин', () => {
    expect(leadWindowShare(hist)).toBe(1)
    expect(leadWindowShare([...hist, { from_s: 300, to_s: 360, count: 50 }])).toBe(0.5)
    expect(leadWindowShare([])).toBeNull()
  })

  it('медиана по середине корзины', () => {
    expect(leadMedianS(hist)).toBe(690)
    expect(leadMedianS([])).toBeNull()
  })
})

describe('качество', () => {
  it('выигрыш у baseline', () => {
    expect(maeGain(45, 60)).toBeCloseTo(0.25)
    expect(maeGain(80, 60)).toBeCloseTo(-1 / 3)
    expect(maeGain(null, 60)).toBeNull()
  })

  it('итог закрытого прогноза', () => {
    expect(outcome(makePrediction({ risk: 'red', actual_delay_s: 200 }))).toBe('hit')
    expect(outcome(makePrediction({ risk: 'green', actual_delay_s: 200 }))).toBe('miss')
    expect(outcome(makePrediction({ risk: 'yellow', actual_delay_s: 10 }))).toBe('false_alarm')
    expect(outcome(makePrediction({ risk: 'yellow', actual_delay_s: 90 }))).toBe('ok')
    expect(outcome(makePrediction({ risk: 'green', actual_delay_s: -30 }))).toBe('ok')
    expect(outcome(makePrediction({ actual_delay_s: null }))).toBe('pending')
  })

  it('факт в интервале P10–P90', () => {
    expect(inInterval(makePrediction({ p10: 30, p90: 150, actual_delay_s: 130 }))).toBe(true)
    expect(inInterval(makePrediction({ p10: 30, p90: 150, actual_delay_s: 151 }))).toBe(false)
    expect(inInterval(makePrediction({ p10: null }))).toBeNull()
  })
})
