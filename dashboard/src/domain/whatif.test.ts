import { describe, expect, it } from 'vitest'
import type { Scenario } from '../api/types'
import {
  candidateRequests,
  computeDelta,
  deltaScore,
  deltaTone,
  formatDeltaValue,
  formatMetric,
  headwaysByStop,
  meanWait,
  relativeChange,
  reserveCandidates,
  suggestReserve,
  verdictTone,
} from './whatif'

function scenario(patch: Partial<Scenario> = {}): Scenario {
  return {
    headways: [],
    mean_wait_s: 300,
    max_gap_s: 900,
    late_stops: 4,
    bunching_pairs: 1,
    vehicles: [],
    ...patch,
  }
}

describe('дельты What-if', () => {
  it('«после − до» по всем метрикам', () => {
    const d = computeDelta(
      scenario(),
      scenario({ mean_wait_s: 240, max_gap_s: 480, late_stops: 4, bunching_pairs: 2 }),
    )
    expect(d).toEqual({ mean_wait_s: -60, max_gap_s: -420, late_stops: 0, bunching_pairs: 1 })
  })

  it('меньше — лучше; для секунд — допуск 5 с', () => {
    expect(deltaTone('mean_wait_s', -60)).toBe('better')
    expect(deltaTone('mean_wait_s', 4)).toBe('same')
    expect(deltaTone('max_gap_s', 30)).toBe('worse')
    expect(deltaTone('late_stops', 0)).toBe('same')
    expect(deltaTone('bunching_pairs', 1)).toBe('worse')
    expect(deltaTone('late_stops', -2)).toBe('better')
  })

  it('общий итог', () => {
    expect(verdictTone({ mean_wait_s: -60, max_gap_s: -300, late_stops: 0, bunching_pairs: 0 })).toBe(
      'better',
    )
    expect(verdictTone({ mean_wait_s: -60, max_gap_s: 0, late_stops: 0, bunching_pairs: 1 })).toBe('mixed')
    expect(verdictTone({ mean_wait_s: 30, max_gap_s: 0, late_stops: 1, bunching_pairs: 0 })).toBe('worse')
    expect(verdictTone({ mean_wait_s: 2, max_gap_s: -3, late_stops: 0, bunching_pairs: 0 })).toBe('same')
  })

  it('форматирование значений и дельт', () => {
    expect(formatMetric('mean_wait_s', 252)).toBe('4 мин 12 с')
    expect(formatMetric('late_stops', 3)).toBe('3')
    expect(formatMetric('max_gap_s', 10603)).toBe('177 мин')
    expect(formatDeltaValue('mean_wait_s', -1498)).toBe('−25 мин')
    expect(formatDeltaValue('mean_wait_s', -65)).toBe('−1 мин 05 с')
    expect(formatDeltaValue('max_gap_s', 420)).toBe('+7 мин')
    expect(formatDeltaValue('bunching_pairs', -1)).toBe('−1')
    expect(formatDeltaValue('late_stops', 0)).toBe('0')
    expect(formatDeltaValue('mean_wait_s', 0.4)).toBe('0 с')
  })

  it('относительное изменение', () => {
    expect(relativeChange(300, 240)).toBeCloseTo(-0.2)
    expect(relativeChange(0, 10)).toBeNull()
  })
})

describe('интервалы и ожидание', () => {
  it('ожидание при случайном подходе: Σh²/(2Σh)', () => {
    expect(meanWait([600, 600, 600])).toBe(300)
    // неравномерные интервалы увеличивают ожидание
    expect(meanWait([60, 1140])).toBeGreaterThan(300)
    expect(meanWait([])).toBe(0)
  })

  it('средний и максимальный интервал по остановкам в порядке маршрута', () => {
    const s = scenario({
      headways: [
        { stop_key: 'b', t: '2026-01-06T08:10:00Z', gap_s: 600 },
        { stop_key: 'a', t: '2026-01-06T08:05:00Z', gap_s: 300 },
        { stop_key: 'b', t: '2026-01-06T08:30:00Z', gap_s: 1200 },
      ],
    })
    expect(headwaysByStop(s, ['a', 'b', 'c'])).toEqual([
      { stop_key: 'a', mean_gap_s: 300, max_gap_s: 300, n: 1 },
      { stop_key: 'b', mean_gap_s: 900, max_gap_s: 1200, n: 2 },
      { stop_key: 'c', mean_gap_s: null, max_gap_s: null, n: 0 },
    ])
  })
})

describe('подсказка «резерв в наибольший разрыв»', () => {
  const baseline = scenario({
    headways: [
      { stop_key: 'a', t: '2026-01-06T08:20:00Z', gap_s: 600 },
      { stop_key: 'c', t: '2026-01-06T08:40:00Z', gap_s: 1200 },
    ],
  })

  it('резерв приходит в середину наибольшего разрыва', () => {
    const s = suggestReserve(baseline, Date.parse('2026-01-06T08:00:00Z'))
    expect(s).toEqual({ from_stop_key: 'c', depart_at: '2026-01-06T08:30:00.000Z', gap_s: 1200 })
  })

  it('середина будущей части разрыва: резерв не едет следом за подходящим ТС', () => {
    // разрыв на «c» идёт с 08:20 до 08:40; сейчас 08:35 — резерв посередине оставшихся 5 мин, а не «сейчас»
    const s = suggestReserve(baseline, Date.parse('2026-01-06T08:35:00Z'))
    expect(s?.depart_at).toBe('2026-01-06T08:37:30.000Z')
  })

  it('без интервалов — нет подсказки', () => {
    expect(suggestReserve(scenario(), 0)).toBeNull()
  })
})

describe('перебор вариантов резерва', () => {
  const baseline = scenario({
    headways: [
      { stop_key: 'a', t: '2026-01-06T08:20:00Z', gap_s: 600 },
      { stop_key: 'c', t: '2026-01-06T08:40:00Z', gap_s: 1200 },
      { stop_key: 'c', t: '2026-01-06T09:00:00Z', gap_s: 900 },
      { stop_key: 'b', t: '2026-01-06T08:30:00Z', gap_s: 300 },
    ],
  })

  it('наибольшие разрывы на разных остановках, со сдвигами отправления', () => {
    const list = reserveCandidates(baseline, Date.parse('2026-01-06T08:00:00Z'), 2, [0, -120])
    expect(list.map((c) => [c.from_stop_key, c.depart_at.slice(11, 16)])).toEqual([
      ['c', '08:30'],
      ['c', '08:28'],
      ['a', '08:15'],
      ['a', '08:13'],
    ])
  })

  it('прошедшие и короткие разрывы не предлагаются', () => {
    // в 08:35 разрывы на «a» и «b» уже закрыты, на «c» больше всего осталось до 09:00 (08:45–09:00)
    const list = reserveCandidates(baseline, Date.parse('2026-01-06T08:35:00Z'), 3, [0])
    expect(list.map((c) => [c.from_stop_key, c.depart_at.slice(11, 19)])).toEqual([['c', '08:52:30']])
  })

  it('оценка: ожидание важнее разрыва, сбивка — сильный штраф', () => {
    const better = deltaScore({ mean_wait_s: -60, max_gap_s: -300, late_stops: 0, bunching_pairs: 0 })
    const withBunching = deltaScore({ mean_wait_s: -60, max_gap_s: -300, late_stops: 0, bunching_pairs: 1 })
    expect(better).toBeLessThan(0)
    expect(withBunching).toBeGreaterThan(better)
  })
})

describe('подбор действия', () => {
  it('резерв в разрывы и придержка каждого ТС', () => {
    const base = {
      route_id: 'R1',
      at: '2026-01-06T08:00:00Z',
      action: 'add_vehicle' as const,
      params: {},
      horizon_min: 60,
    }
    const baseline = scenario({ headways: [{ stop_key: 'c', t: '2026-01-06T08:40:00Z', gap_s: 1200 }] })
    const list = candidateRequests(base, baseline, [7, 9], Date.parse('2026-01-06T08:00:00Z'))
    expect(list.filter((r) => r.action === 'add_vehicle').map((r) => r.params.depart_at)).toEqual([
      '2026-01-06T08:30:00.000Z',
      '2026-01-06T08:28:00.000Z',
      '2026-01-06T08:32:00.000Z',
    ])
    expect(list.filter((r) => r.action === 'hold').map((r) => [r.params.tr_id, r.params.hold_s])).toEqual([
      [7, 120],
      [7, 180],
      [7, 300],
      [9, 120],
      [9, 180],
      [9, 300],
    ])
    expect(list.every((r) => r.route_id === 'R1' && r.horizon_min === 60)).toBe(true)
  })
})
