import { describe, expect, it } from 'vitest'
import type { StringlineOut } from '../api/types'
import {
  buildStringline,
  bunchingPairs,
  shortStopName,
  splitSegments,
  stopAxis,
  visibleExtent,
  withBreaks,
} from './stringline'

const T0 = Date.parse('2026-01-06T08:00:00Z')
const at = (min: number) => new Date(T0 + min * 60_000).toISOString()

/** Маршрут из 4 остановок, два ТС с плановым интервалом 10 мин; второе догоняет первое. */
function sample(): StringlineOut {
  return {
    route_id: 'R1',
    stream_time: at(20),
    stops: [
      { stop_key: 'd', name: 'D', seq: 3 },
      { stop_key: 'a', name: 'A', seq: 0 },
      { stop_key: 'b', name: 'B', seq: 1 },
      { stop_key: 'c', name: 'C', seq: 2 },
    ],
    trips: [
      {
        tr_id: 1,
        planned: [0, 1, 2, 3].map((seq) => ({ t: at(seq * 5), seq })),
        // опаздывает: +4 мин к остановке 2
        actual: [
          { t: at(1), seq: 0 },
          { t: at(8), seq: 1 },
          { t: at(14), seq: 2 },
        ],
        forecast: [{ t: at(22), seq: 3, p10: at(21), p90: at(24) }],
      },
      {
        tr_id: 2,
        planned: [0, 1, 2, 3].map((seq) => ({ t: at(10 + seq * 5), seq })),
        actual: [
          { t: at(10), seq: 0 },
          { t: at(14.5), seq: 1 },
        ],
        forecast: [
          { t: at(19), seq: 2, p10: at(18), p90: at(20.5) },
          { t: at(22.5), seq: 3, p10: at(21.5), p90: at(24.5) },
        ],
      },
    ],
  }
}

describe('buildStringline', () => {
  const model = buildStringline(sample())

  it('остановки по порядку и время «сейчас»', () => {
    expect(model.stops.map((s) => s.stop_key)).toEqual(['a', 'b', 'c', 'd'])
    expect(model.now).toBe(T0 + 20 * 60_000)
  })

  it('нитки: план, факт и прогноз, продолжающий факт', () => {
    expect(model.planned).toHaveLength(2)
    expect(model.actual.map((t) => t.trId)).toEqual([1, 2])
    const fc1 = model.forecast.find((t) => t.trId === 1)
    // прогноз начинается с последней фактической точки
    expect(fc1?.points[0]).toEqual([T0 + 14 * 60_000, 2])
    expect(fc1?.points[1]).toEqual([T0 + 22 * 60_000, 3])
  })

  it('опоздания > 2 мин по факту относительно плана', () => {
    expect(model.late.map((l) => [l.trId, l.seq, Math.round(l.delayS)])).toEqual([
      [1, 1, 180],
      [1, 2, 240],
    ])
    expect(model.lastDelay.get(1)).toBe(240)
    expect(model.lastDelay.get(2)).toBe(-30)
  })

  it('полоса P10–P90 — многоугольник по часовой стрелке', () => {
    const band = model.bands.find((b) => b.trId === 2)
    expect(band?.polygon).toEqual([
      [T0 + 18 * 60_000, 2],
      [T0 + 21.5 * 60_000, 3],
      [T0 + 24.5 * 60_000, 3],
      [T0 + 20.5 * 60_000, 2],
    ])
    // у ТС 1 одна точка прогноза — полосы нет
    expect(model.bands.some((b) => b.trId === 1)).toBe(false)
  })

  it('сбивка: прибытия разных ТС ближе 35% планового интервала', () => {
    // остановка 3: прогноз ТС 1 — 08:22, ТС 2 — 08:22:30 при плане через 10 мин
    expect(model.bunching).toHaveLength(1)
    expect(model.bunching[0]).toMatchObject({ trA: 1, trB: 2, seq: 3, gapS: 30, predicted: true })
    expect(bunchingPairs(model.bunching)).toEqual([[1, 2]])
  })

  it('диапазон времени охватывает все точки и интервалы', () => {
    expect(model.range).toEqual([T0, T0 + 25 * 60_000])
  })

  it('пустой ответ', () => {
    const empty = buildStringline({ route_id: 'R0', stops: [], trips: [], stream_time: null })
    expect(empty.range).toBeNull()
    expect(empty.now).toBeNull()
    expect(empty.bunching).toEqual([])
  })
})

describe('разрывы ниток', () => {
  it('разрыв при переходе через конец круга и при длинном отстое', () => {
    const pts = [
      { t: 0, seq: 0 },
      { t: 60_000, seq: 1 },
      { t: 120_000, seq: 0 },
      { t: 180_000, seq: 1 },
      { t: 180_000 + 25 * 60_000, seq: 2 },
    ]
    expect(withBreaks(pts, 20 * 60)).toEqual([
      [0, 0],
      [60_000, 1],
      null,
      [120_000, 0],
      [180_000, 1],
      null,
      [1_680_000, 2],
    ])
  })

  it('splitSegments делит по разрывам и отбрасывает пустые', () => {
    expect(splitSegments([null, [1, 0], [2, 1], null, null, [3, 0]])).toEqual([
      [
        [1, 0],
        [2, 1],
      ],
      [[3, 0]],
    ])
    expect(splitSegments([])).toEqual([])
  })

  it('пары сбивки уникальны независимо от порядка', () => {
    const p = { t: 0, seq: 0, gapS: 10, predicted: false }
    expect(
      bunchingPairs([
        { ...p, trA: 2, trB: 1 },
        { ...p, trA: 1, trB: 2 },
        { ...p, trA: 3, trB: 1 },
      ]),
    ).toEqual([
      [1, 2],
      [1, 3],
    ])
  })
})

describe('автоподгонка осей и прореживание', () => {
  it('границы данных в окне', () => {
    const model = buildStringline(sample())
    const ext = visibleExtent(model, [T0 + 9 * 60_000, T0 + 16 * 60_000])
    expect(ext.t).toEqual([T0 + 10 * 60_000, T0 + 15 * 60_000])
    expect(ext.seq).toEqual([0, 3])
    expect(visibleExtent(model, [0, 1])).toEqual({ t: null, seq: null })
  })

  it('подсветка опоздания: участки > 2 мин вместе с точкой перед ними', () => {
    const model = buildStringline(sample())
    expect(model.lateRuns).toEqual([
      {
        trId: 1,
        points: [
          [T0 + 1 * 60_000, 0],
          [T0 + 8 * 60_000, 1],
          [T0 + 14 * 60_000, 2],
        ],
      },
    ])
    expect(model.factDelay.get(`1:${T0 + 14 * 60_000}`)).toBe(240)
  })
})

describe('прогноз от позиции ТС', () => {
  const data = sample()
  data.trips[1].now = { t: at(18.5), seq: 1.5, delay_s: 30 }
  const model = buildStringline(data)

  it('пунктир начинается там, где ТС сейчас, прогнозы в прошлом отбрасываются', () => {
    const fc2 = model.forecast.find((t) => t.trId === 2)
    expect(fc2?.points).toEqual([
      [T0 + 18.5 * 60_000, 1.5],
      [T0 + 19 * 60_000, 2],
      [T0 + 22.5 * 60_000, 3],
    ])
  })

  it('связка от последнего прохода до позиции ТС', () => {
    expect(model.links).toEqual([
      {
        trId: 2,
        points: [
          [T0 + 14.5 * 60_000, 1],
          [T0 + 18.5 * 60_000, 1.5],
        ],
      },
    ])
  })

  it('полоса P10–P90 расходится конусом от позиции ТС', () => {
    const band = model.bands.find((b) => b.trId === 2)
    expect(band?.polygon[0]).toEqual([T0 + 18.5 * 60_000, 1.5])
    expect(band?.polygon[band.polygon.length - 1]).toEqual([T0 + 18.5 * 60_000, 1.5])
  })
})

describe('ось остановок', () => {
  it('одна и та же остановка подряд — одна строка; одноимённые через дорогу — разные', () => {
    const axis = stopAxis([
      { stop_key: '37.41505,55.74185', name: 'Ярцевская ул., д.25, к.3', seq: 0 },
      { stop_key: '37.41000,55.74000', name: 'Кунцевская ул., д.1', seq: 1 },
      { stop_key: '37.41463,55.74160', name: 'Ярцевская ул., д.25, к.3', seq: 3 },
      { stop_key: '37.41505,55.74185', name: 'Ярцевская ул., д.25, к.3', seq: 2 },
      { stop_key: '37.42000,55.74500', name: 'Верейская ул., д.35', seq: 4 },
      { stop_key: '37.42100,55.74420', name: 'Верейская ул., д.35', seq: 5 },
    ])
    expect(axis.rows.map((r) => r.seqs)).toEqual([[0], [1], [2, 3], [4], [5]])
    expect(axis.rowOf(3)).toBe(2)
    expect(axis.rowOf(1.5)).toBe(1.5)
    expect(axis.rowOf(3.5)).toBe(2.5)
  })

  it('короткие названия', () => {
    expect(shortStopName('Молодогвардейская ул., д.25, к.1')).toBe('Молодогвардейская 25к1')
    expect(shortStopName('ул. Ивана Франко, д.32, к.1')).toBe('Ивана Франко 32к1')
    expect(shortStopName('Зеленоград, 16-й микрорайон, д.1624')).toBe('16 мкр 1624')
    expect(shortStopName('Пятницкое шоссе, д.5')).toBe('Пятницкое ш. 5')
    expect(shortStopName('Остановка №5')).toBe('Остановка №5')
  })

  it('конец круга и начало следующего в одну минуту плана — без вертикали через весь график', () => {
    const data = sample()
    const at0 = (min: number) => new Date(T0 + min * 60_000).toISOString()
    data.trips = [
      {
        tr_id: 9,
        planned: [
          { t: at0(0), seq: 2 },
          { t: at0(5), seq: 3 },
          { t: at0(5), seq: 0 },
          { t: at0(8), seq: 1 },
        ],
        actual: [],
        forecast: [],
      },
    ]
    const model = buildStringline(data)
    expect(model.planned[0].points).toEqual([
      [T0, 2],
      [T0 + 5 * 60_000, 3],
      null,
      [T0 + 5 * 60_000, 0],
      [T0 + 8 * 60_000, 1],
    ])
  })

  it('две остановки в одну минуту посреди маршрута — по порядку', () => {
    const data = sample()
    const at0 = (min: number) => new Date(T0 + min * 60_000).toISOString()
    data.trips = [
      {
        tr_id: 9,
        planned: [
          { t: at0(0), seq: 0 },
          { t: at0(2), seq: 2 },
          { t: at0(2), seq: 1 },
          { t: at0(4), seq: 3 },
        ],
        actual: [],
        forecast: [],
      },
    ]
    expect(buildStringline(data).planned[0].points).toEqual([
      [T0, 0],
      [T0 + 2 * 60_000, 1],
      [T0 + 2 * 60_000, 2],
      [T0 + 4 * 60_000, 3],
    ])
  })

  it('две остановки в одну минуту плана не рвут нитку', () => {
    const points = [
      { t: 0, seq: 1 },
      { t: 60_000, seq: 3 },
      { t: 60_000, seq: 2 },
    ]
    const data = sample()
    data.trips = [
      {
        tr_id: 9,
        planned: points.map((p) => ({ t: new Date(T0 + p.t).toISOString(), seq: p.seq })),
        actual: [],
        forecast: [],
      },
    ]
    const model = buildStringline(data)
    expect(model.planned[0].points).toEqual([
      [T0, 1],
      [T0 + 60_000, 2],
      [T0 + 60_000, 3],
    ])
  })
})
