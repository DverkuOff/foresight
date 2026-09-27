import { describe, expect, it } from 'vitest'
import { makeIncident } from '../test/factories'
import { compareIncidents, filterIncidents, sortIncidents } from './incidents'

describe('sortIncidents (порядок контракта)', () => {
  it('красные выше жёлтых независимо от вероятности', () => {
    const items = [
      makeIncident({ incident_id: 'y', risk: 'yellow', p_late: 0.59, pred_delay_s: 119 }),
      makeIncident({ incident_id: 'r', risk: 'red', p_late: 0.2, pred_delay_s: 130 }),
      makeIncident({ incident_id: 'g', risk: 'green', p_late: 0.99 }),
    ]
    expect(sortIncidents(items).map((i) => i.incident_id)).toEqual(['r', 'y', 'g'])
  })

  it('внутри уровня — по p_late, затем по pred_delay_s по убыванию', () => {
    const items = [
      makeIncident({ incident_id: 'a', risk: 'red', p_late: 0.7, pred_delay_s: 150 }),
      makeIncident({ incident_id: 'b', risk: 'red', p_late: 0.9, pred_delay_s: 130 }),
      makeIncident({ incident_id: 'c', risk: 'red', p_late: 0.7, pred_delay_s: 300 }),
    ]
    expect(sortIncidents(items).map((i) => i.incident_id)).toEqual(['b', 'c', 'a'])
  })

  it('без вероятности (сбивка) — после инцидентов с вероятностью того же уровня', () => {
    const items = [
      makeIncident({ incident_id: 'bun', kind: 'bunching', risk: 'red', p_late: null, pred_delay_s: 500 }),
      makeIncident({ incident_id: 'del', risk: 'red', p_late: 0.61, pred_delay_s: 100 }),
    ]
    expect(sortIncidents(items).map((i) => i.incident_id)).toEqual(['del', 'bun'])
  })

  it('при равенстве — ближайшее событие раньше, порядок стабилен', () => {
    const a = makeIncident({ incident_id: 'a', time_to_event_s: 800 })
    const b = makeIncident({ incident_id: 'b', time_to_event_s: 650 })
    expect(compareIncidents(a, b)).toBeGreaterThan(0)
    expect(sortIncidents([a, b]).map((i) => i.incident_id)).toEqual(['b', 'a'])
  })

  it('не мутирует исходный массив', () => {
    const items = [makeIncident({ incident_id: 'y' }), makeIncident({ incident_id: 'r', risk: 'red' })]
    sortIncidents(items)
    expect(items[0].incident_id).toBe('y')
  })
})

describe('filterIncidents', () => {
  const items = [
    makeIncident({ incident_id: '1', risk: 'red', tr_id: 122048, route_id: 'R1' }),
    makeIncident({ incident_id: '2', risk: 'yellow', tr_id: 5001, route_id: 'R12' }),
    makeIncident({ incident_id: '3', risk: 'red', kind: 'bunching', tr_id: 7, route_id: 'R3' }),
  ]

  it('по уровню и типу', () => {
    expect(filterIncidents(items, 'red').map((i) => i.incident_id)).toEqual(['1', '3'])
    expect(filterIncidents(items, 'yellow').map((i) => i.incident_id)).toEqual(['2'])
    expect(filterIncidents(items, 'bunching').map((i) => i.incident_id)).toEqual(['3'])
  })

  it('поиск по ТС, маршруту и остановке', () => {
    expect(filterIncidents(items, 'all', '1220').map((i) => i.incident_id)).toEqual(['1'])
    expect(filterIncidents(items, 'all', 'r12').map((i) => i.incident_id)).toEqual(['2'])
    expect(filterIncidents(items, 'all', 'R1').map((i) => i.incident_id)).toEqual(['1'])
    expect(filterIncidents(items, 'all', 'каширское')).toHaveLength(3)
  })
})
