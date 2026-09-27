/** Смоук-тест mock-мира: симулятор из фикстур dataset/test отдаёт данные строго по контракту и «живёт». */
import { describe, expect, it } from 'vitest'
import type { WsMessage } from '../api/types'
import { computeDelta } from '../domain/whatif'
import type { HonestyFixture, WorldFixture } from './fixtures'
import honestyJson from './fixtures/honesty.json'
import worldJson from './fixtures/world.json'
import { Simulator } from './simulator'

function makeSim(fleet = 0) {
  return new Simulator(worldJson as unknown as WorldFixture, honestyJson as unknown as HonestyFixture, {
    fleet,
  })
}

describe('Simulator', () => {
  const sim = makeSim()

  it('маршруты с линиями и остановками по порядку', () => {
    expect(sim.routes.length).toBeGreaterThan(3)
    for (const r of sim.routes) {
      expect(r.route_id).toMatch(/^R\d+$/)
      expect(r.line.length).toBeGreaterThanOrEqual(2)
      expect(r.stops.map((s) => s.seq)).toEqual(r.stops.map((_, i) => i))
    }
  })

  it('ТС на карте с координатами, прогнозами и риском', () => {
    const list = sim.vehicleList()
    expect(list.count).toBeGreaterThan(20)
    const online = list.vehicles.filter((v) => v.status === 'online')
    expect(online.length).toBeGreaterThan(10)
    for (const v of online) {
      expect(v.lat).not.toBeNull()
      expect(v.lon).not.toBeNull()
    }
    expect(list.vehicles.some((v) => v.pred_delay_s !== null)).toBe(true)
  })

  it('есть активные инциденты: красные/жёлтые с причиной и участком', () => {
    const items = sim.incidentList('active')
    expect(items.length).toBeGreaterThan(0)
    expect(items.some((i) => i.risk === 'red')).toBe(true)
    for (const inc of items) {
      expect(['red', 'yellow']).toContain(inc.risk)
      expect(inc.cause.text.length).toBeGreaterThan(0)
      expect(inc.cause.factors.length).toBeLessThanOrEqual(3)
    }
    // заголовок инцидента — прогноз на горизонт: «… через 10–15 мин», а не на уже прошедший план
    const lead = items
      .filter((i) => i.kind === 'delay')
      .map((i) => i.time_to_event_s)
      .sort((a, b) => a - b)
    expect(lead[Math.floor(lead.length / 2)]).toBeGreaterThan(9 * 60)
    const detail = sim.incidentDetail(items[0].incident_id)
    expect(detail?.history.length).toBeGreaterThan(0)
  })

  it('алерты выдаются только заранее: план через 10–15 мин, «задним числом» — 0', () => {
    const alerts = sim.alertList({ limit: 1000 })
    expect(alerts.length).toBeGreaterThan(0)
    for (const a of alerts) {
      const lead = (Date.parse(a.planned_at) - Date.parse(a.issued_at)) / 1000
      expect(lead).toBeGreaterThan(600)
      expect(lead).toBeLessThanOrEqual(900)
    }
    const h = sim.horizon()
    expect(h.retroactive).toBe(0)
    expect(h.closed).toBeGreaterThan(0)
    expect(h.online_mae_s).not.toBeNull()
  })

  it('время идёт: сообщения WS по контракту', () => {
    const got: WsMessage[] = []
    const unsubscribe = sim.subscribe((m) => got.push(m))
    const before = sim.now
    sim.advance(5 * 60_000)
    unsubscribe()
    expect(sim.now).toBe(before + 5 * 60_000)
    const types = new Set(got.map((m) => m.type))
    expect(types.has('incident')).toBe(true)
    expect(types.has('prediction_closed') || types.has('alert')).toBe(true)
  })

  it('нитка и What-if по маршруту', () => {
    const route = sim.routes.find((r) => r.tr_ids.length >= 3) ?? sim.routes[0]
    const sl = sim.stringline(route.route_id, sim.now - 3600_000, sim.now + 1200_000)
    expect(sl?.trips.length).toBeGreaterThan(0)
    const out = sim.whatIf({
      route_id: route.route_id,
      action: 'add_vehicle',
      params: { from_stop_key: route.stops[0].stop_key },
      horizon_min: 60,
    })
    expect(out).not.toBeNull()
    if (out) expect(out.delta).toEqual(computeDelta(out.baseline, out.scenario))
    expect(sim.whatIf({ route_id: 'R-none', action: 'hold', params: {}, horizon_min: 60 })).toBeNull()
  })

  it('нагрузочный режим: 300 ТС', () => {
    const big = makeSim(300)
    expect(big.vehicleList().count).toBe(300)
  })
})

describe('демонстрация деградации', () => {
  it('chaos=redis: health «degraded», Redis недоступен, ответы помечены degraded', () => {
    const sim = makeSim()
    sim.degraded = true
    expect(sim.health().status).toBe('degraded')
    expect(sim.health().dependencies.redis.state).toBe('down')
    expect(sim.perf().deps.redis).toBe('down')
    expect(sim.vehicleList().degraded).toBe(true)
  })
})
