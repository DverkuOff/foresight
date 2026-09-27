/** Mock-сервер по путям контракта `docs/api-contract.md` (обработчики MSW поверх симулятора). */
import { http, HttpResponse, type RequestHandler } from 'msw'
import type { WhatIfRequest } from '../api/types'
import { iso, type Simulator } from './simulator'

function num(value: string | null): number | null {
  if (value === null || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function notFound(detail: string) {
  return HttpResponse.json({ detail }, { status: 404 })
}

export function createHandlers(sim: Simulator): RequestHandler[] {
  const meta = () => ({ stream_time: iso(sim.now), degraded: sim.degraded })
  return [
    http.get('*/health', () => HttpResponse.json(sim.health())),
    http.get('*/api/vehicles', () => HttpResponse.json(sim.vehicleList())),
    http.get('*/api/vehicles/:unitId', ({ params }) => {
      const v = sim.vehicleList().vehicles.find((x) => String(x.unit_id) === params.unitId)
      return v ? HttpResponse.json(v) : notFound(`unit ${String(params.unitId)} not found`)
    }),
    http.get('*/api/routes', () => HttpResponse.json(sim.routes)),
    http.get('*/api/incidents', ({ request }) => {
      const url = new URL(request.url)
      const status = url.searchParams.get('status') === 'all' ? 'all' : 'active'
      const limit = num(url.searchParams.get('limit')) ?? 200
      return HttpResponse.json({ ...meta(), items: sim.incidentList(status, limit) })
    }),
    http.get('*/api/incidents/:id', ({ params }) => {
      const detail = sim.incidentDetail(String(params.id))
      return detail ? HttpResponse.json(detail) : notFound(`incident ${String(params.id)} not found`)
    }),
    http.get('*/api/predictions', ({ request }) => {
      const url = new URL(request.url)
      const s = url.searchParams.get('status')
      const status = s === 'open' || s === 'closed' ? s : null
      const items = sim.predictionList(
        status,
        num(url.searchParams.get('tr_id')),
        num(url.searchParams.get('limit')) ?? 100,
      )
      return HttpResponse.json({ ...meta(), items })
    }),
    http.get('*/api/alerts', ({ request }) => {
      const url = new URL(request.url)
      const since = url.searchParams.get('since')
      const items = sim.alertList({
        since: since ? Date.parse(since) : null,
        level: url.searchParams.get('level'),
        limit: num(url.searchParams.get('limit')) ?? 200,
      })
      return HttpResponse.json({ ...meta(), items })
    }),
    http.get('*/api/stringline', ({ request }) => {
      const url = new URL(request.url)
      const routeId = url.searchParams.get('route_id') ?? ''
      const from = Date.parse(url.searchParams.get('from') ?? '') || sim.now - 60 * 60_000
      const to = Date.parse(url.searchParams.get('to') ?? '') || sim.now + 20 * 60_000
      const out = sim.stringline(routeId, from, to)
      return out ? HttpResponse.json(out) : notFound(`route ${routeId} not found`)
    }),
    http.get('*/api/metrics/horizon', () => HttpResponse.json(sim.horizon())),
    http.get('*/api/metrics/perf', () => HttpResponse.json(sim.perf())),
    http.post('*/api/whatif', async ({ request }) => {
      const body = (await request.json()) as WhatIfRequest
      const out = sim.whatIf(body)
      return out ? HttpResponse.json(out) : notFound(`route ${body.route_id} not found`)
    }),
    http.get('*/api/replay/status', () =>
      HttpResponse.json({
        state: 'running',
        mode: 'mock',
        split: 'test',
        speed: 6,
        loop: true,
        start: '06:00',
        cycle: 1,
        units: sim.vehicleList().count,
        connections_active: sim.vehicleList().count,
        progress: 0.3,
        packets_sent: 0,
        data_time: iso(sim.now),
        error: null,
        epoch: sim.epoch,
      }),
    ),
    http.post('*/api/replay/:action', () => HttpResponse.json({ state: 'running', mode: 'mock' })),
    http.get('*/api/admin/settings', () => HttpResponse.json(settings)),
    http.put('*/api/admin/settings', async ({ request }) => {
      settings = { ...((await request.json()) as typeof settings), updated_at: iso(sim.now) }
      return HttpResponse.json(settings)
    }),
    http.get('*/api/admin/models', () =>
      HttpResponse.json({
        active: 'v2-holdout',
        ml: 'up',
        versions: MOCK_MODELS.map((m) => ({ ...m, active: m.version === 'v2-holdout' })),
      }),
    ),
    http.get('*/api/admin/services', () =>
      HttpResponse.json({
        checked_at: iso(sim.now),
        services: ['api', 'ingest', 'redis', 'postgres', 'predictor', 'ml-service', 'replayer'].map(
          (name) => ({
            name,
            state: sim.degraded && name === 'redis' ? 'down' : 'up',
            detail: 'mock',
            latency_ms: null,
          }),
        ),
      }),
    ),
    http.get('*/api/admin/units', () =>
      HttpResponse.json(
        sim.vehicleList().vehicles.map((v) => ({
          unit_id: v.unit_id,
          tr_id: v.tr_id,
          route_id: v.route_id ?? null,
          scheduled: true,
          status: v.status,
          risk: v.risk ?? null,
          last_packet_at: v.last_packet_at ?? null,
        })),
      ),
    ),
    http.get('*/api/admin/journal', ({ request }) => {
      const kind = new URL(request.url).searchParams.get('kind')
      const items =
        kind === 'predictions'
          ? sim.predictionList(null, null, 500)
          : sim.alertList({ since: null, level: null, limit: 500 })
      return HttpResponse.json({ count: items.length, items })
    }),
  ]
}

let settings = {
  risk: { red_delay_s: 120, red_p_late: 0.6, green_delay_s: 60, green_p_late: 0.3 },
  alert: { min_level: 'yellow', min_p_late: 0 },
  updated_at: null as string | null,
  applies_within_s: 30,
}

const MOCK_MODELS = [
  // метрики из models/*/manifest.json
  { version: 'v1', model: 'catboost_mae_ensemble', precision: 'fp32', cv_mae: 77.0, test_mae: 75.8 },
  { version: 'v1-holdout', model: 'catboost_mae_ensemble', precision: 'fp32', cv_mae: 77.0, test_mae: 75.8 },
  { version: 'v2', model: 'catboost_gru_ensemble', precision: 'int8', cv_mae: 76.7, test_mae: 75.6 },
  { version: 'v2-holdout', model: 'catboost_gru_ensemble', precision: 'int8', cv_mae: 76.7, test_mae: 75.6 },
].map((m) => ({
  ...m,
  created_at: null,
  description: 'mock',
  components: [],
  baseline_test_mae: 93.4,
  online_mae_s: null,
  online_closed: null,
}))
