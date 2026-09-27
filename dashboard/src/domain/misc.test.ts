import { describe, expect, it } from 'vitest'
import { sparkPath } from '../components/sparkPath'
import { grafanaDashboardUrl, grafanaPanelUrl } from './grafana'
import { depStatus, perfStatus } from './perf'
import { busiestRoute, nearestStopSeq, problemRoute, routeShortName } from './routes'

describe('маршруты', () => {
  it('короткое имя без префикса', () => {
    expect(routeShortName({ route_id: 'R1', name: 'Маршрут R1: Ореховый бульв. — Серпуховская ул.' })).toBe(
      'Ореховый бульв. — Серпуховская ул.',
    )
    expect(routeShortName({ route_id: 'R2', name: 'Маршрут R2: ' })).toBe('R2')
  })

  it('маршрут с наибольшим числом проблем (красные весят вдвое)', () => {
    const routes = [{ route_id: 'R1' }, { route_id: 'R2' }, { route_id: 'R3' }]
    const incidents = [
      { route_id: 'R2', risk: 'yellow' },
      { route_id: 'R2', risk: 'yellow' },
      { route_id: 'R3', risk: 'red' },
      { route_id: 'R3', risk: 'yellow' },
      { route_id: 'R9', risk: 'red' },
    ]
    expect(busiestRoute(routes, incidents)).toBe('R3')
    expect(busiestRoute(routes, [])).toBe('R1')
    expect(busiestRoute([], incidents)).toBeNull()
  })
})

describe('производительность', () => {
  it('статус по целям наблюдаемости', () => {
    expect(perfStatus('e2e_p95_s', 0.3)).toBe('ok')
    expect(perfStatus('e2e_p95_s', 0.7)).toBe('warn')
    expect(perfStatus('e2e_p95_s', 1)).toBe('bad')
    expect(perfStatus('consumer_lag', 600)).toBe('bad')
    expect(perfStatus('inference_p95_ms', 99)).toBe('ok')
    expect(perfStatus('ingest_pps', 1e6)).toBe('none')
    expect(perfStatus('tick_p95_ms', null)).toBe('none')
    expect(depStatus('up')).toBe('ok')
    expect(depStatus('down')).toBe('bad')
    expect(depStatus(undefined)).toBe('none')
  })

  it('URL панели Grafana: d-solo, тёмная тема, киоск', () => {
    const url = grafanaPanelUrl('/grafana/', { uid: 'foresight-model', panelId: 7, title: 'x' })
    expect(url.startsWith('/grafana/d-solo/foresight-model/foresight-model?')).toBe(true)
    const params = new URLSearchParams(url.split('?')[1])
    expect(params.get('panelId')).toBe('7')
    expect(params.get('theme')).toBe('light')
    expect(params.get('from')).toBe('now-15m')
    expect(params.has('kiosk')).toBe(true)
    expect(grafanaDashboardUrl('/grafana', 'foresight-overview')).toBe(
      '/grafana/d/foresight-overview/foresight-overview?orgId=1',
    )
  })

  it('спарклайн рвётся на пропусках', () => {
    expect(sparkPath([0, 10], 100, 10, 10)).toBe('M0.0,9.0L100.0,1.0')
    expect(sparkPath([1, null, 1], 100, 10, 10)).toBe('M0.0,8.2M100.0,8.2')
    expect(sparkPath([5], 100, 10, 10)).toBe('')
  })
})

describe('маршрут по проблемам и привязка к остановке', () => {
  it('сначала маршрут со сбивкой', () => {
    const routes = [{ route_id: 'R1' }, { route_id: 'R2' }]
    expect(
      problemRoute(routes, [
        { route_id: 'R1', risk: 'red' },
        { route_id: 'R2', risk: 'yellow', kind: 'bunching' },
      ]),
    ).toBe('R2')
    expect(problemRoute(routes, [{ route_id: 'R1', risk: 'red', kind: 'delay' }])).toBe('R1')
  })

  it('ближайшая остановка', () => {
    const stops = [
      { seq: 0, lat: 55.6, lon: 37.6 },
      { seq: 1, lat: 55.61, lon: 37.62 },
      { seq: 2, lat: 55.62, lon: 37.64 },
    ]
    expect(nearestStopSeq(stops, 55.611, 37.619)).toBe(1)
    expect(nearestStopSeq([], 55, 37)).toBeNull()
  })
})
