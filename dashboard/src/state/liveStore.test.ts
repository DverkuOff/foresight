import { describe, expect, it, vi } from 'vitest'
import { makeAlert, makeIncident, makePrediction, makeVehicle } from '../test/factories'
import { createLiveStore, initialLiveState, MAX_ALERTS, reduceWs, setAlerts } from './liveStore'

const ST = '2026-01-06T08:15:00Z'

describe('reduceWs', () => {
  it('snapshot заменяет ТС, delta дополняет', () => {
    let s = reduceWs(
      initialLiveState,
      {
        type: 'snapshot',
        stream_time: ST,
        vehicles: [makeVehicle({ unit_id: 1 }), makeVehicle({ unit_id: 2 })],
      },
      1000,
    )
    expect([...s.vehicles.keys()]).toEqual([1, 2])
    expect(s.ready).toBe(true)
    s = reduceWs(
      s,
      {
        type: 'delta',
        stream_time: ST,
        vehicles: [makeVehicle({ unit_id: 2, lat: 1 }), makeVehicle({ unit_id: 3 })],
      },
      2000,
    )
    expect([...s.vehicles.keys()]).toEqual([1, 2, 3])
    expect(s.vehicles.get(2)?.lat).toBe(1)
    s = reduceWs(s, { type: 'snapshot', stream_time: ST, vehicles: [makeVehicle({ unit_id: 3 })] }, 3000)
    expect([...s.vehicles.keys()]).toEqual([3])
  })

  it('часы потока: отмечает, когда время сдвинулось', () => {
    let s = reduceWs(initialLiveState, { type: 'clock', stream_time: ST, epoch: 1 }, 1000)
    expect(s).toMatchObject({ streamTime: ST, streamAdvancedAt: 1000, lastMessageAt: 1000, epoch: 1 })
    s = reduceWs(s, { type: 'clock', stream_time: ST, epoch: 1 }, 5000)
    expect(s.streamAdvancedAt).toBe(1000)
    expect(s.lastMessageAt).toBe(5000)
    s = reduceWs(s, { type: 'clock', stream_time: '2026-01-06T08:15:30Z', epoch: 1 }, 6000)
    expect(s.streamAdvancedAt).toBe(6000)
  })

  it('инциденты: open / update / close', () => {
    let s = reduceWs(
      initialLiveState,
      { type: 'incident', stream_time: ST, action: 'open', incident: makeIncident({ incident_id: 'a' }) },
      1,
    )
    s = reduceWs(
      s,
      {
        type: 'incident',
        stream_time: ST,
        action: 'update',
        incident: makeIncident({ incident_id: 'a', risk: 'red' }),
      },
      2,
    )
    expect(s.incidents.get('a')?.risk).toBe('red')
    s = reduceWs(
      s,
      { type: 'incident', stream_time: ST, action: 'close', incident: makeIncident({ incident_id: 'a' }) },
      3,
    )
    expect(s.incidents.size).toBe(0)
  })

  it('алерты: новые первыми, без дублей, с ограничением длины', () => {
    let s = reduceWs(
      initialLiveState,
      { type: 'alert', stream_time: ST, alert: makeAlert({ alert_id: 'a1' }) },
      1,
    )
    s = reduceWs(s, { type: 'alert', stream_time: ST, alert: makeAlert({ alert_id: 'a2' }) }, 2)
    s = reduceWs(s, { type: 'alert', stream_time: ST, alert: makeAlert({ alert_id: 'a1', level: 'red' }) }, 3)
    expect(s.alerts.map((a) => [a.alert_id, a.level])).toEqual([
      ['a1', 'red'],
      ['a2', 'yellow'],
    ])
    const many = Array.from({ length: MAX_ALERTS + 20 }, (_, i) => makeAlert({ alert_id: `x${i}` }))
    expect(setAlerts(s, many).alerts).toHaveLength(MAX_ALERTS)
  })

  it('закрытые прогнозы копятся для «Честности прогноза»', () => {
    const s = reduceWs(
      initialLiveState,
      { type: 'prediction_closed', stream_time: ST, prediction: makePrediction() },
      1,
    )
    expect(s.closed).toHaveLength(1)
  })

  it('смена эпохи (перезапуск потока) сбрасывает инциденты и алерты', () => {
    let s = reduceWs(initialLiveState, { type: 'clock', stream_time: ST, epoch: 1 }, 1)
    s = reduceWs(s, { type: 'incident', stream_time: ST, action: 'open', incident: makeIncident() }, 2)
    s = reduceWs(s, { type: 'alert', stream_time: ST, alert: makeAlert() }, 3)
    s = reduceWs(s, { type: 'clock', stream_time: '2026-01-06T07:30:00Z', epoch: 2 }, 4)
    expect(s.incidents.size).toBe(0)
    expect(s.alerts).toHaveLength(0)
    expect(s.epoch).toBe(2)
    expect(s.streamTime).toBe('2026-01-06T07:30:00Z')
  })
})

describe('createLiveStore', () => {
  it('уведомляет подписчиков только при изменении', () => {
    const store = createLiveStore()
    const listener = vi.fn()
    const unsubscribe = store.subscribe(listener)
    store.dispatch({ type: 'clock', stream_time: ST, epoch: 1 }, 10)
    expect(listener).toHaveBeenCalledTimes(1)
    store.setState((s) => s)
    expect(listener).toHaveBeenCalledTimes(1)
    unsubscribe()
    store.reset()
    expect(listener).toHaveBeenCalledTimes(1)
    expect(store.getState()).toBe(initialLiveState)
  })
})

describe('пакетные уведомления', () => {
  it('серия сообщений — одно уведомление подписчиков', () => {
    const queue: (() => void)[] = []
    const store = createLiveStore(initialLiveState, (flush) => queue.push(flush))
    const listener = vi.fn()
    store.subscribe(listener)
    for (let i = 0; i < 50; i += 1) {
      store.dispatch(
        {
          type: 'incident',
          stream_time: ST,
          action: 'open',
          incident: makeIncident({ incident_id: `i${i}` }),
        },
        i,
      )
    }
    expect(listener).not.toHaveBeenCalled()
    expect(queue).toHaveLength(1)
    queue[0]()
    expect(listener).toHaveBeenCalledTimes(1)
    expect(store.getState().incidents.size).toBe(50)
  })
})
