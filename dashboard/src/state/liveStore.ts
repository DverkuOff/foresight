/**
 * Живое состояние диспетчерского экрана: ТС, активные инциденты, алерты, закрытые прогнозы, часы потока.
 * Наполняется REST-запросами при (пере)подключении и сообщениями `/ws`. Хранилище внешнее
 * (`useSyncExternalStore`), чтобы частые обновления ТС не шли через контекст React.
 */
import { useSyncExternalStore } from 'react'
import type { AlertOut, IncidentOut, PredictionOut, VehicleOut, WsMessage } from '../api/types'
import type { WsState } from '../api/ws'
import { toMs } from '../domain/format'

export const MAX_ALERTS = 500
export const MAX_CLOSED = 200

export interface LiveState {
  vehicles: ReadonlyMap<number, VehicleOut>
  incidents: ReadonlyMap<string, IncidentOut>
  /** Новые первыми. */
  alerts: readonly AlertOut[]
  /** Закрытые прогнозы, новые первыми. */
  closed: readonly PredictionOut[]
  streamTime: string | null
  epoch: number | null
  degraded: boolean
  /** Клиентское время последнего сообщения WS (мс). */
  lastMessageAt: number | null
  /** Клиентское время, когда часы потока последний раз сдвинулись вперёд. */
  streamAdvancedAt: number | null
  ws: WsState
  /** Получен первый снимок ТС. */
  ready: boolean
  /** Список инцидентов получен (REST или WS) — можно выбирать маршрут «по проблемам». */
  incidentsReady: boolean
}

export const initialLiveState: LiveState = {
  vehicles: new Map(),
  incidents: new Map(),
  alerts: [],
  closed: [],
  streamTime: null,
  epoch: null,
  degraded: false,
  lastMessageAt: null,
  streamAdvancedAt: null,
  ws: { status: 'connecting', attempt: 0, nextRetryAt: null, openedAt: null },
  ready: false,
  incidentsReady: false,
}

function withClock(state: LiveState, streamTime: string | null | undefined, now: number): LiveState {
  if (!streamTime) return state
  const next = toMs(streamTime)
  const prev = toMs(state.streamTime)
  if (next === null) return state
  if (prev !== null && next === prev) return state
  return { ...state, streamTime, streamAdvancedAt: now }
}

export function applyVehicles(
  state: LiveState,
  vehicles: VehicleOut[],
  mode: 'snapshot' | 'delta',
  streamTime: string | null | undefined,
  degraded: boolean | undefined,
  now: number,
): LiveState {
  const map = mode === 'snapshot' ? new Map<number, VehicleOut>() : new Map(state.vehicles)
  for (const v of vehicles) map.set(v.unit_id, v)
  const next = { ...state, vehicles: map, ready: true, degraded: degraded ?? state.degraded }
  return withClock(next, streamTime, now)
}

export function applyIncident(
  state: LiveState,
  action: 'open' | 'update' | 'close',
  incident: IncidentOut,
): LiveState {
  const map = new Map(state.incidents)
  if (action === 'close') map.delete(incident.incident_id)
  else map.set(incident.incident_id, incident)
  return { ...state, incidents: map }
}

export function setIncidents(state: LiveState, items: IncidentOut[]): LiveState {
  return { ...state, incidents: new Map(items.map((i) => [i.incident_id, i])), incidentsReady: true }
}

export function addAlert(state: LiveState, alert: AlertOut): LiveState {
  const rest = state.alerts.filter((a) => a.alert_id !== alert.alert_id)
  return { ...state, alerts: [alert, ...rest].slice(0, MAX_ALERTS) }
}

export function setAlerts(state: LiveState, items: AlertOut[]): LiveState {
  const sorted = [...items].sort((a, b) => (toMs(b.issued_at) ?? 0) - (toMs(a.issued_at) ?? 0))
  return { ...state, alerts: sorted.slice(0, MAX_ALERTS) }
}

export function addClosed(state: LiveState, prediction: PredictionOut): LiveState {
  const rest = state.closed.filter((p) => p.prediction_id !== prediction.prediction_id)
  return { ...state, closed: [prediction, ...rest].slice(0, MAX_CLOSED) }
}

export function setClosed(state: LiveState, items: PredictionOut[]): LiveState {
  const sorted = [...items].sort((a, b) => (toMs(b.closed_at) ?? 0) - (toMs(a.closed_at) ?? 0))
  return { ...state, closed: sorted.slice(0, MAX_CLOSED) }
}

/** Применить сообщение `/ws`. */
export function reduceWs(state: LiveState, msg: WsMessage, now: number): LiveState {
  const touched = { ...state, lastMessageAt: now }
  switch (msg.type) {
    case 'snapshot':
    case 'delta':
      return applyVehicles(touched, msg.vehicles, msg.type, msg.stream_time, msg.degraded, now)
    case 'alert':
      return withClock(addAlert(touched, msg.alert), msg.stream_time, now)
    case 'incident':
      return withClock(applyIncident(touched, msg.action, msg.incident), msg.stream_time, now)
    case 'prediction_closed':
      return withClock(addClosed(touched, msg.prediction), msg.stream_time, now)
    case 'clock': {
      const epochChanged = state.epoch !== null && msg.epoch !== state.epoch
      const base = epochChanged
        ? // сброс часов потока (перезапуск replayer): старые инциденты и алерты больше не актуальны
          { ...touched, incidents: new Map(), alerts: [], epoch: msg.epoch, streamTime: null }
        : { ...touched, epoch: msg.epoch }
      return withClock(base, msg.stream_time, now)
    }
  }
}

type Listener = () => void

export interface LiveStore {
  getState(): LiveState
  setState(update: (state: LiveState) => LiveState): void
  subscribe(listener: Listener): () => void
  dispatch(msg: WsMessage, now?: number): void
  reset(): void
}

/**
 *  schedule — отложенное уведомление подписчиков: пачка сообщений WS (десятки инцидентов за тик) даёт одну
 *   перерисовку вместо десятков. Без него подписчики уведомляются сразу (тесты).
 */
export function createLiveStore(
  initial: LiveState = initialLiveState,
  schedule?: (flush: () => void) => void,
): LiveStore {
  let state = initial
  let pending = false
  const listeners = new Set<Listener>()
  const notify = () => {
    pending = false
    listeners.forEach((l) => l())
  }
  const store: LiveStore = {
    getState: () => state,
    setState(update) {
      const next = update(state)
      if (next === state) return
      state = next
      if (!schedule) notify()
      else if (!pending) {
        pending = true
        schedule(notify)
      }
    },
    subscribe(listener) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    dispatch(msg, now = Date.now()) {
      store.setState((s) => reduceWs(s, msg, now))
    },
    reset() {
      store.setState(() => initial)
    },
  }
  return store
}

/** Состояние экрана: уведомления подписчиков не чаще раза в 50 мс. */
export const liveStore = createLiveStore(initialLiveState, (flush) => setTimeout(flush, 50))

/**
 * Подписка на срез состояния. Селектор обязан возвращать уже существующие ссылки (поля состояния) или примитивы —
 * производные массивы считаются в компоненте через `useMemo`.
 */
export function useLive<T>(selector: (state: LiveState) => T): T {
  return useSyncExternalStore(
    liveStore.subscribe,
    () => selector(liveStore.getState()),
    () => selector(liveStore.getState()),
  )
}
