/**
 * Поток данных диспетчерского экрана: REST-снимок при старте и при каждом (пере)подключении WS,
 * затем сообщения `/ws`. Пока WS нет — REST-опрос раз в 10 с (деградация без потери картинки).
 */
import { useCallback, useEffect, useState } from 'react'
import { fetchAlerts, fetchClosedPredictions, fetchIncidents, fetchVehicles } from '../api/queries'
import { useWebSocket, type SocketFactory } from '../api/ws'
import { parseWsMessage } from '../api/wsMessages'
import { wsUrl } from '../config'
import { applyVehicles, liveStore, setAlerts, setClosed, setIncidents } from './liveStore'

export async function refreshLiveState(): Promise<void> {
  const [vehicles, incidents, alerts, closed] = await Promise.allSettled([
    fetchVehicles(),
    fetchIncidents('active'),
    fetchAlerts(),
    fetchClosedPredictions(),
  ])
  const now = Date.now()
  liveStore.setState((state) => {
    let next = state
    if (vehicles.status === 'fulfilled') {
      const v = vehicles.value
      next = applyVehicles(next, v.vehicles, 'snapshot', v.stream_time, v.degraded, now)
    }
    if (incidents.status === 'fulfilled') {
      next = setIncidents(next, incidents.value.items)
      if (incidents.value.degraded) next = { ...next, degraded: true }
    }
    if (alerts.status === 'fulfilled') next = setAlerts(next, alerts.value.items)
    if (closed.status === 'fulfilled') next = setClosed(next, closed.value.items)
    return next
  })
}

export function useLiveFeed(createSocket: SocketFactory): void {
  const refresh = useCallback(() => {
    void refreshLiveState()
  }, [])

  const onMessage = useCallback(
    (data: unknown) => {
      const msg = parseWsMessage(data)
      if (!msg) return
      const prevEpoch = liveStore.getState().epoch
      liveStore.dispatch(msg)
      if (msg.type === 'clock' && prevEpoch !== null && msg.epoch !== prevEpoch) refresh()
    },
    [refresh],
  )

  const [url] = useState(() => wsUrl())
  const ws = useWebSocket({ url, onMessage, onOpen: refresh, createSocket })

  useEffect(() => {
    liveStore.setState((s) => ({ ...s, ws }))
  }, [ws])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    if (ws.status === 'open') return undefined
    const timer = setInterval(refresh, 10_000)
    return () => clearInterval(timer)
  }, [ws.status, refresh])
}

/** Текущее время, обновляемое с периодом `intervalMs` (для «устарело», «через N с»). */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(timer)
  }, [intervalMs])
  return now
}
