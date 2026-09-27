/** Сводное состояние системы для шапки и KPI: связь, свежесть данных, деградация. */
import { useHealth } from '../api/queries'
import { useLive } from './liveStore'

export const STALE_MESSAGE_MS = 10_000
export const STALE_STREAM_MS = 30_000

const DEP_LABEL: Record<string, string> = {
  redis: 'Redis',
  postgres: 'PostgreSQL',
  ml: 'ML-сервис',
  ingest: 'приём NDTP',
  predictor: 'predictor',
}

export interface SystemStatus {
  connection: 'open' | 'connecting' | 'lost'
  retryInS: number | null
  /** Данные не обновлялись (сек) или null, если свежие. */
  staleForS: number | null
  degraded: boolean
  degradedParts: string[]
  apiDown: boolean
}

export function useSystemStatus(now: number): SystemStatus {
  const ws = useLive((s) => s.ws)
  const lastMessageAt = useLive((s) => s.lastMessageAt)
  const streamAdvancedAt = useLive((s) => s.streamAdvancedAt)
  const storeDegraded = useLive((s) => s.degraded)
  const health = useHealth()

  const connection = ws.status === 'open' ? 'open' : ws.status === 'connecting' ? 'connecting' : 'lost'
  const retryInS = ws.nextRetryAt ? Math.max(0, Math.ceil((ws.nextRetryAt - now) / 1000)) : null

  let staleForS: number | null = null
  if (connection === 'open') {
    const msgAge = lastMessageAt ? now - lastMessageAt : 0
    const streamAge = streamAdvancedAt ? now - streamAdvancedAt : 0
    if (msgAge > STALE_MESSAGE_MS || streamAge > STALE_STREAM_MS) {
      staleForS = Math.round(Math.max(msgAge, streamAge) / 1000)
    }
  }

  const parts: string[] = []
  const h = health.data
  if (h) {
    for (const [name, dep] of Object.entries(h.dependencies ?? {})) {
      if (dep.state === 'down') parts.push(DEP_LABEL[name] ?? name)
    }
  }
  const apiDown = health.isError && !health.data
  if (storeDegraded && parts.length === 0) parts.push('часть данных из кэша')
  const degraded = storeDegraded || h?.status === 'degraded' || parts.length > 0
  return { connection, retryInS, staleForS, degraded, degradedParts: parts, apiDown }
}
