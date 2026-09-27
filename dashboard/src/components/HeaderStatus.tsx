import {
  ApiOutlined,
  ExperimentOutlined,
  HistoryOutlined,
  WarningOutlined,
  WifiOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Tooltip } from 'antd'
import { apiGet } from '../api/client'
import type { ReplayStatus } from '../api/types'
import { config } from '../config'
import { formatClock, formatDate } from '../domain/format'
import { useLive } from '../state/liveStore'
import { useNow } from '../state/useLiveFeed'
import { useSystemStatus } from '../state/useSystemStatus'

/** Правая часть шапки: демо-режим, связь, свежесть, деградация, часы потока. */
export function HeaderStatus() {
  const now = useNow(1000)
  const status = useSystemStatus(now)
  const streamTime = useLive((s) => s.streamTime)
  const epoch = useLive((s) => s.epoch)
  // ускорение демо-потока: часы потока идут быстрее настенных — подписываем, чтобы это не удивляло
  const replay = useQuery({
    queryKey: ['replay-status'],
    queryFn: ({ signal }) => apiGet<ReplayStatus>('/api/replay/status', undefined, signal),
    refetchInterval: 15_000,
    retry: 0,
  })
  const speed = replay.data?.state === 'running' && replay.data.speed > 1 ? replay.data.speed : null

  return (
    <div className="app-header__status">
      {config.apiMode === 'mock' ? (
        <Tooltip title="Mock-режим: реальные треки и расписание из dataset/test, прогнозы синтетические, часть ТС — клоны для плотности демо">
          <span className="pill pill--demo">
            <ExperimentOutlined /> ДЕМО-ДАННЫЕ
          </span>
        </Tooltip>
      ) : null}

      {status.connection === 'open' ? (
        <Tooltip title="Потоковое соединение с сервером (WebSocket) активно">
          <span className="pill pill--ok">
            <span className="dot dot--pulse" style={{ background: 'var(--green)' }} />
            <WifiOutlined /> Онлайн
          </span>
        </Tooltip>
      ) : status.connection === 'connecting' ? (
        <span className="pill">
          <WifiOutlined /> Подключение…
        </span>
      ) : (
        <Tooltip title="Показано последнее известное состояние; переподключение автоматически">
          <span className="pill pill--bad" role="status">
            <ApiOutlined /> Нет связи с сервером
            {status.retryInS !== null ? ` · повтор через ${status.retryInS} с` : ''}
          </span>
        </Tooltip>
      )}

      {status.staleForS !== null ? (
        <Tooltip title="Время потока не сдвигается: источник телеметрии остановлен или отстаёт">
          <span className="pill pill--warn" role="status">
            <HistoryOutlined /> Данные устарели · {status.staleForS} с
          </span>
        </Tooltip>
      ) : null}

      {status.degraded || status.apiDown ? (
        <Tooltip
          title={
            status.apiDown
              ? 'API не отвечает на /health'
              : `Работает в режиме деградации: ${status.degradedParts.join(', ') || 'см. «Производительность»'}`
          }
        >
          <span className="pill pill--warn" role="status">
            <WarningOutlined /> Деградация
            {status.degradedParts.length ? `: ${status.degradedParts.slice(0, 2).join(', ')}` : ''}
          </span>
        </Tooltip>
      ) : null}

      <Tooltip
        title={
          speed !== null
            ? `Демо-поток: исторический день ${replay.data?.split ?? ''} проигрывается ×${speed} — минута на часах
              стены = ${speed} мин данных. Эпоха часов: ${epoch ?? '—'}`
            : epoch !== null
              ? `Эпоха часов потока: ${epoch}`
              : undefined
        }
      >
        <div className="clock">
          <span className="clock__time">{formatClock(streamTime)}</span>
          <span className="clock__label">
            время потока · {formatDate(streamTime)}
            {speed !== null ? <b className="clock__speed"> · ×{speed}</b> : null}
          </span>
        </div>
      </Tooltip>
    </div>
  )
}
