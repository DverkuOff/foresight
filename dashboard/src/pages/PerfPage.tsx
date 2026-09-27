/**
 * «Производительность»: ключевые метрики из `GET /api/metrics/perf` (со спарклайнами за последние минуты),
 * состояние зависимостей и встроенные панели Grafana (`/grafana/d-solo/...`, светлая тема, киоск).
 */
import {
  ApiOutlined,
  CarOutlined,
  ClockCircleOutlined,
  DashboardOutlined,
  ExportOutlined,
  FundProjectionScreenOutlined,
  HourglassOutlined,
  NodeIndexOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Skeleton, Tooltip } from 'antd'
import { useState, type ReactNode } from 'react'
import { usePerf } from '../api/queries'
import type { PerfOut } from '../api/types'
import { Sparkline } from '../components/Sparkline'
import { StatCard } from '../components/StatCard'
import { config } from '../config'
import { formatNumber } from '../domain/format'
import { GRAFANA_DASHBOARDS, grafanaDashboardUrl, grafanaPanelUrl, PERF_PANELS } from '../domain/grafana'
import {
  DEP_STATE_LABEL,
  DEPENDENCIES,
  depStatus,
  perfStatus,
  type PerfKey,
  type PerfStatus,
} from '../domain/perf'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { palette } from '../theme'

const SAMPLES = 120

const STATUS_COLOR: Record<PerfStatus, string> = {
  ok: RISK_COLOR.green,
  warn: RISK_COLOR.yellow,
  bad: RISK_COLOR.red,
  none: palette.accent,
}

/** Текст статуса на белом — темнее цвета точки. */
const STATUS_TEXT: Record<PerfStatus, string> = {
  ok: RISK_TEXT_COLOR.green,
  warn: RISK_TEXT_COLOR.yellow,
  bad: RISK_TEXT_COLOR.red,
  none: palette.faint,
}

interface MetricDef {
  key: PerfKey
  label: string
  icon: ReactNode
  format: (v: number) => string
  unit: string
  target: number | null
  sub: string
}

const METRICS: MetricDef[] = [
  {
    key: 'ingest_pps',
    label: 'Приём телеметрии',
    icon: <ThunderboltOutlined />,
    format: (v) => formatNumber(v, v < 10 ? 1 : 0),
    unit: 'пакетов/с',
    target: null,
    sub: 'навигационные пакеты NDTP',
  },
  {
    key: 'e2e_p95_s',
    label: 'Задержка обработки p95',
    icon: <ClockCircleOutlined />,
    format: (v) => formatNumber(v, 2),
    unit: 'с',
    target: 1,
    sub: 'приём пакета → обработка в predictor · цель < 1 с',
  },
  {
    key: 'inference_p95_ms',
    label: 'Инференс модели p95',
    icon: <FundProjectionScreenOutlined />,
    format: (v) => formatNumber(v, 0),
    unit: 'мс',
    target: 100,
    sub: 'POST /predict, пакет · цель < 100 мс',
  },
  {
    key: 'tick_p95_ms',
    label: 'Тик прогнозов p95',
    icon: <HourglassOutlined />,
    format: (v) => formatNumber(v, 0),
    unit: 'мс',
    target: 100,
    sub: 'признаки + модель для всех ТС · цель < 1 с',
  },
  {
    key: 'consumer_lag',
    label: 'Лаг очереди',
    icon: <NodeIndexOutlined />,
    format: (v) => formatNumber(v, 0),
    unit: 'в очереди',
    target: null,
    sub: 'событий Redis Streams → predictor · норма ≈ 0',
  },
  {
    key: 'vehicles_online',
    label: 'ТС на связи',
    icon: <CarOutlined />,
    format: (v) => formatNumber(v, 0),
    unit: 'ТС',
    target: null,
    sub: 'свежие координаты',
  },
]

async function fetchGrafanaHealth(base: string, signal: AbortSignal): Promise<boolean> {
  const timeout = AbortSignal.timeout(4000)
  const res = await fetch(`${base}/api/health`, { signal: AbortSignal.any([signal, timeout]) })
  if (!res.ok) return false
  const body: unknown = await res.json().catch(() => null)
  return typeof body === 'object' && body !== null && 'database' in body
}

function useGrafanaAvailable() {
  return useQuery({
    queryKey: ['grafana-health', config.grafanaUrl],
    queryFn: ({ signal }) => fetchGrafanaHealth(config.grafanaUrl, signal),
    retry: 0,
    refetchInterval: 30_000,
    staleTime: 20_000,
  })
}

function GrafanaPanels() {
  const available = useGrafanaAvailable()
  const base = config.grafanaUrl
  const openLink = grafanaDashboardUrl(base, 'foresight-overview')

  return (
    <section className="panel" style={{ marginTop: 12 }}>
      <div className="panel__head">
        <DashboardOutlined />
        <span className="panel__title">Grafana · метрики Prometheus</span>
        <span className="panel__hint">последние 15 минут · обновление 5 с</span>
        <span style={{ flex: 1 }} />
        <Button size="small" icon={<ExportOutlined />} href={openLink} target="_blank" rel="noreferrer">
          Открыть в Grafana
        </Button>
      </div>
      {available.isLoading ? (
        <div style={{ padding: 16 }}>
          <Skeleton active />
        </div>
      ) : available.data ? (
        <div className="grafana-grid">
          {PERF_PANELS.map((p) => (
            <div key={`${p.uid}-${p.panelId}`} className="grafana-cell">
              <iframe
                className="grafana-frame"
                title={p.title}
                src={grafanaPanelUrl(base, p)}
                loading="lazy"
                referrerPolicy="same-origin"
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="grafana-off">
          <ApiOutlined className="grafana-off__icon" />
          <div>
            <div className="grafana-off__title">Grafana сейчас недоступна</div>
            <div className="grafana-off__text">
              {config.apiMode === 'mock'
                ? 'Демо-режим: стек наблюдаемости (Prometheus + Grafana) не запущен. Карточки выше считает mock-сервер.'
                : `Панели встраиваются из ${base}/ — проверьте, что сервис grafana запущен и разрешено встраивание.`}{' '}
              На живом стенде здесь шесть панелей: пропускная способность, задержка обработки, инференс, лаг
              очереди, онлайн-MAE и доступность зависимостей.
            </div>
            <div className="grafana-off__links">
              {GRAFANA_DASHBOARDS.map((d) => (
                <a key={d.uid} href={grafanaDashboardUrl(base, d.uid)} target="_blank" rel="noreferrer">
                  {d.title}
                </a>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

export default function PerfPage() {
  const perf = usePerf()
  const [seen, setSeen] = useState(0)
  const [history, setHistory] = useState<PerfOut[]>([])

  // история для спарклайнов (обновление состояния при новом ответе — во время рендера, без эффекта)
  const data = perf.data
  if (data && perf.dataUpdatedAt !== seen) {
    setSeen(perf.dataUpdatedAt)
    setHistory((h) => [...h, data].slice(-SAMPLES))
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="page-head__title">Производительность и надёжность</div>
          <div className="page-head__sub">
            Ключевые показатели конвейера «приём → шина → прогноз → дашборд»; обновление каждые 3 секунды.
          </div>
        </div>
      </div>

      <div className="grid-6">
        {METRICS.map((m) => {
          const value = data?.[m.key] ?? null
          const status = perfStatus(m.key, value)
          const color = STATUS_COLOR[status]
          return (
            <StatCard
              key={m.key}
              label={m.label}
              icon={m.icon}
              accent={color}
              value={value === null ? '—' : m.format(value)}
              unit={value === null ? undefined : m.unit}
              sub={m.sub}
            >
              <Sparkline
                values={history.map((h) => h[m.key])}
                color={color}
                target={m.target}
                ariaLabel={`${m.label}: история`}
              />
            </StatCard>
          )
        })}
      </div>

      <section className="panel deps" style={{ marginTop: 12 }} aria-label="Зависимости">
        <div className="panel__head">
          <span className="panel__title">Сервисы и зависимости</span>
          <span className="panel__hint">
            при отказе любой из них система работает в режиме деградации, а не падает
          </span>
        </div>
        <div className="deps__row">
          {DEPENDENCIES.map((d, i) => {
            const state = data?.deps[d.key]
            const status = depStatus(state)
            const color = status === 'none' ? palette.faint : STATUS_COLOR[status]
            return (
              <div key={d.key} className="deps__item-wrap">
                {i > 0 ? <span className="deps__arrow">→</span> : null}
                <Tooltip title={d.hint}>
                  <div className="deps__item" style={{ borderColor: `${color}66` }}>
                    <span
                      className={`dot ${status === 'bad' ? 'dot--pulse' : ''}`}
                      style={{ background: color, width: 10, height: 10 }}
                    />
                    <div>
                      <div className="deps__name">{d.label}</div>
                      <div className="deps__state" style={{ color: STATUS_TEXT[status] }}>
                        {state ? DEP_STATE_LABEL[state] : 'нет данных'}
                      </div>
                    </div>
                  </div>
                </Tooltip>
              </div>
            )
          })}
        </div>
      </section>

      <GrafanaPanels />
    </div>
  )
}
