import { AimOutlined, BulbOutlined, LineChartOutlined, NodeCollapseOutlined } from '@ant-design/icons'
import { Button, Drawer, Skeleton, Space, Tooltip } from 'antd'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { useIncident } from '../api/queries'
import type { IncidentOut, Risk, RouteOut } from '../api/types'
import { causeInfo } from '../domain/causes'
import {
  formatClock,
  formatDelay,
  formatHm,
  formatIn,
  formatPercent,
  stopLabel,
  vehicleLabel,
} from '../domain/format'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { routeEnds } from '../domain/routes'
import { useLive } from '../state/liveStore'
import { CauseIcon } from './CauseIcon'
import { DeviationChart } from './DeviationChart'
import { RiskTag } from './RiskDot'

interface Props {
  incident: IncidentOut | null
  route: RouteOut | undefined
  open: boolean
  onClose: () => void
  onFocus: () => void
}

function Fact({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div>
      <div className="fact__label">{label}</div>
      <div className="fact__value" title={typeof value === 'string' ? value : undefined}>
        {value}
      </div>
      {hint ? <div className="fact__label">{hint}</div> : null}
    </div>
  )
}

export function IncidentDrawer({ incident, route, open, onClose, onFocus }: Props) {
  const navigate = useNavigate()
  const detail = useIncident(open && incident ? incident.incident_id : null)
  const streamTime = useLive((s) => s.streamTime)
  const inc = detail.data ?? incident
  const color = inc ? RISK_COLOR[inc.risk] : undefined
  const textColor = inc ? RISK_TEXT_COLOR[inc.risk] : undefined
  const bunching = inc?.kind === 'bunching'
  // цвет самого прогноза — по его значению; уровень инцидента держится гистерезисом и может отставать
  const ownRisk: Risk | null =
    inc && !bunching
      ? inc.pred_delay_s > 120 || (inc.p_late ?? 0) > 0.6
        ? 'red'
        : inc.pred_delay_s < 60 && (inc.p_late ?? 0) < 0.3
          ? 'green'
          : 'yellow'
      : null
  const easing = ownRisk === 'green' && inc?.risk !== 'green'
  const ownColor = ownRisk ? RISK_TEXT_COLOR[ownRisk] : textColor
  const steps = inc
    ? easing
      ? ['Действий не требуется: прогноз вернулся в норму, инцидент закроется сам.']
      : (inc.cause.recommendation || causeInfo(inc.cause.code).recommendation)
          .split('\n')
          .map((x) => x.trim())
          .filter((x) => x && x !== '—')
    : []
  const maxFactor = inc ? Math.max(1, ...inc.cause.factors.map((f) => Math.abs(f.contribution_s))) : 1

  return (
    <Drawer
      open={open}
      onClose={onClose}
      mask={false}
      size={typeof window !== 'undefined' && window.innerWidth < 1500 ? 420 : 480}
      placement="right"
      destroyOnHidden
      title={
        inc ? (
          <Space size={10} wrap>
            <RiskTag risk={inc.risk} />
            <span style={{ fontSize: 17 }}>{vehicleLabel(inc.tr_id, inc.unit_id)}</span>
            {inc.route_id ? (
              <span
                className="inc__route"
                style={{ background: `${route?.color ?? '#0ea5e9'}33`, fontSize: 14 }}
              >
                {inc.route_id}
              </span>
            ) : null}
            {route ? <span className="drawer-ends">{routeEnds(route)}</span> : null}
          </Space>
        ) : (
          'Инцидент'
        )
      }
      extra={
        <Tooltip title="Показать ТС на карте">
          <Button icon={<AimOutlined />} onClick={onFocus} aria-label="Показать на карте" />
        </Tooltip>
      }
      styles={{ body: { paddingTop: 0 } }}
    >
      {!inc ? (
        <Skeleton active />
      ) : (
        <>
          <div className="card-section">
            {bunching ? (
              <>
                <div className="card-section__title">Сбивка ТС</div>
                <div className="forecast-big" style={{ color: textColor }}>
                  <NodeCollapseOutlined />{' '}
                  {formatDelay(inc.cause.factors[0]?.contribution_s ?? null).replace('+', '')}
                </div>
                <div style={{ color: 'var(--muted)', marginTop: 4 }}>
                  интервал до ТС {inc.related_tr_id ?? '—'} впереди
                  {inc.cause.factors[1]
                    ? ` при плановом ${formatDelay(inc.cause.factors[1].contribution_s).replace('+', '')}`
                    : ''}
                </div>
              </>
            ) : (
              <>
                <div className="card-section__title">Прогноз опоздания</div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'space-between',
                    gap: 16,
                  }}
                >
                  <div>
                    <div className="forecast-big" style={{ color: ownColor }}>
                      {formatDelay(inc.pred_delay_s)}
                    </div>
                    <div style={{ color: 'var(--muted)', marginTop: 4 }}>
                      {inc.p10 !== null && inc.p90 !== null ? (
                        <>
                          интервал P10–P90: <span className="nowrap">{formatDelay(inc.p10)}</span> …{' '}
                          <span className="nowrap">{formatDelay(inc.p90)}</span>
                        </>
                      ) : (
                        'интервал не рассчитан моделью'
                      )}
                    </div>
                  </div>
                  {inc.p_late !== null ? (
                    <div className="prob">
                      <div className="prob__value" style={{ color: ownColor }}>
                        {formatPercent(inc.p_late)}
                      </div>
                      <div className="prob__label">вероятность опоздания больше 2 мин</div>
                    </div>
                  ) : null}
                </div>
                {easing ? (
                  <div className="easing-note">
                    ↓ Риск спадает: прогноз уже в норме. Инцидент закроется, если так будет несколько проверок
                    подряд.
                  </div>
                ) : null}
              </>
            )}
          </div>

          <div className="card-section">
            <div className="facts">
              <Fact label="Целевая остановка" value={stopLabel(inc.target_stop.name)} />
              <Fact
                label="План прибытия"
                value={formatHm(inc.target_stop.planned_at)}
                hint={formatIn(inc.time_to_event_s)}
              />
              <Fact label="Текущее отклонение" value={formatDelay(inc.vehicle.current_delay_s)} />
              <Fact
                label="Прогноз выдан"
                value={formatClock(inc.issued_at)}
                hint={`за ${Math.max(0, Math.round((Date.parse(inc.target_stop.planned_at) - Date.parse(inc.issued_at)) / 60000))} мин до плана`}
              />
              <Fact
                label="Участок"
                value={inc.segment ? `${inc.segment.from_stop} → ${inc.segment.to_stop}` : '—'}
              />
              <Fact
                label="Скорость"
                value={inc.vehicle.speed_kmh !== null ? `${inc.vehicle.speed_kmh} км/ч` : '—'}
              />
            </div>
          </div>

          <div className="card-section">
            <div className="card-section__title">Причина</div>
            <div
              style={{
                display: 'flex',
                gap: 10,
                alignItems: 'center',
                fontSize: 15,
                fontWeight: 600,
                marginBottom: 12,
              }}
            >
              <CauseIcon code={inc.cause.code} text={inc.cause.text} style={{ fontSize: 18, color }} />
              {inc.cause.code === 'unknown'
                ? 'Явной причины нет'
                : inc.cause.text || causeInfo(inc.cause.code).text}
            </div>
            {inc.cause.factors.length ? (
              <div className="factors-title">
                {bunching ? (
                  'Интервалы'
                ) : (
                  <>
                    <span>Что повлияло на прогноз</span>
                    <Tooltip title="Насколько каждый признак сдвинул прогноз: плюс — в сторону опоздания, минус — в сторону нагона. Это вклад в прогноз, а не значение признака.">
                      <span className="factors-title__unit">вклад в прогноз, с</span>
                    </Tooltip>
                  </>
                )}
              </div>
            ) : null}
            {inc.cause.factors.map((f) => (
              <div className="factor" key={f.feature}>
                <span>{f.label}</span>
                <span className="factor__value">
                  {bunching
                    ? formatDelay(f.contribution_s)
                    : `${f.contribution_s > 0 ? '+' : ''}${Math.round(f.contribution_s)} с`}
                </span>
                <div className="factor__bar">
                  <div
                    className="factor__fill"
                    style={{
                      width: `${(Math.abs(f.contribution_s) / maxFactor) * 100}%`,
                      background: f.contribution_s >= 0 ? (color ?? '#f5b90b') : '#22c55e',
                    }}
                  />
                </div>
              </div>
            ))}
            {inc.cause.factors.length === 0 ? (
              <div style={{ color: 'var(--muted)' }}>Вклад признаков недоступен</div>
            ) : null}
          </div>

          <div className="card-section">
            <div className="card-section__title">Рекомендация диспетчеру</div>
            <div className="recommendation">
              <BulbOutlined style={{ color: '#0284c7', fontSize: 18, marginTop: 2 }} />
              {steps.length > 1 ? (
                <ol className="recommendation__steps">
                  {steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              ) : (
                <span>{steps[0] ?? 'Наблюдать.'}</span>
              )}
            </div>
          </div>

          <div className="card-section">
            <div className="card-section__title">Отклонение за 30 мин и прогноз на 15 мин</div>
            {detail.data ? (
              <DeviationChart
                history={detail.data.history}
                // журнал прогнозов пуст (перезапуск, задержка записи) — прогноз самого инцидента
                forecast={
                  detail.data.forecast.length || bunching
                    ? detail.data.forecast
                    : [
                        {
                          stop_id: inc.target_stop.stop_id,
                          name: inc.target_stop.name,
                          planned_at: inc.target_stop.planned_at,
                          pred_delay_s: inc.pred_delay_s,
                          p10: inc.p10,
                          p90: inc.p90,
                        },
                      ]
                }
                now={streamTime}
              />
            ) : (
              <Skeleton.Node active style={{ width: '100%', height: 200 }} />
            )}
            {detail.isError ? <div style={{ color: 'var(--muted)' }}>Детали инцидента недоступны</div> : null}
          </div>

          <div className="card-section" style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <Button
              type="primary"
              icon={<LineChartOutlined />}
              disabled={!inc.route_id}
              onClick={() =>
                navigate(`/stringline?route=${encodeURIComponent(inc.route_id ?? '')}&tr=${inc.tr_id}`)
              }
            >
              График движения маршрута
            </Button>
            <Button icon={<AimOutlined />} onClick={onFocus}>
              На карте
            </Button>
          </div>
          <div style={{ color: 'var(--faint)', fontSize: 13 }}>id инцидента {inc.incident_id}</div>
        </>
      )}
    </Drawer>
  )
}
