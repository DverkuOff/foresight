import { NodeCollapseOutlined, SearchOutlined } from '@ant-design/icons'
import { Empty, Input, Segmented } from 'antd'
import { memo, useMemo, useState, type CSSProperties } from 'react'
import type { IncidentOut, RouteOut } from '../api/types'
import { causeInfo } from '../domain/causes'
import {
  formatDelayShort,
  formatHm,
  formatIn,
  formatPercent,
  stopLabel,
  vehicleLabel,
} from '../domain/format'
import { filterIncidents, type IncidentFilter } from '../domain/incidents'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { routeEnds } from '../domain/routes'
import { CauseIcon } from './CauseIcon'
import { RiskDot } from './RiskDot'

interface Props {
  incidents: IncidentOut[]
  routes: Map<string, RouteOut>
  selectedId: string | null
  onSelect: (incident: IncidentOut) => void
  /** Текст пустого списка, когда данных нет (нет связи, ожидание снимка). */
  unavailableText?: string | null
}

/** Строка списка; memo — строки не перерисовываются при каждом обновлении положения ТС. */
const IncidentRow = memo(function IncidentRow({
  incident,
  route,
  selected,
  onSelect,
}: {
  incident: IncidentOut
  route: RouteOut | undefined
  selected: boolean
  onSelect: (incident: IncidentOut) => void
}) {
  const color = RISK_COLOR[incident.risk]
  const bunching = incident.kind === 'bunching'
  // прогноз уже в норме, а уровень держится гистерезисом: инцидент закроется через несколько тиков
  const easing =
    !bunching && incident.pred_delay_s < 60 && (incident.p_late === null || incident.p_late < 0.3)
  return (
    <button
      type="button"
      className={`inc ${selected ? 'inc--selected' : ''}`}
      style={{ '--risk': color, '--risk-text': RISK_TEXT_COLOR[incident.risk] } as CSSProperties}
      onClick={() => onSelect(incident)}
      aria-pressed={selected}
    >
      <div className="inc__top">
        <span className="inc__vehicle">{vehicleLabel(incident.tr_id, incident.unit_id)}</span>
        {incident.route_id ? (
          <span className="inc__route" style={{ background: `${route?.color ?? '#0ea5e9'}33` }}>
            {incident.route_id}
          </span>
        ) : null}
        {route ? <span className="inc__ends">{routeEnds(route)}</span> : null}
        {bunching ? (
          <span className="inc__route" style={{ background: '#7c3aed26', color: '#6d28d9' }}>
            <NodeCollapseOutlined /> сбивка
          </span>
        ) : null}
      </div>
      <div className="inc__headline">
        {bunching ? (
          <>
            <span className="inc__delay">Сбивка</span> с ТС {incident.related_tr_id ?? '—'} у{' '}
            {stopLabel(incident.target_stop.name)} {formatIn(incident.time_to_event_s)}
          </>
        ) : (
          <>
            <span className="inc__delay">{formatDelayShort(incident.pred_delay_s)}</span> к{' '}
            {stopLabel(incident.target_stop.name)} {formatIn(incident.time_to_event_s)}
          </>
        )}
      </div>
      <div className="inc__side">
        <span className="inc__prob">{incident.p_late !== null ? formatPercent(incident.p_late) : '—'}</span>
        <span className="inc__prob-label">{incident.p_late !== null ? 'вероятн.' : 'интервал'}</span>
      </div>
      <div className="inc__meta">
        {easing ? (
          <span
            className="inc__easing"
            title="Прогноз уже в норме. Инцидент закроется, если так будет несколько проверок подряд"
          >
            ↓ риск спадает
          </span>
        ) : null}
        <CauseIcon code={incident.cause.code} text={incident.cause.text} />
        <span>{causeInfo(incident.cause.code).short}</span>
        <span>·</span>
        <span>план {formatHm(incident.target_stop.planned_at)}</span>
        {/* в узкой строке сжимается время выдачи, а не причина */}
        <span className="inc__issued">· выдан {formatHm(incident.issued_at)}</span>
      </div>
    </button>
  )
})

export const IncidentList = memo(function IncidentList({
  incidents,
  routes,
  selectedId,
  onSelect,
  unavailableText = null,
}: Props) {
  const [filter, setFilter] = useState<IncidentFilter>('all')
  const [query, setQuery] = useState('')
  const counts = useMemo(
    () => ({
      red: incidents.filter((i) => i.risk === 'red').length,
      yellow: incidents.filter((i) => i.risk === 'yellow').length,
      bunching: incidents.filter((i) => i.kind === 'bunching').length,
    }),
    [incidents],
  )
  const visible = useMemo(() => filterIncidents(incidents, filter, query), [incidents, filter, query])

  return (
    <section className="panel incidents" aria-label="Проблемные ТС">
      <div className="panel__head">
        <span className="panel__title">Проблемные ТС</span>
        <span className="panel__hint">по убыванию риска · {incidents.length}</span>
      </div>
      <div className="incidents__filters">
        <Segmented<IncidentFilter>
          size="small"
          block
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: `Все ${incidents.length}` },
            {
              value: 'red',
              label: (
                <span className="seg-count" title="Высокий риск">
                  <RiskDot risk="red" size={8} /> {counts.red}
                </span>
              ),
            },
            {
              value: 'yellow',
              label: (
                <span className="seg-count" title="Риск опоздания">
                  <RiskDot risk="yellow" size={8} /> {counts.yellow}
                </span>
              ),
            },
            {
              value: 'bunching',
              label: (
                <span className="seg-count" title="Сбивка ТС">
                  <NodeCollapseOutlined /> {counts.bunching}
                </span>
              ),
            },
          ]}
        />
        <Input
          size="small"
          allowClear
          placeholder="ТС, маршрут"
          prefix={<SearchOutlined />}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Поиск по ТС и маршруту"
        />
      </div>
      <div className="incidents__list">
        {visible.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              unavailableText ??
              (incidents.length === 0
                ? 'Проблемных ТС нет — все идут по графику'
                : 'Нет инцидентов по фильтру')
            }
            style={{ marginTop: 40 }}
          />
        ) : (
          visible.map((inc) => (
            <IncidentRow
              key={inc.incident_id}
              incident={inc}
              route={inc.route_id ? routes.get(inc.route_id) : undefined}
              selected={inc.incident_id === selectedId}
              onSelect={onSelect}
            />
          ))
        )}
      </div>
    </section>
  )
})
