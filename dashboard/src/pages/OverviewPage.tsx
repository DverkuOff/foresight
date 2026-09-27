/** «Оперативная обстановка»: KPI сверху, карта слева, проблемные ТС справа, карточка инцидента. */
import { CloseOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useRoutes } from '../api/queries'
import type { IncidentOut, RouteOut, VehicleOut } from '../api/types'
import { IncidentDrawer } from '../components/IncidentDrawer'
import { IncidentList } from '../components/IncidentList'
import { KpiBar } from '../components/KpiBar'
import { RiskTag } from '../components/RiskDot'
import { VehicleMap } from '../components/VehicleMap'
import { formatDelay, formatHm, vehicleLabel } from '../domain/format'
import { sortIncidents } from '../domain/incidents'
import { vehicleRisk } from '../domain/risk'
import { useLive } from '../state/liveStore'
import { useNow } from '../state/useLiveFeed'
import { useSystemStatus } from '../state/useSystemStatus'

function VehicleCard({
  vehicle,
  route,
  onClose,
}: {
  vehicle: VehicleOut
  route?: RouteOut
  onClose: () => void
}) {
  return (
    <div
      className="panel"
      style={{
        position: 'absolute',
        top: 12,
        right: 56,
        zIndex: 3,
        width: 280,
        padding: 12,
        background: 'rgba(255,255,255,0.97)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <b style={{ fontSize: 15 }}>{vehicleLabel(vehicle.tr_id, vehicle.unit_id)}</b>
        {vehicle.route_id ? (
          <span className="inc__route" style={{ background: `${route?.color ?? '#0ea5e9'}33` }}>
            {vehicle.route_id}
          </span>
        ) : null}
        <span style={{ flex: 1 }} />
        <Button size="small" type="text" icon={<CloseOutlined />} onClick={onClose} aria-label="Закрыть" />
      </div>
      <RiskTag risk={vehicleRisk(vehicle)} />
      <div className="facts" style={{ marginTop: 10 }}>
        <div>
          <div className="fact__label">Сейчас</div>
          <div className="fact__value">{formatDelay(vehicle.current_delay_s ?? null)}</div>
        </div>
        <div>
          <div className="fact__label">Прогноз 10–15 мин</div>
          <div className="fact__value">{formatDelay(vehicle.pred_delay_s ?? null)}</div>
        </div>
        <div style={{ gridColumn: '1 / -1' }}>
          <div className="fact__label">Следующая остановка</div>
          <div className="fact__value" style={{ fontSize: 13 }}>
            {vehicle.next_stop
              ? `${vehicle.next_stop.name} · план ${formatHm(vehicle.next_stop.planned_at)}`
              : '—'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function OverviewPage() {
  const now = useNow(1000)
  const status = useSystemStatus(now)
  const routesQuery = useRoutes()
  const vehiclesMap = useLive((s) => s.vehicles)
  const incidentsMap = useLive((s) => s.incidents)
  const ready = useLive((s) => s.ready)
  const [params, setParams] = useSearchParams()
  const [selectedUnit, setSelectedUnit] = useState<number | null>(null)
  const [focusKey, setFocusKey] = useState(0)

  const routes = useMemo(() => routesQuery.data ?? [], [routesQuery.data])
  const routeById = useMemo(() => new Map(routes.map((r) => [r.route_id, r])), [routes])
  const vehicles = useMemo(() => [...vehiclesMap.values()], [vehiclesMap])
  const incidents = useMemo(() => sortIncidents(incidentsMap.values()), [incidentsMap])

  // ?incident=<id> или ?incident=top (для скриншотов и ссылок)
  const incidentParam = params.get('incident')
  const selectedIncidentId = incidentParam === 'top' ? (incidents[0]?.incident_id ?? null) : incidentParam
  const selectedIncident = selectedIncidentId ? (incidentsMap.get(selectedIncidentId) ?? null) : null
  const [lastIncident, setLastIncident] = useState<IncidentOut | null>(null)
  if (selectedIncident && selectedIncident !== lastIncident) setLastIncident(selectedIncident)
  const drawerIncident =
    selectedIncident ??
    (selectedIncidentId && lastIncident?.incident_id === selectedIncidentId ? lastIncident : null)

  useEffect(() => {
    if (incidentParam === 'top' && incidents[0]) {
      setParams({ incident: incidents[0].incident_id }, { replace: true })
    }
  }, [incidentParam, incidents, setParams])

  const selectIncident = useCallback(
    (inc: IncidentOut) => {
      setParams({ incident: inc.incident_id })
      setSelectedUnit(inc.unit_id)
      setFocusKey((k) => k + 1)
    },
    [setParams],
  )

  const onVehicleClick = useCallback(
    (unitId: number) => {
      const v = vehiclesMap.get(unitId)
      const incId = v?.incident_id ?? null
      if (incId && incidentsMap.has(incId)) {
        setParams({ incident: incId })
      } else {
        setParams({})
      }
      setSelectedUnit(unitId)
      setFocusKey((k) => k + 1)
    },
    [vehiclesMap, incidentsMap, setParams],
  )

  const closeDrawer = useCallback(() => {
    setParams({})
    setSelectedUnit(null)
  }, [setParams])

  const highlightedUnit = drawerIncident?.unit_id ?? selectedUnit
  const cardVehicle = !drawerIncident && selectedUnit !== null ? vehiclesMap.get(selectedUnit) : undefined

  return (
    <div className="page page--fixed">
      <KpiBar status={status} />
      <div className="ops">
        <section className="panel map-panel" aria-label="Карта">
          <VehicleMap
            routes={routes}
            vehicles={vehicles}
            incidents={incidents}
            selectedIncident={drawerIncident}
            selectedUnitId={highlightedUnit}
            focusKey={focusKey}
            onVehicleClick={onVehicleClick}
          />
          {cardVehicle ? (
            <VehicleCard
              vehicle={cardVehicle}
              route={cardVehicle.route_id ? routeById.get(cardVehicle.route_id) : undefined}
              onClose={() => setSelectedUnit(null)}
            />
          ) : null}
          {status.connection === 'lost' || status.staleForS !== null ? (
            <div className="map-banner" role="status">
              <span className={`pill ${status.connection === 'lost' ? 'pill--bad' : 'pill--warn'}`}>
                {status.connection === 'lost'
                  ? 'Нет связи с сервером — показано последнее известное положение ТС'
                  : `Данные устарели: поток не обновлялся ${status.staleForS} с`}
              </span>
            </div>
          ) : null}
        </section>
        <IncidentList
          incidents={incidents}
          routes={routeById}
          selectedId={drawerIncident?.incident_id ?? null}
          onSelect={selectIncident}
          unavailableText={
            status.connection === 'lost' && incidents.length === 0
              ? 'Нет связи с сервером — список проблемных ТС недоступен'
              : !ready
                ? 'Ожидание данных потока…'
                : null
          }
        />
      </div>
      <IncidentDrawer
        open={drawerIncident !== null}
        incident={drawerIncident}
        route={drawerIncident?.route_id ? routeById.get(drawerIncident.route_id) : undefined}
        onClose={closeDrawer}
        onFocus={() => {
          if (drawerIncident) setSelectedUnit(drawerIncident.unit_id)
          setFocusKey((k) => k + 1)
        }}
      />
    </div>
  )
}
