/**
 * Карта маршрутной сети (MapLibre GL): линии маршрутов, ТС с цветом риска и стрелкой курса, пульсация у красных,
 * подсветка участков инцидентов. Данные обновляются через `setData` GeoJSON-источников (слои не пересоздаются),
 * поэтому сотни ТС обновляются без лагов. Без интернета карта работает без подложки.
 */
import { AimOutlined } from '@ant-design/icons'
import { Button, Tooltip } from 'antd'
import type { Feature, FeatureCollection, LineString, Point } from 'geojson'
import {
  type GeoJSONSource,
  LngLatBounds,
  Map as MapLibreMap,
  Marker,
  NavigationControl,
  Popup,
  setWorkerUrl,
  type StyleSpecification,
} from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useRef, useState } from 'react'
import type { IncidentOut, Risk, RouteOut, VehicleOut } from '../api/types'
import { config } from '../config'
import { formatDelayShort, vehicleLabel } from '../domain/format'
import { RISK_COLOR, RISK_LABEL, RISK_TEXT_COLOR, vehicleRisk } from '../domain/risk'
import { addBaseStyle, addRasterBase, BASE, fetchBaseStyle } from './basemap'
import { escapeHtml } from './chartStyle'

setWorkerUrl(workerUrl)

const RISKS: Risk[] = ['green', 'yellow', 'red', 'unknown']
const MOSCOW: [number, number] = [37.62, 55.75]

interface Props {
  routes: RouteOut[]
  vehicles: VehicleOut[]
  incidents: IncidentOut[]
  selectedIncident: IncidentOut | null
  selectedUnitId: number | null
  /** Меняется, когда нужно отцентрировать карту на выбранном ТС. */
  focusKey: number
  onVehicleClick: (unitId: number) => void
}

function buildStyle(): StyleSpecification {
  // только фон: подложка (векторная или растровая, см. ./basemap) добавляется под слои приложения после загрузки
  return {
    version: 8,
    sources: {},
    layers: [{ id: 'background', type: 'background', paint: { 'background-color': BASE.background } }],
  }
}

/** Иконки ТС: круг цвета риска со стрелкой курса (или без стрелки, если курс неизвестен). */
function vehicleIcon(color: string, withArrow: boolean): ImageData | null {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.lineJoin = 'round'
  if (withArrow) {
    ctx.beginPath()
    ctx.moveTo(32, 3)
    ctx.lineTo(43, 21)
    ctx.lineTo(21, 21)
    ctx.closePath()
    ctx.fillStyle = color
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 4
    ctx.stroke()
    ctx.fill()
  }
  ctx.beginPath()
  ctx.arc(32, 34, 15, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 5
  ctx.stroke()
  ctx.fill()
  ctx.beginPath()
  ctx.arc(32, 34, 5, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.75)'
  ctx.fill()
  return ctx.getImageData(0, 0, size, size)
}

/** Линии маршрутов: каждое направление отдельно — остановки обратного направления тоже на линии. */
function routesGeoJson(routes: RouteOut[]): FeatureCollection<LineString> {
  return {
    type: 'FeatureCollection',
    features: routes.flatMap((r) =>
      (r.directions?.length ? r.directions.map((d) => d.line) : [r.line])
        .filter((line) => line.length >= 2)
        .map((line): Feature<LineString> => ({
          type: 'Feature',
          properties: { route_id: r.route_id, color: r.color, name: r.name },
          geometry: { type: 'LineString', coordinates: line },
        })),
    ),
  }
}

function stopsGeoJson(routes: RouteOut[]): FeatureCollection<Point> {
  const seen = new Set<string>()
  const features: Feature<Point>[] = []
  for (const r of routes) {
    for (const s of r.stops) {
      if (seen.has(s.stop_key)) continue
      seen.add(s.stop_key)
      features.push({
        type: 'Feature',
        properties: { name: s.name, color: r.color },
        geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
      })
    }
  }
  return { type: 'FeatureCollection', features }
}

function vehiclesGeoJson(vehicles: VehicleOut[]): FeatureCollection<Point> {
  const features: Feature<Point>[] = []
  for (const v of vehicles) {
    if (v.lat === null || v.lon === null) continue
    const risk = vehicleRisk(v)
    const hasCourse = v.course_deg !== null && (v.speed_kmh ?? 0) > 1
    features.push({
      type: 'Feature',
      id: v.unit_id,
      properties: {
        unit_id: v.unit_id,
        tr_id: v.tr_id,
        route_id: v.route_id ?? '',
        risk,
        icon: `veh-${risk}${hasCourse ? '' : '-dot'}`,
        course: v.course_deg ?? 0,
        faded: v.status !== 'online',
        delay: v.pred_delay_s ?? v.current_delay_s ?? null,
        order: risk === 'red' ? 3 : risk === 'yellow' ? 2 : risk === 'green' ? 1 : 0,
      },
      geometry: { type: 'Point', coordinates: [v.lon, v.lat] },
    })
  }
  return { type: 'FeatureCollection', features }
}

function segmentsGeoJson(
  incidents: IncidentOut[],
  selected: IncidentOut | null,
): FeatureCollection<LineString> {
  return {
    type: 'FeatureCollection',
    features: incidents
      .filter((i) => i.segment && i.segment.line.length >= 2)
      .map((i) => ({
        type: 'Feature',
        properties: { risk: i.risk, selected: selected?.incident_id === i.incident_id },
        geometry: { type: 'LineString', coordinates: i.segment?.line ?? [] },
      })),
  }
}

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }

function setSource(map: MapLibreMap, id: string, data: FeatureCollection): void {
  const source = map.getSource(id) as GeoJSONSource | undefined
  void source?.setData(data)
}

export function VehicleMap(props: Props) {
  const { routes, vehicles, incidents, selectedIncident, selectedUnitId, focusKey, onVehicleClick } = props
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState<string | null>(null)
  const clickRef = useRef(onVehicleClick)
  const vehiclesRef = useRef(vehicles)
  const fitted = useRef(false)
  const pulses = useRef(new Map<number, Marker>())

  useEffect(() => {
    clickRef.current = onVehicleClick
    vehiclesRef.current = vehicles
  })

  // --- инициализация карты (один раз)
  useEffect(() => {
    const el = container.current
    if (!el) return undefined
    let map: MapLibreMap
    try {
      map = new MapLibreMap({
        container: el,
        style: buildStyle(),
        center: MOSCOW,
        zoom: 10,
        minZoom: 8,
        maxZoom: 18,
        attributionControl: { compact: false },
        dragRotate: false,
        pitchWithRotate: false,
        fadeDuration: 0,
      })
    } catch (err) {
      queueMicrotask(() => setFailed(err instanceof Error ? err.message : String(err)))
      return undefined
    }
    mapRef.current = map
    map.touchZoomRotate.disableRotation()
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right')
    map.on('error', () => {
      // недоступные тайлы подложки не мешают работе карты (офлайн)
    })
    const popup = new Popup({ closeButton: false, closeOnClick: false, offset: 14, maxWidth: '260px' })

    map.on('load', () => {
      for (const risk of RISKS) {
        const arrow = vehicleIcon(RISK_COLOR[risk], true)
        const dot = vehicleIcon(RISK_COLOR[risk], false)
        if (arrow) map.addImage(`veh-${risk}`, arrow, { pixelRatio: 2 })
        if (dot) map.addImage(`veh-${risk}-dot`, dot, { pixelRatio: 2 })
      }
      map.addSource('routes', { type: 'geojson', data: EMPTY })
      map.addSource('stops', { type: 'geojson', data: EMPTY })
      map.addSource('segments', { type: 'geojson', data: EMPTY })
      map.addSource('vehicles', { type: 'geojson', data: EMPTY })
      map.addSource('target', { type: 'geojson', data: EMPTY })

      map.addLayer({
        id: 'routes-casing',
        type: 'line',
        source: 'routes',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': '#ffffff',
          'line-width': ['interpolate', ['linear'], ['zoom'], 9, 5.5, 13, 8, 16, 11],
          'line-opacity': 0.9,
        },
      })
      map.addLayer({
        id: 'routes',
        type: 'line',
        source: 'routes',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': ['interpolate', ['linear'], ['zoom'], 9, 3, 13, 4.5, 16, 6.5],
          'line-opacity': 0.9,
        },
      })
      map.addLayer({
        id: 'stops',
        type: 'circle',
        source: 'stops',
        minzoom: 12.5,
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 12.5, 2, 16, 4.5],
          'circle-color': '#ffffff',
          'circle-stroke-color': ['get', 'color'],
          'circle-stroke-width': 1.5,
        },
      })
      map.addLayer({
        id: 'segments-glow',
        type: 'line',
        source: 'segments',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['match', ['get', 'risk'], 'red', RISK_COLOR.red, RISK_COLOR.yellow],
          'line-width': ['case', ['get', 'selected'], 16, 10],
          'line-opacity': ['case', ['get', 'selected'], 0.35, 0.18],
          'line-blur': 6,
        },
      })
      map.addLayer({
        id: 'segments',
        type: 'line',
        source: 'segments',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': ['match', ['get', 'risk'], 'red', RISK_COLOR.red, RISK_COLOR.yellow],
          'line-width': ['case', ['get', 'selected'], 5, 3],
          'line-opacity': ['case', ['get', 'selected'], 1, 0.75],
        },
      })
      map.addLayer({
        id: 'target',
        type: 'circle',
        source: 'target',
        paint: {
          'circle-radius': 9,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#0f1b2d',
          'circle-stroke-width': 2.5,
        },
      })
      map.addLayer({
        id: 'vehicle-selected',
        type: 'circle',
        source: 'vehicles',
        filter: ['==', ['get', 'unit_id'], -1],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 9, 11, 14, 17],
          'circle-color': 'rgba(14,165,233,0.16)',
          'circle-stroke-color': '#0284c7',
          'circle-stroke-width': 2,
        },
      })
      map.addLayer({
        id: 'vehicles',
        type: 'symbol',
        source: 'vehicles',
        layout: {
          'icon-image': ['get', 'icon'],
          'icon-size': ['interpolate', ['linear'], ['zoom'], 8, 0.85, 11, 1.05, 14, 1.25, 17, 1.4],
          'icon-rotate': ['get', 'course'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'symbol-sort-key': ['get', 'order'],
        },
        paint: {
          'icon-opacity': ['case', ['get', 'faded'], 0.45, 1],
        },
      })

      map.on('click', 'vehicles', (e) => {
        const f = e.features?.[0]
        const unit = f?.properties?.unit_id
        if (unit !== undefined) clickRef.current(Number(unit))
      })
      map.on('mouseenter', 'vehicles', () => {
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mousemove', 'vehicles', (e) => {
        const f = e.features?.[0]
        if (!f || f.geometry.type !== 'Point') return
        const unit = Number(f.properties?.unit_id)
        const v = vehiclesRef.current.find((x) => x.unit_id === unit)
        if (!v) return
        const risk = vehicleRisk(v)
        const lines = [
          `<b>${escapeHtml(vehicleLabel(v.tr_id, v.unit_id))}</b>${v.route_id ? ` · ${escapeHtml(v.route_id)}` : ''}`,
          `<span style="color:${RISK_TEXT_COLOR[risk]}">● ${escapeHtml(RISK_LABEL[risk])}</span>`,
          v.pred_delay_s !== null && v.pred_delay_s !== undefined
            ? `Прогноз: ${escapeHtml(formatDelayShort(v.pred_delay_s))}`
            : '',
          v.current_delay_s !== null && v.current_delay_s !== undefined
            ? `Сейчас: ${escapeHtml(formatDelayShort(v.current_delay_s))}`
            : '',
          v.next_stop ? `След.: ${escapeHtml(v.next_stop.name)}` : '',
          v.speed_kmh !== null ? `${v.speed_kmh} км/ч` : '',
        ].filter(Boolean)
        popup
          .setLngLat(f.geometry.coordinates as [number, number])
          .setHTML(lines.join('<br/>'))
          .addTo(map)
      })
      map.on('mouseleave', 'vehicles', () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      })
      setLoaded(true)
      void fetchBaseStyle(config.basemapStyle).then((base) => {
        if (mapRef.current !== map) return // the map was removed meanwhile
        try {
          if (base) addBaseStyle(map, base, 'routes-casing')
          else addRasterBase(map, 'routes-casing')
        } catch {
          // подложка не обязательна: маршруты и ТС видны и на фоне
        }
      })
    })

    const pulseMarkers = pulses.current
    return () => {
      popup.remove()
      pulseMarkers.forEach((m) => m.remove())
      pulseMarkers.clear()
      map.remove()
      mapRef.current = null
      fitted.current = false
    }
  }, [])

  // --- маршруты
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    setSource(map, 'routes', routesGeoJson(routes))
    setSource(map, 'stops', stopsGeoJson(routes))
    if (!fitted.current && routes.length) {
      const bounds = new LngLatBounds()
      routes.forEach((r) => r.line.forEach((p) => bounds.extend(p)))
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, { padding: 48, duration: 0, maxZoom: 13 })
        fitted.current = true
      }
    }
  }, [loaded, routes])

  // --- ТС и пульсация красных
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    setSource(map, 'vehicles', vehiclesGeoJson(vehicles))
    const markers = pulses.current
    const alive = new Set<number>()
    for (const v of vehicles) {
      if (v.lat === null || v.lon === null || vehicleRisk(v) !== 'red' || v.status === 'offline') continue
      alive.add(v.unit_id)
      const existing = markers.get(v.unit_id)
      if (existing) existing.setLngLat([v.lon, v.lat])
      else {
        const el = document.createElement('div')
        el.className = 'pulse-ring'
        markers.set(v.unit_id, new Marker({ element: el }).setLngLat([v.lon, v.lat]).addTo(map))
      }
    }
    for (const [unit, marker] of markers) {
      if (!alive.has(unit)) {
        marker.remove()
        markers.delete(unit)
      }
    }
  }, [loaded, vehicles])

  // --- участки инцидентов и выбранное ТС
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    setSource(map, 'segments', segmentsGeoJson(incidents, selectedIncident))
    setSource(
      map,
      'target',
      selectedIncident
        ? {
            type: 'FeatureCollection',
            features: [
              {
                type: 'Feature',
                properties: {},
                geometry: {
                  type: 'Point',
                  coordinates: [selectedIncident.target_stop.lon, selectedIncident.target_stop.lat],
                },
              },
            ],
          }
        : EMPTY,
    )
    map.setFilter('vehicle-selected', ['==', ['get', 'unit_id'], selectedUnitId ?? -1])
  }, [loaded, incidents, selectedIncident, selectedUnitId])

  // --- центрирование на выбранном ТС
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map || selectedUnitId === null) return
    const v = vehiclesRef.current.find((x) => x.unit_id === selectedUnitId)
    const lon = v?.lon ?? selectedIncident?.vehicle.lon ?? null
    const lat = v?.lat ?? selectedIncident?.vehicle.lat ?? null
    if (lon === null || lat === null) return
    // инцидент: показать целиком ТС, проблемный участок и целевую остановку; иначе — приблизиться к ТС
    const inc = selectedIncident && selectedIncident.unit_id === selectedUnitId ? selectedIncident : null
    if (inc && inc.segment && inc.segment.line.length >= 2) {
      const bounds = new LngLatBounds([lon, lat], [lon, lat])
      inc.segment.line.forEach((p) => bounds.extend(p))
      bounds.extend([inc.target_stop.lon, inc.target_stop.lat])
      map.fitBounds(bounds, {
        padding: { top: 70, bottom: 70, left: 70, right: 110 },
        maxZoom: 15.5,
        duration: 700,
      })
      return
    }
    map.easeTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 13.5), duration: 700 })
    // только при смене выбора / явном запросе
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, selectedUnitId, focusKey])

  const fitAll = () => {
    const map = mapRef.current
    if (!map || !routes.length) return
    const bounds = new LngLatBounds()
    routes.forEach((r) => r.line.forEach((p) => bounds.extend(p)))
    map.fitBounds(bounds, { padding: 48, duration: 600, maxZoom: 13 })
  }

  return (
    <>
      <div ref={container} className="map-canvas" aria-label="Карта маршрутной сети" />
      {failed ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'grid',
            placeItems: 'center',
            color: 'var(--muted)',
          }}
        >
          Карта недоступна в этом браузере (нужен WebGL): {failed}
        </div>
      ) : null}
      <div className="map-toolbar">
        <Tooltip title="Показать всю сеть">
          <Button size="small" icon={<AimOutlined />} onClick={fitAll}>
            Вся сеть
          </Button>
        </Tooltip>
      </div>
      <div className="map-legend" aria-label="Легенда">
        {(['green', 'yellow', 'red', 'unknown'] as Risk[]).map((r) => (
          <div key={r} className="map-legend__row">
            <span className="dot" style={{ background: RISK_COLOR[r], width: 10, height: 10 }} />
            {r === 'green'
              ? 'Норма (< 1 мин)'
              : r === 'yellow'
                ? 'Риск опоздания'
                : r === 'red'
                  ? 'Высокий риск (> 2 мин)'
                  : 'Нет прогноза'}
          </div>
        ))}
        <div className="map-legend__row">
          <span
            className="map-legend__line"
            style={{ background: RISK_COLOR.red, boxShadow: '0 0 6px rgba(239, 68, 68, 0.6)' }}
          />
          Участок инцидента
        </div>
        <div className="map-legend__row">
          <span className="map-legend__line" style={{ background: '#4C8DF6' }} />
          Линии маршрутов
        </div>
      </div>
    </>
  )
}
