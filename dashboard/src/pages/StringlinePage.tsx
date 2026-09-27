/**
 * График движения маршрута («время × остановки»): плановые нитки (серые), фактические по ТС (цветные,
 * опоздание > 2 мин подсвечено), где ТС сейчас и прогноз от этой точки на 15 минут (пунктир + конус P10–P90),
 * сбивка ТС. Зум и панорама — колесо/перетаскивание.
 */
import { NodeCollapseOutlined, ReloadOutlined, WarningOutlined } from '@ant-design/icons'
import { Button, Empty, Segmented, Select, Spin, Tooltip } from 'antd'
import type { CustomSeriesRenderItem } from 'echarts'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useRoutes, useStringline } from '../api/queries'
import type { Risk, RouteOut, VehicleOut } from '../api/types'
import { axisLine, axisText, escapeHtml, timeAxisLabel, tooltipBase } from '../components/chartStyle'
import { EChart, type ChartClick, type ChartOption } from '../components/EChart'
import { RiskDot } from '../components/RiskDot'
import { formatClock, formatDelay, formatDelayShort, formatDuration, stopLabel, toMs } from '../domain/format'
import { sortIncidents } from '../domain/incidents'
import { nearestStopSeq, routeShortName } from '../domain/routes'
import {
  buildStringline,
  bunchingPairs,
  positionBetween,
  shortStopName,
  splitSegments,
  stopAxis,
  upcomingStops,
  visibleExtent,
  type BunchingPoint,
  type StopAxis,
  type StringlineModel,
} from '../domain/stringline'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { useLive } from '../state/liveStore'
import { useAutoRoute } from '../state/useAutoRoute'
import { palette, threadColors } from '../theme'

const HISTORY_OPTIONS = [
  { value: 30, label: '30 мин' },
  { value: 60, label: '1 ч' },
  { value: 120, label: '2 ч' },
]
const AHEAD_MIN = 20
/** Правая граница графика: горизонт прогноза 15 мин и немного воздуха. */
const VIEW_AHEAD_MIN = 17
const BUNCHING_COLOR = '#c026d3'
/** Подсказки легенды: наведите на пункт — что он значит. */
const LEGEND_HINTS: Record<string, string> = {
  План: 'Где ТС должно быть по расписанию: плановое время прохода каждой остановки.',
  Факт: 'Где ТС было на самом деле: проходы остановок, восстановленные по GPS.',
  Опоздание: 'Участки, где ТС шло позже расписания больше чем на 2 минуты.',
  'Нет отметок': 'Промежуток без отметок о проходе остановок: ТС стояло на конечной или пропадал GPS.',
  'ТС сейчас': 'Где ТС находится сейчас; рядом — его текущее отклонение от расписания.',
  Прогноз: 'Когда ТС приедет на следующие остановки по прогнозу модели.',
  'P10–P90': 'Диапазон, в который фактическое время прибытия попадёт с вероятностью 80 %.',
  Сбивка: 'Место, где два ТС маршрута идут вплотную друг к другу — за ними растёт разрыв.',
}
const LATE_GLOW = 'rgba(239,68,68,0.35)'
const PLAN_COLOR = '#94a3b8'
/** Цвет первого ТС маршрута — им же нарисованы значки легенды. */
const FACT_COLOR = threadColors[0]
const REPLACE_SERIES = ['series']
const GRID = { left: 214, right: 30, top: 40, bottom: 34 }
/** Минимум пикселей по вертикали на подпись остановки. */
const LABEL_PX = 21

/** `#rrggbb` → `rgba(r,g,b,a)`. */
function withAlpha(hex: string, alpha: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex)
  if (!m) return hex
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

function delayColor(delayS: number | null | undefined): string {
  if (delayS === undefined || delayS === null) return palette.muted
  if (delayS > 120) return RISK_COLOR.red
  if (delayS >= 60) return RISK_COLOR.yellow
  return RISK_COLOR.green
}

/** Цвет нитки ТС по порядку на маршруте (стабилен между обновлениями). */
function colorMap(model: StringlineModel | null, trIds: number[]): Map<number, string> {
  const ids = new Set<number>(trIds)
  model?.actual.forEach((t) => ids.add(t.trId))
  model?.planned.forEach((t) => ids.add(t.trId))
  const sorted = [...ids].sort((a, b) => a - b)
  return new Map(sorted.map((id, i) => [id, threadColors[i % threadColors.length]]))
}

function buildOption(
  model: StringlineModel,
  axis: StopAxis,
  colors: Map<number, string>,
  selectedTr: number | null,
  range: [number, number],
  bunching: BunchingPoint[],
  plotHeight: number,
): ChartOption {
  const { rows, rowOf } = axis
  const rowName = (row: number) => rows[Math.max(0, Math.min(rows.length - 1, Math.round(row)))]?.name ?? ''
  const toRow = (p: [number, number]): [number, number] => [p[0], rowOf(p[1])]
  // оси подгоняются под данные в окне: пустые часы и неиспользуемая часть кругового маршрута не тратят место
  const extent = visibleExtent(model, range)
  const last = Math.max(1, rows.length - 1)
  const xMin = extent.t ? Math.max(range[0], extent.t[0] - 3 * 60_000) : range[0]
  const xMax = model.now !== null ? Math.min(range[1], model.now + VIEW_AHEAD_MIN * 60_000) : range[1]
  const yMin = extent.seq ? Math.max(0, Math.floor(rowOf(extent.seq[0])) - 1) : 0
  const yMax = extent.seq ? Math.min(last, Math.ceil(rowOf(extent.seq[1])) + 1) : last
  const fit = Math.max(4, Math.floor(plotHeight / LABEL_PX))
  const labelStep = Math.max(1, Math.ceil((yMax - yMin + 1) / fit))
  const dim = (trId: number) => selectedTr !== null && selectedTr !== trId
  const series: NonNullable<ChartOption['series']> = []

  // плановые нитки
  for (const thread of model.planned) {
    splitSegments(thread.points).forEach((seg, i) => {
      series.push({
        id: `plan:${thread.trId}:${i}`,
        name: 'План',
        type: 'line',
        data: seg.map(toRow),
        symbol: 'circle',
        symbolSize: 4,
        silent: true,
        z: 1,
        lineStyle: {
          color: PLAN_COLOR,
          width: selectedTr === thread.trId ? 2.6 : 2,
          opacity: dim(thread.trId) ? 0.3 : 0.9,
        },
        itemStyle: { color: PLAN_COLOR, opacity: dim(thread.trId) ? 0.3 : 0.9 },
        emphasis: { disabled: true },
      })
    })
  }

  // полосы P10–P90 прогноза (конус от позиции ТС)
  const bands = model.bands
  const renderBand: CustomSeriesRenderItem = (params, api) => {
    const band = bands[params.dataIndex]
    if (!band) return null
    const color = colors.get(band.trId) ?? FACT_COLOR
    return {
      type: 'polygon',
      shape: { points: band.polygon.map((p) => api.coord(toRow(p))) },
      style: {
        fill: withAlpha(color, dim(band.trId) ? 0.05 : 0.2),
        stroke: withAlpha(color, dim(band.trId) ? 0.1 : 0.55),
        lineWidth: 1,
        lineDash: [3, 3],
      },
      silent: true,
    }
  }
  series.push({
    id: 'bands',
    name: 'P10–P90',
    type: 'custom',
    renderItem: renderBand,
    data: bands.map((b) => toRow(b.polygon[0])),
    clip: true,
    silent: true,
    z: 2,
    itemStyle: { color: 'rgba(100,116,139,0.35)' },
    tooltip: { show: false },
  })

  // подсветка опоздания > 2 мин под фактической ниткой
  model.lateRuns.forEach((run, i) => {
    const pts = run.points.filter((p): p is [number, number] => p !== null).map(toRow)
    series.push({
      id: `late:${run.trId}:${i}`,
      name: 'Опоздание',
      type: 'line',
      data: pts,
      symbol: pts.length === 1 ? 'circle' : 'none',
      symbolSize: 10,
      silent: true,
      z: 3,
      lineStyle: {
        color: LATE_GLOW,
        width: 11,
        cap: 'round',
        join: 'round',
        opacity: dim(run.trId) ? 0.3 : 1,
      },
      itemStyle: { color: LATE_GLOW },
      emphasis: { disabled: true },
    })
  })

  // фактические нитки
  for (const thread of model.actual) {
    const color = colors.get(thread.trId) ?? FACT_COLOR
    const selected = selectedTr === thread.trId
    splitSegments(thread.points).forEach((seg, i) => {
      series.push({
        id: `fact:${thread.trId}:${i}`,
        name: 'Факт',
        type: 'line',
        data: seg.map(toRow),
        symbol: 'circle',
        symbolSize: selected ? 7 : 5,
        z: selected ? 6 : 4,
        lineStyle: { color, width: selected ? 4 : 3, opacity: dim(thread.trId) ? 0.22 : 1 },
        itemStyle: { color, opacity: dim(thread.trId) ? 0.22 : 1 },
        emphasis: { focus: 'series', lineStyle: { width: 3.4 } },
      })
    })
  }

  // стоянка или пропуск отметок: пунктирная связка от последнего прохода до позиции ТС
  model.links.forEach((link, i) => {
    const color = colors.get(link.trId) ?? FACT_COLOR
    series.push({
      id: `gap:${link.trId}:${i}`,
      name: 'Нет отметок',
      type: 'line',
      data: link.points.filter((p): p is [number, number] => p !== null).map(toRow),
      symbol: 'none',
      z: 4,
      lineStyle: { color, width: 1.5, type: [2, 4], opacity: dim(link.trId) ? 0.15 : 0.7 },
      emphasis: { disabled: true },
    })
  })

  // прогноз: пунктир от позиции ТС
  for (const thread of model.forecast) {
    const color = colors.get(thread.trId) ?? FACT_COLOR
    splitSegments(thread.points).forEach((seg, i) => {
      series.push({
        id: `fc:${thread.trId}:${i}`,
        name: 'Прогноз',
        type: 'line',
        data: seg.map(toRow),
        symbol: 'emptyCircle',
        symbolSize: 7,
        z: 5,
        lineStyle: { color, width: 2.6, type: [8, 5], opacity: dim(thread.trId) ? 0.2 : 1 },
        itemStyle: { color, opacity: dim(thread.trId) ? 0.2 : 1 },
      })
    })
  }

  // где ТС сейчас — с текущим отклонением рядом
  series.push({
    id: 'pos',
    name: 'ТС сейчас',
    type: 'scatter',
    symbol: 'circle',
    symbolSize: 14,
    z: 9,
    data: model.positions.map((p) => ({
      value: [p.t, rowOf(p.seq), p.delayS ?? Number.NaN, p.trId],
      itemStyle: {
        color: colors.get(p.trId) ?? FACT_COLOR,
        opacity: dim(p.trId) ? 0.3 : 1,
        borderColor: '#ffffff',
        borderWidth: 2.5,
        shadowBlur: 12,
        shadowColor: colors.get(p.trId) ?? FACT_COLOR,
      },
      label: {
        show: p.delayS !== null && !dim(p.trId),
        position: 'left' as const,
        distance: 10,
        formatter: formatDelayShort(p.delayS),
        color: delayColor(p.delayS),
        fontSize: 13,
        fontWeight: 700,
        backgroundColor: 'rgba(255,255,255,0.94)',
        borderColor: palette.border,
        borderWidth: 1,
        borderRadius: 5,
        padding: [3, 6],
      },
    })),
  })
  series.push({
    id: 'bunching',
    name: 'Сбивка',
    type: 'scatter',
    symbol: 'diamond',
    symbolSize: 16,
    z: 8,
    data: bunching.map((b) => [b.t, rowOf(b.seq), b.gapS, b.trA, b.trB, b.predicted ? 1 : 0]),
    itemStyle: {
      color: BUNCHING_COLOR,
      borderColor: '#ffffff',
      borderWidth: 1.5,
      shadowBlur: 10,
      shadowColor: BUNCHING_COLOR,
    },
  })

  // «сейчас» и зона прогноза
  if (model.now !== null) {
    series.push({
      id: 'now',
      name: 'сейчас',
      type: 'line',
      data: [[model.now, yMin]],
      symbol: 'none',
      silent: true,
      tooltip: { show: false },
      lineStyle: { opacity: 0 },
      markLine: {
        silent: true,
        symbol: 'none',
        label: { show: false },
        lineStyle: { color: palette.accentStrong, width: 1.5, type: 'solid', opacity: 0.8 },
        data: [{ xAxis: model.now }],
      },
      markArea: {
        silent: true,
        itemStyle: { color: 'rgba(14,165,233,0.05)' },
        label: {
          color: palette.accentStrong,
          fontSize: 13,
          position: 'insideTopLeft',
          distance: 8,
          formatter: `сейчас ${formatClock(model.now).slice(0, 5)} → прогноз 15 мин`,
        },
        data: [[{ xAxis: model.now }, { xAxis: model.now + 15 * 60_000 }]],
      },
    })
  }

  const tooltipFormatter = (raw: unknown): string => {
    const p = raw as { seriesId?: string; value?: unknown }
    const id = p.seriesId ?? ''
    const v = Array.isArray(p.value) ? (p.value as number[]) : []
    const [t, row] = v
    const where = `${escapeHtml(stopLabel(rowName(row)))} · ${formatClock(t)}`
    if (id === 'pos')
      return `<b>ТС ${v[3]}</b> сейчас · к ${escapeHtml(stopLabel(rowName(Math.ceil(row))))}${
        Number.isFinite(v[2]) ? `<br/>отклонение от графика <b>${escapeHtml(formatDelay(v[2]))}</b>` : ''
      }`
    if (id === 'bunching')
      return `<b style="color:${BUNCHING_COLOR}">Сбивка</b>: ТС ${v[3]} и ТС ${v[4]}<br/>интервал ${escapeHtml(formatDuration(v[2]))} · ${
        v[5] ? 'по прогнозу' : 'по факту'
      }<br/>${where}`
    const [kind, tr] = id.split(':')
    if (kind === 'fact') {
      const delay = model.factDelay.get(`${tr}:${t}`)
      return `<b>ТС ${escapeHtml(tr ?? '')}</b> · проход остановки<br/>${where}${
        delay === undefined
          ? ''
          : `<br/>отклонение <b style="color:${delayColor(delay)}">${escapeHtml(formatDelay(delay))}</b>`
      }`
    }
    if (kind === 'gap')
      return `<b>ТС ${escapeHtml(tr ?? '')}</b><br/>нет отметок о проходе остановок: стоянка или пропуск GPS`
    const label = kind === 'fc' ? 'прогноз прибытия' : 'план'
    return `<b>ТС ${escapeHtml(tr ?? '')}</b> · ${label}<br/>${where}`
  }

  return {
    animation: false,
    grid: GRID,
    legend: {
      // пустой маршрут: легенда из пары пунктов без линий только путает
      show: model.planned.length + model.actual.length + model.forecast.length > 0,
      // в одну строку: перенос на вторую наезжал на график; короткие имена влезают с 1280 px, уже — листается
      type: 'scroll',
      top: 6,
      left: 16,
      right: GRID.right,
      pageIconColor: palette.muted,
      pageTextStyle: { color: palette.muted },
      textStyle: { color: palette.muted, fontSize: 14 },
      itemWidth: 18,
      itemHeight: 8,
      itemGap: 14,
      tooltip: {
        show: true,
        ...tooltipBase,
        formatter: (p: unknown) => {
          const name = (p as { name?: string }).name ?? ''
          return `<b>${escapeHtml(name)}</b><br/>${escapeHtml(LEGEND_HINTS[name] ?? '')}`
        },
      },
      data: [
        // значки легенды — те же цвета и штрихи, что на графике
        { name: 'План', itemStyle: { color: PLAN_COLOR }, lineStyle: { color: PLAN_COLOR } },
        { name: 'Факт', itemStyle: { color: FACT_COLOR }, lineStyle: { color: FACT_COLOR, width: 3 } },
        { name: 'Опоздание', icon: 'roundRect', itemStyle: { color: 'rgba(239,68,68,0.55)' } },
        {
          name: 'Нет отметок',
          icon: 'path://M0,4 L3,4 L3,6 L0,6 Z M6,4 L9,4 L9,6 L6,6 Z M12,4 L15,4 L15,6 L12,6 Z',
          itemStyle: { color: FACT_COLOR },
        },
        { name: 'ТС сейчас', itemStyle: { color: FACT_COLOR, borderColor: '#ffffff', borderWidth: 2 } },
        {
          name: 'Прогноз',
          itemStyle: { color: FACT_COLOR },
          lineStyle: { color: FACT_COLOR, type: 'dashed' },
        },
        { name: 'P10–P90', icon: 'roundRect', itemStyle: { color: 'rgba(13,148,136,0.3)' } },
        // пункт легенды без точек на графике только путает
        ...(bunching.length ? [{ name: 'Сбивка', itemStyle: { color: BUNCHING_COLOR } }] : []),
      ],
    },
    tooltip: { ...tooltipBase, trigger: 'item', confine: true, formatter: tooltipFormatter },
    xAxis: {
      type: 'time',
      min: xMin,
      max: xMax,
      axisLabel: timeAxisLabel,
      axisLine,
      splitLine: { show: true, lineStyle: { color: palette.borderSoft } },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      min: yMin,
      max: yMax,
      interval: labelStep,
      axisLabel: {
        ...axisText,
        // последняя подпись вне шага наезжает на соседнюю
        showMaxLabel: (yMax - yMin) % labelStep === 0,
        formatter: (value: number) =>
          Number.isInteger(value) && rows[value]
            ? `${truncate(shortStopName(rows[value].name), 24)} {n|${value + 1}}`
            : '',
        rich: { n: { color: palette.faint, fontSize: 11, width: 20, align: 'right' } },
      },
      axisLine,
      axisTick: { show: false },
      splitLine: { lineStyle: { color: palette.border } },
      splitArea: { show: true, areaStyle: { color: ['rgba(100,116,139,0.04)', 'rgba(0,0,0,0)'] } },
      minorTick: { show: false, splitNumber: labelStep },
      minorSplitLine: { show: labelStep > 1, lineStyle: { color: palette.borderSoft } },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      {
        type: 'inside',
        yAxisIndex: 0,
        filterMode: 'none',
        zoomOnMouseWheel: 'shift',
        moveOnMouseWheel: false,
      },
    ],
    series,
  }
}

function VehicleCard({
  trId,
  color,
  delayS,
  nextStop,
  vehicle,
  selected,
  onClick,
}: {
  trId: number
  color: string
  delayS: number | undefined
  nextStop: string | null
  vehicle: VehicleOut | undefined
  selected: boolean
  onClick: () => void
}) {
  const risk: Risk = vehicle?.risk ?? 'unknown'
  const pred = vehicle?.pred_delay_s ?? null
  return (
    <button
      type="button"
      className={`sl-veh ${selected ? 'sl-veh--selected' : ''}`}
      onClick={onClick}
      aria-pressed={selected}
    >
      <span className="sl-veh__head">
        <span className="sl-veh__swatch" style={{ background: color }} />
        <span className="sl-veh__name">ТС {trId}</span>
      </span>
      <span className="sl-veh__stop" title={nextStop ?? undefined}>
        {nextStop ? `к ${stopLabel(shortStopName(nextStop))}` : 'позиция неизвестна'}
      </span>
      <span className="sl-veh__row">
        <span>отклонение сейчас</span>
        <b style={{ color: delayColor(delayS) }}>{delayS === undefined ? '—' : formatDelayShort(delayS)}</b>
      </span>
      <span className="sl-veh__row">
        <span>
          <RiskDot risk={risk} size={8} /> через 10–15 мин
        </span>
        <b style={{ color: RISK_TEXT_COLOR[risk] }}>{pred === null ? '—' : formatDelayShort(pred)}</b>
      </span>
    </button>
  )
}

/** Высота области графика: сколько подписей остановок поместится. */
function usePlotHeight(): [(el: HTMLElement | null) => void, number] {
  const [height, setHeight] = useState(600)
  const [el, setEl] = useState<HTMLElement | null>(null)
  useEffect(() => {
    if (!el) return undefined
    const update = () => setHeight(Math.max(120, el.clientHeight - GRID.top - GRID.bottom - 8))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [el])
  return [setEl, height]
}

export default function StringlinePage() {
  const [params, setParams] = useSearchParams()
  const routesQuery = useRoutes()
  const incidentsMap = useLive((s) => s.incidents)
  const liveVehicles = useLive((s) => s.vehicles)
  const streamTime = useLive((s) => s.streamTime)
  const [historyMin, setHistoryMin] = useState(30)
  const [zoomKey, setZoomKey] = useState(0)

  const routes = useMemo(() => routesQuery.data ?? [], [routesQuery.data])
  const incidents = useMemo(() => sortIncidents(incidentsMap.values()), [incidentsMap])
  const routeParam = params.get('route')
  const trRaw = Number(params.get('tr'))
  const selectedTr = Number.isFinite(trRaw) && trRaw > 0 ? trRaw : null

  const updateParams = useCallback(
    (patch: Record<string, string | null>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [k, v] of Object.entries(patch)) {
            if (v === null) next.delete(k)
            else next.set(k, v)
          }
          return next
        },
        { replace: true },
      )
    },
    [setParams],
  )

  // по умолчанию — маршрут со сбивкой (её видно на нитке), иначе с наибольшим числом проблем
  const defaultRoute = useAutoRoute(routes)
  const routeId = routeParam && routes.some((r) => r.route_id === routeParam) ? routeParam : null

  useEffect(() => {
    if (!routeId && defaultRoute) updateParams({ route: defaultRoute })
  }, [routeId, defaultRoute, updateParams])

  const route: RouteOut | undefined = routes.find((r) => r.route_id === routeId)
  const nowMs = toMs(streamTime)
  const anchor = nowMs === null ? null : Math.floor(nowMs / 60_000) * 60_000
  const from = anchor === null ? null : new Date(anchor - historyMin * 60_000).toISOString()
  const to = anchor === null ? null : new Date(anchor + AHEAD_MIN * 60_000).toISOString()
  const query = useStringline(routeId, from, to)

  const data = query.data?.route_id === routeId ? query.data : undefined
  const vehicleByTr = useMemo(() => {
    const map = new Map<number, VehicleOut>()
    for (const v of liveVehicles.values()) if (v.tr_id !== null) map.set(v.tr_id, v)
    return map
  }, [liveVehicles])
  // где ТС между остановками — по GPS (с шагом 0,1 остановки, чтобы график не перерисовывался каждую секунду)
  const gpsKey = useMemo(() => {
    if (!data || !route) return ''
    const parts: string[] = []
    for (const trip of data.trips) {
      if (!trip.now) continue
      const v = vehicleByTr.get(trip.tr_id)
      const seq = positionBetween(Math.round(trip.now.seq + 0.5), v?.lat, v?.lon, route.stops)
      if (seq !== null) parts.push(`${trip.tr_id}:${seq}`)
    }
    return parts.join(',')
  }, [data, route, vehicleByTr])
  const model = useMemo(() => {
    if (!data) return null
    const gps = new Map(
      gpsKey ? gpsKey.split(',').map((p) => p.split(':').map(Number) as [number, number]) : [],
    )
    const trips = data.trips.map((trip) => {
      const seq = gps.get(trip.tr_id)
      return seq === undefined || !trip.now ? trip : { ...trip, now: { ...trip.now, seq } }
    })
    return buildStringline({ ...data, trips })
  }, [data, gpsKey])
  const axis = useMemo(() => (model ? stopAxis(model.stops) : null), [model])
  const [plotRef, plotHeight] = usePlotHeight()
  const colors = useMemo(() => colorMap(model, route?.tr_ids ?? []), [model, route])
  const range = useMemo<[number, number] | null>(
    () => (anchor === null ? null : [anchor - historyMin * 60_000, anchor + AHEAD_MIN * 60_000]),
    [anchor, historyMin],
  )
  const routeIncidents = useMemo(() => incidents.filter((i) => i.route_id === routeId), [incidents, routeId])
  // сбивка: найденная по ниткам + инциденты «сбивка» backend (он смотрит дальше горизонта графика)
  const allBunching = useMemo(() => {
    const now = model?.now ?? null
    const fromIncidents: BunchingPoint[] = []
    if (route && now !== null) {
      for (const inc of routeIncidents) {
        if (inc.kind !== 'bunching') continue
        const seq = nearestStopSeq(route.stops, inc.target_stop.lat, inc.target_stop.lon)
        if (seq === null) continue
        const gap = inc.cause.factors.find((f) => f.feature === 'headway_s')?.contribution_s ?? 0
        fromIncidents.push({
          trA: inc.related_tr_id ?? inc.tr_id,
          trB: inc.tr_id,
          t: now + inc.time_to_event_s * 1000,
          seq,
          gapS: gap,
          predicted: true,
        })
      }
    }
    return [...(model?.bunching ?? []), ...fromIncidents]
  }, [model, route, routeIncidents])
  const option = useMemo(
    () =>
      model && axis && range
        ? buildOption(model, axis, colors, selectedTr, range, allBunching, plotHeight)
        : null,
    [model, axis, colors, selectedTr, range, allBunching, plotHeight],
  )

  const pairs = useMemo(() => bunchingPairs(allBunching), [allBunching])
  const nextStopByTr = useMemo(() => {
    const map = new Map<number, string>()
    if (!model || !axis) return map
    for (const p of model.positions) {
      const live = vehicleByTr.get(p.trId)?.next_stop?.name
      const row = axis.rows[Math.ceil(axis.rowOf(p.seq))]
      if (live || row) map.set(p.trId, live ?? row.name)
    }
    return map
  }, [model, axis, vehicleByTr])
  const vehicles = useMemo(
    () => [...new Set((data?.trips ?? []).map((t) => t.tr_id))].sort((a, b) => a - b),
    [data],
  )
  const lateNow = model ? [...model.lastDelay.values()].filter((d) => d > 120).length : 0
  // таблица ближайших остановок — для выбранного ТС, иначе для первого ТС с прогнозом
  const focusTr = selectedTr ?? model?.forecast.find((t) => t.points.length)?.trId ?? vehicles[0] ?? null
  const upcoming = useMemo(
    () => (data && axis && focusTr !== null ? upcomingStops(data, focusTr, axis, 6) : []),
    [data, axis, focusTr],
  )
  // в окне ни плана, ни ТС: смена маршрута закончилась или ещё не началась
  const idle =
    model !== null &&
    !model.planned.length &&
    !model.actual.length &&
    !model.forecast.length &&
    !model.positions.length

  const toggleTr = useCallback(
    (trId: number) => updateParams({ tr: selectedTr === trId ? null : String(trId) }),
    [selectedTr, updateParams],
  )
  const onChartClick = useCallback(
    (p: ChartClick) => {
      const [kind, tr] = (p.seriesId ?? '').split(':')
      if ((kind === 'fact' || kind === 'fc' || kind === 'plan' || kind === 'gap') && tr) toggleTr(Number(tr))
    },
    [toggleTr],
  )

  return (
    <div className="page page--fixed">
      <div className="panel toolbar sl-toolbar">
        <span className="toolbar__label">Маршрут</span>
        <Select
          value={routeId ?? undefined}
          loading={routesQuery.isLoading}
          onChange={(value: string) => updateParams({ route: value, tr: null })}
          style={{ width: 'clamp(240px, 26vw, 420px)' }}
          showSearch={{ optionFilterProp: 'label' }}
          placeholder="Выберите маршрут"
          options={routes.map((r) => ({
            value: r.route_id,
            label: `${r.route_id} · ${routeShortName(r)}`,
          }))}
          optionRender={(opt) => {
            const r = routes.find((x) => x.route_id === opt.value)
            return (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <span className="route-swatch" style={{ background: r?.color }} />
                {opt.label}
              </span>
            )
          }}
        />
        <span className="toolbar__label">История</span>
        <Segmented
          size="middle"
          value={historyMin}
          onChange={(v) => setHistoryMin(Number(v))}
          options={HISTORY_OPTIONS}
        />
        <Tooltip title="Сбросить масштаб">
          <Button
            icon={<ReloadOutlined />}
            onClick={() => setZoomKey((k) => k + 1)}
            aria-label="Сбросить масштаб"
          />
        </Tooltip>
        <span style={{ flex: 1 }} />
        <Tooltip title="ТС маршрута в окне графика">
          <span className="chip">ТС: {vehicles.length}</span>
        </Tooltip>
        <Tooltip title="ТС, опаздывающие сейчас больше чем на 2 минуты">
          <span className={`chip ${lateNow ? 'chip--bad' : ''}`}>
            <WarningOutlined /> опаздывают: {lateNow}
          </span>
        </Tooltip>
        <Tooltip title="Пары ТС, идущих вплотную (по факту и по прогнозу)">
          <span className={`chip ${pairs.length ? 'chip--bunch' : ''}`}>
            <NodeCollapseOutlined /> сбивок: {pairs.length}
          </span>
        </Tooltip>
      </div>

      <div className="sl-body">
        <section className="panel sl-chart" aria-label="График движения" ref={plotRef}>
          {idle ? (
            <div className="sl-empty">
              <div className="sl-empty__card">
                <b>Сейчас на маршруте {routeId} нет рейсов</b>
                <span>
                  В окне графика нет ни плановых рейсов, ни ТС на линии: смена закончилась или ещё не
                  началась.
                </span>
                {defaultRoute && defaultRoute !== routeId ? (
                  <Button type="primary" onClick={() => updateParams({ route: defaultRoute, tr: null })}>
                    Открыть {defaultRoute} — маршрут, где сейчас есть проблемы
                  </Button>
                ) : null}
              </div>
            </div>
          ) : null}
          {option && model ? (
            model.stops.length ? (
              <EChart
                key={`${routeId}-${zoomKey}`}
                option={option}
                height="100%"
                notMerge={false}
                replaceMerge={REPLACE_SERIES}
                onClick={onChartClick}
                ariaLabel="График движения: время по горизонтали, остановки маршрута по вертикали"
              />
            ) : (
              <Empty description="У маршрута нет остановок" style={{ marginTop: 120 }} />
            )
          ) : query.isError ? (
            <Empty description="График недоступен: сервер не отвечает" style={{ marginTop: 120 }} />
          ) : nowMs === null ? (
            <Empty description="Ожидание данных потока…" style={{ marginTop: 120 }} />
          ) : (
            <div style={{ display: 'grid', placeItems: 'center', height: '100%' }}>
              <Spin />
            </div>
          )}
        </section>

        <aside className="panel sl-side" aria-label="ТС маршрута">
          <div className="panel__head">
            <span className="panel__title">ТС на маршруте</span>
          </div>
          <div className="sl-side__list">
            {vehicles.length === 0 ? (
              <div className="muted-note">Нет ТС в окне графика</div>
            ) : (
              vehicles.map((trId) => (
                <VehicleCard
                  key={trId}
                  trId={trId}
                  color={colors.get(trId) ?? FACT_COLOR}
                  delayS={model?.lastDelay.get(trId)}
                  nextStop={nextStopByTr.get(trId) ?? null}
                  vehicle={vehicleByTr.get(trId)}
                  selected={selectedTr === trId}
                  onClick={() => toggleTr(trId)}
                />
              ))
            )}
          </div>
          {upcoming.length ? (
            <div className="sl-side__section sl-next">
              <div className="card-section__title">Ближайшие остановки · ТС {focusTr}</div>
              <div className="sl-next__head">
                <span>план</span>
                <span>прогноз</span>
                <span>откл.</span>
              </div>
              {upcoming.map((u) => (
                <div key={`${u.seq}-${u.forecast}`} className="sl-next__row">
                  <span className="sl-next__name" title={u.name}>
                    {shortStopName(u.name)}
                  </span>
                  <span className="sl-next__times">
                    <span>{u.plan === null ? '—' : formatClock(u.plan).slice(0, 5)}</span>
                    <b>{formatClock(u.forecast).slice(0, 5)}</b>
                    <b style={{ color: delayColor(u.delayS) }}>{formatDelayShort(u.delayS)}</b>
                  </span>
                </div>
              ))}
            </div>
          ) : null}
          {pairs.length ? (
            <div className="sl-side__section">
              <div className="card-section__title">Сбивка</div>
              {pairs.slice(0, 4).map(([a, b]) => {
                const point = allBunching.find(
                  (p) => (p.trA === a && p.trB === b) || (p.trA === b && p.trB === a),
                )
                return (
                  <div key={`${a}-${b}`} className="sl-pair">
                    <NodeCollapseOutlined style={{ color: BUNCHING_COLOR }} />
                    <span>
                      ТС {a} и ТС {b}
                      {point ? ` · ${formatDuration(point.gapS)}${point.predicted ? ', прогноз' : ''}` : ''}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : null}
          <div className="sl-side__section sl-help">
            <div className="card-section__title">Как читать</div>
            <p>
              <span className="sl-help__line" style={{ background: PLAN_COLOR }} /> план,{' '}
              <span className="sl-help__line" style={{ background: FACT_COLOR }} /> факт ТС. Правее плана —
              опаздывает, пологая — идёт медленно.
            </p>
            <p>
              <span className="sl-help__line sl-help__line--glow" /> подсветка — опоздание больше 2 минут.
            </p>
            <p>
              <span className="sl-help__dot" /> ТС сейчас; от него пунктир — прогноз, конус — интервал
              P10–P90.
            </p>
            <p>Точечная линия — нет отметок о проходе остановок: ТС стояло или пропал GPS.</p>
            <p>
              Линии сходятся — <b style={{ color: BUNCHING_COLOR }}>сбивка</b>: ТС идут вплотную, за ними
              растёт разрыв.
            </p>
          </div>
        </aside>
      </div>
    </div>
  )
}
