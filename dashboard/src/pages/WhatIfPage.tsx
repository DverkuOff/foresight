/**
 * What-if: «что будет, если…» — выпустить резервный автобус или задержать автобус на остановке. Итог одной
 * фразой, таблица «без действия / с действием» (ожидание, самый долгий перерыв, опоздания, сбивка) и график
 * «где будут автобусы» на выбранный период.
 */
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  BulbOutlined,
  CarOutlined,
  ExperimentOutlined,
  MinusOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  ConfigProvider,
  Empty,
  InputNumber,
  Segmented,
  Select,
  Slider,
  Spin,
  Tooltip,
} from 'antd'
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router'
import { apiPost } from '../api/client'
import { useRoutes, useWhatIf } from '../api/queries'
import type { RouteOut, Scenario, WhatIfOut, WhatIfRequest } from '../api/types'
import {
  axisLine,
  axisText,
  escapeHtml,
  legendBase,
  splitLine,
  timeAxisLabel,
  tooltipBase,
} from '../components/chartStyle'
import { EChart, type ChartOption } from '../components/EChart'
import { formatClock, formatHm, formatPercent, stopLabel, toMs } from '../domain/format'
import { sortIncidents } from '../domain/incidents'
import { routeShortName } from '../domain/routes'
import {
  candidateRequests,
  deltaTone,
  formatDeltaValue,
  formatMetric,
  deltaScore,
  relativeChange,
  verdictTone,
  WHATIF_METRICS,
  type VerdictTone,
} from '../domain/whatif'
import { RISK_TEXT_COLOR } from '../domain/risk'
import { useLive } from '../state/liveStore'
import { useAutoRoute } from '../state/useAutoRoute'
import { shortStopName } from '../domain/stringline'
import { palette } from '../theme'

type Action = WhatIfRequest['action']

interface FormState {
  action: Action
  fromStopKey: string | null
  departInMin: number
  holdTr: number | null
  holdMin: number
  horizonMin: number
}

const HORIZONS = [
  { value: 30, label: '30 мин' },
  { value: 60, label: '1 час' },
  { value: 90, label: '1,5 часа' },
]
const RESERVE_COLOR = '#db2777'
const AFTER_COLOR = palette.accent

const VERDICT: Record<VerdictTone, { title: string; type: 'success' | 'warning' | 'error' | 'info' }> = {
  better: { title: 'станет лучше', type: 'success' },
  mixed: { title: 'часть показателей лучше, часть хуже', type: 'warning' },
  worse: { title: 'станет хуже', type: 'error' },
  same: { title: 'заметного эффекта нет', type: 'info' },
}

const ACTIONS: { value: Action; title: string; text: string; icon: ReactNode }[] = [
  {
    value: 'add_vehicle',
    title: 'Выпустить резервный автобус',
    text: 'Ещё один автобус выходит на линию с выбранной остановки и закрывает большой перерыв между автобусами.',
    icon: <PlayCircleOutlined />,
  },
  {
    value: 'hold',
    title: 'Задержать автобус на остановке',
    text: 'Автобус стоит несколько минут, чтобы не догонять идущий впереди: так автобусы не идут вплотную.',
    icon: <PauseCircleOutlined />,
  },
]

function actionTitle(request: WhatIfRequest, route: RouteOut | undefined): string {
  if (request.action === 'hold') {
    return `Если задержать автобус ТС ${request.params.tr_id ?? '—'} на ${Math.round((request.params.hold_s ?? 0) / 60)} мин`
  }
  const stop = route?.stops.find((s) => s.stop_key === request.params.from_stop_key)
  return `Если выпустить резервный автобус с ${stopLabel(stop?.name)} в ${formatHm(request.params.depart_at ?? null)}`
}

/** Итог одной фразой: что изменится для пассажиров. */
function verdictText(result: WhatIfOut): string {
  const b = result.baseline
  const a = result.scenario
  const parts: string[] = []
  const changed = (key: (typeof WHATIF_METRICS)[number]['key']) =>
    deltaTone(key, result.delta[key]) !== 'same'
  if (changed('mean_wait_s')) {
    const rel = relativeChange(b.mean_wait_s, a.mean_wait_s)
    parts.push(
      `пассажиры будут ждать автобус в среднем ${formatMetric('mean_wait_s', a.mean_wait_s)} вместо ` +
        `${formatMetric('mean_wait_s', b.mean_wait_s)}${rel === null ? '' : ` (${rel > 0 ? '+' : '−'}${formatPercent(Math.abs(rel))})`}`,
    )
  }
  if (changed('max_gap_s'))
    parts.push(
      `самый долгий перерыв между автобусами — ${formatMetric('max_gap_s', a.max_gap_s)} вместо ` +
        formatMetric('max_gap_s', b.max_gap_s),
    )
  if (changed('late_stops')) parts.push(`опозданий больше 2 мин — ${a.late_stops} вместо ${b.late_stops}`)
  if (changed('bunching_pairs'))
    parts.push(`пар автобусов вплотную — ${a.bunching_pairs} вместо ${b.bunching_pairs}`)
  if (!parts.length) return 'Показатели почти не меняются.'
  const text = parts.join('; ')
  return `${text[0].toUpperCase()}${text.slice(1)}.`
}

function threadsOption(result: WhatIfOut, route: RouteOut, request: WhatIfRequest): ChartOption {
  const stops = [...route.stops].sort((a, b) => a.seq - b.seq)
  const seqOf = new Map(stops.map((s) => [s.stop_key, s.seq]))
  const names = new Map(stops.map((s) => [s.seq, s.name]))
  const hold = request.action === 'hold'
  const changed = (trId: number | null) => trId === null || (hold && trId === request.params.tr_id)
  const series: NonNullable<ChartOption['series']> = []
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  const segments = (v: Scenario['vehicles'][number]) => {
    const pts = v.arrivals
      // по ключу остановку, которую ТС проходит дважды (конечная, общая в обе стороны), не различить — берём seq
      .map((a) => [toMs(a.t), a.seq ?? seqOf.get(a.stop_key)] as const)
      .filter((p): p is readonly [number, number] => p[0] !== null && p[1] !== undefined)
      .sort((a, b) => a[0] - b[0])
    const out: [number, number][][] = []
    let cur: [number, number][] = []
    for (const p of pts) {
      if (cur.length && p[1] < cur[cur.length - 1][1]) {
        out.push(cur)
        cur = []
      }
      cur.push([p[0], p[1]])
    }
    if (cur.length) out.push(cur)
    // одиночная точка у края горизонта — не линия, а шум
    const lines = out.filter((seg) => seg.length > 1)
    for (const seg of lines)
      for (const [, seq] of seg) {
        lo = Math.min(lo, seq)
        hi = Math.max(hi, seq)
      }
    return lines
  }
  const OTHERS = 'Другие автобусы маршрута'
  const BEFORE = 'Этот автобус без задержки'
  const AFTER = 'Этот автобус с задержкой'
  const RESERVE = 'Резервный автобус'
  result.baseline.vehicles.forEach((v, i) => {
    segments(v).forEach((seg, j) => {
      series.push({
        id: `b:${v.tr_id ?? 'r'}:${i}:${j}`,
        name: changed(v.tr_id) ? BEFORE : OTHERS,
        type: 'line',
        data: seg,
        symbol: 'none',
        lineStyle: {
          color: changed(v.tr_id) ? '#64748b' : '#a3adb9',
          width: changed(v.tr_id) ? 2.2 : 2,
          type: changed(v.tr_id) ? 'dashed' : 'solid',
        },
        itemStyle: { color: '#a3adb9' },
      })
    })
  })
  result.scenario.vehicles.forEach((v, i) => {
    if (!changed(v.tr_id)) return
    const segs = segments(v)
    segs.forEach((seg, j) => {
      const reserve = v.tr_id === null
      const color = reserve ? RESERVE_COLOR : AFTER_COLOR
      series.push({
        id: `s:${v.tr_id ?? 'r'}:${i}:${j}`,
        name: reserve ? RESERVE : AFTER,
        type: 'line',
        data: seg,
        symbol: 'circle',
        symbolSize: 6,
        z: 5,
        lineStyle: { color, width: 3.4 },
        itemStyle: { color },
        // подпись прямо на линии — без поиска в легенде
        endLabel: {
          show: j === segs.length - 1,
          formatter: reserve ? 'резервный' : 'с задержкой',
          color,
          fontSize: 14,
          fontWeight: 600,
        },
      })
    })
  })
  const now = toMs(request.at ?? null)
  if (now !== null)
    series.push({
      id: 'now',
      name: 'сейчас',
      type: 'line',
      data: [],
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: palette.accentStrong, width: 1.5, type: 'solid' },
        label: { formatter: 'сейчас', color: palette.accentStrong, fontSize: 13, position: 'insideStartTop' },
        data: [{ xAxis: now }],
      },
    })
  // ось остановок — только по той части маршрута, где есть движение
  const last = Math.max(1, stops.length - 1)
  const yMin = Number.isFinite(lo) ? Math.max(0, lo - 1) : 0
  const yMax = Number.isFinite(hi) ? Math.min(last, hi + 1) : last
  const interval = Math.max(1, Math.ceil((yMax - yMin + 1) / 9))
  return {
    animation: false,
    grid: { left: 210, right: 110, top: 40, bottom: 30 },
    legend: { ...legendBase, data: hold ? [OTHERS, BEFORE, AFTER] : [OTHERS, RESERVE] },
    tooltip: {
      ...tooltipBase,
      trigger: 'item',
      formatter: (raw: unknown) => {
        const p = raw as { seriesName: string; value: [number, number] }
        return `<b>${escapeHtml(p.seriesName)}</b><br/>${escapeHtml(stopLabel(names.get(p.value[1])))} · ${formatClock(p.value[0])}`
      },
    },
    xAxis: {
      type: 'time',
      // с момента расчёта: слева «сейчас», дальше — что будет
      min: now ?? undefined,
      axisLabel: timeAxisLabel,
      axisLine,
      splitLine: { show: true, lineStyle: { color: palette.borderSoft } },
    },
    yAxis: {
      type: 'value',
      inverse: true,
      min: yMin,
      max: yMax,
      interval,
      axisLabel: {
        ...axisText,
        // последняя подпись вне шага наезжает на соседнюю
        showMaxLabel: (yMax - yMin) % interval === 0,
        formatter: (v: number) => {
          const name = shortStopName(names.get(v) ?? '')
          return `${name.length > 26 ? `${name.slice(0, 25)}…` : name} {n|${v + 1}}`
        },
        rich: { n: { color: palette.faint, fontSize: 11, width: 20, align: 'right' } },
      },
      splitLine,
    },
    series,
  }
}

/** Перебрать варианты действий на сервере и выбрать лучший по сводной оценке дельты. */
async function pickBest(candidates: WhatIfRequest[]): Promise<WhatIfRequest | null> {
  const results = await Promise.allSettled(candidates.map((c) => apiPost<WhatIfOut>('/api/whatif', c)))
  let best: WhatIfRequest | null = null
  let bestScore = Number.POSITIVE_INFINITY
  results.forEach((r, i) => {
    if (r.status !== 'fulfilled') return
    const score = deltaScore(r.value.delta)
    if (score < bestScore) {
      bestScore = score
      best = candidates[i]
    }
  })
  return best
}

/** Через столько минут потока расчёт What-if устарел: автобусы уже в других местах. */
const STALE_MIN = 10

/** Форма по запросу What-if (после подбора действия). */
function formFromRequest(prev: FormState, request: WhatIfRequest, now: number): FormState {
  if (request.action === 'hold') {
    return {
      ...prev,
      action: 'hold',
      holdTr: request.params.tr_id ?? prev.holdTr,
      holdMin: Math.max(1, Math.round((request.params.hold_s ?? 180) / 60)),
    }
  }
  const depart = toMs(request.params.depart_at ?? null) ?? now
  return {
    ...prev,
    action: 'add_vehicle',
    fromStopKey: request.params.from_stop_key ?? prev.fromStopKey,
    departInMin: Math.max(0, Math.round((depart - now) / 60_000)),
  }
}

/** Сравнение «без действия / с действием» таблицей: для решения «делать или нет» читается быстрее графиков. */
function ComparisonTable({ result, horizonMin }: { result: WhatIfOut; horizonMin: number }) {
  return (
    <section className="panel wi-compare">
      <div className="panel__head">
        <span className="panel__title">Без действия и с действием</span>
        <span className="panel__hint">на ближайшие {horizonMin} мин · во всех строках меньше — лучше</span>
      </div>
      <table className="wi-table">
        <thead>
          <tr>
            <th>Показатель</th>
            <th className="wi-table__num">Без действия</th>
            <th className="wi-table__num">С действием</th>
            <th className="wi-table__num">Изменение</th>
          </tr>
        </thead>
        <tbody>
          {WHATIF_METRICS.map((m) => {
            const before = result.baseline[m.key]
            const after = result.scenario[m.key]
            const delta = result.delta[m.key]
            const tone = deltaTone(m.key, delta)
            const rel = relativeChange(before, after)
            const color =
              tone === 'better'
                ? RISK_TEXT_COLOR.green
                : tone === 'worse'
                  ? RISK_TEXT_COLOR.red
                  : palette.muted
            const Icon = tone === 'same' ? MinusOutlined : delta < 0 ? ArrowDownOutlined : ArrowUpOutlined
            return (
              <tr key={m.key}>
                <td>
                  <span className="wi-table__label">{m.label}</span>
                  <span className="wi-table__hint">{m.hint}</span>
                </td>
                <td className="wi-table__num wi-table__before">{formatMetric(m.key, before)}</td>
                <td className="wi-table__num wi-table__after">{formatMetric(m.key, after)}</td>
                <td className="wi-table__num" style={{ color }}>
                  {tone === 'same' ? (
                    'без изменений'
                  ) : (
                    <>
                      <Icon /> {formatDeltaValue(m.key, delta)}
                      {rel !== null && m.unit === 's' ? ` · ${formatPercent(Math.abs(rel))}` : ''}
                    </>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}

export default function WhatIfPage() {
  const [params, setParams] = useSearchParams()
  const routesQuery = useRoutes()
  const incidentsMap = useLive((s) => s.incidents)
  const streamTime = useLive((s) => s.streamTime)
  const mutation = useWhatIf()
  const { mutate } = mutation
  const [form, setForm] = useState<FormState>({
    action: 'add_vehicle',
    fromStopKey: null,
    departInMin: 5,
    holdTr: null,
    holdMin: 3,
    horizonMin: 60,
  })
  const autoRan = useRef<string | null>(null)
  const [pickNote, setPickNote] = useState<string | null>(null)

  const routes = useMemo(() => routesQuery.data ?? [], [routesQuery.data])
  const incidents = useMemo(() => sortIncidents(incidentsMap.values()), [incidentsMap])
  const bunching = useMemo(() => incidents.filter((i) => i.kind === 'bunching'), [incidents])
  const routeParam = params.get('route')
  const autoRoute = useAutoRoute(routes)
  const routeId = routeParam && routes.some((r) => r.route_id === routeParam) ? routeParam : null

  // маршрут по умолчанию фиксируется в адресе, чтобы не «прыгал» при обновлении инцидентов
  useEffect(() => {
    if (!routeId && autoRoute) setParams({ route: autoRoute }, { replace: true })
  }, [routeId, autoRoute, setParams])
  const route = routes.find((r) => r.route_id === routeId)
  const stops = useMemo(() => (route ? [...route.stops].sort((a, b) => a.seq - b.seq) : []), [route])
  const routeBunching = bunching.find((b) => b.route_id === routeId)

  const buildRequest = useCallback(
    (f: FormState): WhatIfRequest | null => {
      const now = toMs(streamTime)
      if (!route || now === null) return null
      const at = new Date(now).toISOString()
      if (f.action === 'hold') {
        const tr = f.holdTr ?? routeBunching?.tr_id ?? route.tr_ids[0] ?? null
        if (tr === null) return null
        return {
          route_id: route.route_id,
          at,
          action: 'hold',
          params: { tr_id: tr, hold_s: f.holdMin * 60 },
          horizon_min: f.horizonMin,
        }
      }
      return {
        route_id: route.route_id,
        at,
        action: 'add_vehicle',
        params: {
          from_stop_key: f.fromStopKey ?? stops[0]?.stop_key,
          depart_at: new Date(now + f.departInMin * 60_000).toISOString(),
        },
        horizon_min: f.horizonMin,
      }
    },
    [route, stops, streamTime, routeBunching],
  )

  const run = useCallback(
    (f: FormState) => {
      const request = buildRequest(f)
      if (request) mutate(request)
    },
    [buildRequest, mutate],
  )

  /**
   * Подбор действия: базовый расчёт, затем перебор вариантов (резерв в наибольшие разрывы, придержка каждого ТС)
   * и показ лучшего по сводной оценке.
   */
  const runBest = useCallback(
    (f: FormState) => {
      const request = buildRequest({ ...f, action: 'add_vehicle' })
      if (!request || !route) return
      setPickNote(null)
      mutate(request, {
        onSuccess: (data) => {
          const now = toMs(request.at ?? null) ?? 0
          const candidates = candidateRequests(request, data.baseline, route.tr_ids, now + 60_000)
          if (!candidates.length) return
          // перебор — отдельными запросами; итоговый mutate — уже вне колбэков первого
          void pickBest(candidates).then((picked) => {
            if (!picked) return
            // отправление — целые минуты от «сейчас», как в форме: иначе вывод и поле расходятся на минуту
            const departMs = toMs(picked.params.depart_at ?? null)
            const best: WhatIfRequest =
              picked.action === 'add_vehicle' && departMs !== null
                ? {
                    ...picked,
                    params: {
                      ...picked.params,
                      depart_at: new Date(
                        now + Math.max(0, Math.round((departMs - now) / 60_000)) * 60_000,
                      ).toISOString(),
                    },
                  }
                : picked
            setForm((prev) => formFromRequest(prev, best, now))
            setPickNote(
              `Подобрано автоматически — лучший из ${candidates.length} вариантов: резервный автобус в самые большие перерывы или задержка каждого автобуса на 2–5 мин.`,
            )
            mutate(best)
          })
        },
      })
    },
    [buildRequest, mutate, route],
  )

  // первый расчёт — автоматически (лучшее действие), чтобы экран не был пустым
  useEffect(() => {
    if (!route || streamTime === null || autoRan.current === route.route_id) return
    autoRan.current = route.route_id
    runBest(form)
  }, [route, streamTime, form, runBest])

  const result = mutation.data
  const request = mutation.variables

  const applySuggestion = () => runBest(form)

  const setRoute = (id: string) => {
    setParams({ route: id }, { replace: true })
    setForm((f) => ({ ...f, fromStopKey: null, holdTr: null }))
    setPickNote(null)
    autoRan.current = null
  }

  const threadsChart = useMemo(
    () => (result && route && request ? threadsOption(result, route, request) : null),
    [result, route, request],
  )
  const tone = result ? verdictTone(result.delta) : null
  const sameRoute = request?.route_id === routeId
  const requestAt = toMs(request?.at ?? null)
  const nowMs = toMs(streamTime)
  const staleMin =
    requestAt !== null && nowMs !== null ? Math.max(0, Math.round((nowMs - requestAt) / 60_000)) : null

  return (
    <div className="page page--fixed">
      <div className="wi-body">
        <aside className="panel wi-form" aria-label="Параметры сценария">
          <div className="panel__head">
            <ExperimentOutlined />
            <span className="panel__title">Сценарий</span>
            <span className="panel__hint">
              {request?.at ? `расчёт на ${formatClock(request.at)}` : `сейчас ${formatClock(streamTime)}`}
            </span>
          </div>
          <ConfigProvider componentSize="large">
            <div className="wi-form__body">
              <label className="field">
                <span className="field__label">Маршрут</span>
                <Select
                  value={routeId ?? undefined}
                  onChange={setRoute}
                  loading={routesQuery.isLoading}
                  showSearch={{ optionFilterProp: 'label' }}
                  options={routes.map((r) => ({
                    value: r.route_id,
                    label: `${r.route_id} · ${routeShortName(r)}`,
                  }))}
                />
              </label>

              <div className="field">
                <span className="field__label">Что сделать</span>
                <div className="wi-choice" role="radiogroup" aria-label="Что сделать">
                  {ACTIONS.map((a) => (
                    <button
                      key={a.value}
                      type="button"
                      role="radio"
                      aria-checked={form.action === a.value}
                      className={`wi-choice__item ${form.action === a.value ? 'is-active' : ''}`}
                      onClick={() => setForm((f) => ({ ...f, action: a.value }))}
                    >
                      <span className="wi-choice__title">
                        {a.icon} {a.title}
                      </span>
                      <span className="wi-choice__text">{a.text}</span>
                    </button>
                  ))}
                </div>
              </div>

              {form.action === 'add_vehicle' ? (
                <>
                  <label className="field">
                    <span className="field__label">С какой остановки выпустить</span>
                    <Select
                      value={form.fromStopKey ?? stops[0]?.stop_key}
                      onChange={(v: string) => setForm((f) => ({ ...f, fromStopKey: v }))}
                      showSearch={{ optionFilterProp: 'label' }}
                      options={stops.map((s) => ({ value: s.stop_key, label: `${s.seq + 1}. ${s.name}` }))}
                    />
                  </label>
                  <label className="field">
                    <span className="field__label">
                      Через сколько минут, выйдет в{' '}
                      {/* от момента последнего расчёта: при ускоренном потоке «сейчас» убегает, а время в заголовке
                        результата — то, с которым считали */}
                      <b className="num">
                        {formatHm(
                          (toMs(request?.at ?? null) ?? toMs(streamTime) ?? 0) + form.departInMin * 60_000,
                        )}
                      </b>
                    </span>
                    <InputNumber
                      min={0}
                      max={60}
                      value={form.departInMin}
                      onChange={(v) => setForm((f) => ({ ...f, departInMin: v ?? 0 }))}
                      style={{ width: '100%' }}
                    />
                  </label>
                </>
              ) : (
                <>
                  <label className="field">
                    <span className="field__label">Какой автобус задержать</span>
                    <Select
                      value={form.holdTr ?? routeBunching?.tr_id ?? route?.tr_ids[0]}
                      onChange={(v: number) => setForm((f) => ({ ...f, holdTr: v }))}
                      options={(route?.tr_ids ?? []).map((id) => ({
                        value: id,
                        label:
                          routeBunching?.tr_id === id
                            ? `ТС ${id} — догоняет ТС ${routeBunching.related_tr_id ?? ''}`
                            : `ТС ${id}`,
                      }))}
                    />
                  </label>
                  <div className="field">
                    <span className="field__label">
                      На сколько задержать на ближайшей остановке: <b>{form.holdMin} мин</b>
                    </span>
                    <Slider
                      min={1}
                      max={10}
                      value={form.holdMin}
                      onChange={(v) => setForm((f) => ({ ...f, holdMin: v }))}
                    />
                  </div>
                </>
              )}

              <div className="field">
                <span className="field__label">За какой период сравнить</span>
                <Segmented
                  block
                  value={form.horizonMin}
                  onChange={(v) => setForm((f) => ({ ...f, horizonMin: Number(v) }))}
                  options={HORIZONS}
                />
              </div>

              <Button
                type="primary"
                size="large"
                block
                icon={<ExperimentOutlined />}
                loading={mutation.isPending}
                disabled={!route || streamTime === null}
                onClick={() => run(form)}
              >
                Посчитать эффект
              </Button>
              <Tooltip title="Система сама переберёт выпуск резервного автобуса и задержку каждого автобуса и покажет лучший вариант">
                <Button
                  block
                  icon={<BulbOutlined />}
                  disabled={!route || mutation.isPending}
                  onClick={applySuggestion}
                >
                  Подобрать лучшее действие
                </Button>
              </Tooltip>
              {pickNote ? <div className="wi-note">{pickNote}</div> : null}
              {routeBunching && form.action !== 'hold' ? (
                <div className="wi-hint">
                  <CarOutlined /> На маршруте автобусы идут вплотную: ТС {routeBunching.tr_id} догоняет ТС{' '}
                  {routeBunching.related_tr_id}. Попробуйте «Задержать автобус на остановке».
                </div>
              ) : null}
            </div>
          </ConfigProvider>
        </aside>

        <section className="wi-result" aria-label="Результат">
          {mutation.isError ? (
            <Alert
              type="error"
              showIcon
              title="Расчёт не выполнен"
              description={String(mutation.error.message)}
            />
          ) : null}
          {result && request && sameRoute && tone ? (
            <>
              {staleMin !== null && staleMin >= STALE_MIN ? (
                <Alert
                  type="warning"
                  showIcon
                  title={`Расчёт сделан на ${formatHm(request.at ?? null)}: с тех пор прошло ${staleMin} мин потока`}
                  description="Автобусы уже в других местах — пересчитайте для текущей обстановки."
                  action={
                    <Button onClick={() => run(form)} disabled={mutation.isPending}>
                      Пересчитать
                    </Button>
                  }
                />
              ) : null}
              {result.baseline.vehicles.every((v) => v.arrivals.length === 0) ? (
                // по плану в горизонте нет рейсов (перерыв, ночь): сравнивать нечего — не таблица из нулей
                <Alert
                  className="wi-verdict"
                  type="info"
                  showIcon
                  title={`По плану в ближайшие ${request.horizon_min} мин на маршруте ${request.route_id} нет рейсов`}
                  description="Сравнивать нечего: ТС маршрута в это время не выходят на линию (перерыв в расписании). Выберите период побольше или другой маршрут."
                />
              ) : (
                <>
                  <Alert
                    className="wi-verdict"
                    type={VERDICT[tone].type}
                    showIcon
                    title={
                      <span>
                        <b>{actionTitle(request, route)}</b> — {VERDICT[tone].title}
                      </span>
                    }
                    description={verdictText(result)}
                  />
                  <ComparisonTable result={result} horizonMin={request.horizon_min} />
                  <section className="panel wi-chart">
                    <div className="panel__head">
                      <span className="panel__title">Где будут автобусы</span>
                      <span className="panel__hint">
                        по горизонтали — время, по вертикали — остановки маршрута; каждая линия — один автобус
                      </span>
                    </div>
                    {threadsChart ? (
                      <EChart
                        option={threadsChart}
                        height="100%"
                        ariaLabel="Где будут автобусы: без действия и с действием"
                      />
                    ) : null}
                  </section>
                </>
              )}
            </>
          ) : mutation.isPending ? (
            <div style={{ display: 'grid', placeItems: 'center', height: '100%' }}>
              <Spin size="large" />
            </div>
          ) : (
            <Empty
              style={{ marginTop: 120 }}
              description="Выберите маршрут и что сделать, затем нажмите «Посчитать эффект»"
            />
          )}
        </section>
      </div>
    </div>
  )
}
