/**
 * «Честность прогноза»: доказательство, что прогноз выдаётся заранее и сверяется с фактом.
 * Гистограмма заблаговременности (окно 10–15 мин), алерты «задним числом» (должно быть 0), онлайн-MAE против
 * baseline `cur_dev_s`, доля опозданий, предупреждённых заранее, и лента последних закрытых прогнозов.
 */
import {
  AimOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  FieldTimeOutlined,
  FundOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons'
import { Empty, Segmented, Table, Tooltip, type TableColumnsType } from 'antd'
import { useMemo, useState } from 'react'
import { useHorizon } from '../api/queries'
import type { HorizonOut, LeadBucket, PredictionOut } from '../api/types'
import {
  axisLine,
  axisText as baseAxisText,
  legendBase as baseLegend,
  splitLine,
  tooltipBase,
} from '../components/chartStyle'
import { EChart, type ChartOption } from '../components/EChart'
import { RiskDot } from '../components/RiskDot'
import { StatCard } from '../components/StatCard'
import { formatClock, formatDelay, formatNumber, formatPercent, stopLabel, toMs } from '../domain/format'
import {
  bucketInWindow,
  inInterval,
  leadMedianS,
  leadWindowShare,
  maeGain,
  outcome,
  OUTCOME_LABEL,
  type Outcome,
} from '../domain/honesty'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { useLive } from '../state/liveStore'
import { palette } from '../theme'

const WINDOWS = [
  { value: 900, label: '15 мин' },
  { value: 3600, label: '1 час' },
  { value: 86_400, label: 'сутки' },
]

const OUTCOME_COLOR: Record<Outcome, string> = {
  hit: RISK_COLOR.green,
  ok: palette.muted,
  miss: RISK_COLOR.red,
  false_alarm: RISK_COLOR.yellow,
  pending: palette.faint,
}

function secondsText(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${formatNumber(value)}\u00a0с`
}

function minutesLabel(s: number): string {
  const m = s / 60
  return Number.isInteger(m) ? String(m) : m.toFixed(1)
}

// страница для показа на экране: подписи графиков крупнее общих
const axisText = { ...baseAxisText, fontSize: 15 }
const legendBase = {
  ...baseLegend,
  textStyle: { ...baseLegend.textStyle, fontSize: 15 },
  itemWidth: 16,
  itemHeight: 10,
}
const CHART_HEIGHT = 340

const LEAD_PLAN = 'до планового прибытия'
const LEAD_FACT = 'до фактического прохода'

function leadOption(h: HorizonOut): ChartOption {
  const plan = new Map(h.lead_hist.map((b) => [b.from_s, b]))
  const fact = new Map((h.actual_lead_hist ?? []).map((b) => [b.from_s, b]))
  // общая ось: минуты от 0 до конца самой длинной серии (пустые корзины — 0)
  const lo = Math.min(...[...plan.keys(), ...fact.keys()].filter((v) => v >= 0), 600)
  const hi = Math.max(...[...plan.values(), ...fact.values()].map((b) => b.to_s), 900)
  const step = h.lead_hist[0] ? h.lead_hist[0].to_s - h.lead_hist[0].from_s : 60
  const buckets: LeadBucket[] = []
  for (let f = Math.floor(lo / step) * step; f < hi; f += step)
    buckets.push({ from_s: f, to_s: f + step, count: 0 })
  const categories = buckets.map((b) => `${minutesLabel(b.from_s)}–${minutesLabel(b.to_s)}`)
  const first = buckets.findIndex((b) => bucketInWindow(b))
  let last = -1
  buckets.forEach((b, i) => {
    if (bucketInWindow(b)) last = i
  })
  const hasFact = fact.size > 0
  return {
    animation: false,
    grid: { left: 44, right: 16, top: hasFact ? 40 : 30, bottom: 40 },
    legend: hasFact ? { ...legendBase, data: [LEAD_PLAN, LEAD_FACT] } : undefined,
    tooltip: {
      ...tooltipBase,
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (raw: unknown) => {
        const items = (Array.isArray(raw) ? raw : [raw]) as {
          dataIndex: number
          seriesName: string
          value: number
        }[]
        const b = buckets[items[0]?.dataIndex ?? -1]
        if (!b) return ''
        const lines = items.map((p) => `${p.seriesName}: <b>${p.value}</b>`)
        return `Выдан за <b>${minutesLabel(b.from_s)}–${minutesLabel(b.to_s)} мин</b><br/>${lines.join('<br/>')}`
      },
    },
    xAxis: {
      type: 'category',
      data: categories,
      name: 'за сколько минут до события выдан прогноз',
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: axisText,
      axisLabel: { ...axisText, formatter: (v: string) => v.split('–')[0] },
      axisLine,
      axisTick: { show: false },
    },
    yAxis: { type: 'value', axisLabel: axisText, splitLine, minInterval: 1 },
    series: [
      {
        type: 'bar',
        name: LEAD_PLAN,
        barGap: '10%',
        barCategoryGap: '22%',
        itemStyle: { color: RISK_COLOR.green },
        data: buckets.map((b) => ({
          value: plan.get(b.from_s)?.count ?? 0,
          itemStyle: {
            color: bucketInWindow(b) ? RISK_COLOR.green : '#cbd5e1',
            borderRadius: [3, 3, 0, 0],
          },
        })),
        markArea:
          first >= 0
            ? {
                silent: true,
                itemStyle: {
                  color: 'rgba(34,197,94,0.08)',
                  borderColor: 'rgba(34,197,94,0.35)',
                  borderWidth: 1,
                },
                label: {
                  color: '#15803d',
                  fontSize: 15,
                  position: 'insideTopLeft',
                  formatter: 'окно 10–15 мин',
                },
                data: [[{ xAxis: categories[first] }, { xAxis: categories[last] }]],
              }
            : undefined,
      },
      ...(hasFact
        ? [
            {
              type: 'bar' as const,
              name: LEAD_FACT,
              itemStyle: { color: palette.accent, borderRadius: [3, 3, 0, 0] },
              data: buckets.map((b) => fact.get(b.from_s)?.count ?? 0),
            },
          ]
        : []),
    ],
  }
}

function maeOption(h: HorizonOut): ChartOption {
  const rows = [...h.mae_by_hour].sort((a, b) => a.hour - b.hour)
  return {
    animation: false,
    grid: { left: 48, right: 16, top: 34, bottom: 28 },
    legend: { ...legendBase, data: ['Форсайт (онлайн-MAE)', 'Baseline cur_dev_s'] },
    tooltip: {
      ...tooltipBase,
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (raw: unknown) => {
        const list = (Array.isArray(raw) ? raw : [raw]) as {
          dataIndex: number
          seriesName: string
          value: number | null
        }[]
        const row = rows[list[0]?.dataIndex ?? -1]
        if (!row) return ''
        const lines = list.map((p) => `${p.seriesName}: <b>${secondsText(p.value)}</b>`)
        return [
          `<b>${String(row.hour).padStart(2, '0')}:00–${String(row.hour + 1).padStart(2, '0')}:00</b>`,
          ...lines,
          `закрыто прогнозов: ${row.n}`,
        ].join('<br/>')
      },
    },
    xAxis: {
      type: 'category',
      data: rows.map((r) => `${String(r.hour).padStart(2, '0')}:00`),
      axisLabel: axisText,
      axisLine,
      axisTick: { show: false },
    },
    yAxis: { type: 'value', name: 'MAE, с', nameTextStyle: axisText, axisLabel: axisText, splitLine },
    series: [
      {
        type: 'bar',
        name: 'Форсайт (онлайн-MAE)',
        data: rows.map((r) => r.mae_s),
        itemStyle: { color: palette.accent, borderRadius: [3, 3, 0, 0] },
        barGap: '12%',
      },
      {
        type: 'bar',
        name: 'Baseline cur_dev_s',
        data: rows.map((r) => r.baseline_s),
        itemStyle: { color: '#94a3b8', borderRadius: [3, 3, 0, 0] },
      },
    ],
  }
}

const columns: TableColumnsType<PredictionOut> = [
  {
    title: 'Закрыт',
    dataIndex: 'closed_at',
    width: 84,
    render: (v: string | null) => <span className="num">{formatClock(v)}</span>,
  },
  {
    title: 'ТС',
    key: 'tr',
    width: 150,
    render: (_: unknown, p) => (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
        <RiskDot risk={p.risk} size={8} />
        {p.tr_id}
        {p.route_id ? (
          <span className="inc__route" style={{ background: palette.accentSoft, color: palette.accentText }}>
            {p.route_id}
          </span>
        ) : null}
      </span>
    ),
  },
  {
    title: 'Остановка',
    dataIndex: 'target_stop_name',
    ellipsis: true,
    render: (v: string) => stopLabel(v),
  },
  {
    title: 'Выдан за',
    dataIndex: 'lead_s',
    width: 84,
    align: 'right',
    render: (v: number) => <span className="num">{formatNumber(v / 60, 1)} мин</span>,
  },
  {
    title: 'Прогноз',
    dataIndex: 'pred_delay_s',
    width: 104,
    align: 'right',
    render: (v: number) => <span className="num">{formatDelay(v)}</span>,
  },
  {
    title: 'Факт',
    dataIndex: 'actual_delay_s',
    width: 104,
    align: 'right',
    render: (v: number | null) => <span className="num">{formatDelay(v)}</span>,
  },
  {
    title: 'Ошибка',
    key: 'err',
    width: 90,
    align: 'right',
    render: (_: unknown, p) => {
      const err =
        p.abs_error_s ?? (p.actual_delay_s !== null ? Math.abs(p.actual_delay_s - p.pred_delay_s) : null)
      const color =
        err === null
          ? palette.muted
          : err <= 60
            ? RISK_TEXT_COLOR.green
            : err <= 120
              ? RISK_TEXT_COLOR.yellow
              : RISK_TEXT_COLOR.red
      const within = inInterval(p)
      return (
        <Tooltip
          title={
            within === null
              ? undefined
              : within
                ? 'Факт внутри интервала P10–P90'
                : 'Факт вне интервала P10–P90'
          }
        >
          <span className="num" style={{ color, display: 'inline-flex', gap: 6, alignItems: 'center' }}>
            {err === null ? '—' : `${Math.round(err)} с`}
            {within === null ? null : within ? (
              <CheckCircleFilled style={{ color: RISK_COLOR.green, fontSize: 15 }} />
            ) : (
              <CloseCircleFilled style={{ color: palette.faint, fontSize: 15 }} />
            )}
          </span>
        </Tooltip>
      )
    },
  },
  {
    title: 'Итог',
    key: 'outcome',
    width: 156,
    render: (_: unknown, p) => {
      const o = outcome(p)
      return (
        <span className="outcome" style={{ color: OUTCOME_COLOR[o], borderColor: `${OUTCOME_COLOR[o]}55` }}>
          {OUTCOME_LABEL[o]}
        </span>
      )
    },
  },
]

export default function HonestyPage() {
  const [windowS, setWindowS] = useState(3600)
  const horizon = useHorizon(windowS)
  const closed = useLive((s) => s.closed)
  const h = horizon.data

  const leadChart = useMemo(() => (h ? leadOption(h) : null), [h])
  const maeChart = useMemo(() => (h ? maeOption(h) : null), [h])
  const share = h ? leadWindowShare(h.lead_hist) : null
  const median = h ? leadMedianS(h.lead_hist) : null
  const factMedian = h?.actual_lead_hist?.length ? leadMedianS(h.actual_lead_hist) : null
  const gain = h ? maeGain(h.online_mae_s, h.baseline_mae_s) : null
  const retro = h?.retroactive ?? null
  const rows = useMemo(
    () => [...closed].sort((a, b) => (toMs(b.closed_at) ?? 0) - (toMs(a.closed_at) ?? 0)).slice(0, 60),
    [closed],
  )

  return (
    <div className="page page--large">
      <div className="page-head">
        <div>
          <div className="page-head__title">Прогноз выдаётся заранее и проверяется фактом</div>
          <div className="page-head__sub">
            Каждый прогноз создаётся для остановки с планом строго через 10–15 минут и закрывается, когда ТС
            её проходит.
          </div>
        </div>
        <span style={{ flex: 1 }} />
        <span className="toolbar__label">Окно</span>
        <Segmented size="large" value={windowS} onChange={(v) => setWindowS(Number(v))} options={WINDOWS} />
      </div>

      <div className="grid-5">
        <StatCard
          label="Алерты задним числом"
          icon={<SafetyCertificateOutlined />}
          accent={retro ? RISK_COLOR.red : RISK_COLOR.green}
          value={
            <span
              className="hero-zero"
              style={{ color: retro ? RISK_TEXT_COLOR.red : RISK_TEXT_COLOR.green }}
            >
              {retro ?? '—'}
            </span>
          }
          sub={
            retro
              ? 'есть алерты, выданные после планового прибытия'
              : 'ни одного алерта после события — должно быть 0'
          }
          tooltip="Алерт «задним числом» — выданный позже планового прибытия ТС на остановку"
        />
        <StatCard
          label="Онлайн-MAE прогноза"
          icon={<AimOutlined />}
          accent={palette.accent}
          value={secondsText(h?.online_mae_s)}
          sub={
            <>
              <span className="nowrap">baseline cur_dev_s: {secondsText(h?.baseline_mae_s)}</span>{' '}
              {gain !== null ? (
                <b
                  className="nowrap"
                  style={{ color: gain > 0 ? RISK_TEXT_COLOR.green : RISK_TEXT_COLOR.red }}
                >
                  {gain > 0 ? 'лучше' : 'хуже'} на {formatPercent(Math.abs(gain))}
                </b>
              ) : null}
            </>
          }
          tooltip="Средняя абсолютная ошибка закрытых прогнозов против наивного прогноза «отклонение сохранится» (cur_dev_s)"
        />
        <StatCard
          label="Опоздания с алертом"
          icon={<FundOutlined />}
          accent={RISK_COLOR.yellow}
          value={formatPercent(h?.warned_share)}
          sub="опозданий > 2 мин, о которых алерт предупредил за 10–15 мин"
          tooltip="Доля фактических опозданий более 2 минут, для которых заранее был выдан жёлтый или красный алерт"
        />
        <StatCard
          label="Заблаговременность"
          icon={<FieldTimeOutlined />}
          accent={RISK_COLOR.green}
          value={median === null ? '—' : `${(median / 60).toFixed(1).replace('.', ',')} мин`}
          sub={
            share === null
              ? 'прогнозов пока нет'
              : `${formatPercent(share)} — в окне 10–15 мин до плана` +
                (factMedian === null
                  ? ''
                  : ` · до факта ${(factMedian / 60).toFixed(1).replace('.', ',')} мин`)
          }
          tooltip="Медиана времени от выдачи прогноза до планового прибытия; «до факта» — до фактического прохода остановки"
        />
        <StatCard
          label="Закрыто прогнозов"
          icon={<CheckCircleFilled />}
          accent="#64748b"
          value={formatNumber(h?.closed)}
          sub="сверено с фактом прохождения остановки"
        />
      </div>

      <div className="grid-2" style={{ marginTop: 12 }}>
        <section className="panel">
          <div className="panel__head">
            <span className="panel__title">Заблаговременность прогнозов</span>
            <span className="panel__hint">сколько минут оставалось до события в момент выдачи</span>
          </div>
          {leadChart ? (
            <EChart
              option={leadChart}
              height={CHART_HEIGHT}
              ariaLabel="Гистограмма заблаговременности прогнозов"
            />
          ) : (
            <Empty
              style={{ padding: 60 }}
              description={horizon.isError ? 'Метрики недоступны' : 'Загрузка…'}
            />
          )}
        </section>
        <section className="panel">
          <div className="panel__head">
            <span className="panel__title">Ошибка по часам: Форсайт против baseline</span>
            <span className="panel__hint">ниже — лучше</span>
          </div>
          {maeChart && h?.mae_by_hour.length ? (
            <EChart option={maeChart} height={CHART_HEIGHT} ariaLabel="MAE по часам: модель и baseline" />
          ) : (
            <Empty style={{ padding: 60 }} description="Закрытых прогнозов пока нет" />
          )}
        </section>
      </div>

      <section className="panel" style={{ marginTop: 12 }}>
        <div className="panel__head">
          <span className="panel__title">Последние закрытые прогнозы</span>
          <span className="panel__hint">прогноз против факта · обновляется в реальном времени</span>
        </div>
        <Table<PredictionOut>
          className="ribbon"
          size="middle"
          rowKey="prediction_id"
          columns={columns}
          dataSource={rows}
          pagination={false}
          scroll={{ y: 460 }}
          locale={{ emptyText: 'Прогнозы ещё не закрывались — дождитесь прохождения остановок' }}
        />
      </section>
    </div>
  )
}
