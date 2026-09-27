/** Мини-график карточки инцидента: отклонение ТС за 30 мин и прогноз на 15 мин с интервалом P10–P90. */
import { useMemo } from 'react'
import type { DelayPoint, ForecastPoint } from '../api/types'
import { formatDelay, formatHm, toMs } from '../domain/format'
import { RISK_COLOR } from '../domain/risk'
import { palette } from '../theme'
import { axisLine, axisText, splitLine, timeAxisLabel, tooltipBase } from './chartStyle'
import { EChart, type ChartOption } from './EChart'

interface Props {
  history: DelayPoint[]
  forecast: ForecastPoint[]
  now: string | null
  height?: number
}

const toMin = (s: number) => Math.round((s / 60) * 100) / 100

export function DeviationChart({ history, forecast, now, height = 200 }: Props) {
  const option = useMemo<ChartOption>(() => {
    const hist = history
      .map((p) => [toMs(p.t), toMin(p.delay_s)] as [number | null, number])
      .filter((p): p is [number, number] => p[0] !== null)
    const last = hist.length ? hist[hist.length - 1] : null
    const fc = forecast
      .map((f) => ({ t: toMs(f.planned_at), pred: f.pred_delay_s, p10: f.p10, p90: f.p90 }))
      .filter((f): f is { t: number; pred: number; p10: number | null; p90: number | null } => f.t !== null)
      .sort((a, b) => a.t - b.t)
    const predLine: [number, number][] = [
      ...(last ? [last] : []),
      ...fc.map((f): [number, number] => [f.t, toMin(f.pred)]),
    ]
    const lower: [number, number][] = [
      ...(last ? [last] : []),
      ...fc.map((f): [number, number] => [f.t, toMin(f.p10 ?? f.pred)]),
    ]
    const band: [number, number][] = [
      ...(last ? [[last[0], 0] as [number, number]] : []),
      ...fc.map((f): [number, number] => [f.t, toMin((f.p90 ?? f.pred) - (f.p10 ?? f.pred))]),
    ]
    const nowMs = toMs(now)
    return {
      animation: false,
      grid: { left: 44, right: 12, top: 18, bottom: 26 },
      tooltip: {
        ...tooltipBase,
        trigger: 'axis',
        formatter: (params: unknown) => {
          const list = Array.isArray(params)
            ? (params as { seriesName: string; value: [number, number] }[])
            : []
          const t = list[0]?.value[0]
          const rows = list
            .filter((p) => p.seriesName !== 'P10' && p.seriesName !== 'Интервал')
            .map((p) => `${p.seriesName}: <b>${formatDelay(p.value[1] * 60)}</b>`)
          return [`<b>${formatHm(t)}</b>`, ...rows].join('<br/>')
        },
      },
      xAxis: {
        type: 'time',
        axisLabel: timeAxisLabel,
        axisLine,
        splitLine: { show: false },
        min: nowMs !== null ? nowMs - 30 * 60_000 : undefined,
        max: nowMs !== null ? nowMs + 16 * 60_000 : undefined,
      },
      yAxis: {
        type: 'value',
        name: 'мин',
        nameTextStyle: axisText,
        axisLabel: { ...axisText, formatter: (v: number) => (v > 0 ? `+${v}` : `${v}`) },
        splitLine,
      },
      series: [
        {
          name: 'P10',
          type: 'line',
          data: lower,
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          silent: true,
        },
        {
          name: 'Интервал',
          type: 'line',
          data: band,
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(245, 185, 11, 0.18)' },
          silent: true,
        },
        {
          name: 'Отклонение',
          type: 'line',
          data: hist,
          symbol: 'circle',
          symbolSize: 5,
          lineStyle: { color: palette.accentStrong, width: 2 },
          itemStyle: { color: palette.accentStrong },
          markLine: {
            silent: true,
            symbol: 'none',
            label: { color: palette.faint, fontSize: 12, formatter: '{b}' },
            data: [
              {
                name: '1 мин',
                yAxis: 1,
                lineStyle: { color: RISK_COLOR.green, type: 'dashed', opacity: 0.5 },
              },
              { name: '2 мин', yAxis: 2, lineStyle: { color: RISK_COLOR.red, type: 'dashed', opacity: 0.6 } },
              ...(nowMs !== null
                ? [
                    {
                      name: 'сейчас',
                      xAxis: nowMs,
                      lineStyle: { color: palette.muted, type: 'solid' as const, opacity: 0.5 },
                    },
                  ]
                : []),
            ],
          },
        },
        {
          name: 'Прогноз',
          type: 'line',
          data: predLine,
          symbol: 'emptyCircle',
          symbolSize: 6,
          lineStyle: { color: RISK_COLOR.yellow, width: 2, type: 'dashed' },
          itemStyle: { color: RISK_COLOR.yellow },
        },
      ],
    }
  }, [history, forecast, now])

  return (
    <EChart
      option={option}
      height={height}
      ariaLabel="Отклонение от графика за 30 минут и прогноз на 15 минут"
    />
  )
}
