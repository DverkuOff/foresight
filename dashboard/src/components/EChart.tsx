/** Лёгкая обёртка ECharts (tree-shaking: только нужные графики и компоненты). */
import { BarChart, CustomChart, LineChart, ScatterChart } from 'echarts/charts'
import type {
  BarSeriesOption,
  CustomSeriesOption,
  LineSeriesOption,
  ScatterSeriesOption,
} from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
} from 'echarts/components'
import type {
  DataZoomComponentOption,
  GridComponentOption,
  LegendComponentOption,
  TooltipComponentOption,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import type { ComposeOption, ECharts } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  CanvasRenderer,
])

export type ChartOption = ComposeOption<
  | LineSeriesOption
  | BarSeriesOption
  | ScatterSeriesOption
  | CustomSeriesOption
  | GridComponentOption
  | TooltipComponentOption
  | LegendComponentOption
  | DataZoomComponentOption
>

/** Параметры события клика по элементу графика (подмножество, которое нужно дашборду). */
export interface ChartClick {
  seriesId?: string
  seriesName?: string
  dataIndex?: number
  value?: unknown
}

interface Props {
  option: ChartOption
  height: number | string
  /**
   * Полная замена опций (по умолчанию). Для живых графиков с зумом — `false` + `replaceMerge: ['series']`:
   * серии заменяются, а состояние зума и легенды сохраняется.
   */
  notMerge?: boolean
  replaceMerge?: string[]
  className?: string
  ariaLabel?: string
  onClick?: (params: ChartClick) => void
}

export function EChart({
  option,
  height,
  notMerge = true,
  replaceMerge,
  className,
  ariaLabel,
  onClick,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<ECharts | null>(null)
  const clickRef = useRef(onClick)

  useEffect(() => {
    clickRef.current = onClick
  }, [onClick])

  useEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const instance = echarts.init(el, null, { renderer: 'canvas' })
    chart.current = instance
    instance.on('click', (params) => {
      const p = params as ChartClick
      clickRef.current?.({
        seriesId: p.seriesId,
        seriesName: p.seriesName,
        dataIndex: p.dataIndex,
        value: p.value,
      })
    })
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(el)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    chart.current?.setOption(option, { notMerge, replaceMerge, lazyUpdate: true })
  }, [option, notMerge, replaceMerge])

  return (
    <div
      ref={ref}
      className={className}
      role="img"
      aria-label={ariaLabel}
      style={{ height, width: '100%', minHeight: 0 }}
    />
  )
}
