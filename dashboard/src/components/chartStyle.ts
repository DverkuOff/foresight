/** Общий стиль графиков ECharts в светлой теме дашборда. */
import { formatHm } from '../domain/format'
import { palette } from '../theme'

export const axisText = { color: palette.muted, fontSize: 13 }

export const axisLine = { lineStyle: { color: palette.border } }

export const splitLine = { lineStyle: { color: palette.borderSoft } }

export const tooltipBase = {
  backgroundColor: 'rgba(255, 255, 255, 0.98)',
  borderColor: palette.border,
  textStyle: { color: palette.text, fontSize: 14 },
  extraCssText: 'box-shadow: 0 8px 24px rgba(15,27,45,.14); border-radius: 8px;',
}

export const legendBase = {
  textStyle: { color: palette.muted, fontSize: 14 },
  itemWidth: 14,
  itemHeight: 8,
  top: 0,
}

export const timeAxisLabel = {
  ...axisText,
  formatter: (value: number) => formatHm(value),
  hideOverlap: true,
}

export function escapeHtml(text: string): string {
  return text.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c,
  )
}
