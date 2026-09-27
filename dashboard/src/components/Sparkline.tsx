/** Спарклайн на SVG (без библиотеки графиков): история метрики за последние минуты. */
import { sparkPath } from './sparkPath'

interface Props {
  values: readonly (number | null)[]
  color: string
  height?: number
  /** Горизонтальная линия цели (например, 1 с для задержки). */
  target?: number | null
  ariaLabel?: string
}

export function Sparkline({ values, color, height = 28, target = null, ariaLabel }: Props) {
  const width = 200
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v))
  const max = Math.max(1e-9, ...finite, target ?? 0) * 1.1
  const path = sparkPath(values, width, height, max)
  const targetY = target !== null ? height - (target / max) * (height - 2) - 1 : null
  return (
    <svg
      className="sparkline"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      {targetY !== null ? (
        <line
          x1={0}
          x2={width}
          y1={targetY}
          y2={targetY}
          stroke="#94a3b8"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
      ) : null}
      {path ? (
        <>
          <path d={`${path}L${width},${height}L0,${height}Z`} fill={color} opacity={0.12} />
          <path d={path} fill="none" stroke={color} strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
        </>
      ) : null}
    </svg>
  )
}
