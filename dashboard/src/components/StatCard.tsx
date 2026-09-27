import { Tooltip } from 'antd'
import type { CSSProperties, ReactNode } from 'react'

interface Props {
  label: string
  icon?: ReactNode
  value: ReactNode
  unit?: string
  sub?: ReactNode
  /** Цвет полосы слева (статус метрики). */
  accent?: string
  tooltip?: string
  /** Дополнительное содержимое под значением (спарклайн, чипы). */
  children?: ReactNode
  className?: string
}

/** Карточка метрики: подпись, крупное значение, пояснение; полоса слева — статус. */
export function StatCard({ label, icon, value, unit, sub, accent, tooltip, children, className }: Props) {
  const body = (
    <div
      className={`panel stat-card ${className ?? ''}`}
      style={{ '--kpi-accent': accent ?? 'var(--border)' } as CSSProperties}
    >
      <div className="stat-card__label">
        {icon}
        {label}
      </div>
      <div className="stat-card__value">
        {value}
        {unit ? <span className="kpi__unit">{unit}</span> : null}
      </div>
      {sub ? <div className="stat-card__sub">{sub}</div> : null}
      {children}
    </div>
  )
  return tooltip ? (
    <Tooltip title={tooltip} placement="bottom">
      {body}
    </Tooltip>
  ) : (
    body
  )
}
