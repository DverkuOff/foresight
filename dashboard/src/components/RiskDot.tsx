import type { Risk } from '../api/types'
import { RISK_COLOR, RISK_LABEL } from '../domain/risk'

export function RiskDot({ risk, size = 10 }: { risk: Risk; size?: number }) {
  return (
    <span
      className="risk-dot"
      role="img"
      aria-label={RISK_LABEL[risk]}
      title={RISK_LABEL[risk]}
      style={{ background: RISK_COLOR[risk], width: size, height: size }}
    />
  )
}

export function RiskTag({ risk }: { risk: Risk }) {
  const color = RISK_COLOR[risk]
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '0 8px',
        height: 22,
        borderRadius: 11,
        fontSize: 14,
        fontWeight: 600,
        // на белом текст темнее самого цвета риска (styles.css: --risk-text-*)
        color: `var(--risk-text-${risk}, ${color})`,
        background: `${color}1f`,
        border: `1px solid ${color}55`,
        whiteSpace: 'nowrap',
      }}
    >
      <RiskDot risk={risk} size={8} />
      {RISK_LABEL[risk]}
    </span>
  )
}
