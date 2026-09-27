import {
  ClockCircleOutlined,
  DashboardOutlined,
  DisconnectOutlined,
  FieldTimeOutlined,
  NodeCollapseOutlined,
  QuestionCircleOutlined,
  RiseOutlined,
} from '@ant-design/icons'
import { Tooltip } from 'antd'
import type { CSSProperties, ReactNode } from 'react'
import type { CauseCode } from '../api/types'
import { causeInfo } from '../domain/causes'

const ICONS: Record<CauseCode, ReactNode> = {
  dwell_long: <ClockCircleOutlined />,
  slow_segment: <DashboardOutlined />,
  layover: <FieldTimeOutlined />,
  accumulated_delay: <RiseOutlined />,
  bunching: <NodeCollapseOutlined />,
  gps_lost: <DisconnectOutlined />,
  unknown: <QuestionCircleOutlined />,
}

interface Props {
  code: CauseCode
  text?: string
  withLabel?: boolean
  style?: CSSProperties
}

/** Иконка причины с подсказкой (текст причины из контракта). */
export function CauseIcon({ code, text, withLabel = false, style }: Props) {
  const info = causeInfo(code)
  const icon = ICONS[code] ?? ICONS.unknown
  return (
    <Tooltip title={text ?? info.text}>
      <span
        style={{ display: 'inline-flex', alignItems: 'center', gap: 6, ...style }}
        aria-label={text ?? info.text}
      >
        {icon}
        {withLabel ? <span>{info.short}</span> : null}
      </span>
    </Tooltip>
  )
}
