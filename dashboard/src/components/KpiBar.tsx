import {
  AlertOutlined,
  CarOutlined,
  ClockCircleOutlined,
  HeartOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { Tooltip } from 'antd'
import { useMemo } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { formatDelay, vehicleLabel } from '../domain/format'
import { alertsInWindow, computeKpis } from '../domain/kpi'
import { RISK_COLOR, RISK_TEXT_COLOR } from '../domain/risk'
import { useLive } from '../state/liveStore'
import type { SystemStatus } from '../state/useSystemStatus'

function Kpi(props: {
  label: string
  icon: ReactNode
  accent: string
  value: ReactNode
  unit?: string
  sub?: ReactNode
  tooltip: string
}) {
  return (
    <Tooltip title={props.tooltip} placement="bottom">
      <div className="panel kpi" style={{ '--kpi-accent': props.accent } as CSSProperties}>
        <div className="kpi__label">
          {props.icon}
          {props.label}
        </div>
        <div className="kpi__value">
          {props.value}
          {props.unit ? <span className="kpi__unit">{props.unit}</span> : null}
        </div>
        {props.sub ? <div className="kpi__sub">{props.sub}</div> : null}
      </div>
    </Tooltip>
  )
}

export function KpiBar({ status }: { status: SystemStatus }) {
  const vehicles = useLive((s) => s.vehicles)
  const alerts = useLive((s) => s.alerts)
  const streamTime = useLive((s) => s.streamTime)
  const k = useMemo(() => computeKpis(vehicles.values()), [vehicles])
  const alerts15 = useMemo(() => alertsInWindow(alerts, streamTime), [alerts, streamTime])

  const atRisk = k.red + k.yellow
  const systemOk =
    status.connection === 'open' && !status.degraded && !status.apiDown && status.staleForS === null
  const systemText =
    status.connection === 'lost'
      ? 'Нет связи'
      : status.degraded || status.apiDown
        ? 'Деградация'
        : status.staleForS !== null
          ? 'Устарело'
          : status.connection === 'connecting'
            ? 'Подключение'
            : 'Норма'
  const systemColor = systemOk
    ? RISK_COLOR.green
    : status.connection === 'lost'
      ? RISK_COLOR.red
      : RISK_COLOR.yellow

  return (
    <div className="kpi-row">
      <Kpi
        label="ТС в работе"
        icon={<CarOutlined />}
        accent="#0ea5e9"
        value={k.inWork}
        unit={`из ${k.total}`}
        sub={
          <>
            <span>на связи</span>
            {k.stale ? <span>· {k.stale} без свежих данных</span> : null}
          </>
        }
        tooltip="ТС со свежими координатами (статус online) из всех известных устройств"
      />
      <Kpi
        label="В риске"
        icon={<WarningOutlined />}
        accent={atRisk ? RISK_COLOR.red : RISK_COLOR.green}
        value={<span style={{ color: atRisk ? '#dc2626' : undefined }}>{atRisk}</span>}
        unit="ТС"
        sub={
          <>
            <span style={{ color: RISK_TEXT_COLOR.red }}>● {k.red} высокий</span>
            <span style={{ color: RISK_TEXT_COLOR.yellow }}>● {k.yellow} риск</span>
          </>
        }
        tooltip="Красные: прогноз опоздания > 2 мин или вероятность > 60%. Жёлтые: промежуточный риск"
      />
      <Kpi
        label="Средний прогноз опоздания"
        icon={<ClockCircleOutlined />}
        accent={RISK_COLOR.yellow}
        value={k.meanPredS === null ? '—' : formatDelay(Math.round(k.meanPredS / 5) * 5)}
        sub={
          k.maxPred ? (
            <>
              <span>макс. {formatDelay(k.maxPred.value)}</span>
              <span>· {vehicleLabel(k.maxPred.vehicle.tr_id)}</span>
            </>
          ) : (
            <span>прогнозов пока нет</span>
          )
        }
        tooltip="Среднее по прогнозам на ближайшую остановку в окне 10–15 мин по всем ТС с прогнозом"
      />
      <Kpi
        label="Алерты за 15 мин"
        icon={<AlertOutlined />}
        accent={alerts15.red ? RISK_COLOR.red : '#0ea5e9'}
        value={alerts15.total}
        sub={<span>{alerts15.red} красных · по времени потока</span>}
        tooltip="Предупреждения, выданные за последние 15 минут времени потока"
      />
      <Kpi
        label="Состояние системы"
        icon={<HeartOutlined />}
        accent={systemColor}
        value={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontSize: 22 }}>
            <span
              className={`dot ${systemOk ? '' : 'dot--pulse'}`}
              style={{ width: 12, height: 12, background: systemColor, boxShadow: `0 0 10px ${systemColor}` }}
            />
            {systemText}
          </span>
        }
        sub={
          <span>
            {status.degradedParts.length
              ? status.degradedParts.join(', ')
              : systemOk
                ? 'все сервисы в работе'
                : status.apiDown
                  ? 'API не отвечает'
                  : 'см. индикаторы в шапке'}
          </span>
        }
        tooltip="Связь с сервером, свежесть потока и зависимости (Redis, PostgreSQL, ML)"
      />
    </div>
  )
}
