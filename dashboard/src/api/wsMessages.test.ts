import { describe, expect, it } from 'vitest'
import { makeAlert, makeIncident, makePrediction, makeVehicle } from '../test/factories'
import { parseWsMessage } from './wsMessages'

describe('parseWsMessage', () => {
  it('snapshot / delta с ТС; числовой incident_id приводится к строке', () => {
    const msg = parseWsMessage({
      type: 'delta',
      stream_time: '2026-01-06T08:15:00Z',
      version: 7,
      vehicles: [{ ...makeVehicle(), incident_id: 42 }],
    })
    expect(msg).toMatchObject({ type: 'delta', version: 7, degraded: false })
    expect(msg?.type === 'delta' ? msg.vehicles[0].incident_id : null).toBe('42')
  })

  it('alert во вложенном поле и «плоский»', () => {
    const alert = makeAlert({ alert_id: 'a-9' })
    expect(parseWsMessage({ type: 'alert', stream_time: null, alert })).toMatchObject({
      type: 'alert',
      alert: { alert_id: 'a-9' },
    })
    expect(parseWsMessage({ type: 'alert', stream_time: null, ...alert, alert_id: 9 })).toMatchObject({
      type: 'alert',
      alert: { alert_id: '9', level: 'yellow' },
    })
  })

  it('incident требует корректное действие', () => {
    const incident = { ...makeIncident(), incident_id: 5 }
    expect(parseWsMessage({ type: 'incident', action: 'open', incident })).toMatchObject({
      type: 'incident',
      action: 'open',
      incident: { incident_id: '5' },
    })
    expect(parseWsMessage({ type: 'incident', action: 'delete', incident })).toBeNull()
  })

  it('prediction_closed и clock', () => {
    expect(parseWsMessage({ type: 'prediction_closed', prediction: makePrediction() })?.type).toBe(
      'prediction_closed',
    )
    expect(parseWsMessage({ type: 'clock', stream_time: '2026-01-06T08:15:00Z', epoch: 3 })).toEqual({
      type: 'clock',
      stream_time: '2026-01-06T08:15:00Z',
      epoch: 3,
    })
  })

  it('мусор отбрасывается', () => {
    expect(parseWsMessage(null)).toBeNull()
    expect(parseWsMessage('snapshot')).toBeNull()
    expect(parseWsMessage({ type: 'unknown' })).toBeNull()
    expect(parseWsMessage({ type: 'snapshot', vehicles: 'x' })).toBeNull()
  })
})
