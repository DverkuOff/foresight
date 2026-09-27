import { describe, expect, it } from 'vitest'
import {
  formatClock,
  formatDelay,
  formatDelayShort,
  formatDuration,
  formatHm,
  formatIn,
  formatMinSec,
  formatPercent,
  formatSignedSeconds,
  incidentHeadline,
  stopLabel,
  toMs,
  vehicleLabel,
} from './format'

describe('«+3 мин к ост. X через 12 мин»', () => {
  it('заголовок инцидента', () => {
    expect(incidentHeadline(180, 'Каширское ш., д.49', 720)).toBe(
      '+3 мин к ост. Каширское ш., д.49 через 12 мин',
    )
  })

  it('не дублирует «ост.» и подписывает пустое имя', () => {
    expect(incidentHeadline(-45, 'Ост. Кленовый бульвар', 40)).toBe(
      '−45 с к Ост. Кленовый бульвар через 40 с',
    )
    expect(stopLabel('  ')).toBe('ост. без названия')
    expect(stopLabel('остановка «Школа»')).toBe('остановка «Школа»')
    expect(stopLabel('Остановка №48')).toBe('ост. №48')
  })

  it('короткое отклонение: до минуты — секунды шагом 5, дальше минуты', () => {
    expect(formatDelayShort(180)).toBe('+3 мин')
    expect(formatDelayShort(150)).toBe('+2,5 мин')
    expect(formatDelayShort(126)).toBe('+2,1 мин')
    expect(formatDelayShort(95)).toBe('+1,6 мин')
    expect(formatDelayShort(89)).toBe('+1,5 мин')
    expect(formatDelayShort(720)).toBe('+12 мин')
    expect(formatDelayShort(42)).toBe('+40 с')
    expect(formatDelayShort(-61)).toBe('−1 мин')
    expect(formatDelayShort(2)).toBe('0 с')
    expect(formatDelayShort(null)).toBe('—')
  })

  it('время до события', () => {
    expect(formatIn(720)).toBe('через 12 мин')
    expect(formatIn(42)).toBe('через 40 с')
    expect(formatIn(10)).toBe('сейчас')
    expect(formatIn(-300)).toBe('5 мин назад')
    expect(formatIn(undefined)).toBe('—')
  })
})

describe('форматирование величин', () => {
  it('точное отклонение со знаком', () => {
    expect(formatDelay(200)).toBe('+3 мин 20 с')
    expect(formatDelay(-45)).toBe('−45 с')
    expect(formatDelay(120)).toBe('+2 мин')
    expect(formatDelay(0)).toBe('0 с')
  })

  it('длительности', () => {
    expect(formatDuration(45)).toBe('45 с')
    expect(formatDuration(720)).toBe('12 мин')
    expect(formatDuration(3900)).toBe('1 ч 05 мин')
    expect(formatMinSec(252)).toBe('4 мин 12 с')
    expect(formatMinSec(-65)).toBe('1 мин 05 с')
    expect(formatMinSec(420)).toBe('7 мин')
    expect(formatMinSec(9)).toBe('9 с')
  })

  it('время потока — в UTC (наивное время датасета)', () => {
    expect(formatClock('2026-01-06T08:15:30Z')).toBe('08:15:30')
    expect(formatHm('2026-01-06T23:59:59Z')).toBe('23:59')
    expect(formatClock(null)).toBe('—')
    expect(toMs('not a date')).toBeNull()
  })

  it('проценты, секунды со знаком, подпись ТС', () => {
    expect(formatPercent(0.625)).toBe('63%')
    expect(formatPercent(0.625, 1)).toBe('62.5%')
    expect(formatSignedSeconds(-40.4)).toBe('−40')
    expect(formatSignedSeconds(125)).toBe('+125')
    expect(vehicleLabel(122048)).toBe('ТС 122048')
    expect(vehicleLabel(null, 894032)).toBe('Устройство 894032')
  })
})
