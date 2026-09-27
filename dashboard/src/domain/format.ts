/**
 * Форматирование для диспетчера. Время потока показывается как в данных: наивное время датасета передаётся в UTC,
 * поэтому часы форматируются в зоне UTC.
 */

const MINUS = '−'

const clockFmt = new Intl.DateTimeFormat('ru-RU', {
  timeZone: 'UTC',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})
const hmFmt = new Intl.DateTimeFormat('ru-RU', {
  timeZone: 'UTC',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})
const dateFmt = new Intl.DateTimeFormat('ru-RU', { timeZone: 'UTC', day: '2-digit', month: 'long' })

export function toMs(value: string | number | Date | null | undefined): number | null {
  if (value === null || value === undefined) return null
  const ms = value instanceof Date ? value.getTime() : typeof value === 'number' ? value : Date.parse(value)
  return Number.isFinite(ms) ? ms : null
}

/** 08:15:30 */
export function formatClock(value: string | number | Date | null | undefined): string {
  const ms = toMs(value)
  return ms === null ? '—' : clockFmt.format(ms)
}

/** 08:15 */
export function formatHm(value: string | number | Date | null | undefined): string {
  const ms = toMs(value)
  return ms === null ? '—' : hmFmt.format(ms)
}

/** 6 января */
export function formatDate(value: string | number | Date | null | undefined): string {
  const ms = toMs(value)
  return ms === null ? '—' : dateFmt.format(ms)
}

function sign(value: number): string {
  return value > 0 ? '+' : value < 0 ? MINUS : ''
}

/**
 * Короткое отклонение: `+3 мин`, `+2,1 мин`, `+45 с`, `−12 мин`, `0 с`.
 * До минуты — секунды (округление до 5 с), до 10 минут — с десятыми, дальше — целые минуты.
 */
export function formatDelayShort(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const abs = Math.abs(seconds)
  if (abs < 60) {
    const s = Math.round(abs / 5) * 5
    return s === 0 ? '0 с' : `${sign(seconds)}${s} с`
  }
  if (abs < 600) {
    // 95 с и 126 с не должны оба стать «+2 мин» разного цвета риска
    const m = Math.round(abs / 6) / 10
    return `${sign(seconds)}${Number.isInteger(m) ? m : m.toFixed(1).replace('.', ',')} мин`
  }
  return `${sign(seconds)}${Math.round(abs / 60)} мин`
}

/** Точное отклонение: `+3 мин 20 с`, `−45 с`. */
export function formatDelay(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const abs = Math.round(Math.abs(seconds))
  const m = Math.floor(abs / 60)
  const s = abs % 60
  if (abs === 0) return '0 с'
  if (m === 0) return `${sign(seconds)}${s} с`
  return s === 0 ? `${sign(seconds)}${m} мин` : `${sign(seconds)}${m} мин ${s} с`
}

/** Длительность без знака: `12 мин`, `45 с`, `1 ч 05 мин`. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const abs = Math.round(Math.abs(seconds))
  if (abs < 60) return `${abs} с`
  const m = Math.round(abs / 60)
  if (m < 60) return `${m} мин`
  return `${Math.floor(m / 60)} ч ${String(m % 60).padStart(2, '0')} мин`
}

/** Время до события: `через 12 мин`, `через 40 с`, `сейчас`, `12 мин назад`. */
export function formatIn(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  if (Math.abs(seconds) < 15) return 'сейчас'
  if (seconds < 0) return `${formatDuration(-seconds)} назад`
  if (seconds < 60) return `через ${Math.round(seconds / 5) * 5} с`
  return `через ${Math.round(seconds / 60)} мин`
}

/** «Ост. Каширское ш., д.49» — без дублирования «ост.». */
export function stopLabel(name: string | null | undefined): string {
  const clean = (name ?? '').trim()
  if (clean === '') return 'ост. без названия'
  // «к Остановка №48» не по-русски: безымянная остановка из расписания — «к ост. №48»
  const numbered = /^остановка\s*(№\s*\d+.*)$/i.exec(clean)
  if (numbered) return `ост. ${numbered[1]}`
  return /^ост(\.|ановка)/i.test(clean) ? clean : `ост. ${clean}`
}

/** Заголовок проблемы: `+3 мин к ост. X через 12 мин`. */
export function incidentHeadline(predDelayS: number, stopName: string, timeToEventS: number): string {
  return `${formatDelayShort(predDelayS)} к ${stopLabel(stopName)} ${formatIn(timeToEventS)}`
}

export function formatPercent(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '—'
  return `${(p * 100).toFixed(digits)}%`
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

/** Секунды со знаком для компактных таблиц: `+125`, `−40`. */
export function formatSignedSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const r = Math.round(value)
  return `${sign(r)}${Math.abs(r)}`
}

export function vehicleLabel(trId: number | null | undefined, unitId?: number | null): string {
  if (trId !== null && trId !== undefined) return `ТС ${trId}`
  return unitId !== null && unitId !== undefined ? `Устройство ${unitId}` : 'ТС'
}

/** Длительность без знака с секундами: `4 мин 12 с`, `45 с`, `7 мин`. */
export function formatMinSec(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const abs = Math.round(Math.abs(seconds))
  const m = Math.floor(abs / 60)
  const s = abs % 60
  if (m === 0) return `${s} с`
  return s === 0 ? `${m} мин` : `${m} мин ${String(s).padStart(2, '0')} с`
}
