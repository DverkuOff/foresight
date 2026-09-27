/**
 * Построение серий графика «нитка» (время × остановки маршрута) из ответа `GET /api/stringline`.
 * Независимо от библиотеки графиков: страница превращает модель в опции ECharts.
 */
import type { RouteStop, StringlineOut, StringlineStop } from '../api/types'
import { toMs } from './format'

/** [время, мс; номер остановки] или разрыв линии. */
export type ThreadPoint = [number, number] | null

export interface Thread {
  trId: number
  points: ThreadPoint[]
}

export interface ForecastBand {
  trId: number
  /** Многоугольник полосы P10–P90: [время, seq] по часовой стрелке. */
  polygon: [number, number][]
}

export interface LatePoint {
  trId: number
  t: number
  seq: number
  delayS: number
}

export interface BunchingPoint {
  trA: number
  trB: number
  t: number
  seq: number
  gapS: number
  predicted: boolean
}

export interface VehiclePosition {
  trId: number
  t: number
  seq: number
  delayS: number | null
}

export interface StringlineModel {
  stops: StringlineStop[]
  planned: Thread[]
  actual: Thread[]
  forecast: Thread[]
  bands: ForecastBand[]
  late: LatePoint[]
  /** Участки фактической нитки с опозданием > порога (подсветка). */
  lateRuns: Thread[]
  /** Отклонение от плана в каждой фактической точке: ключ `trId:t`. */
  factDelay: Map<string, number>
  /** Связка от последнего прохода остановки до позиции ТС (стоянка или пропуск детектора). */
  links: Thread[]
  bunching: BunchingPoint[]
  /** Отклонение последней фактической точки по ТС. */
  lastDelay: Map<number, number>
  /** Где ТС сейчас (видно и там, где детектор пропустил остановки и у факта разрыв). */
  positions: VehiclePosition[]
  range: [number, number] | null
  now: number | null
}

export interface StringlineOptions {
  /** Разрыв линии, если между соседними точками больше этого (отстой). */
  breakGapS?: number
  /** Порог опоздания, с. */
  lateS?: number
  /** Сбивка: интервал меньше этой доли планового. */
  bunchingShare?: number
  /** Сбивка при неизвестном плановом интервале, с. */
  bunchingFallbackS?: number
}

/** Дольше этого без отметок связку «последний проход → позиция» не рисуем. */
const LINK_MAX_S = 60 * 60

const DEFAULTS: Required<StringlineOptions> = {
  breakGapS: 20 * 60,
  lateS: 120,
  bunchingShare: 0.35,
  bunchingFallbackS: 90,
}

interface RawPoint {
  t: number
  seq: number
}

function toRaw(points: { t: string; seq: number }[]): RawPoint[] {
  const out: RawPoint[] = []
  for (const p of points) {
    const t = toMs(p.t)
    if (t !== null && Number.isFinite(p.seq)) out.push({ t, seq: p.seq })
  }
  if (!out.length) return out
  // при равном времени — по порядку остановок: две остановки в одну минуту плана не рвут нитку; на стыке
  // кругов (конец и начало маршрута в одну минуту) сначала конец круга, потом начало следующего
  const seqs = out.map((p) => p.seq)
  const half = (Math.max(...seqs) - Math.min(...seqs)) / 2
  out.sort((a, b) => a.t - b.t)
  const sorted: RawPoint[] = []
  for (let i = 0; i < out.length;) {
    let j = i
    while (j < out.length && out[j].t === out[i].t) j += 1
    const group = out.slice(i, j)
    const lo = Math.min(...group.map((p) => p.seq))
    const hi = Math.max(...group.map((p) => p.seq))
    // стык кругов — номера в одной минуте отстоят больше чем на полмаршрута
    const wraps = hi - lo > half
    const nextLoop = (p: RawPoint) => Number(wraps && p.seq - lo <= half)
    group.sort((a, b) => nextLoop(a) - nextLoop(b) || a.seq - b.seq)
    sorted.push(...group)
    i = j
  }
  return sorted
}

function isBreak(prev: RawPoint, p: RawPoint, breakGapS: number): boolean {
  return p.seq < prev.seq || p.t - prev.t > breakGapS * 1000
}

/**
 * Линия с разрывами: разрыв при переходе через конец кругового маршрута (номер остановки уменьшился)
 * и при длинной паузе (отстой).
 */
export function withBreaks(points: RawPoint[], breakGapS: number = DEFAULTS.breakGapS): ThreadPoint[] {
  const out: ThreadPoint[] = []
  let prev: RawPoint | null = null
  for (const p of points) {
    if (prev && isBreak(prev, p, breakGapS)) out.push(null)
    out.push([p.t, p.seq])
    prev = p
  }
  return out
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const s = [...values].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

/** Плановое время той же остановки того же рейса, ближайшее к фактическому (в пределах 30 мин). */
function matchPlanned(planned: RawPoint[], p: RawPoint): number | null {
  let best: number | null = null
  for (const q of planned) {
    if (q.seq !== p.seq) continue
    const d = Math.abs(q.t - p.t)
    if (d <= 30 * 60 * 1000 && (best === null || d < Math.abs(best - p.t))) best = q.t
  }
  return best
}

export function buildStringline(data: StringlineOut, options: StringlineOptions = {}): StringlineModel {
  const opt = { ...DEFAULTS, ...options }
  const stops = [...data.stops].sort((a, b) => a.seq - b.seq)
  const planned: Thread[] = []
  const actual: Thread[] = []
  const forecast: Thread[] = []
  const bands: ForecastBand[] = []
  const late: LatePoint[] = []
  const lateRuns: Thread[] = []
  const factDelay = new Map<string, number>()
  const links: Thread[] = []
  const lastDelay = new Map<number, number>()
  const positions: VehiclePosition[] = []
  const arrivalsBySeq = new Map<number, { trId: number; t: number; predicted: boolean }[]>()
  const plannedBySeq = new Map<number, number[]>()
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY

  const track = (t: number) => {
    lo = Math.min(lo, t)
    hi = Math.max(hi, t)
  }
  const addArrival = (seq: number, trId: number, t: number, predicted: boolean) => {
    const list = arrivalsBySeq.get(seq) ?? []
    list.push({ trId, t, predicted })
    arrivalsBySeq.set(seq, list)
  }

  for (const trip of data.trips) {
    const pl = toRaw(trip.planned)
    const ac = toRaw(trip.actual)
    const fc = toRaw(trip.forecast)
    pl.forEach((p) => {
      track(p.t)
      const list = plannedBySeq.get(p.seq) ?? []
      list.push(p.t)
      plannedBySeq.set(p.seq, list)
    })
    ac.forEach((p) => track(p.t))
    fc.forEach((p) => track(p.t))
    const nowT = trip.now ? toMs(trip.now.t) : null
    if (trip.now && nowT !== null) {
      const delayS = trip.now.delay_s ?? null
      positions.push({ trId: trip.tr_id, t: nowT, seq: trip.now.seq, delayS })
      track(nowT)
    }
    if (pl.length) planned.push({ trId: trip.tr_id, points: withBreaks(pl, opt.breakGapS) })
    if (ac.length) actual.push({ trId: trip.tr_id, points: withBreaks(ac, opt.breakGapS) })

    // подсветка опоздания: подряд идущие точки > порога вместе с точкой перед ними
    let run: ThreadPoint[] = []
    const flushRun = () => {
      if (run.length) lateRuns.push({ trId: trip.tr_id, points: run })
      run = []
    }
    ac.forEach((p, i) => {
      addArrival(p.seq, trip.tr_id, p.t, false)
      const prev = i > 0 ? ac[i - 1] : null
      if (prev && isBreak(prev, p, opt.breakGapS)) flushRun()
      const plan = matchPlanned(pl, p)
      if (plan === null) {
        flushRun()
        return
      }
      const delayS = (p.t - plan) / 1000
      factDelay.set(`${trip.tr_id}:${p.t}`, delayS)
      lastDelay.set(trip.tr_id, delayS)
      if (delayS > opt.lateS) {
        late.push({ trId: trip.tr_id, t: p.t, seq: p.seq, delayS })
        if (!run.length && prev && !isBreak(prev, p, opt.breakGapS)) run.push([prev.t, prev.seq])
        run.push([p.t, p.seq])
      } else flushRun()
    })
    flushRun()

    const lastFact = ac.length ? ac[ac.length - 1] : null
    if (
      trip.now &&
      nowT !== null &&
      lastFact &&
      nowT > lastFact.t &&
      nowT - lastFact.t <= LINK_MAX_S * 1000 &&
      trip.now.seq >= lastFact.seq
    ) {
      links.push({
        trId: trip.tr_id,
        points: [
          [lastFact.t, lastFact.seq],
          [nowT, trip.now.seq],
        ],
      })
    }

    // прогноз идёт от того места, где ТС сейчас; если позиции нет — от последней фактической точки
    const here: RawPoint | null =
      trip.now && nowT !== null ? { t: nowT, seq: trip.now.seq } : ac.length ? ac[ac.length - 1] : null
    const ahead = here && trip.now ? fc.filter((p) => p.t > here.t) : fc
    if (ahead.length) {
      const joined = here && ahead[0].seq >= here.seq ? [here, ...ahead] : ahead
      forecast.push({ trId: trip.tr_id, points: withBreaks(joined, opt.breakGapS) })
      for (const p of ahead) addArrival(p.seq, trip.tr_id, p.t, true)
      // полоса P10–P90: по сегментам без переходов через конец круга; от позиции ТС — конусом
      const raw = trip.forecast
        .map((f) => ({ seq: f.seq, t10: toMs(f.p10), t90: toMs(f.p90), t: toMs(f.t) }))
        .filter(
          (f): f is { seq: number; t10: number; t90: number; t: number } =>
            f.t10 !== null && f.t90 !== null && f.t !== null && (!here || !trip.now || f.t > here.t),
        )
        .sort((a, b) => a.t - b.t)
      let segment: typeof raw =
        here && trip.now && raw.length && raw[0].seq >= here.seq
          ? [{ seq: here.seq, t10: here.t, t90: here.t, t: here.t }]
          : []
      const flush = () => {
        if (segment.length >= 2) {
          const polygon: [number, number][] = [
            ...segment.map((f): [number, number] => [f.t10, f.seq]),
            ...[...segment].reverse().map((f): [number, number] => [f.t90, f.seq]),
          ]
          bands.push({ trId: trip.tr_id, polygon })
        }
        segment = []
      }
      for (const f of raw) {
        if (segment.length && f.seq < segment[segment.length - 1].seq) flush()
        // раньше «сейчас» ТС уже не приедет: нижняя граница полосы не уходит в прошлое
        segment.push(here && trip.now ? { ...f, t10: Math.max(f.t10, here.t) } : f)
        track(f.t10)
        track(f.t90)
      }
      flush()
    }
  }

  // сбивка: соседние прибытия разных ТС на одну остановку ближе доли планового интервала
  const bunching: BunchingPoint[] = []
  for (const [seq, arrivals] of arrivalsBySeq) {
    const plannedTimes = [...(plannedBySeq.get(seq) ?? [])].sort((a, b) => a - b)
    const gaps = plannedTimes
      .slice(1)
      .map((t, i) => (t - plannedTimes[i]) / 1000)
      .filter((g) => g > 30)
    const plannedHeadway = median(gaps)
    const threshold = plannedHeadway ? plannedHeadway * opt.bunchingShare : opt.bunchingFallbackS
    const sorted = [...arrivals].sort((a, b) => a.t - b.t)
    for (let i = 1; i < sorted.length; i += 1) {
      const a = sorted[i - 1]
      const b = sorted[i]
      if (a.trId === b.trId) continue
      const gapS = (b.t - a.t) / 1000
      if (gapS < threshold) {
        bunching.push({ trA: a.trId, trB: b.trId, t: b.t, seq, gapS, predicted: a.predicted || b.predicted })
      }
    }
  }
  bunching.sort((x, y) => x.t - y.t)
  // последнее отклонение ТС — текущее (по детектору), если оно есть: свежее последней фактической точки
  for (const p of positions) if (p.delayS !== null) lastDelay.set(p.trId, p.delayS)

  return {
    stops,
    planned,
    actual,
    forecast,
    bands,
    late,
    lateRuns,
    factDelay,
    links,
    bunching,
    lastDelay,
    positions,
    range: Number.isFinite(lo) ? [lo, hi] : null,
    now: toMs(data.stream_time),
  }
}

/** Непрерывные отрезки нитки: разрывы (`null`) делят линию, одиночные точки сохраняются. */
export function splitSegments(points: ThreadPoint[]): [number, number][][] {
  const out: [number, number][][] = []
  let current: [number, number][] = []
  for (const p of points) {
    if (p === null) {
      if (current.length) out.push(current)
      current = []
    } else current.push(p)
  }
  if (current.length) out.push(current)
  return out
}

/** Пары ТС в сбивке (уникальные). */
export function bunchingPairs(points: BunchingPoint[]): [number, number][] {
  const seen = new Set<string>()
  const out: [number, number][] = []
  for (const p of points) {
    const [a, b] = p.trA < p.trB ? [p.trA, p.trB] : [p.trB, p.trA]
    const key = `${a}-${b}`
    if (!seen.has(key)) {
      seen.add(key)
      out.push([a, b])
    }
  }
  return out
}

/** Границы данных в окне времени: для автоподгонки осей (время и диапазон остановок). */
export function visibleExtent(
  model: Pick<StringlineModel, 'planned' | 'actual' | 'forecast'>,
  window: [number, number],
): { t: [number, number] | null; seq: [number, number] | null } {
  let tLo = Number.POSITIVE_INFINITY
  let tHi = Number.NEGATIVE_INFINITY
  let sLo = Number.POSITIVE_INFINITY
  let sHi = Number.NEGATIVE_INFINITY
  for (const threads of [model.planned, model.actual, model.forecast]) {
    for (const thread of threads) {
      for (const p of thread.points) {
        if (p === null || p[0] < window[0] || p[0] > window[1]) continue
        tLo = Math.min(tLo, p[0])
        tHi = Math.max(tHi, p[0])
        sLo = Math.min(sLo, p[1])
        sHi = Math.max(sHi, p[1])
      }
    }
  }
  return {
    t: Number.isFinite(tLo) ? [tLo, tHi] : null,
    seq: Number.isFinite(sLo) ? [sLo, sHi] : null,
  }
}

/** Подряд идущие записи расписания ближе этого — одна физическая остановка (одна строка оси). */
const SAME_STOP_M = 60

function keyCoords(key: string): [number, number] | null {
  const m = /^(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/.exec(key)
  return m ? [Number(m[1]), Number(m[2])] : null
}

/** Та же остановка: тот же ключ или то же название в пределах 60 м (две записи конечной и т.п.). */
export function sameStop(a: StringlineStop, b: StringlineStop): boolean {
  if (a.stop_key === b.stop_key) return true
  if (a.name !== b.name) return false
  const pa = keyCoords(a.stop_key)
  const pb = keyCoords(b.stop_key)
  if (!pa || !pb) return false
  const dx = (pa[0] - pb[0]) * 111_320 * Math.cos((pa[1] * Math.PI) / 180)
  const dy = (pa[1] - pb[1]) * 110_540
  return Math.hypot(dx, dy) <= SAME_STOP_M
}

export interface StopRow {
  /** Номера остановок маршрута (seq), показанных этой строкой. */
  seqs: number[]
  name: string
}

export interface StopAxis {
  rows: StopRow[]
  /** Строка оси для номера остановки; дробный номер (ТС между остановками) — дробная строка. */
  rowOf: (seq: number) => number
}

/** Ось остановок «нитки»: повтор одной и той же остановки подряд — одна строка, без «ступеньки» на месте. */
export function stopAxis(stops: readonly StringlineStop[]): StopAxis {
  const sorted = [...stops].sort((a, b) => a.seq - b.seq)
  const rows: StopRow[] = []
  const bySeq = new Map<number, number>()
  let prev: StringlineStop | null = null
  for (const stop of sorted) {
    if (prev && rows.length && sameStop(prev, stop)) rows[rows.length - 1].seqs.push(stop.seq)
    else rows.push({ seqs: [stop.seq], name: stop.name })
    bySeq.set(stop.seq, rows.length - 1)
    prev = stop
  }
  const rowOf = (seq: number): number => {
    const exact = bySeq.get(seq)
    if (exact !== undefined) return exact
    const lo = bySeq.get(Math.floor(seq))
    const hi = bySeq.get(Math.ceil(seq))
    if (lo !== undefined && hi !== undefined) return lo + (hi - lo) * (seq - Math.floor(seq))
    return lo ?? hi ?? seq
  }
  return { rows, rowOf }
}

/** Короткое название для оси: `Молодогвардейская ул., д.25, к.1` → `Молодогвардейская 25к1`. */
export function shortStopName(name: string): string {
  return name
    .replace(/,?\s*д\.\s*(?=\d)/g, ' ')
    .replace(/,?\s*к\.\s*(?=\d)/g, 'к')
    .replace(/,?\s*стр\.\s*(?=\d)/g, 'с')
    .replace(/(^|\s)ул\.\s*/g, '$1')
    .replace(/(\d+)-й микрорайон/g, '$1 мкр')
    .replace(/шоссе/g, 'ш.')
    .replace(/^Зеленоград,\s*/, '')
    .replace(/\s+,/g, ',')
    .replace(/\s{2,}/g, ' ')
    .replace(/[\s,]+$/, '')
    .trim()
}

export interface UpcomingStop {
  seq: number
  name: string
  /** Плановое прибытие, мс (null — план не найден). */
  plan: number | null
  /** Прогноз прибытия, мс. */
  forecast: number
  p10: number | null
  p90: number | null
  /** Прогноз − план, с. */
  delayS: number | null
}

/**
 * Ближайшие остановки ТС по прогнозу: план, прогноз и отклонение — то же, что на графике, но читается таблицей.
 * Повтор одной и той же остановки подряд (две записи расписания) показывается один раз.
 */
export function upcomingStops(data: StringlineOut, trId: number, axis: StopAxis, limit = 6): UpcomingStop[] {
  const trip = data.trips.find((t) => t.tr_id === trId)
  if (!trip) return []
  const names = new Map(data.stops.map((s) => [s.seq, s.name]))
  const planned = toRaw(trip.planned)
  const now = toMs(data.stream_time)
  const points = trip.forecast
    .map((f) => ({ seq: f.seq, t: toMs(f.t), p10: toMs(f.p10), p90: toMs(f.p90) }))
    .filter((f): f is { seq: number; t: number; p10: number | null; p90: number | null } => f.t !== null)
    .filter((f) => now === null || f.t > now)
    .sort((a, b) => a.t - b.t)
  const out: UpcomingStop[] = []
  let lastRow: number | null = null
  for (const f of points) {
    const row = axis.rowOf(f.seq)
    if (row === lastRow) continue
    lastRow = row
    const plan = matchPlanned(planned, { t: f.t, seq: f.seq })
    out.push({
      seq: f.seq,
      name: names.get(f.seq) ?? `остановка ${f.seq + 1}`,
      plan,
      forecast: f.t,
      p10: f.p10,
      p90: f.p90,
      delayS: plan === null ? null : (f.t - plan) / 1000,
    })
    if (out.length >= limit) break
  }
  return out
}

function distanceM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dx = (lon2 - lon1) * 111_320 * Math.cos((lat1 * Math.PI) / 180)
  const dy = (lat2 - lat1) * 110_540
  return Math.hypot(dx, dy)
}

/**
 * Где ТС между предыдущей и следующей остановкой по GPS: `seq` предыдущей + доля пути (0…1).
 * `nextSeq` — следующая остановка; `null`, если координат нет или ТС далеко от отрезка (не по пути).
 */
export function positionBetween(
  nextSeq: number,
  lat: number | null | undefined,
  lon: number | null | undefined,
  stops: readonly RouteStop[],
): number | null {
  if (lat === null || lat === undefined || lon === null || lon === undefined || nextSeq < 1) return null
  const next = stops.find((s) => s.seq === nextSeq)
  const prev = stops.find((s) => s.seq === nextSeq - 1)
  if (!next || !prev) return null
  const toPrev = distanceM(lat, lon, prev.lat, prev.lon)
  const toNext = distanceM(lat, lon, next.lat, next.lon)
  const span = distanceM(prev.lat, prev.lon, next.lat, next.lon)
  // по пути: сумма расстояний до концов не сильно больше самого отрезка
  if (toPrev + toNext > Math.max(3 * span, span + 600)) return null
  const share = toPrev + toNext > 0 ? toPrev / (toPrev + toNext) : 0.5
  return nextSeq - 1 + Math.round(Math.min(1, Math.max(0, share)) * 10) / 10
}
