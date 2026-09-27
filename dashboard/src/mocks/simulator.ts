/**
 * Симулятор «живого» потока для mock-режима. Всё считается из фикстур `dataset/test`:
 * ТС едут по реальным GPS-трекам, прохождение остановок — по факту расписания, прогноз на окно (t+10, t+15] —
 * синтетический (факт + детерминированный шум, сходится по мере приближения), причины — правила по треку.
 * Часы потока идут с ускорением; по окончании окна фикстур — сброс часов (новая эпоха), как при перезапуске replayer.
 */
import type {
  AlertLevel,
  AlertOut,
  Cause,
  CauseCode,
  CauseFactor,
  DelayPoint,
  ForecastPoint,
  HorizonOut,
  IncidentDetailOut,
  IncidentOut,
  LeadBucket,
  LinkStatus,
  LngLat,
  MaeByHour,
  PerfOut,
  PredictionOut,
  Risk,
  RouteOut,
  RouteStop,
  StringlineOut,
  StringlineTrip,
  VehicleListOut,
  VehicleOut,
  WhatIfOut,
  WhatIfRequest,
  WsMessage,
} from '../api/types'
import { CAUSES } from '../domain/causes'
import { compareIncidents } from '../domain/incidents'
import { riskLevel } from '../domain/risk'
import type { HonestyFixture, SeedPrediction, SourceFixture, WorldFixture } from './fixtures'
import { gauss, hash01, normCdf } from './random'
import { runWhatIf, type EngineArrival, type EngineVehicle } from './whatifEngine'

export const TICK_MS = 30_000
const WINDOW_FROM_MS = 10 * 60_000
const WINDOW_TO_MS = 15 * 60_000
const WARMUP_MS = 30 * 60_000
const LATE_S = 120
/** Максимальное «облегчение» опоздания клона ТС, с. */
const CLONE_RELIEF_S = 240
/** Повторный алерт по тому же ТС того же или меньшего уровня — не чаще, мс. */
const ALERT_COOLDOWN_MS = 10 * 60_000

export function iso(ms: number): string {
  return new Date(ms).toISOString().replace('.000Z', 'Z')
}

interface RouteModel {
  route: RouteOut
  stops: RouteStop[]
  /** Плановое время хода seq → seq+1, с. */
  runS: number[]
}

interface Source {
  trId: number
  routeId: string | null
  t: Float64Array
  lon: Float64Array
  lat: Float64Array
  course: Float64Array
  speed: Float64Array
  fixture: SourceFixture
}

interface StopEvent {
  stopId: number
  seq: number
  stop: RouteStop
  plan: number
  fact: number | null
}

interface SimVehicle {
  unitId: number
  trId: number
  clone: boolean
  source: Source
  route: RouteModel | null
  dTrack: number
  dPlan: number
  stops: StopEvent[]
}

interface Position {
  lon: number | null
  lat: number | null
  course: number | null
  speed: number | null
  valid: boolean
  status: LinkStatus
  eventTime: number | null
  gpsAgeS: number
  packets: number
}

interface DelayState {
  currentDelayS: number
  lastPassed: StopEvent | null
  next: StopEvent | null
}

interface OpenPrediction {
  id: string
  key: string
  vehicle: SimVehicle
  stop: StopEvent
  issuedAt: number
  baselineS: number
  pred: number
  p10: number
  p90: number
  pLate: number
  risk: Risk
  alerted: AlertLevel | null
  cause: Cause
}

type ClosedPrediction = PredictionOut & { baseline_delay_s: number }

export interface SimOptions {
  /** Всего ТС (дополнительные клоны для нагрузочной проверки карты), 0 — как в фикстурах. */
  fleet?: number
}

function lowerBound(arr: Float64Array, x: number): number {
  // индекс последнего элемента <= x, -1 если нет
  let lo = 0
  let hi = arr.length - 1
  let ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] <= x) {
      ans = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return ans
}

function bearing(lon1: number, lat1: number, lon2: number, lat2: number): number {
  const toRad = Math.PI / 180
  const y = Math.sin((lon2 - lon1) * toRad) * Math.cos(lat2 * toRad)
  const x =
    Math.cos(lat1 * toRad) * Math.sin(lat2 * toRad) -
    Math.sin(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.cos((lon2 - lon1) * toRad)
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360
}

function distM(lon1: number, lat1: number, lon2: number, lat2: number): number {
  const kx = 111_320 * Math.cos((((lat1 + lat2) / 2) * Math.PI) / 180)
  return Math.hypot((lon2 - lon1) * kx, (lat2 - lat1) * 110_540)
}

function round1(x: number): number {
  return Math.round(x * 10) / 10
}

function median(values: number[]): number | null {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export class Simulator {
  readonly t0: number
  readonly end: number
  now: number
  epoch = 1
  readonly routes: RouteOut[]
  private readonly routeModels = new Map<string, RouteModel>()
  private readonly vehicles: SimVehicle[] = []
  private readonly seeds: SeedPrediction[]
  private open = new Map<string, OpenPrediction>()
  private closed: ClosedPrediction[] = []
  private alerts: AlertOut[] = []
  /** Последний алерт по ТС: уровень и время (подавление повторов). */
  private lastAlert = new Map<number, { level: AlertLevel; t: number }>()
  private incidents = new Map<string, IncidentOut>()
  private closedIncidents: IncidentOut[] = []
  private nextTick: number
  private listeners = new Set<(msg: WsMessage) => void>()
  private silent = false
  /** Демонстрация деградации (mock `?chaos=redis`): Redis «недоступен», api отдаёт последнее состояние. */
  degraded = false

  constructor(world: WorldFixture, honesty: HonestyFixture, options: SimOptions = {}) {
    this.t0 = Date.parse(world.meta.moment)
    this.end = this.t0 + world.meta.window_s * 1000
    this.routes = world.routes
    this.seeds = honesty.closed
    for (const route of world.routes) {
      const stops = [...route.stops].sort((a, b) => a.seq - b.seq)
      this.routeModels.set(route.route_id, { route, stops, runS: [] })
    }
    const sources = new Map<number, Source>()
    for (const f of world.sources) {
      const n = f.track.length
      const src: Source = {
        trId: f.tr_id,
        routeId: f.route_id,
        t: new Float64Array(n),
        lon: new Float64Array(n),
        lat: new Float64Array(n),
        course: new Float64Array(n),
        speed: new Float64Array(n),
        fixture: f,
      }
      f.track.forEach(([t, lon, lat, course, speed], i) => {
        src.t[i] = this.t0 + t * 1000
        src.lon[i] = lon
        src.lat[i] = lat
        src.course[i] = course
        src.speed[i] = speed
      })
      sources.set(f.tr_id, src)
    }
    // плановое время хода между соседними остановками маршрута (медиана по расписанию источников)
    const runs = new Map<string, Map<number, number[]>>()
    for (const f of world.sources) {
      if (!f.route_id) continue
      const bySeq = runs.get(f.route_id) ?? new Map<number, number[]>()
      const rows = [...f.stops].sort((a, b) => a[2] - b[2])
      for (let i = 1; i < rows.length; i += 1) {
        const dt = rows[i][2] - rows[i - 1][2]
        if (rows[i][1] === rows[i - 1][1] + 1 && dt >= 0 && dt < 600) {
          const list = bySeq.get(rows[i - 1][1]) ?? []
          list.push(dt)
          bySeq.set(rows[i - 1][1], list)
        }
      }
      runs.set(f.route_id, bySeq)
    }
    for (const [routeId, model] of this.routeModels) {
      const bySeq = runs.get(routeId)
      model.runS = model.stops.map((s) => median(bySeq?.get(s.seq) ?? []) ?? 75)
    }

    const addVehicle = (
      unitId: number,
      trId: number,
      sourceId: number,
      dTrack: number,
      dPlan: number,
      clone: boolean,
    ) => {
      const source = sources.get(sourceId)
      if (!source) return
      // обычный клон повторял бы опоздание источника; план клона сдвигаем на 0…4 мин позже, чтобы риск по
      // парку был разнообразным (пара «сбивка» из фикстур — с заданным сдвигом, её не трогаем)
      if (clone && dTrack === dPlan) dPlan = dTrack - Math.round(hash01(`relief-${unitId}`) * CLONE_RELIEF_S)
      const route = source.routeId ? (this.routeModels.get(source.routeId) ?? null) : null
      const stops: StopEvent[] = []
      if (route) {
        for (const [stopId, seq, plan, fact] of source.fixture.stops) {
          const stop = route.stops[seq]
          if (!stop) continue
          stops.push({
            stopId: clone ? stopId * 100 + (unitId % 100) : stopId,
            seq,
            stop,
            plan: this.t0 + (plan - dPlan) * 1000,
            fact: fact === null ? null : this.t0 + (fact - dTrack) * 1000,
          })
        }
        stops.sort((a, b) => a.plan - b.plan)
      }
      this.vehicles.push({
        unitId,
        trId,
        clone,
        source,
        route,
        dTrack: dTrack * 1000,
        dPlan: dPlan * 1000,
        stops,
      })
    }
    for (const v of world.vehicles) addVehicle(v.unit_id, v.tr_id, v.source, v.d_track, v.d_plan, v.clone)
    const scheduled = world.sources.filter((s) => s.route_id && s.stops.length)
    for (let i = 0; options.fleet && this.vehicles.length < options.fleet && scheduled.length; i += 1) {
      const src = scheduled[i % scheduled.length]
      const d = Math.round((hash01(`fleet-${i}`) - 0.5) * 3600)
      addVehicle(8_200_000 + i, 8_300_000 + i, src.tr_id, d, d, true)
    }
    for (const model of this.routeModels.values()) {
      model.route = {
        ...model.route,
        tr_ids: this.vehicles.filter((v) => v.route === model).map((v) => v.trId),
      }
    }
    this.routes = [...this.routeModels.values()].map((m) => m.route)

    this.now = this.t0
    this.nextTick = this.t0
    this.reset()
  }

  // ------------------------------------------------------------------ жизненный цикл

  subscribe(listener: (msg: WsMessage) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  private emit(msg: WsMessage): void {
    if (this.silent) return
    this.listeners.forEach((l) => l(msg))
  }

  /** Разослать сообщение подписчикам (mock-сокетам). */
  broadcast(msg: WsMessage): void {
    this.emit(msg)
  }

  /** Сброс к началу окна с «прогревом» (история алертов, закрытых прогнозов и инцидентов за 30 мин). */
  reset(): void {
    this.open = new Map()
    this.closed = []
    this.alerts = []
    this.lastAlert = new Map()
    this.incidents = new Map()
    this.closedIncidents = []
    this.silent = true
    this.now = this.t0 - WARMUP_MS
    this.nextTick = this.now
    this.advanceTo(this.t0)
    this.silent = false
  }

  /** Сдвинуть часы потока на `dtMs`; тики прогнозов — каждые 30 с времени потока. */
  advance(dtMs: number): void {
    const target = this.now + dtMs
    if (target >= this.end) {
      this.epoch += 1
      this.reset()
      this.emit({ type: 'clock', stream_time: iso(this.now), epoch: this.epoch })
      this.emit(this.vehicleMessage('snapshot'))
      return
    }
    this.advanceTo(target)
  }

  private advanceTo(target: number): void {
    while (this.nextTick <= target) {
      this.now = this.nextTick
      this.tick(this.now)
      this.nextTick += TICK_MS
    }
    this.now = target
  }

  // ------------------------------------------------------------------ состояние ТС

  private position(v: SimVehicle, t: number): Position {
    const src = v.source
    const s = t + v.dTrack
    const i = lowerBound(src.t, s)
    const n = src.t.length
    if (i < 0) {
      return {
        lon: null,
        lat: null,
        course: null,
        speed: null,
        valid: false,
        status: 'offline',
        eventTime: null,
        gpsAgeS: Infinity,
        packets: 0,
      }
    }
    const age = (s - src.t[i]) / 1000
    const j = i + 1 < n ? i + 1 : -1
    let lon = src.lon[i]
    let lat = src.lat[i]
    let course = src.course[i]
    let speed = src.speed[i]
    let eventTime = src.t[i]
    let valid = age <= 60
    if (j >= 0 && src.t[j] - src.t[i] <= 90_000) {
      const k = (s - src.t[i]) / (src.t[j] - src.t[i])
      lon = src.lon[i] + (src.lon[j] - src.lon[i]) * k
      lat = src.lat[i] + (src.lat[j] - src.lat[i]) * k
      speed = src.speed[i] + (src.speed[j] - src.speed[i]) * k
      if (distM(src.lon[i], src.lat[i], src.lon[j], src.lat[j]) > 8) {
        course = bearing(src.lon[i], src.lat[i], src.lon[j], src.lat[j])
      }
      eventTime = s
      valid = true
    }
    const status: LinkStatus = age <= 60 || valid ? 'online' : age <= 300 ? 'stale' : 'offline'
    return {
      lon,
      lat,
      course: Math.round(course),
      speed: Math.max(0, Math.round(speed)),
      valid,
      status,
      eventTime: eventTime - v.dTrack,
      gpsAgeS: valid ? 0 : age,
      packets: i + 1,
    }
  }

  private delayState(v: SimVehicle, t: number): DelayState {
    let lastPassed: StopEvent | null = null
    for (const s of v.stops) {
      if (s.fact !== null && s.fact <= t && (lastPassed === null || s.fact >= (lastPassed.fact ?? 0)))
        lastPassed = s
    }
    const currentDelayS =
      lastPassed && lastPassed.fact !== null ? (lastPassed.fact - lastPassed.plan) / 1000 : 0
    let next: StopEvent | null = null
    for (const s of v.stops) {
      if (lastPassed && s.plan < lastPassed.plan) continue
      if (s === lastPassed) continue
      if (s.fact === null ? s.plan + currentDelayS * 1000 > t : s.fact > t) {
        next = s
        break
      }
    }
    return { currentDelayS, lastPassed, next }
  }

  /** Синтетический прогноз задержки на остановку: сходится к факту по мере приближения. */
  private forecast(v: SimVehicle, stop: StopEvent, t: number, currentDelayS: number) {
    const actual = stop.fact !== null ? (stop.fact - stop.plan) / 1000 : currentDelayS
    const lead = Math.max(0, (stop.plan - t) / 1000)
    const w = Math.min(0.95, Math.max(0.3, 0.9 - (lead / 900) * 0.4))
    const noise = gauss(`${v.unitId}:${stop.stopId}`) * (20 + (lead / 900) * 30)
    const pred = currentDelayS + w * (actual - currentDelayS) + noise
    const sigma = 45 + 0.2 * Math.abs(pred) + (lead / 900) * 15
    const pLate = 1 - normCdf((LATE_S - pred) / sigma)
    return { pred, sigma, pLate, p10: pred - 1.2816 * sigma, p90: pred + 1.2816 * sigma }
  }

  private speedStats(v: SimVehicle, t: number) {
    const src = v.source
    const s = t + v.dTrack
    const hi = lowerBound(src.t, s)
    let sum = 0
    let n = 0
    let dwellFrom = s
    let dwelling = true
    for (let i = hi; i >= 0 && src.t[i] >= s - 300_000; i -= 1) {
      sum += src.speed[i]
      n += 1
      if (dwelling && src.speed[i] < 3) dwellFrom = src.t[i]
      else dwelling = false
    }
    return { meanSpeed: n ? sum / n : null, dwellS: (s - dwellFrom) / 1000 }
  }

  private cause(v: SimVehicle, t: number, ds: DelayState, target: StopEvent, pred: number): Cause {
    const pos = this.position(v, t)
    const { meanSpeed, dwellS } = this.speedStats(v, t)
    let layoverS = 0
    let prev: StopEvent | null = null
    for (const s of v.stops) {
      if (s.plan < t - 60_000 || s.plan > target.plan) continue
      if (prev && s.plan - prev.plan > 300_000) layoverS = Math.max(layoverS, (s.plan - prev.plan) / 1000)
      prev = s
    }
    const past = this.delayState(v, t - 10 * 60_000).currentDelayS
    const slope = ds.currentDelayS - past
    let code: CauseCode
    if (!pos.valid && pos.gpsAgeS > 60) code = 'gps_lost'
    else if (layoverS > 0 && pred - ds.currentDelayS > 20) code = 'layover'
    else if (dwellS >= 100) code = 'dwell_long'
    else if (meanSpeed !== null && meanSpeed < 12) code = 'slow_segment'
    else if (ds.currentDelayS > 60 || pred > 90) code = 'accumulated_delay'
    else code = 'unknown'
    const factors: CauseFactor[] = [
      { feature: 'cur_dev_s', label: 'Текущее отклонение', contribution_s: 0.55 * ds.currentDelayS },
      { feature: 'dev_slope', label: 'Рост отклонения за 10 мин', contribution_s: 0.45 * slope },
      { feature: 'stop_dur', label: 'Простой на остановке', contribution_s: 0.4 * Math.min(dwellS, 600) },
      {
        feature: 'spd_300',
        label: 'Средняя скорость за 5 мин',
        contribution_s: meanSpeed === null ? 0 : (18 - meanSpeed) * 2.5,
      },
      {
        feature: 'layover_ahead',
        label: 'Отстой на конечной впереди',
        contribution_s: layoverS > 0 ? 0.5 * (pred - ds.currentDelayS) : 0,
      },
      {
        feature: 'gps_age',
        label: 'Возраст GPS-фиксации',
        contribution_s: Number.isFinite(pos.gpsAgeS) ? 0.3 * pos.gpsAgeS : 0,
      },
    ]
    const top = factors
      .filter((f) => Math.abs(f.contribution_s) >= 1)
      .sort((a, b) => Math.abs(b.contribution_s) - Math.abs(a.contribution_s))
      .slice(0, 3)
      .map((f) => ({ ...f, contribution_s: round1(f.contribution_s) }))
    const info = CAUSES[code]
    return { code, text: info.text, recommendation: info.recommendation, factors: top }
  }

  private segment(v: SimVehicle, from: StopEvent | null, to: StopEvent) {
    const route = v.route
    if (!route) return null
    const n = route.stops.length
    const start = from ? from.seq : (to.seq - 1 + n) % n
    const line: LngLat[] = []
    let seq = start
    for (let guard = 0; guard <= n && line.length < 40; guard += 1) {
      const s = route.stops[seq]
      line.push([s.lon, s.lat])
      if (seq === to.seq) break
      seq = (seq + 1) % n
    }
    return { from_stop: route.stops[start].name, to_stop: to.stop.name, line }
  }

  private predictionsOf(v: SimVehicle): OpenPrediction[] {
    const out: OpenPrediction[] = []
    for (const p of this.open.values()) if (p.vehicle === v) out.push(p)
    return out.sort((a, b) => a.stop.plan - b.stop.plan)
  }

  /**
   * Текущий прогноз ТС — самый свежий, на горизонт 10–15 мин (карточка: «+3 мин к ост. X через 12 мин»):
   * открытый прогноз с наибольшим плановым временем. Прогнозы на уже прошедшие по плану остановки
   * остаются открытыми до факта, но в заголовок не попадают.
   */
  private currentPrediction(v: SimVehicle): OpenPrediction | null {
    const list = this.predictionsOf(v)
    return list[list.length - 1] ?? null
  }

  // ------------------------------------------------------------------ тик прогнозов

  private tick(t: number): void {
    // 1. новые и обновлённые прогнозы на окно (t+10, t+15]
    for (const v of this.vehicles) {
      if (!v.route || !v.stops.length) continue
      const pos = this.position(v, t)
      if (pos.status === 'offline') continue
      const ds = this.delayState(v, t)
      for (const stop of v.stops) {
        if (stop.plan <= t + WINDOW_FROM_MS || stop.plan > t + WINDOW_TO_MS) continue
        if (stop.fact !== null && stop.fact <= t) continue
        const key = `${v.unitId}:${stop.stopId}`
        const f = this.forecast(v, stop, t, ds.currentDelayS)
        const risk = riskLevel(f.pred, f.pLate)
        const cause = this.cause(v, t, ds, stop, f.pred)
        let p = this.open.get(key)
        if (!p) {
          p = {
            id: `p-${v.unitId}-${stop.stopId}-${this.epoch}`,
            key,
            vehicle: v,
            stop,
            issuedAt: t,
            baselineS: ds.currentDelayS,
            pred: f.pred,
            p10: f.p10,
            p90: f.p90,
            pLate: f.pLate,
            risk,
            alerted: null,
            cause,
          }
          this.open.set(key, p)
        } else {
          Object.assign(p, { pred: f.pred, p10: f.p10, p90: f.p90, pLate: f.pLate, risk, cause })
        }
        const level: AlertLevel | null = risk === 'red' ? 'red' : risk === 'yellow' ? 'yellow' : null
        // один алерт на ТС за 10 мин (повышение жёлтый → красный — сразу), как у диспетчерской системы
        const prev = this.lastAlert.get(v.unitId)
        const repeat =
          level !== null &&
          prev !== undefined &&
          t - prev.t < ALERT_COOLDOWN_MS &&
          (prev.level === 'red' || level === 'yellow')
        if (level && repeat && p.alerted === null) p.alerted = level
        if (level && !repeat && (p.alerted === null || (p.alerted === 'yellow' && level === 'red'))) {
          p.alerted = level
          this.lastAlert.set(v.unitId, { level, t })
          const alert: AlertOut = {
            alert_id: `a-${p.id}-${level}`,
            prediction_id: p.id,
            tr_id: v.trId,
            route_id: v.route.route.route_id,
            level,
            cause,
            issued_at: iso(t),
            planned_at: iso(stop.plan),
            pred_delay_s: round1(f.pred),
            p_late: Math.round(f.pLate * 1000) / 1000,
            acknowledged: false,
          }
          this.alerts.unshift(alert)
          if (this.alerts.length > 1000) this.alerts.length = 1000
          this.emit({ type: 'alert', stream_time: iso(t), alert })
        }
      }
    }
    // 2. закрытие прогнозов фактом
    for (const [key, p] of this.open) {
      const { stop } = p
      if (stop.fact !== null && stop.fact <= t) {
        const actual = (stop.fact - stop.plan) / 1000
        const closed: ClosedPrediction = {
          ...this.predictionOut(p),
          status: 'closed',
          actual_delay_s: round1(actual),
          abs_error_s: round1(Math.abs(actual - p.pred)),
          closed_at: iso(stop.fact),
          baseline_delay_s: p.baselineS,
        }
        this.closed.push(closed)
        this.open.delete(key)
        const { baseline_delay_s: _omit, ...prediction } = closed
        void _omit
        this.emit({ type: 'prediction_closed', stream_time: iso(t), prediction })
      } else if (stop.fact === null && stop.plan + 20 * 60_000 < t) {
        this.open.delete(key)
      }
    }
    // 3. инциденты «опоздание»
    const active = new Map<string, IncidentOut>()
    for (const v of this.vehicles) {
      const p = this.currentPrediction(v)
      if (!p || (p.risk !== 'red' && p.risk !== 'yellow')) continue
      const inc = this.delayIncident(v, p, t)
      active.set(inc.incident_id, inc)
    }
    // 4. сбивка
    for (const inc of this.bunchingIncidents(t)) active.set(inc.incident_id, inc)
    for (const [id, inc] of active) {
      const action = this.incidents.has(id) ? 'update' : 'open'
      this.incidents.set(id, inc)
      this.emit({ type: 'incident', stream_time: iso(t), action, incident: inc })
    }
    for (const [id, inc] of this.incidents) {
      if (active.has(id)) continue
      this.incidents.delete(id)
      this.closedIncidents.unshift(inc)
      if (this.closedIncidents.length > 200) this.closedIncidents.length = 200
      this.emit({ type: 'incident', stream_time: iso(t), action: 'close', incident: inc })
    }
  }

  private predictionOut(p: OpenPrediction): PredictionOut {
    const v = p.vehicle
    return {
      prediction_id: p.id,
      tr_id: v.trId,
      unit_id: v.unitId,
      route_id: v.route?.route.route_id ?? null,
      target_stop_id: p.stop.stopId,
      target_stop_name: p.stop.stop.name,
      planned_at: iso(p.stop.plan),
      issued_at: iso(p.issuedAt),
      lead_s: Math.round((p.stop.plan - p.issuedAt) / 1000),
      pred_delay_s: round1(p.pred),
      p10: round1(p.p10),
      p50: round1(p.pred),
      p90: round1(p.p90),
      p_late: Math.round(p.pLate * 1000) / 1000,
      risk: p.risk,
      model_version: 'mock',
      source: 'model',
      status: 'open',
      actual_delay_s: null,
      abs_error_s: null,
      closed_at: null,
    }
  }

  private delayIncident(v: SimVehicle, p: OpenPrediction, t: number): IncidentOut {
    const pos = this.position(v, t)
    const ds = this.delayState(v, t)
    const route = v.route?.route
    return {
      incident_id: `inc-${v.unitId}-${this.epoch}`,
      kind: 'delay',
      tr_id: v.trId,
      unit_id: v.unitId,
      route_id: route?.route_id ?? null,
      route_name: route?.name ?? null,
      risk: p.risk,
      target_stop: {
        stop_id: p.stop.stopId,
        name: p.stop.stop.name,
        lat: p.stop.stop.lat,
        lon: p.stop.stop.lon,
        planned_at: iso(p.stop.plan),
      },
      pred_delay_s: round1(p.pred),
      p10: round1(p.p10),
      p90: round1(p.p90),
      p_late: Math.round(p.pLate * 1000) / 1000,
      cause: p.cause,
      segment: this.segment(v, ds.lastPassed, p.stop),
      issued_at: iso(p.issuedAt),
      time_to_event_s: Math.round((p.stop.plan - t) / 1000),
      vehicle: {
        lat: pos.lat,
        lon: pos.lon,
        course_deg: pos.course,
        speed_kmh: pos.speed,
        current_delay_s: round1(ds.currentDelayS),
      },
      related_tr_id: null,
    }
  }

  /** Сбивка: на общей ближайшей остановке ожидаемый интервал меньше половины планового и меньше 5 мин. */
  private bunchingIncidents(t: number): IncidentOut[] {
    const byRoute = new Map<RouteModel, { v: SimVehicle; ds: DelayState }[]>()
    for (const v of this.vehicles) {
      if (!v.route || !v.stops.length) continue
      if (this.position(v, t).status === 'offline') continue
      const list = byRoute.get(v.route) ?? []
      list.push({ v, ds: this.delayState(v, t) })
      byRoute.set(v.route, list)
    }
    const out: IncidentOut[] = []
    const seen = new Set<string>()
    for (const [route, list] of byRoute) {
      const byStop = new Map<string, { v: SimVehicle; ds: DelayState; stop: StopEvent; est: number }[]>()
      for (const item of list) {
        const upcoming = item.v.stops.filter((s) => s.plan + item.ds.currentDelayS * 1000 > t).slice(0, 3)
        for (const stop of upcoming) {
          const est = stop.plan + item.ds.currentDelayS * 1000
          if (est - t > 20 * 60_000) continue
          const arr = byStop.get(stop.stop.stop_key) ?? []
          arr.push({ ...item, stop, est })
          byStop.set(stop.stop.stop_key, arr)
        }
      }
      for (const arr of byStop.values()) {
        arr.sort((a, b) => a.est - b.est)
        for (let i = 1; i < arr.length; i += 1) {
          const lead = arr[i - 1]
          const follow = arr[i]
          if (lead.v === follow.v) continue
          const headway = (follow.est - lead.est) / 1000
          const planned = Math.abs(follow.stop.plan - lead.stop.plan) / 1000
          if (planned < 120 || headway >= 0.5 * planned || headway >= 300) continue
          const id = `bun-${follow.v.unitId}-${lead.v.unitId}-${this.epoch}`
          if (seen.has(id)) continue
          seen.add(id)
          const pos = this.position(follow.v, t)
          const holdMin = Math.max(1, Math.round((planned / 2 - headway) / 60))
          const cause: Cause = {
            code: 'bunching',
            text: `${CAUSES.bunching.text}: ТС ${lead.v.trId} впереди в ${Math.round(headway / 60)} мин при плане ${Math.round(planned / 60)} мин`,
            recommendation: `Придержать ТС ${follow.v.trId} на ост. ${follow.stop.stop.name} на ~${holdMin} мин, чтобы выровнять интервал; при росте разрыва сзади — выпуск резерва`,
            factors: [
              {
                feature: 'headway_s',
                label: 'Интервал до впереди идущего ТС',
                contribution_s: Math.round(headway),
              },
              {
                feature: 'planned_headway_s',
                label: 'Плановый интервал',
                contribution_s: Math.round(planned),
              },
              {
                feature: 'leader_delay_s',
                label: `Опоздание ТС ${lead.v.trId} впереди`,
                contribution_s: round1(lead.ds.currentDelayS),
              },
            ],
          }
          const risk: Risk = headway < 0.25 * planned ? 'red' : 'yellow'
          out.push({
            incident_id: id,
            kind: 'bunching',
            tr_id: follow.v.trId,
            unit_id: follow.v.unitId,
            route_id: route.route.route_id,
            route_name: route.route.name,
            risk,
            target_stop: {
              stop_id: follow.stop.stopId,
              name: follow.stop.stop.name,
              lat: follow.stop.stop.lat,
              lon: follow.stop.stop.lon,
              planned_at: iso(follow.stop.plan),
            },
            pred_delay_s: round1(follow.ds.currentDelayS),
            p10: null,
            p90: null,
            p_late: null,
            cause,
            segment: this.segment(follow.v, follow.ds.lastPassed, follow.stop),
            issued_at: iso(t),
            time_to_event_s: Math.round((follow.est - t) / 1000),
            vehicle: {
              lat: pos.lat,
              lon: pos.lon,
              course_deg: pos.course,
              speed_kmh: pos.speed,
              current_delay_s: round1(follow.ds.currentDelayS),
            },
            related_tr_id: lead.v.trId,
          })
        }
      }
    }
    return out
  }

  // ------------------------------------------------------------------ REST

  vehicleOut(v: SimVehicle, t = this.now): VehicleOut {
    const pos = this.position(v, t)
    const ds = v.route ? this.delayState(v, t) : null
    const p = this.currentPrediction(v)
    let incidentId: string | null = null
    for (const inc of this.incidents.values()) {
      if (inc.unit_id === v.unitId && (incidentId === null || inc.kind === 'delay'))
        incidentId = inc.incident_id
    }
    const risk: Risk | null = v.route && v.stops.length ? (p?.risk ?? 'unknown') : 'unknown'
    return {
      unit_id: v.unitId,
      tr_id: v.trId,
      status: pos.status,
      connected: pos.status !== 'offline',
      lat: pos.lat === null ? null : Math.round(pos.lat * 1e6) / 1e6,
      lon: pos.lon === null ? null : Math.round(pos.lon * 1e6) / 1e6,
      valid: pos.valid,
      speed_kmh: pos.speed,
      course_deg: pos.course,
      event_time: pos.eventTime === null ? null : iso(Math.round(pos.eventTime / 1000) * 1000),
      received_at: new Date().toISOString(),
      age_s: pos.eventTime === null ? null : Math.max(0, Math.round((t - pos.eventTime) / 1000)),
      packets: pos.packets,
      reconnects: 0,
      route_id: v.route?.route.route_id ?? null,
      risk,
      current_delay_s: ds && ds.lastPassed ? round1(ds.currentDelayS) : null,
      pred_delay_s: p ? round1(p.pred) : null,
      p_late: p ? Math.round(p.pLate * 1000) / 1000 : null,
      next_stop:
        ds && ds.next
          ? {
              stop_id: ds.next.stopId,
              stop_key: ds.next.stop.stop_key,
              name: ds.next.stop.name,
              planned_at: iso(ds.next.plan),
            }
          : null,
      incident_id: incidentId,
    }
  }

  vehiclesOut(): VehicleOut[] {
    return this.vehicles.map((v) => this.vehicleOut(v))
  }

  vehicleMessage(type: 'snapshot' | 'delta'): WsMessage {
    return {
      type,
      stream_time: iso(this.now),
      server_time: new Date().toISOString(),
      version: Math.round(this.now / 1000),
      degraded: this.degraded,
      vehicles: this.vehiclesOut(),
    }
  }

  vehicleList(): VehicleListOut {
    const vehicles = this.vehiclesOut()
    const counts: Record<LinkStatus, number> = { online: 0, stale: 0, offline: 0 }
    vehicles.forEach((v) => {
      counts[v.status] += 1
    })
    return {
      count: vehicles.length,
      server_time: new Date().toISOString(),
      stream_time: iso(this.now),
      status_counts: counts,
      degraded: this.degraded,
      synced_at: new Date().toISOString(),
      vehicles,
    }
  }

  incidentList(status: 'active' | 'all', limit = 200): IncidentOut[] {
    const active = [...this.incidents.values()].sort(compareIncidents)
    const items = status === 'all' ? [...active, ...this.closedIncidents] : active
    return items.slice(0, limit)
  }

  incidentDetail(id: string): IncidentDetailOut | null {
    const inc = this.incidents.get(id) ?? this.closedIncidents.find((i) => i.incident_id === id)
    if (!inc) return null
    const v = this.vehicles.find((x) => x.unitId === inc.unit_id)
    if (!v) return { ...inc, history: [], forecast: [] }
    const t = this.now
    const ds = this.delayState(v, t)
    const history: DelayPoint[] = []
    for (const s of v.stops) {
      if (s.fact !== null && s.fact <= t && s.fact >= t - 30 * 60_000) {
        history.push({ t: iso(s.fact), delay_s: round1((s.fact - s.plan) / 1000) })
      }
    }
    history.sort((a, b) => Date.parse(a.t) - Date.parse(b.t))
    history.push({ t: iso(t), delay_s: round1(ds.currentDelayS) })
    const forecast: ForecastPoint[] = []
    for (const s of v.stops) {
      if (s.plan <= t || s.plan > t + WINDOW_TO_MS) continue
      if (s.fact !== null && s.fact <= t) continue
      const f = this.forecast(v, s, t, ds.currentDelayS)
      forecast.push({
        stop_id: s.stopId,
        name: s.stop.name,
        planned_at: iso(s.plan),
        pred_delay_s: round1(f.pred),
        p10: round1(f.p10),
        p90: round1(f.p90),
      })
    }
    return { ...inc, history, forecast }
  }

  alertList(options: { since?: number | null; level?: string | null; limit?: number }): AlertOut[] {
    return this.alerts
      .filter((a) => (options.since ? Date.parse(a.issued_at) >= options.since : true))
      .filter((a) => (options.level ? a.level === options.level : true))
      .slice(0, options.limit ?? 200)
  }

  predictionList(status: 'open' | 'closed' | null, trId: number | null, limit = 100): PredictionOut[] {
    const strip = ({ baseline_delay_s: _b, ...rest }: ClosedPrediction): PredictionOut => {
      void _b
      return rest
    }
    const closed = [...this.seeds, ...this.closed]
      .sort((a, b) => Date.parse(b.closed_at ?? '') - Date.parse(a.closed_at ?? ''))
      .map(strip)
    const open = [...this.open.values()].map((p) => this.predictionOut(p))
    const items = status === 'open' ? open : status === 'closed' ? closed : [...open, ...closed]
    return items.filter((p) => (trId === null ? true : p.tr_id === trId)).slice(0, limit)
  }

  horizon(): HorizonOut {
    const all = [...this.seeds, ...this.closed]
    const leadHist: LeadBucket[] = []
    for (let from = 0; from < 20 * 60; from += 60) leadHist.push({ from_s: from, to_s: from + 60, count: 0 })
    const byHour = new Map<number, { err: number; base: number; n: number }>()
    let err = 0
    let base = 0
    let late = 0
    let warned = 0
    for (const p of all) {
      // корзины (from, to]: заблаговременность ровно 15 мин — в последней корзине окна 10–15
      const bucket = leadHist[Math.min(leadHist.length - 1, Math.max(0, Math.ceil(p.lead_s / 60) - 1))]
      bucket.count += 1
      const actual = p.actual_delay_s ?? 0
      const e = Math.abs(actual - p.pred_delay_s)
      const b = Math.abs(actual - p.baseline_delay_s)
      err += e
      base += b
      const hour = new Date(p.planned_at).getUTCHours()
      const h = byHour.get(hour) ?? { err: 0, base: 0, n: 0 }
      h.err += e
      h.base += b
      h.n += 1
      byHour.set(hour, h)
      if (actual > LATE_S) {
        late += 1
        if (p.risk === 'red' || p.risk === 'yellow') warned += 1
      }
    }
    const maeByHour: MaeByHour[] = [...byHour.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([hour, h]) => ({ hour, mae_s: round1(h.err / h.n), baseline_s: round1(h.base / h.n), n: h.n }))
    const retroactive = this.alerts.filter((a) => Date.parse(a.issued_at) > Date.parse(a.planned_at)).length
    return {
      closed: all.length,
      online_mae_s: all.length ? round1(err / all.length) : null,
      baseline_mae_s: all.length ? round1(base / all.length) : null,
      warned_share: late ? Math.round((warned / late) * 1000) / 1000 : null,
      retroactive,
      lead_hist: leadHist,
      mae_by_hour: maeByHour,
    }
  }

  perf(): PerfOut {
    const online = this.vehiclesOut().filter((v) => v.status === 'online').length
    const wave = (k: number) => 0.5 + 0.5 * Math.sin(this.now / 7000 + k)
    return {
      ingest_pps: round1(online * 0.085 * (0.9 + 0.2 * wave(1))),
      e2e_p95_s: Math.round((0.22 + 0.12 * wave(2)) * 1000) / 1000,
      inference_p95_ms: round1(14 + 9 * wave(3)),
      tick_p95_ms: round1(38 + 22 * wave(4)),
      consumer_lag: Math.round(3 * wave(5)),
      vehicles_online: online,
      deps: {
        redis: this.degraded ? 'down' : 'up',
        postgres: 'up',
        ml: 'up',
        ingest: 'up',
        predictor: this.degraded ? 'degraded' : 'up',
      },
    }
  }

  stringline(routeId: string, from: number, to: number): StringlineOut | null {
    const model = this.routeModels.get(routeId)
    if (!model) return null
    const t = this.now
    const trips: StringlineTrip[] = []
    for (const v of this.vehicles) {
      if (v.route !== model || !v.stops.length) continue
      const ds = this.delayState(v, t)
      const planned = v.stops
        .filter((s) => s.plan >= from && s.plan <= to)
        .map((s) => ({ t: iso(s.plan), seq: s.seq }))
      const actual = v.stops
        .filter((s) => s.fact !== null && s.fact <= t && s.fact >= from && s.fact <= to)
        .map((s) => ({ t: iso(s.fact ?? 0), seq: s.seq }))
      const forecast = v.stops
        .filter((s) => s.plan > t && s.plan <= t + WINDOW_TO_MS && (s.fact === null || s.fact > t))
        .map((s) => {
          const f = this.forecast(v, s, t, ds.currentDelayS)
          const arrival = Math.max(t, s.plan + f.pred * 1000)
          return {
            t: iso(arrival),
            seq: s.seq,
            p10: iso(Math.max(t, s.plan + f.p10 * 1000)),
            p90: iso(Math.max(t, s.plan + f.p90 * 1000)),
          }
        })
      if (planned.length || actual.length || forecast.length)
        trips.push({ tr_id: v.trId, planned, actual, forecast })
    }
    return {
      route_id: routeId,
      stops: model.stops.map((s) => ({ stop_key: s.stop_key, name: s.name, seq: s.seq })),
      trips,
      stream_time: iso(t),
    }
  }

  whatIf(request: WhatIfRequest): WhatIfOut | null {
    const model = this.routeModels.get(request.route_id)
    if (!model) return null
    const at = request.at ? Date.parse(request.at) : this.now
    const vehicles: EngineVehicle[] = []
    for (const v of this.vehicles) {
      if (v.route !== model || !v.stops.length) continue
      const ds = this.delayState(v, at)
      const arrivals: EngineArrival[] = v.stops
        .filter((s) => s.fact === null || s.fact > at)
        .map((s) => ({
          stopKey: s.stop.stop_key,
          seq: s.seq,
          plan: s.plan,
          est: Math.max(at, s.plan + ds.currentDelayS * 1000),
        }))
      vehicles.push({ trId: v.trId, arrivals })
    }
    return runWhatIf({
      route: { stopKeys: model.stops.map((s) => s.stop_key), runS: model.runS },
      vehicles,
      at,
      request,
    })
  }

  health() {
    return {
      status: this.degraded ? ('degraded' as const) : ('ok' as const),
      service: 'api (mock)',
      version: 'mock',
      uptime_s: Math.round((this.now - this.t0) / 1000),
      dependencies: {
        redis: this.degraded
          ? { state: 'down' as const, outages: 1, error: 'connection refused (mock chaos)' }
          : { state: 'up' as const, outages: 0 },
        postgres: { state: 'up' as const, outages: 0 },
      },
    }
  }
}
