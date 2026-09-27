/** Типы фикстур `scripts/make_dashboard_fixtures.py` (см. docstring скрипта). */
import type { PredictionOut, RouteOut } from '../api/types'

/** [t, с от момента; lon; lat; курс; скорость км/ч] */
export type TrackPoint = [number, number, number, number, number]
/** [stop_id; seq в маршруте; план, с от момента; факт, с от момента | null] */
export type StopRow = [number, number, number, number | null]

export interface SourceFixture {
  tr_id: number
  route_id: string | null
  track: TrackPoint[]
  stops: StopRow[]
}

export interface VehicleFixture {
  unit_id: number
  tr_id: number
  /** tr_id реального ТС-источника трека и расписания. */
  source: number
  /** Сдвиг трека: положение в момент t = положение источника в t + d_track (с). */
  d_track: number
  /** Сдвиг плана: план = план источника − d_plan (с). */
  d_plan: number
  clone: boolean
}

export interface WorldFixture {
  meta: { source: string; moment: string; window_s: number; history_s: number; note: string }
  routes: RouteOut[]
  sources: SourceFixture[]
  vehicles: VehicleFixture[]
}

export type SeedPrediction = PredictionOut & { baseline_delay_s: number }

export interface HonestyFixture {
  meta: { source: string; model_source?: string }
  closed: SeedPrediction[]
}
