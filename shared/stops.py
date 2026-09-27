"""Детектор прохождения плановых остановок по GPS (последовательный map matching).

Геометрия. Для остановки считаем расстояние от её точки до каждого отрезка между соседними GPS-точками
(проекция на отрезок, equirectangular в метрах; отрезки с разрывом > `max_gap_s` не интерполируются).
«Визит» — максимальная серия отрезков ближе `radius_m` (120 м), не прерванная точкой дальше него.
Визит засчитывается, если ТС подошло ближе `match_m` (60 м). Широкий радиус визита объединяет
подход, стоянку и отъезд в один визит, поэтому минимум ищется по всему проезду мимо остановки.

Момент прохода внутри визита (проверено по time_fact_begin на train/test):

* обычная остановка — момент минимального расстояния (интерполяция по отрезку);
* конечная прибытия (после неё плановый разрыв > `terminal_gap_s`) — первый момент, когда ТС подошло
  на `dmin + near_min_m`, т.е. прибытие (отстой после прибытия не сдвигает факт);
* конечная отправления (перед ней плановый разрыв) — последний такой момент в последнем визите до
  прохода следующей остановки, т.е. отправление после отстоя.

Момент прохода берётся только в окне [plan − window_before_s, plan + window_after_s] и не раньше
прохода предыдущей сопоставленной остановки.

Сопоставление событийное и каузальное. Состояние: курсор (момент и отрезок последнего прохода) и список
ещё не решённых остановок в порядке плана. Кандидаты — первые `lookahead` нерешённых остановок и все
остановки с тем же плановым временем (порядок остановок с равным планом неизвестен). Для кандидата
берётся первый засчитанный визит после курсора. Следующее решение — самое раннее из двух событий:

* подтверждение визита кандидата (первая точка вне радиуса или первая точка после закрытия окна):
  остановка сопоставлена, нерешённые остановки с более ранним планом — пропущены;
* закрытие окна первой нерешённой остановки без визитов (первая точка после plan + window_after_s):
  остановка не сопоставлена.

Каждое решение принимается в момент реальной GPS-точки и использует только точки до него. Поэтому для
любого T прогон на точках с time ≤ T даёт те же решения для всех остановок с `confirmed_at ≤ T`.
`confirmed_at` — момент решения: для сопоставленной остановки это подтверждение визита (или более
поздний момент предыдущего решения), для несопоставленной — момент, когда пропуск стал известен.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8
_M_PER_DEG = np.pi / 180.0 * EARTH_RADIUS_M

MODE_MIN, MODE_ARRIVE, MODE_DEPART = 0, 1, 2

RESULT_COLUMNS = ["tt_action_item_id", "time_begin", "pass_time", "dist_m", "confirmed_at"]


@dataclass(frozen=True)
class DetectorParams:
    window_before_s: float = 600.0  # проход не раньше plan − 10 мин
    window_after_s: float = 900.0  # проход не позже plan + 15 мин
    radius_m: float = 120.0  # границы визита и его подтверждение (выход из радиуса)
    match_m: float = 60.0  # визит засчитывается, если ТС подошло к остановке ближе этого
    near_min_m: float = 10.0  # допуск к минимуму расстояния для конечных (прибытие/отправление)
    terminal_gap_s: float = 300.0  # плановый разрыв, после/перед которым остановка считается конечной
    max_gap_s: float = 300.0  # отрезки с большим разрывом по времени не интерполируем
    lookahead: int = 2  # сколько нерешённых остановок одновременно ждём (плюс равные по плану)
    depart_lookahead: int = 3  # сколько следующих остановок смотрим, решая конечную отправления
    context_s: float = 600.0  # запас трека перед окном (начало визита при долгой стоянке)


DEFAULT_PARAMS = DetectorParams()

_EVENT, _PENDING, _NONE = 0, 1, 2

# событие визита: (pass_time, dist_m, confirm_time, pass_seg)
_Event = tuple[float, float, float, int]


def _stop_modes(plan: np.ndarray, gap_s: float) -> np.ndarray:
    """Тип остановки по плану: промежуточная / конечная прибытия / конечная отправления."""
    uniq = np.unique(plan)
    pos = np.searchsorted(uniq, plan)
    prev_gap = np.where(pos > 0, plan - uniq[np.maximum(pos - 1, 0)], np.inf)
    next_gap = np.where(pos < len(uniq) - 1, uniq[np.minimum(pos + 1, len(uniq) - 1)] - plan, np.inf)
    arrive = next_gap > gap_s
    depart = prev_gap > gap_s
    modes = np.full(len(plan), MODE_MIN, dtype=np.int8)
    modes[arrive & ~depart] = MODE_ARRIVE
    modes[depart & ~arrive] = MODE_DEPART
    return modes


class _Matcher:
    def __init__(
        self,
        t: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        plan: np.ndarray,
        slon: np.ndarray,
        slat: np.ndarray,
        params: DetectorParams,
    ) -> None:
        self.t, self.lon, self.lat = t, lon, lat
        self.plan, self.slon, self.slat = plan, slon, slat
        self.p = params
        self.pending_from = np.inf  # ранний возможный проход незавершённого визита (см. visits)
        self.modes = _stop_modes(plan, params.terminal_gap_s)
        self.coslat = np.cos(np.deg2rad(slat))

    # --- геометрия -------------------------------------------------------------------------------
    def _xy(self, k: int, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        x = (self.lon[a:b] - self.slon[k]) * (self.coslat[k] * _M_PER_DEG)
        y = (self.lat[a:b] - self.slat[k]) * _M_PER_DEG
        return x, y

    def _segments(self, k: int, a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Точки [a, b): расстояния точек, отрезков и время ближайшей точки отрезка."""
        x, y = self._xy(k, a, b)
        t = self.t[a:b]
        d = np.hypot(x, y)
        dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
        l2 = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(l2 > 0, -(x[:-1] * dx + y[:-1] * dy) / l2, 0.0)
        u = np.clip(u, 0.0, 1.0)
        # через большой разрыв по времени не интерполируем: берём ближайший конец отрезка
        gap = dt > self.p.max_gap_s
        u = np.where(gap, (d[1:] < d[:-1]).astype(np.float64), u)
        sd = np.hypot(x[:-1] + u * dx, y[:-1] + u * dy)
        st = t[:-1] + u * dt
        return d, sd, st

    def _pass_in_run(self, k: int, sd: np.ndarray) -> int:
        mode = self.modes[k]
        if mode == MODE_MIN:
            return int(np.argmin(sd))
        close = np.flatnonzero(sd < sd.min() + self.p.near_min_m)
        return int(close[0] if mode == MODE_ARRIVE else close[-1])

    # --- поиск визита ----------------------------------------------------------------------------
    def visits(self, k: int, c: int, ct: float, first_only: bool = True) -> tuple[list[_Event], int]:
        """Подходящие визиты к остановке k после курсора (c — отрезок, ct — момент последнего прохода).

        Рассматриваются точки до первой точки после закрытия окна (plan + window_after_s) включительно.
        Момент прохода выбирается среди отрезков визита с временем в [max(plan − before, ct), plan + after].
        Визит подтверждается первой точкой вне радиуса или первой точкой после закрытия окна.
        Возвращает (события (pass_time, dist, confirm_time, pass_seg), статус хвоста: _PENDING — последний
        визит ещё не завершён, иначе _NONE).
        """
        p, t, n = self.p, self.t, len(self.t)
        lo = self.plan[k] - p.window_before_s
        hi = self.plan[k] + p.window_after_s
        a = max(c, int(np.searchsorted(t, lo - p.context_s)) - 1, 0)
        m = int(np.searchsorted(t, hi, side="right"))  # первая точка после закрытия окна
        b = min(n, m + 1)
        found: list[_Event] = []
        if b - a < 2:
            return found, _NONE
        d, sd, st = self._segments(k, a, b)
        r = p.radius_m
        near = sd < r
        inside = d < r
        nseg = len(sd)
        cont_prev = np.zeros(nseg, dtype=bool)
        cont_prev[1:] = inside[1:-1]  # отрезок j продолжает j−1, если общая точка ближе порога
        starts = np.flatnonzero(near & ~cont_prev)
        cont_next = np.zeros(nseg, dtype=bool)
        cont_next[:-1] = inside[1:-1]
        ends = np.flatnonzero(near & ~cont_next)
        min_pass = max(lo, ct)
        for s, e in zip(starts, ends, strict=True):
            if e == nseg - 1 and inside[nseg]:
                if m >= n:
                    # визит ещё идёт, а окно не закрыто; его проход не может быть раньше этого момента
                    self.pending_from = max(float(st[s]), min_pass)
                    return found, _PENDING
                q = m  # ТС ещё у остановки, но окно уже закрыто
            else:
                q = a + e + 1  # первая точка вне радиуса
            seg = np.arange(s, e + 1)
            seg = seg[(st[seg] >= min_pass) & (st[seg] <= hi)]
            if len(seg) == 0 or sd[seg].min() >= p.match_m:
                continue
            j = seg[self._pass_in_run(k, sd[seg])]
            found.append((float(st[j]), float(sd[j]), float(t[q]), a + int(j)))
            if first_only:
                break
        return found, _NONE

    def query(self, k: int, c: int, ct: float) -> tuple[int, _Event | None]:
        """Первый подходящий визит: (_EVENT, событие) | (_PENDING, None) | (_NONE, None)."""
        found, status = self.visits(k, c, ct, first_only=True)
        return (_EVENT, found[0]) if found else (status, None)

    def timeout(self, k: int) -> float:
        """Момент первой точки после закрытия окна остановки k (inf, если такой ещё нет)."""
        idx = int(np.searchsorted(self.t, self.plan[k] + self.p.window_after_s, side="right"))
        return float(self.t[idx]) if idx < len(self.t) else np.inf

    # --- основной цикл ---------------------------------------------------------------------------
    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ns = len(self.plan)
        pass_t = np.full(ns, np.nan)
        dist = np.full(ns, np.nan)
        conf = np.full(ns, np.nan)
        if ns == 0 or len(self.t) < 2:
            return pass_t, dist, conf
        horizon = self.p.window_before_s + self.p.window_after_s
        pending = list(range(ns))  # нерешённые остановки в порядке плана
        c, ct, now = 0, -np.inf, -np.inf
        cache: dict[int, tuple[int, _Event | None]] = {}

        def query(k: int) -> tuple[int, _Event | None]:
            if k not in cache:
                cache[k] = self.query(k, c, ct)
            return cache[k]

        while pending:
            first = pending[0]
            if self.modes[first] == MODE_DEPART and len(pending) > 1:
                # конечная отправления: ТС может несколько раз подъезжать к остановке во время отстоя.
                # Берём последний визит до прохода одной из следующих остановок (первое подтверждение
                # среди них) или, если их нет, до закрытия окна следующей остановки.
                # Момент решения — самое раннее наблюдаемое событие среди следующих остановок
                # (подтверждённый проход или закрытие окна непройденной следующей остановки);
                # проход конечной берём не позже прохода/закрытия, по которому принято решение.
                limit, base = np.inf, np.inf
                for nxt in pending[1 : 1 + self.p.depart_lookahead]:
                    status, ev = query(nxt)
                    if status == _EVENT and ev[2] < base:
                        limit, base = ev[0], ev[2]
                if query(pending[1])[0] == _NONE:
                    to = self.timeout(pending[1])
                    if to < base:
                        limit, base = to, to
                if not np.isfinite(base):
                    break  # решение зависит от ещё не поступивших точек
                found, status = self.visits(first, c, ct, first_only=False)
                if status == _PENDING and self.pending_from <= limit:
                    break  # незавершённый визит к конечной ещё может дать проход до limit
                found = [v for v in found if v[0] <= limit]
                now = max(now, base)
                if found:
                    pt, dm, cf, seg = found[-1]
                    now = max(now, cf)
                    pass_t[first], dist[first] = pt, dm
                    c, ct = seg, pt
                    cache.clear()
                conf[first] = now
                pending.pop(0)
                cache.pop(first, None)
                continue

            best_k, best = -1, None
            # кандидаты: первые `lookahead` нерешённых + все с тем же планом
            # (порядок остановок с равным планом неизвестен)
            last_plan = min(
                self.plan[pending[min(len(pending), self.p.lookahead) - 1]], self.plan[first] + horizon
            )
            for k in pending:
                if self.plan[k] > last_plan:
                    break
                status, ev = query(k)
                if status == _EVENT and (best is None or ev[2] < best[2]):
                    best_k, best = k, ev
            to = self.timeout(first) if cache[first][0] == _NONE else np.inf
            if best is not None and best[2] <= to:
                pt, dm, cf, seg = best
                now = max(now, cf)
                pass_t[best_k], dist[best_k], conf[best_k] = pt, dm, now
                plan_k = self.plan[best_k]
                keep = []
                for k in pending:
                    if k == best_k:
                        continue
                    if self.plan[k] < plan_k:
                        conf[k] = now  # пропущена: ТС уже прошло более позднюю остановку
                    else:
                        keep.append(k)
                pending = keep
                c, ct = seg, pt
                cache.clear()
            elif np.isfinite(to):
                now = max(now, to)
                conf[first] = now
                pending.pop(0)
                cache.pop(first, None)
            else:
                break  # решение зависит от ещё не поступивших точек
        return pass_t, dist, conf


def _prepare_track(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    tcol = "event_time" if "event_time" in track.columns else "time"
    tr = track[[tcol, "lon", "lat"]].dropna()
    times = pd.to_datetime(tr[tcol]).astype("datetime64[ns]").to_numpy().astype(np.int64)
    order = np.argsort(times, kind="stable")
    times = times[order]
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = times[1:] != times[:-1]
    idx = order[keep]
    times = times[keep]
    t0 = int(times[0]) if len(times) else 0
    t = (times - t0) / 1e9
    lon = tr["lon"].to_numpy(dtype=np.float64)[idx]
    lat = tr["lat"].to_numpy(dtype=np.float64)[idx]
    return t, lon, lat, t0


def _to_time(sec: np.ndarray, t0: int) -> np.ndarray:
    out = np.full(len(sec), np.datetime64("NaT", "ns"))
    ok = np.isfinite(sec)
    out[ok] = (t0 + np.round(sec[ok] * 1e9).astype(np.int64)).astype("datetime64[ns]")
    return out


def detect_passages(
    track: pd.DataFrame, stops: pd.DataFrame, params: DetectorParams = DEFAULT_PARAMS
) -> pd.DataFrame:
    """Моменты прохождения плановых остановок одного ТС.

    track: точки ТС — event_time (или time), lon, lat (прочие колонки игнорируются).
    stops: плановые остановки ТС — tt_action_item_id, time_begin, stop_lon, stop_lat.
    Результат (в порядке плана): tt_action_item_id, time_begin, pass_time, dist_m, confirmed_at.
    """
    st = stops[["tt_action_item_id", "time_begin", "stop_lon", "stop_lat"]].copy()
    st["time_begin"] = pd.to_datetime(st["time_begin"]).astype("datetime64[ns]")
    st = st.sort_values(["time_begin", "tt_action_item_id"], kind="stable").reset_index(drop=True)
    t, lon, lat, t0 = _prepare_track(track)
    plan = (st["time_begin"].to_numpy().astype(np.int64) - t0) / 1e9
    slon = st["stop_lon"].to_numpy(dtype=np.float64)
    slat = st["stop_lat"].to_numpy(dtype=np.float64)
    valid = np.isfinite(slon) & np.isfinite(slat)
    pass_t = np.full(len(st), np.nan)
    dist = np.full(len(st), np.nan)
    conf = np.full(len(st), np.nan)
    if valid.any():
        matcher = _Matcher(t, lon, lat, plan[valid], slon[valid], slat[valid], params)
        pass_t[valid], dist[valid], conf[valid] = matcher.run()
    return pd.DataFrame(
        {
            "tt_action_item_id": st["tt_action_item_id"].to_numpy(dtype=np.int64),
            "time_begin": st["time_begin"].to_numpy(),
            "pass_time": _to_time(pass_t, t0),
            "dist_m": dist,
            "confirmed_at": _to_time(conf, t0),
        }
    )


def detect_all(
    traffic: pd.DataFrame, schedule: pd.DataFrame, params: DetectorParams = DEFAULT_PARAMS
) -> pd.DataFrame:
    """detect_passages для всех ТС расписания; добавляет колонку tr_id."""
    groups = dict(tuple(traffic.groupby("tr_id", sort=False)))
    empty = traffic.iloc[:0]
    parts = []
    for tr_id, stops in schedule.groupby("tr_id", sort=True):
        res = detect_passages(groups.get(tr_id, empty), stops, params)
        res.insert(0, "tr_id", np.int64(tr_id))
        parts.append(res)
    if not parts:
        cols = ["tr_id", *RESULT_COLUMNS]
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    return pd.concat(parts, ignore_index=True)
