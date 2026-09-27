"""Маршруты из планового расписания (``docs/api-contract.md`` §1) и их геометрия по реальным GPS-трекам.

В расписании нет номера маршрута, поэтому маршрут выводится из плана:

* **место остановки** — ``stop_key``: координаты точки остановки, округлённые до 5 знаков (~1 м), строкой
  ``"lon,lat"`` (``"37.43071,55.80401"``). Разные плановые прибытия (``tt_action_item_id``) на одну и ту же
  остановку дают один ``stop_key``;
* **маршрут** — ТС с одинаковым набором мест остановок (с точностью до порядка обхода) объединяются в маршрут
  ``R1``, ``R2``, …; номера выдаются по возрастанию наименьшего ``tr_id`` группы, поэтому результат
  детерминирован и не зависит от порядка строк;
* **направления** — порядок остановок берётся из плановой последовательности одного полного рейса ТС группы
  с наименьшим ``tr_id`` (рейс — участок плана между отстоями: разрыв плана > 5 мин), самого частого.
  Если рейс не замкнут, он — прямое направление (``direction = 0``), а обратное (``direction = 1``) — самый
  частый рейс, который начинается у его конечной и заканчивается у его начала (в пределах
  :data:`TERMINAL_M`: платформы конечной бывают разными). Замкнутый рейс (начало и конец ближе
  :data:`LOOP_M`) — это «туда и обратно» без отстоя на дальней конечной либо кольцо: «туда и обратно»
  (большинство остановок второй половины рядом с остановками первой — противоположная сторона улицы)
  делится на дальней от начала остановке на два направления, кольцо остаётся одним. Прямое и обратное —
  **разные линии**, поэтому линия не смешивает «туда» и «обратно»;
* **последовательность остановок** (``stops``, ``seq`` для «нитки») — прямое направление, за ним обратное
  (общая конечная — один раз): круг «туда и обратно»;
* **имя** — «Маршрут R1: <первая конечная> — <вторая конечная>» по адресам остановок (``building_address``):
  вторая конечная — последняя остановка прямого рейса, у кругового рейса — самая далёкая от первой;
* **линия направления** — по дорогам: участок между соседними остановками — реальный путь ТС по GPS между
  моментами их прохода (детектор :mod:`shared.stops`), выбирается проход, чья длина ближе всего к медиане
  качественных проходов (без разрывов GPS), упрощённый алгоритмом Дугласа—Пекера (5 м); если проходов нет —
  прямой отрезок. Поле ``line`` маршрута (совместимость с контрактом) — линия прямого направления.

Геометрия участков строится офлайн по истории (``python -m shared.routes build``: телеметрия и детектор train
и test, кэш — :data:`SEGMENTS_FILE` в репозитории) и не зависит от сплита: участок идентифицируется парой
``stop_key``. Честность прогнозов это не затрагивает: геометрия — только картинка карты. Для самих маршрутов
используются только плановые поля (``tr_id``, ``tt_action_item_id``, ``time_begin``, координаты, адрес) —
факт не нужен. Тот же код у фикстур дашборда (``scripts/make_dashboard_fixtures.py``), поэтому моки и живой
режим показывают одни и те же маршруты.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LAYOVER_GAP_S = 300.0
"""Плановый разрыв больше — отстой на конечной (граница рейса), как в ``shared.features.LAYOVER_S``."""
KEY_DIGITS = 5
"""Знаков после запятой в ``stop_key`` (~1 м)."""
MIN_TRIP_STOPS = 4
"""Рейсы короче (служебные подъезды) не задают последовательность маршрута."""
LOOP_M = 300.0
"""Рейс, чьи первая и последняя остановки ближе, — замкнутый."""
TERMINAL_M = 800.0
"""Обратный рейс начинается у конечной прямого (и заканчивается у его начала) в пределах этого расстояния."""
PAIRED_M = 300.0
"""Остановка обратного пути «напротив» остановки прямого — ближе этого."""
PAIRED_SHARE = 0.5
"""Замкнутый рейс — «туда и обратно», если столько остановок второй половины имеют пару в первой."""
ROUTE_COLORS = (
    "#4C8DF6",
    "#8B7CF6",
    "#2BB5C8",
    "#C07CF0",
    "#5E9FD8",
    "#7A8CFF",
    "#48A6A6",
    "#A58BFF",
    "#3E7CB1",
    "#9DA7FF",
    "#6FB7E0",
    "#B38CD9",
    "#5C7CFA",
)
"""Цвета линий маршрутов: сине-фиолетовая гамма, не пересекается с цветами риска."""

SEGMENTS_FILE = Path(__file__).resolve().parent.parent / "backend" / "assets" / "route_segments.json"
"""Кэш геометрии участков в репозитории (его читают predictor и фикстуры дашборда)."""
SEGMENTS_FORMAT = "foresight-route-segments/1"
SIMPLIFY_M = 5.0
"""Допуск упрощения Дугласа—Пекера, м."""
MAX_STEP_M = 350.0
"""Проход с шагом между соседними GPS-точками больше (разрыв сигнала) — не «качественный»."""
MAX_STEP_S = 60.0
"""... или с разрывом по времени больше."""
MAX_SEGMENT_S = 1800.0
"""Проходы соседних остановок дальше по времени (стоянка, сход с рейса) — не путь участка."""
MAX_DETOUR = 3.0
"""Путь длиннее прямого отрезка в столько раз (+ 300 м) — петля или объезд, не путь участка."""
MAX_SKIP = 3
"""Путь в обход непосещаемых остановок: не больше стольких пропущенных детектором остановок подряд."""
NEAR_M = 60.0
"""Запасной путь без детектора: визит к остановке — точки трека ближе этого."""
MAX_SPEED_MS = 25.0
"""Выброс GPS: до точки от последней нормальной пришлось бы ехать быстрее (90 км/ч) — точка отбрасывается."""
SPIKE_M = 80.0
"""Выброс-«шпилька»: точка уводит путь в сторону и сразу обратно (оба плеча длиннее) — отбрасывается."""
SPIKE_RATIO = 0.35
"""... если хорда соседей короче этой доли суммы плеч (разворот больше ~140°)."""
OSRM_URL = "https://router.project-osrm.org"
"""Маршрутизатор для участков без GPS-трека (только при сборке кэша; работе сервисов интернет не нужен)."""
OSRM_MAX_DETOUR = 3.0
"""Путь OSRM длиннее прямого отрезка в столько раз (+ 500 м) — объезд по односторонним улицам, не берётся."""
MATCH_RADIUS_M = 25.0
"""Погрешность GPS для привязки трека к дорогам (OSRM match), м."""
MATCH_MIN_CONFIDENCE = 0.2
"""Привязка с меньшей уверенностью OSRM не берётся — остаётся трек GPS."""
MATCH_END_M = 30.0
"""Привязанный путь участка должен начинаться и кончаться не дальше этого от его остановок, м (иначе он ушёл
на соседнюю проезжую часть — остановки окажутся в стороне от линии)."""
MATCH_STOP_RADIUS_M = 15.0
"""Погрешность для крайних точек трека — самих остановок: держит привязку у их стороны дороги."""
TRACK_MEAN_M = 25.0
TRACK_MAX_M = 70.0
"""Путь по дорогам принимается, если отходит от трека GPS в среднем не больше ``TRACK_MEAN_M`` и нигде не
больше ``TRACK_MAX_M`` (м) — иначе он срезал по дворам или ушёл на другую улицу."""
MATCH_MAX_POINTS = 10
"""Больше точек публичный OSRM в одном запросе match не принимает (TooBig)."""

_EARTH_M = 6_371_000.0
_M_PER_DEG = math.pi / 180.0 * 6_371_008.8


def stop_key(lon: float, lat: float) -> str:
    """Место остановки по координатам: ``"lon,lat"`` с 5 знаками после запятой.

    >>> stop_key(37.430707051, 55.8040083)
    '37.43071,55.80401'
    """
    return f"{round(float(lon), KEY_DIGITS):.{KEY_DIGITS}f},{round(float(lat), KEY_DIGITS):.{KEY_DIGITS}f}"


def stop_keys(lon: Iterable[float], lat: Iterable[float]) -> list[str]:
    """:func:`stop_key` для массивов координат."""
    return [stop_key(x, y) for x, y in zip(lon, lat, strict=True)]


def segment_key(a: str, b: str) -> str:
    """Ключ участка между местами остановок ``a`` → ``b`` (направленный)."""
    return f"{a}|{b}"


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Расстояние по большому кругу, м."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_M * math.asin(math.sqrt(min(a, 1.0)))


def _local_xy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Координаты в метрах (equirectangular вокруг средней широты) — для упрощения и длин."""
    coslat = math.cos(math.radians(float(np.mean(lat)))) if len(lat) else 1.0
    return np.column_stack([np.asarray(lon) * coslat * _M_PER_DEG, np.asarray(lat) * _M_PER_DEG])


def path_length_m(line: Sequence[Sequence[float]]) -> float:
    """Длина ломаной ``[[lon, lat], …]``, м."""
    if len(line) < 2:
        return 0.0
    arr = np.asarray(line, dtype=np.float64)
    xy = _local_xy(arr[:, 0], arr[:, 1])
    return float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))


def douglas_peucker(line: Sequence[Sequence[float]], eps_m: float = SIMPLIFY_M) -> list[list[float]]:
    """Упрощение ломаной ``[[lon, lat], …]`` алгоритмом Дугласа—Пекера с допуском ``eps_m`` метров.

    Концы сохраняются всегда; точка остаётся, если отклоняется от хорды своего участка больше ``eps_m``.
    """
    pts = [list(map(float, p)) for p in line]
    if len(pts) < 3:
        return pts
    arr = np.asarray(pts, dtype=np.float64)
    xy = _local_xy(arr[:, 0], arr[:, 1])
    keep = np.zeros(len(xy), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = xy[b] - xy[a]
        rel = xy[a + 1 : b] - xy[a]
        norm = float(np.hypot(*seg))
        if norm == 0.0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > eps_m:
            k = a + 1 + i
            keep[k] = True
            stack += [(a, k), (k, b)]
    return [pts[i] for i in np.flatnonzero(keep)]


def split_trips(plan_s: np.ndarray) -> list[slice]:
    """Рейсы ТС: участки плана (упорядоченного по времени) между разрывами больше :data:`LAYOVER_GAP_S`.

    Args:
        plan_s: плановые моменты остановок ТС, секунды, по возрастанию.

    Returns:
        Срезы рейсов в порядке плана.
    """
    n = len(plan_s)
    if n == 0:
        return []
    cuts = np.flatnonzero(np.diff(np.asarray(plan_s, dtype=np.float64)) > LAYOVER_GAP_S) + 1
    bounds = [0, *cuts.tolist(), n]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def key_distance_m(a: str, b: str) -> float:
    """Расстояние между местами остановок, м (одинаковые — 0; ключ не из координат — бесконечность)."""
    if a == b:
        return 0.0
    try:
        (lon1, lat1), (lon2, lat2) = _coords(a), _coords(b)
    except ValueError:
        return math.inf
    return haversine_m(lon1, lat1, lon2, lat2)


def split_loop(trip: Sequence[str]) -> int | None:
    """Где делить замкнутый рейс на «туда» и «обратно»: индекс дальней от начала остановки.

    Returns:
        Индекс или ``None`` — кольцо (остановки второй половины не идут «напротив» остановок первой) или
        места остановок не из координат.
    """
    if len(trip) < MIN_TRIP_STOPS:
        return None
    dist = [key_distance_m(trip[0], k) for k in trip]
    if not all(math.isfinite(d) for d in dist):
        return None
    far = int(np.argmax(dist))
    there, back = trip[1:far], trip[far + 1 : -1]
    if not there or not back:
        return None
    paired = sum(min(key_distance_m(k, o) for o in there) <= PAIRED_M for k in back)
    return far if paired >= PAIRED_SHARE * len(back) else None


def route_directions(keys: Sequence[str], plan_s: np.ndarray) -> list[list[str]]:
    """Направления маршрута по плану одного ТС: прямое и (если есть) обратное, в порядке обхода.

    Основной рейс — самый частый не короче :data:`MIN_TRIP_STOPS` (при равенстве — встретившийся раньше).
    Незамкнутый основной рейс — прямое направление, обратное — самый частый рейс, начинающийся у его конечной
    и заканчивающийся у его начала (:data:`TERMINAL_M`). Замкнутый (:data:`LOOP_M`) рейс «туда и обратно»
    делится на дальней остановке (:func:`split_loop`): его половины — два направления с общей конечной;
    кольцо — одно направление.

    Args:
        keys: ``stop_key`` остановок ТС в порядке плана.
        plan_s: их плановые моменты, секунды.

    Returns:
        Одно или два направления — списки ``stop_key`` в порядке обхода (пусто, если у ТС нет остановок).
    """
    trips = [tuple(keys[s]) for s in split_trips(plan_s)]
    long = [t for t in trips if len(t) >= MIN_TRIP_STOPS] or trips
    if not long:
        return []
    counts = Counter(long)
    main, _ = counts.most_common(1)[0]
    if key_distance_m(main[0], main[-1]) <= LOOP_M:
        far = split_loop(main)
        return [list(main)] if far is None else [list(main[: far + 1]), list(main[far:])]
    back = [
        t
        for t, _ in counts.most_common()
        if t != main
        and key_distance_m(t[0], main[-1]) <= TERMINAL_M
        and key_distance_m(t[-1], main[0]) <= TERMINAL_M
    ]
    return [list(main), list(back[0])] if back else [list(main)]


def join_directions(dirs: Sequence[Sequence[str]]) -> list[str]:
    """Последовательность маршрута из направлений: подряд, общая конечная на стыке — один раз."""
    out: list[str] = []
    for d in dirs:
        out += list(d[1:]) if out and d and d[0] == out[-1] else list(d)
    return out


def route_sequence(keys: Sequence[str], plan_s: np.ndarray) -> list[str]:
    """Последовательность мест остановок маршрута (ось «нитки»): прямое направление, затем обратное
    (:func:`join_directions`).

    Args:
        keys: ``stop_key`` остановок ТС в порядке плана.
        plan_s: их плановые моменты, секунды.

    Returns:
        ``stop_key`` в порядке обхода (пусто, если у ТС нет остановок).
    """
    return join_directions(route_directions(keys, plan_s))


def assign_seq(keys: Sequence[str], plan_s: np.ndarray, sequence: Sequence[str]) -> np.ndarray:
    """Номер остановки в последовательности маршрута для каждой плановой остановки ТС.

    Сопоставление последовательное внутри рейса: следующее вхождение ``stop_key`` после текущей позиции
    (круговые маршруты проходят одно место дважды). Остановки вне последовательности — ``-1``.

    Args:
        keys: ``stop_key`` остановок ТС в порядке плана.
        plan_s: их плановые моменты, секунды.
        sequence: последовательность маршрута (:func:`route_sequence`).

    Returns:
        Массив ``int64`` длины ``len(keys)``.
    """
    out = np.full(len(keys), -1, dtype=np.int64)
    positions: dict[str, list[int]] = {}
    for j, key in enumerate(sequence):
        positions.setdefault(key, []).append(j)
    for trip in split_trips(plan_s):
        pos: int | None = None
        for i in range(trip.start, trip.stop):
            candidates = positions.get(keys[i])
            if not candidates:
                continue
            if pos is None:
                pos = candidates[0]
            else:
                ahead = [j for j in candidates if j > pos]
                pos = ahead[0] if ahead else candidates[0]
            out[i] = pos
    return out


# ---- geometry of the segments ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentPass:
    """Один проход участка по GPS: путь и его качество."""

    line: tuple[tuple[float, float], ...]
    length_m: float
    max_step_m: float
    max_step_s: float
    straight_m: float

    @property
    def good(self) -> bool:
        """Без разрывов сигнала и без петель."""
        return (
            self.max_step_m <= MAX_STEP_M
            and self.max_step_s <= MAX_STEP_S
            and self.length_m <= MAX_DETOUR * self.straight_m + 300.0
        )


def _interp(ts: np.ndarray, v: np.ndarray, t: float) -> float:
    return float(np.interp(t, ts, v))


def despike(lon: np.ndarray, lat: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Маска точек пути без выбросов GPS; концы (позиции у остановок) остаются всегда.

    Точка отбрасывается, если от последней оставленной до неё скорость выше :data:`MAX_SPEED_MS` (скачок
    координат, в том числе серия точек «в стороне»), и если она — вершина «шпильки»: оба плеча длиннее
    :data:`SPIKE_M`, а соседи почти совпадают (путь уходит в сторону и возвращается).
    """
    n = len(lon)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    xy = _local_xy(lon, lat)
    last = 0
    for k in range(1, n - 1):
        dt = max(float(ts[k] - ts[last]), 1.0)
        if float(np.hypot(*(xy[k] - xy[last]))) / dt > MAX_SPEED_MS:
            keep[k] = False
        else:
            last = k
    changed = True
    while changed:
        changed = False
        idx = np.flatnonzero(keep)
        for a, k, b in zip(idx, idx[1:], idx[2:], strict=False):
            d1 = float(np.hypot(*(xy[k] - xy[a])))
            d2 = float(np.hypot(*(xy[b] - xy[k])))
            chord = float(np.hypot(*(xy[b] - xy[a])))
            if d1 > SPIKE_M and d2 > SPIKE_M and chord < SPIKE_RATIO * (d1 + d2):
                keep[k] = False
                changed = True
                break
    return keep


def segment_passes(
    traffic: pd.DataFrame, schedule: pd.DataFrame, passages: pd.DataFrame
) -> dict[str, list[SegmentPass]]:
    """Проходы участков «остановка → следующая по плану остановка» по трекам ТС.

    Путь участка — позиция ТС в момент прохода первой остановки (интерполяция по треку), GPS-точки между
    моментами проходов и позиция в момент прохода второй. Берутся остановки одного рейса, сопоставленные
    детектором, с проходами по возрастанию не дальше :data:`MAX_SEGMENT_S`: соседние по плану и, если ТС
    не проехало мимо промежуточных остановок (детектор их пропустил: остановка в стороне от пути, разрыв
    GPS), — следующая сопоставленная через не больше :data:`MAX_SKIP` пропущенных (участок ``a → c``
    вместо ``a → b → c``; :func:`_path` берёт его, если пути ``a → b`` нет).

    Args:
        traffic: очищенная телеметрия (``shared.data.load_traffic``): tr_id, event_time, lon, lat.
        schedule: плановое расписание: tr_id, tt_action_item_id, time_begin, stop_lon, stop_lat.
        passages: проходы детектора (``shared.stops.detect_all``): tr_id, tt_action_item_id, pass_time.

    Returns:
        ``{segment_key: [проходы]}``.
    """
    plan = schedule[["tr_id", "tt_action_item_id", "time_begin", "stop_lon", "stop_lat"]].copy()
    plan["tb"] = pd.to_datetime(plan["time_begin"]).astype("datetime64[ns]").astype(np.int64) / 1e9
    pas = passages[["tr_id", "tt_action_item_id", "pass_time"]].dropna(subset=["pass_time"]).copy()
    pas["pass_s"] = pd.to_datetime(pas["pass_time"]).astype("datetime64[ns]").astype(np.int64) / 1e9
    by = ["tr_id", "tt_action_item_id"]
    plan = plan.merge(pas[[*by, "pass_s"]], on=by, how="left")
    plan = plan.sort_values(["tr_id", "tb", "tt_action_item_id"], kind="stable")
    tracks = {int(k): g for k, g in traffic.groupby("tr_id", sort=False)}
    out: dict[str, list[SegmentPass]] = {}
    for tr_id, st in plan.groupby("tr_id", sort=True):
        g = tracks.get(int(tr_id))
        if g is None or len(g) < 2:
            continue
        ts = pd.to_datetime(g["event_time"]).astype("datetime64[ns]").to_numpy().astype(np.int64) / 1e9
        order = np.argsort(ts, kind="stable")
        ts, lon, lat = ts[order], g["lon"].to_numpy(np.float64)[order], g["lat"].to_numpy(np.float64)[order]
        tb, ps = st["tb"].to_numpy(np.float64), st["pass_s"].to_numpy(np.float64)
        slon, slat = st["stop_lon"].to_numpy(np.float64), st["stop_lat"].to_numpy(np.float64)
        keys = stop_keys(np.nan_to_num(slon), np.nan_to_num(slat))
        n = len(st)
        for i in range(n - 1):
            if not np.isfinite(ps[i]):
                continue
            j = i + 1  # the next matched stop of the same trip, skipping at most MAX_SKIP unmatched ones
            while j < n and j - i <= MAX_SKIP + 1 and tb[j] - tb[j - 1] <= LAYOVER_GAP_S:
                if np.isfinite(ps[j]):
                    break
                j += 1
            else:
                continue
            if j >= n or j - i > MAX_SKIP + 1 or not np.isfinite(ps[j]):
                continue
            a, b = ps[i], ps[j]
            if b <= a or b - a > MAX_SEGMENT_S or keys[i] == keys[j]:
                continue  # the same place twice
            if a < ts[0] or b > ts[-1]:
                continue
            inner = np.flatnonzero((ts > a) & (ts < b))
            pt_t = np.r_[a, ts[inner], b]
            pt_lon = np.r_[_interp(ts, lon, a), lon[inner], _interp(ts, lon, b)]
            pt_lat = np.r_[_interp(ts, lat, a), lat[inner], _interp(ts, lat, b)]
            ok = despike(pt_lon, pt_lat, pt_t)
            pt_t, pt_lon, pt_lat = pt_t[ok], pt_lon[ok], pt_lat[ok]
            xy = _local_xy(pt_lon, pt_lat)
            steps = np.hypot(*np.diff(xy, axis=0).T)
            line = tuple(
                (round(float(x), 6), round(float(y), 6)) for x, y in zip(pt_lon, pt_lat, strict=True)
            )
            out.setdefault(segment_key(keys[i], keys[j]), []).append(
                SegmentPass(
                    line=line,
                    length_m=float(steps.sum()),
                    max_step_m=float(steps.max()) if len(steps) else 0.0,
                    max_step_s=float(np.diff(pt_t).max()) if len(pt_t) > 1 else 0.0,
                    straight_m=haversine_m(slon[i], slat[i], slon[j], slat[j]),
                )
            )
    return out


def choose_pass(passes: Sequence[SegmentPass]) -> SegmentPass | None:
    """Представительный проход участка: среди качественных (иначе среди всех) — ближайший к медиане длины.

    При равенстве — с меньшим максимальным шагом, затем с меньшей длиной (детерминированно).
    """
    if not passes:
        return None
    pool = [p for p in passes if p.good] or list(passes)
    median = float(np.median([p.length_m for p in pool]))
    return min(pool, key=lambda p: (round(abs(p.length_m - median), 3), p.max_step_m, p.length_m, p.line))


def _runs(mask: np.ndarray, dist: np.ndarray) -> list[int]:
    """Индексы ближайших точек каждого непрерывного участка ``mask`` (визиты к месту)."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
    return [int(run[np.argmin(dist[run])]) for run in np.split(idx, cuts)]


def proximity_passes(
    traffic: pd.DataFrame, schedule: pd.DataFrame, skip: Container[str] = ()
) -> dict[str, list[SegmentPass]]:
    """Проходы участков без детектора: трек ТС от визита к первой остановке до визита ко второй.

    Запасной путь для участков, которые детектор не сопоставил ни разу (ТС шло с опозданием дальше окна
    детектора, остановка в стороне от пути и т. п.), хотя ТС там ездят. Визит — непрерывный отрезок трека
    ближе :data:`NEAR_M` к месту остановки, его ближайшая точка; путь ``a → b`` — от последнего визита к
    ``a`` перед визитом к ``b`` не дольше ``max(300 с, 4 × план участка)`` (так обратный рейс по встречной
    стороне улицы не принимается за прямой). Если ТС ездят мимо этих остановок только в обратном порядке
    (остановки обслуживаются не так, как в плане), берётся путь ``b → a`` в обратном порядке точек: линия
    идёт по улице, а не напрямик сквозь дома.

    Args:
        traffic: очищенная телеметрия: tr_id, event_time, lon, lat.
        schedule: плановое расписание: tr_id, tt_action_item_id, time_begin, stop_lon, stop_lat.
        skip: участки, для которых путь уже есть (по детектору).

    Returns:
        ``{segment_key: [проходы]}``.
    """
    plan = schedule[["tr_id", "tt_action_item_id", "time_begin", "stop_lon", "stop_lat"]].copy()
    plan["tb"] = pd.to_datetime(plan["time_begin"]).astype("datetime64[ns]").astype(np.int64) / 1e9
    plan = plan.sort_values(["tr_id", "tb", "tt_action_item_id"], kind="stable")
    tracks = {int(k): g for k, g in traffic.groupby("tr_id", sort=False)}
    out: dict[str, list[SegmentPass]] = {}
    for tr_id, st in plan.groupby("tr_id", sort=True):
        g = tracks.get(int(tr_id))
        if g is None or len(g) < 2:
            continue
        ts = pd.to_datetime(g["event_time"]).astype("datetime64[ns]").to_numpy().astype(np.int64) / 1e9
        order = np.argsort(ts, kind="stable")
        ts, lon, lat = ts[order], g["lon"].to_numpy(np.float64)[order], g["lat"].to_numpy(np.float64)[order]
        tb = st["tb"].to_numpy(np.float64)
        slon, slat = st["stop_lon"].to_numpy(np.float64), st["stop_lat"].to_numpy(np.float64)
        keys = stop_keys(np.nan_to_num(slon), np.nan_to_num(slat))
        coslat = math.cos(math.radians(float(np.mean(lat))))
        dist: dict[int, np.ndarray] = {}  # distance of every fix to a planned stop, m
        seen: set[str] = set()
        for i in range(len(st) - 1):
            key = segment_key(keys[i], keys[i + 1])
            if key in skip or key in seen or keys[i] == keys[i + 1] or tb[i + 1] - tb[i] > LAYOVER_GAP_S:
                continue
            if not (np.isfinite(slon[i]) and np.isfinite(slon[i + 1])):
                continue
            seen.add(key)
            for k in (i, i + 1):
                if k not in dist:
                    dist[k] = np.hypot((lon - slon[k]) * coslat, lat - slat[k]) * _M_PER_DEG
            limit = max(300.0, 4.0 * (tb[i + 1] - tb[i]))
            spans = _visit_spans(dist[i], dist[i + 1], ts, limit)
            reverse = not spans  # the vehicles drive this street only the other way: its path, reversed
            if reverse:
                spans = [(jb, ia) for ia, jb in _visit_spans(dist[i + 1], dist[i], ts, limit)]
            for ia, jb in spans:
                lo, hi = min(ia, jb), max(ia, jb)
                pt_lon, pt_lat, pt_t = lon[lo : hi + 1], lat[lo : hi + 1], ts[lo : hi + 1]
                ok = despike(pt_lon, pt_lat, pt_t)
                pt_lon, pt_lat, pt_t = pt_lon[ok], pt_lat[ok], pt_t[ok]
                steps = np.hypot(*np.diff(_local_xy(pt_lon, pt_lat), axis=0).T)
                pts = zip(pt_lon, pt_lat, strict=True)
                line = tuple((round(float(x), 6), round(float(y), 6)) for x, y in pts)
                out.setdefault(key, []).append(
                    SegmentPass(
                        line=line[::-1] if reverse else line,
                        length_m=float(steps.sum()),
                        max_step_m=float(steps.max()) if len(steps) else 0.0,
                        max_step_s=float(np.diff(pt_t).max()) if len(pt_t) > 1 else 0.0,
                        straight_m=haversine_m(slon[i], slat[i], slon[i + 1], slat[i + 1]),
                    )
                )
    return out


def _visit_spans(da: np.ndarray, db: np.ndarray, ts: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Отрезки трека ``(ia, jb)``: визит к ``b`` и последний визит к ``a`` перед ним не раньше ``limit``."""
    visits_a = _runs(da < NEAR_M, da)
    out = []
    for jb in _runs(db < NEAR_M, db):
        before = [ia for ia in visits_a if ia < jb and ts[jb] - ts[ia] <= limit]
        if before:
            out.append((before[-1], jb))
    return out


def assemble_segments(
    detected: Mapping[str, Sequence[SegmentPass]],
    fallback: Mapping[str, Sequence[SegmentPass]] | None = None,
    eps_m: float = SIMPLIFY_M,
) -> dict[str, dict[str, Any]]:
    """Геометрия участков из проходов: по детектору, для остальных — запасные (:func:`proximity_passes`).

    Returns:
        ``{segment_key: {line, passes, good, length_m, straight_m, source}}``; ``line`` упрощена
        (:func:`douglas_peucker`), ``source`` — ``detector`` или ``proximity``.
    """
    out: dict[str, dict[str, Any]] = {}
    pools = [("detector", detected), ("proximity", fallback or {})]
    for key in sorted(set(detected) | set(fallback or {})):
        for source, pool in pools:
            passes = list(pool.get(key) or [])
            best = choose_pass(passes)
            if best is None or (source == "proximity" and not best.good):
                continue  # a fallback path must be clean: no GPS gaps, no loops
            out[key] = {
                "line": douglas_peucker(best.line, eps_m),
                "passes": len(passes),
                "good": sum(p.good for p in passes),
                "length_m": round(best.length_m, 1),
                "straight_m": round(best.straight_m, 1),
                "source": source,
            }
            break
    return out


def build_segments(
    traffic: pd.DataFrame,
    schedule: pd.DataFrame,
    passages: pd.DataFrame,
    eps_m: float = SIMPLIFY_M,
    *,
    proximity: bool = True,
) -> dict[str, dict[str, Any]]:
    """Геометрия участков одного сплита (:func:`assemble_segments`)."""
    detected = segment_passes(traffic, schedule, passages)
    fallback = proximity_passes(traffic, schedule, skip=set(detected)) if proximity else None
    return assemble_segments(detected, fallback, eps_m)


def load_segments(path: str | Path | None = None) -> dict[str, list[list[float]]]:
    """Кэш геометрии участков: ``{segment_key: [[lon, lat], …]}``; нет файла — пустой словарь.

    Args:
        path: файл кэша (по умолчанию :data:`SEGMENTS_FILE`).
    """
    p = Path(path) if path else SEGMENTS_FILE
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("format") != SEGMENTS_FORMAT:
        raise ValueError(f"{p}: format {data.get('format')!r} != {SEGMENTS_FORMAT!r}")
    return {k: [list(map(float, pt)) for pt in v["line"]] for k, v in data.get("segments", {}).items()}


# ---- routes ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteStop:
    """Остановка маршрута."""

    stop_key: str
    name: str
    lat: float
    lon: float
    seq: int


@dataclass(frozen=True)
class Direction:
    """Направление маршрута: ``direction`` (0 — прямое, 1 — обратное), остановки по порядку, линия."""

    direction: int
    name: str
    stops: tuple[RouteStop, ...]
    line: tuple[tuple[float, float], ...]
    gps_segments: int
    """Участков, линия которых построена по GPS (остальные — прямые отрезки)."""

    def to_dict(self) -> dict[str, Any]:
        """Словарь в формате ``GET /api/routes`` (элемент ``directions``)."""
        return {
            "direction": self.direction,
            "name": self.name,
            "stops": [
                {"stop_key": s.stop_key, "name": s.name, "lat": s.lat, "lon": s.lon, "seq": s.seq}
                for s in self.stops
            ],
            "line": [list(p) for p in self.line],
            "segments": max(len(self.stops) - 1, 0),
            "gps_segments": self.gps_segments,
        }


@dataclass(frozen=True)
class Route:
    """Маршрут: ``route_id``, имя, ТС, остановки по порядку обхода, направления, цвет (``GET /api/routes``).

    Поле ``line`` — линия прямого направления (совместимость с контрактом).
    """

    route_id: str
    name: str
    tr_ids: tuple[int, ...]
    stops: tuple[RouteStop, ...]
    color: str
    directions: tuple[Direction, ...] = ()

    @property
    def line(self) -> list[list[float]]:
        """Линия маршрута ``[[lon, lat], …]``: прямое направление (по дорогам, если есть геометрия)."""
        if self.directions:
            return [list(p) for p in self.directions[0].line]
        return [[s.lon, s.lat] for s in self.stops]

    @property
    def sequence(self) -> list[str]:
        """``stop_key`` остановок по порядку обхода."""
        return [s.stop_key for s in self.stops]

    def to_dict(self) -> dict[str, Any]:
        """Словарь в формате ``GET /api/routes``."""
        return {
            "route_id": self.route_id,
            "name": self.name,
            "tr_ids": list(self.tr_ids),
            "stops": [
                {"stop_key": s.stop_key, "name": s.name, "lat": s.lat, "lon": s.lon, "seq": s.seq}
                for s in self.stops
            ],
            "line": self.line,
            "directions": [d.to_dict() for d in self.directions],
            "color": self.color,
        }


@dataclass
class RouteMap:
    """Маршруты сплита и принадлежность ТС.

    Attributes:
        routes: маршруты по возрастанию номера.
        by_tr: ``tr_id → route_id``.
        names: ``stop_key → название остановки`` (самый частый непустой адрес).
        segments: геометрия участков по GPS (``segment_key → [[lon, lat], …]``).
    """

    routes: list[Route] = field(default_factory=list)
    by_tr: dict[int, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    segments: dict[str, list[list[float]]] = field(default_factory=dict)

    def route(self, route_id: str | None) -> Route | None:
        """Маршрут по ``route_id`` (``None`` — нет такого)."""
        return next((r for r in self.routes if r.route_id == route_id), None)

    def of(self, tr_id: int) -> Route | None:
        """Маршрут ТС (``None`` — ТС нет в расписании)."""
        return self.route(self.by_tr.get(int(tr_id)))

    def path(self, keys: Sequence[str]) -> list[list[float]]:
        """Линия через места остановок по порядку: участки по GPS, где они есть, иначе прямые отрезки."""
        line, _ = _path(keys, self.segments)
        return [list(p) for p in line]

    @property
    def geometry_source(self) -> dict[str, int]:
        """Сколько участков направлений построено по GPS и сколько — прямыми отрезками."""
        gps = sum(d.gps_segments for r in self.routes for d in r.directions)
        total = sum(max(len(d.stops) - 1, 0) for r in self.routes for d in r.directions)
        return {"segments": total, "gps_segments": gps, "straight_segments": total - gps}


def _coords(key: str) -> tuple[float, float]:
    lon, lat = key.split(",")
    return float(lon), float(lat)


def _path(
    keys: Sequence[str],
    segments: Mapping[str, Sequence[Sequence[float]]],
    straight: list[tuple[str, str]] | None = None,
) -> tuple[tuple[tuple[float, float], ...], int]:
    """Линия через места ``keys`` и число участков по GPS.

    Участок ``a → b`` — его путь по GPS; если его нет — путь в обход непосещаемых остановок ``a → c``
    (ближайшая ``c`` не дальше :data:`MAX_SKIP` остановок, :func:`segment_passes`), он закрывает все
    участки между ``a`` и ``c``; иначе прямой отрезок ``a → b`` (он добавляется в ``straight``).
    """
    line: list[tuple[float, float]] = []
    gps = 0
    i, n = 0, len(keys)
    while i < n - 1:
        span, geom = 1, None
        for k in range(1, min(MAX_SKIP + 1, n - 1 - i) + 1):
            geom = segments.get(segment_key(keys[i], keys[i + k]))
            if geom:
                span = k
                break
        part = [tuple(map(float, p)) for p in geom] if geom else [_coords(keys[i]), _coords(keys[i + 1])]
        gps += span if geom else 0
        if not geom and straight is not None:
            straight.append((keys[i], keys[i + 1]))
        if line and line[-1] == part[0]:
            part = part[1:]
        line += [(round(x, 6), round(y, 6)) for x, y in part]
        i += span
    if not line and keys:
        line = [_coords(keys[0])]
    return tuple(line), gps


def _plan_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    """Плановые поля в едином виде: tr_id, tt_action_item_id, plan_s, lon, lat, name, stop_key."""
    df = pd.DataFrame(
        {
            "tr_id": schedule["tr_id"].astype(np.int64).to_numpy(),
            "tt_action_item_id": schedule["tt_action_item_id"].astype(np.int64).to_numpy(),
            "plan_s": pd.to_datetime(schedule["time_begin"]).astype("datetime64[ns]").astype(np.int64) / 1e9,
            "lon": schedule["stop_lon"].astype(np.float64).to_numpy(),
            "lat": schedule["stop_lat"].astype(np.float64).to_numpy(),
        }
    )
    if "building_address" in schedule.columns:
        names = schedule["building_address"].astype("string").fillna("").str.strip()
        df["name"] = names.to_numpy(dtype=object)
    else:
        df["name"] = ""
    df = df[np.isfinite(df["lon"].to_numpy()) & np.isfinite(df["lat"].to_numpy())]
    df = df.sort_values(["tr_id", "plan_s", "tt_action_item_id"], kind="stable").reset_index(drop=True)
    df["stop_key"] = stop_keys(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df


def stop_names(schedule: pd.DataFrame) -> dict[str, str]:
    """Название каждого места остановки: самый частый непустой адрес (при равенстве — первый по алфавиту)."""
    df = _plan_frame(schedule)
    named = df[df["name"] != ""]
    out: dict[str, str] = {}
    for key, names in named.groupby("stop_key", sort=True)["name"]:
        counts = Counter(names)
        best = max(counts.values())
        out[str(key)] = min(n for n, c in counts.items() if c == best)
    return out


def build_routes(
    schedule: pd.DataFrame, segments: Mapping[str, Sequence[Sequence[float]]] | None = None
) -> RouteMap:
    """Вывести маршруты из планового расписания.

    Args:
        schedule: плановое расписание (``shared.data.load_schedule``): ``tr_id``, ``tt_action_item_id``,
            ``time_begin``, ``stop_lon``, ``stop_lat``; ``building_address`` — для названий (необязательно).
            Колонки факта не читаются.
        segments: геометрия участков (``load_segments``); ``None`` — кэш репозитория (:data:`SEGMENTS_FILE`),
            ``{}`` — прямые отрезки.

    Returns:
        :class:`RouteMap`.
    """
    if segments is None:
        try:
            segments = load_segments()
        except (OSError, ValueError):  # a damaged cache costs the road geometry, not the routes
            segments = {}
    geometry = dict(segments)
    df = _plan_frame(schedule)
    names = stop_names(schedule)
    coords = df.groupby("stop_key", sort=True)[["lon", "lat"]].first()
    groups: dict[frozenset[str], list[int]] = {}
    per_tr: dict[int, pd.DataFrame] = {}
    for tr_id, rows in df.groupby("tr_id", sort=True):
        per_tr[int(tr_id)] = rows
        groups.setdefault(frozenset(rows["stop_key"]), []).append(int(tr_id))
    used = geometry
    result = RouteMap(names=names)
    ordered = sorted(groups.values(), key=min)
    for idx, tr_ids in enumerate(ordered):
        route_id = f"R{idx + 1}"
        rows = per_tr[min(tr_ids)]
        dirs = route_directions(list(rows["stop_key"]), rows["plan_s"].to_numpy())
        sequence = join_directions(dirs)

        def stop(seq: int, key: str) -> RouteStop:
            return RouteStop(
                stop_key=key,
                name=names.get(key) or f"Остановка №{seq + 1}",
                lat=round(float(coords.loc[key, "lat"]), 6),
                lon=round(float(coords.loc[key, "lon"]), 6),
                seq=seq,
            )

        stops = tuple(stop(seq, key) for seq, key in enumerate(sequence))
        directions: list[Direction] = []
        offset = 0
        for d, keys in enumerate(dirs):
            if d and keys and offset < len(sequence) and sequence[offset] != keys[0]:
                offset += 1  # no shared terminal: the direction starts right after the previous one
            dstops = stops[offset : offset + len(keys)]
            offset += len(keys) - 1
            line, gps = _path(keys, used)
            directions.append(
                Direction(
                    direction=d,
                    name=f"{dstops[0].name} → {dstops[-1].name}" if dstops else "",
                    stops=dstops,
                    line=line,
                    gps_segments=gps,
                )
            )
        if stops:
            first = stops[0]
            if sequence[0] == sequence[-1]:
                far = max(stops, key=lambda s: haversine_m(first.lon, first.lat, s.lon, s.lat))
            else:
                far = stops[-1]
            if len(dirs) > 1:
                far = directions[0].stops[-1]
            name = f"Маршрут {route_id}: {first.name} — {far.name}"
        else:
            name = f"Маршрут {route_id}"
        route = Route(
            route_id=route_id,
            name=name,
            tr_ids=tuple(sorted(tr_ids)),
            stops=stops,
            color=ROUTE_COLORS[idx % len(ROUTE_COLORS)],
            directions=tuple(directions),
        )
        result.routes.append(route)
        for tr in tr_ids:
            result.by_tr[tr] = route_id
    result.segments = {k: [list(p) for p in v] for k, v in used.items()}
    return result


def straight_pairs(
    schedule: pd.DataFrame, segments: Mapping[str, Sequence[Sequence[float]]]
) -> list[tuple[str, str]]:
    """Участки направлений маршрутов, которые без геометрии рисуются прямыми отрезками."""
    pairs: list[tuple[str, str]] = []
    for route in build_routes(schedule, segments).routes:
        for d in route.directions:
            _path([s.stop_key for s in d.stops], segments, pairs)
    return list(dict.fromkeys((a, b) for a, b in pairs if a != b))


def osrm_path(a: str, b: str, url: str = OSRM_URL, timeout_s: float = 15.0) -> list[list[float]] | None:
    """Путь по дорогам между местами остановок ``a → b`` (OSRM, профиль driving), упрощённый; ``None`` —
    нет маршрута или он слишком длинный (:data:`OSRM_MAX_DETOUR`)."""
    (lon1, lat1), (lon2, lat2) = _coords(a), _coords(b)
    query = f"{url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    request = urllib.request.Request(query, headers={"User-Agent": "foresight-hackathon/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (the URL is ours)
        data = json.load(response)
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    coords = [[float(x), float(y)] for x, y in data["routes"][0]["geometry"]["coordinates"]]
    line = [[lon1, lat1], *coords, [lon2, lat2]]
    if path_length_m(line) > OSRM_MAX_DETOUR * haversine_m(lon1, lat1, lon2, lat2) + 500.0:
        return None
    return douglas_peucker(line)


def osrm_match(
    line: Sequence[Sequence[float]],
    url: str = OSRM_URL,
    timeout_s: float = 8.0,
    radius_m: float = MATCH_RADIUS_M,
    max_points: int = MATCH_MAX_POINTS,
    ends: tuple[Sequence[float], Sequence[float]] | None = None,
) -> list[list[float]] | None:
    """Путь участка, привязанный к дорогам (OSRM match): дрожь GPS, крючки к остановкам на тротуаре и
    выбросы уходят, линия идёт по улицам. ``None`` — не привязался (несколько кусков, низкая уверенность,
    длина сильно отличается от трека): тогда остаётся трек GPS."""
    pts = [list(map(float, pt)) for pt in line]
    if len(pts) < 2:
        return None
    if len(pts) > max_points:
        step = math.ceil(len(pts) / (max_points - 1))
        pts = pts[:-1:step] + [pts[-1]]
    coords = ";".join(f"{x:.6f},{y:.6f}" for x, y in pts)
    radii = [radius_m] * len(pts)
    radii[0] = radii[-1] = min(radius_m, MATCH_STOP_RADIUS_M)
    radiuses = ";".join(f"{r:g}" for r in radii)
    query = (
        # без tidy: с ним публичный сервер на части треков не отвечает
        f"{url.rstrip('/')}/match/v1/driving/{coords}?overview=full&geometries=geojson&radiuses={radiuses}"
    )
    request = urllib.request.Request(query, headers={"User-Agent": "foresight-hackathon/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (the URL is ours)
        data = json.load(response)
    matchings = data.get("matchings") or []
    if data.get("code") != "Ok" or len(matchings) != 1:
        return None
    if float(matchings[0].get("confidence") or 0.0) < MATCH_MIN_CONFIDENCE:
        return None
    geom = [[float(x), float(y)] for x, y in matchings[0]["geometry"]["coordinates"]]
    gps_m, road_m = path_length_m(pts), path_length_m(geom)
    if len(geom) < 2 or road_m > 1.5 * gps_m + 150.0 or road_m < 0.5 * gps_m - 50.0:
        return None
    if not _ends_near(geom, ends or (pts[0], pts[-1])):
        return None
    return douglas_peucker(geom, 2.0)


def _ends_near(
    geom: Sequence[Sequence[float]],
    ends: tuple[Sequence[float], Sequence[float]],
    limit_m: float = MATCH_END_M,
) -> bool:
    """Путь начинается у первой остановки участка и кончается у второй (:data:`MATCH_END_M`)."""
    (a, b), first, last = ends, geom[0], geom[-1]
    return (
        haversine_m(first[0], first[1], a[0], a[1]) <= limit_m
        and haversine_m(last[0], last[1], b[0], b[1]) <= limit_m
    )


def track_offset_m(geom: Sequence[Sequence[float]], track: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Насколько путь ``geom`` отходит от трека ``track``: (среднее, максимум) расстояния от точек пути (через
    каждые ~10 м) до ломаной трека, м."""
    if len(geom) < 2 or len(track) < 2:
        return 0.0, 0.0
    ref = np.asarray(track, dtype=np.float64)[:, :2]
    pts = np.asarray(geom, dtype=np.float64)[:, :2]
    lat0 = float(ref[:, 1].mean())
    sx = math.cos(math.radians(lat0)) * _M_PER_DEG
    rx, ry = ref[:, 0] * sx, ref[:, 1] * _M_PER_DEG
    dense: list[tuple[float, float]] = []
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:], strict=True):
        a, b = (x1 * sx, y1 * _M_PER_DEG), (x2 * sx, y2 * _M_PER_DEG)
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // 10.0))
        dense += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
    dense.append((pts[-1][0] * sx, pts[-1][1] * _M_PER_DEG))
    q = np.asarray(dense)
    ax, ay, bx, by = rx[:-1], ry[:-1], rx[1:], ry[1:]
    dx, dy = bx - ax, by - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-9)
    u = np.clip(((q[:, :1] - ax) * dx + (q[:, 1:] - ay) * dy) / l2, 0.0, 1.0)
    d = np.hypot(ax + u * dx - q[:, :1], ay + u * dy - q[:, 1:]).min(axis=1)
    return float(d.mean()), float(d.max())


def osrm_route_line(
    line: Sequence[Sequence[float]],
    url: str = OSRM_URL,
    timeout_s: float = 8.0,
    ends: tuple[Sequence[float], Sequence[float]] | None = None,
    *,
    curb: bool = False,
    ratio: tuple[float, float] = (0.0, 1.4),
    slack_m: float = 200.0,
    track: Sequence[Sequence[float]] | None = None,
) -> list[list[float]] | None:
    """Путь участка по дорогам между его остановками (OSRM route). ``curb`` — подъезд к остановкам со
    стороны тротуара: путь идёт по той проезжей части, где стоит остановка, и проходит через неё. Путь
    берётся, если его длина в пределах ``ratio`` от длины трека ``line`` (± ``slack_m``) — иначе ТС ехало
    другими улицами."""
    (lon1, lat1), (lon2, lat2) = ends if ends else (line[0][:2], line[-1][:2])
    query = f"{url.rstrip('/')}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    if curb:
        query += "&approaches=curb;curb"
    request = urllib.request.Request(query, headers={"User-Agent": "foresight-hackathon/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (the URL is ours)
        data = json.load(response)
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    geom = [[float(x), float(y)] for x, y in data["routes"][0]["geometry"]["coordinates"]]
    ref, got = path_length_m(line), path_length_m(geom)
    if len(geom) < 2 or got > ratio[1] * ref + slack_m or got < ratio[0] * ref - slack_m:
        return None
    if not _ends_near(geom, ((lon1, lat1), (lon2, lat2)), 2 * MATCH_END_M):
        return None
    if track is not None:
        # путь должен идти вдоль реального трека: иначе он срезал по дворам или другим улицам
        mean_m, max_m = track_offset_m(geom, track)
        if mean_m > TRACK_MEAN_M or max_m > TRACK_MAX_M:
            return None
    return douglas_peucker(geom, 2.0)


# ---- привязка целого направления к дорогам (map matching) ------------------------------------------------

TRACE_STEP_M = 80.0
"""Шаг точек трека направления для привязки к дорогам, м (длинные направления — реже, до ~900 точек): редкие
точки — привязка выбирает путь, близкий по длине к прямому (основную дорогу), а не каждое колебание GPS у
остановок (заезды во дворы, перескоки между проезжими частями бульвара)."""
TRACE_RADIUS_M = 45.0
"""Погрешность GPS при привязке трека направления (OSRM match), м."""
TRACE_TRACK_MEAN_M = 20.0
TRACE_TRACK_MAX_M = 55.0
"""Привязанный кусок, который отходит от трека GPS участка в среднем дальше ``TRACE_TRACK_MEAN_M`` или где-то
дальше ``TRACE_TRACK_MAX_M`` (м), заменяется треком: автобусу там можно то, чего нельзя легковой машине."""
TRACE_STOP_RADIUS_M = 20.0
"""Погрешность для самих остановок в треке направления (``trace --stops``), м."""
TRACE_STOP_M = 60.0
"""Остановка режет привязанную линию направления, если она не дальше этого от линии, м."""


def encode_polyline(points: Sequence[Sequence[float]], precision: int = 6) -> str:
    """Google encoded polyline ``[[lon, lat], …]`` (формат координат OSRM ``polyline6(…)``)."""
    factor = 10**precision
    out: list[str] = []
    prev_lat = prev_lon = 0
    for lon, lat in points:
        ilat, ilon = round(lat * factor), round(lon * factor)
        for v in (ilat - prev_lat, ilon - prev_lon):
            v = ~(v << 1) if v < 0 else v << 1
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        prev_lat, prev_lon = ilat, ilon
    return "".join(out)


def densify(line: Sequence[Sequence[float]], step_m: float) -> list[list[float]]:
    """Точки ломаной не реже чем через ``step_m`` метров (вершины сохраняются)."""
    out: list[list[float]] = []
    for (x1, y1), (x2, y2) in zip(line[:-1], line[1:], strict=True):
        n = max(1, int(haversine_m(x1, y1, x2, y2) // step_m))
        out += [[x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n] for k in range(n)]
    if line:
        out.append([float(line[-1][0]), float(line[-1][1])])
    return out


def drop_spurs(line: Sequence[Sequence[float]], max_turn_deg: float = 160.0) -> list[list[float]]:
    """Ломаная без «усов» туда-обратно: вершина, где путь разворачивается назад (поворот больше
    ``max_turn_deg``), убирается, пока такие есть — ус съедается до основания (заезд к остановке в тупик)."""
    pts = [list(map(float, p[:2])) for p in line]
    cos_lim = math.cos(math.radians(max_turn_deg))
    changed = True
    while changed and len(pts) > 2:
        changed = False
        arr = np.asarray(pts)
        xy = _local_xy(arr[:, 0], arr[:, 1])
        v1, v2 = xy[1:-1] - xy[:-2], xy[2:] - xy[1:-1]
        n1, n2 = np.hypot(*v1.T), np.hypot(*v2.T)
        cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-9)
        bad = np.nonzero((cos < cos_lim) & (n1 > 0.5) & (n2 > 0.5))[0]
        if len(bad):
            del pts[int(bad[0]) + 1]
            changed = True
    return pts


def drop_loops(
    line: Sequence[Sequence[float]], max_loop_m: float = 150.0, near_m: float = 8.0
) -> list[list[float]]:
    """Ломаная без мелких петель: путь вернулся ближе ``near_m`` к своей точке, пройдя меньше ``max_loop_m`` —
    петля (объезд квартала к остановке, дрожь привязки) вырезается; развороты на конечных длиннее."""
    pts = [list(map(float, p[:2])) for p in line]
    if len(pts) < 4:
        return pts
    arr = np.asarray(pts)
    xy = _local_xy(arr[:, 0], arr[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    out: list[list[float]] = []
    i = 0
    while i < len(pts):
        out.append(pts[i])
        ahead = (
            np.nonzero((cum[i + 1 :] - cum[i] < max_loop_m) & (cum[i + 1 :] - cum[i] > 3 * near_m))[0] + i + 1
        )
        back = [j for j in ahead if math.hypot(*(xy[j] - xy[i])) < near_m]
        i = back[-1] + 1 if back else i + 1
    return out


def osrm_match_trace(
    line: Sequence[Sequence[float]],
    url: str,
    timeout_s: float = 120.0,
    radius_m: float = TRACE_RADIUS_M,
    radii: Sequence[float] | None = None,
) -> list[list[list[float]]]:
    """Привязать длинный трек к дорогам одним запросом OSRM match (свой сервер, ``--max-matching-size``):
    куски привязки по порядку (трек без времени не рвётся по паузам, куски — там, где рядом нет дорог)."""
    coords = urllib.parse.quote(encode_polyline(line), safe="")
    radiuses = ";".join(f"{r:g}" for r in (radii if radii is not None else [radius_m] * len(line)))
    query = (
        f"{url.rstrip('/')}/match/v1/driving/polyline6({coords})?overview=full&geometries=geojson"
        f"&tidy=true&gaps=ignore&radiuses={radiuses}"
    )
    request = urllib.request.Request(query, headers={"User-Agent": "foresight-hackathon/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (the URL is ours)
        data = json.load(response)
    if data.get("code") != "Ok":
        return []
    return [
        [[float(x), float(y)] for x, y in m["geometry"]["coordinates"]] for m in data.get("matchings") or []
    ]


def join_matchings(matchings: Sequence[Sequence[Sequence[float]]], url: str) -> list[list[float]]:
    """Куски привязки направления одной линией: разрыв между концом куска и началом следующего — путь по
    дорогам OSRM (без остановок в роли точек пути); не нашёлся — прямой отрезок."""
    line: list[list[float]] = []
    for m in matchings:
        if line and m:
            (x1, y1), (x2, y2) = line[-1], m[0]
            query = f"{url.rstrip('/')}/route/v1/driving/{x1},{y1};{x2},{y2}?overview=full&geometries=geojson"
            gap: list[list[float]] = []
            try:
                request = urllib.request.Request(query, headers={"User-Agent": "foresight-hackathon/1.0"})
                with urllib.request.urlopen(request, timeout=30.0) as response:  # noqa: S310 (the URL is ours)
                    data = json.load(response)
                if data.get("code") == "Ok" and data.get("routes"):
                    gap = [[float(x), float(y)] for x, y in data["routes"][0]["geometry"]["coordinates"]]
            except (OSError, ValueError):
                gap = []
            line += gap
        line += [list(map(float, pt)) for pt in m]
    return line


def cut_positions(
    line: Sequence[Sequence[float]], stops: Sequence[tuple[float, float]], near_m: float = TRACE_STOP_M
) -> list[float | None]:
    """Где остановки ``stops`` (по порядку) режут линию: позиция ``j + u`` (отрезок ``j``, доля ``u``) —
    первая по ходу линии, не раньше предыдущей остановки, ближе ``near_m`` (ближайшая точка этого подхода);
    ``None`` — остановка дальше от линии."""
    arr = np.asarray(line, dtype=np.float64)
    if len(arr) < 2:
        return [None] * len(stops)
    sx = math.cos(math.radians(float(arr[:, 1].mean()))) * _M_PER_DEG
    x, y = arr[:, 0] * sx, arr[:, 1] * _M_PER_DEG
    ax, ay, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
    l2 = np.maximum(dx * dx + dy * dy, 1e-9)
    out: list[float | None] = []
    pos = 0.0
    for lon, lat in stops:
        px, py = lon * sx, lat * _M_PER_DEG
        u = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        d = np.hypot(ax + u * dx - px, ay + u * dy - py)
        j0 = int(pos)
        hits = np.nonzero(d[j0:] < near_m)[0]
        if not len(hits):
            out.append(None)
            continue
        j = j0 + int(hits[0])
        while j + 1 < len(d) and d[j + 1] <= d[j]:
            j += 1
        at = max(j + float(u[j]), pos)
        out.append(at)
        pos = at
    return out


def _point_at(arr: np.ndarray, at: float) -> list[float]:
    j = min(int(at), len(arr) - 2)
    u = at - j
    return [
        float(arr[j, 0] + u * (arr[j + 1, 0] - arr[j, 0])),
        float(arr[j, 1] + u * (arr[j + 1, 1] - arr[j, 1])),
    ]


def trace_segments(
    line: Sequence[Sequence[float]], keys: Sequence[str], near_m: float = TRACE_STOP_M
) -> list[tuple[int, int, list[list[float]]]]:
    """Участки ``keys[i] → keys[j]`` — куски привязанной линии направления между местами остановок на ней:
    соседних, а если остановка на линию не легла — через неё (до :data:`MAX_SKIP` остановок, линия всё равно
    идёт по дороге); кусок, длина которого не похожа на путь через остановки, не берётся."""
    arr = np.asarray(line, dtype=np.float64)
    cuts = cut_positions(line, [_coords(k) for k in keys], near_m)
    have = [i for i, c in enumerate(cuts) if c is not None]
    out: list[tuple[int, int, list[list[float]]]] = []
    k = 0
    while k < len(have) - 1:
        i, step = have[k], 1
        # ближайшая по ходу остановка, до которой кусок похож на путь через остановки; иначе — через неё
        for n in range(k + 1, len(have)):
            j = have[n]
            ca, cb = cuts[i], cuts[j]
            if j - i > MAX_SKIP:
                break
            if ca is None or cb is None or cb <= ca or keys[i] == keys[j]:
                continue
            piece = [_point_at(arr, ca), *arr[int(ca) + 1 : int(cb) + 1].tolist(), _point_at(arr, cb)]
            pairs = zip(keys[i:j], keys[i + 1 : j + 1], strict=True)
            straight = sum(haversine_m(*_coords(a), *_coords(b)) for a, b in pairs)
            length = path_length_m(piece)
            if length > 2.5 * straight + 300.0 or length < 0.6 * straight - 30.0:
                continue
            out.append((i, j, douglas_peucker(piece, 2.0)))
            step = n - k
            break
        k += step
    return out


def _cmd_trace(args: argparse.Namespace) -> None:
    """Геометрия участков привязкой целых направлений к дорогам: трек GPS направления (кэш ``--gps`` из
    ``build``, без привязки) → OSRM match одним запросом → куски между остановками. Остановки не становятся
    точками пути — линия не заезжает к ним по проездам. Остальные участки — как в кэше ``--segments``."""
    from shared.data import load_schedule

    gps = load_segments(args.gps)
    src = Path(args.segments or SEGMENTS_FILE)
    data = json.loads(src.read_text(encoding="utf-8"))
    segments = data["segments"]
    traced: dict[str, list[list[float]]] = {}
    skipped: set[str] = set()
    # трек GPS участка, а где его нет (ТС без валидного GPS, дыры у конечных) — путь по дорогам из кэша
    lines = {**{k: v["line"] for k, v in segments.items()}, **gps}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        routes = build_routes(load_schedule(split), lines)
        for route in routes.routes:
            for d in route.directions:
                keys = [s.stop_key for s in d.stops]
                at_stop = {tuple(round(v, 6) for v in _coords(k)) for k in keys}
                # точки-сами остановки (стоят на тротуаре: крючки путей OSRM к ним) не привязываются
                pts = drop_spurs([list(p) for p in d.line if tuple(round(v, 6) for v in p) not in at_stop])
                if len(pts) < 2:
                    continue
                step = max(TRACE_STEP_M, path_length_m(pts) / 900.0)
                dense = densify(pts, step)
                radii = [TRACE_RADIUS_M] * len(dense)
                if args.stops:
                    # сами остановки — точки трека с узким радиусом: привязка держит ту проезжую часть, у
                    # которой они стоят (не перескакивает на соседнюю, если GPS шёл посередине бульвара)
                    at = cut_positions(dense, [_coords(k) for k in keys], 40.0)
                    found = sorted(
                        ((p, k) for p, k in zip(at, keys, strict=True) if p is not None), reverse=True
                    )
                    for p, k in found:
                        dense.insert(int(p) + 1, list(_coords(k)))
                        radii.insert(int(p) + 1, TRACE_STOP_RADIUS_M)
                try:
                    matchings = osrm_match_trace(dense, args.osrm, radii=radii)
                except (OSError, ValueError) as exc:
                    print(f"{route.route_id}/{d.direction}: {exc}", flush=True)
                    continue
                got = 0
                for m in [join_matchings(matchings, args.osrm)] if args.join and matchings else matchings:
                    for i, j, piece in trace_segments(drop_spurs(drop_loops(m)), keys):
                        traced.setdefault(segment_key(keys[i], keys[j]), piece)
                        if j - i > 1:  # через непривязанные остановки: их старые участки не рисуются
                            skipped.update(
                                segment_key(a, b) for a, b in zip(keys[i:j], keys[i + 1 : j + 1], strict=True)
                            )
                        got += j - i
                where = f"{split} {route.route_id}/{d.direction}"
                print(f"{where}: {len(matchings)} matchings, {got}/{len(keys) - 1} segments", flush=True)
    for key in skipped - set(traced):
        segments.pop(key, None)
    # остальные участки (старые пути): без крючков к самим остановкам и без усов туда-обратно
    for key, seg in segments.items():
        if key in traced:
            continue
        ends = {tuple(round(v, 6) for v in _coords(k)) for k in key.split("|")}
        body = [p for p in seg["line"] if tuple(round(v, 6) for v in p) not in ends]
        seg["line"] = drop_spurs(drop_loops(body if len(body) >= 2 else seg["line"]))
    # привязка по дорогам для легковых: где автобусам разрешено больше (встречная полоса, одностороннее
    # «кроме автобусов»), она уходит на соседнюю проезжую часть — там верим треку GPS
    kept_gps = 0
    for key in list(traced):
        track = gps.get(key)
        if not track or len(track) < 2:
            continue
        mean_m, max_m = track_offset_m(traced[key], track)
        if mean_m > TRACE_TRACK_MEAN_M or max_m > TRACE_TRACK_MAX_M:
            traced[key] = drop_spurs(drop_loops([list(map(float, pt)) for pt in track]))
            kept_gps += 1
    print(f"GPS instead of the matching (off the track): {kept_gps}")
    for key, line in traced.items():
        a, b = key.split("|")
        seg = segments.setdefault(key, {"passes": 0, "good": 0, "source": "osrm"})
        seg["line"] = line
        seg["length_m"] = round(path_length_m(line), 1)
        seg["straight_m"] = round(haversine_m(*_coords(a), *_coords(b)), 1)
        seg["matched"] = "trace"
    data["rule"] = data.get("rule", "").split("; then snapped")[0] + (
        "; then snapped to roads by map matching whole directions (the GPS track of a direction → one OSRM "
        "match, cut at the stops; matched: trace), the other segments as before (route / match / GPS)"
    )
    data["snapped_at"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    Path(args.out or src).write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8"
    )
    by_way = Counter(str(v.get("matched") or "gps") for v in segments.values())
    print(f"traced {len(traced)}; {dict(by_way)} of {len(segments)} segments → {args.out or src}")


# ---- CLI -------------------------------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> None:
    from shared.data import load_schedule, load_traffic
    from shared.stops import detect_all

    detected: dict[str, list[SegmentPass]] = {}
    data = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        traffic = load_traffic(split)
        plan = load_schedule(split).drop(columns=["time_fact_begin", "manual_fill"], errors="ignore")
        passages = detect_all(traffic, plan)
        for key, passes in segment_passes(traffic, plan, passages).items():
            detected.setdefault(key, []).extend(passes)
        data.append((traffic, plan))
        print(f"{split}: {len(traffic)} points, {int(passages['pass_time'].notna().sum())} passes")
    fallback: dict[str, list[SegmentPass]] = {}
    for traffic, plan in data:
        for key, passes in proximity_passes(traffic, plan, skip=set(detected)).items():
            fallback.setdefault(key, []).extend(passes)
    out = assemble_segments(detected, fallback, args.eps)
    if args.osrm:
        lines = {k: v["line"] for k, v in out.items()}
        pairs = list(dict.fromkeys(p for _, plan in data for p in straight_pairs(plan, lines)))
        added = 0
        for a, b in pairs:
            line = None
            for attempt in range(3):
                try:
                    line = osrm_path(a, b, args.osrm)
                    break
                except (OSError, ValueError) as exc:
                    print(f"OSRM {a} → {b} (attempt {attempt + 1}): {exc}")
                finally:
                    time.sleep(1.0)  # the public OSRM server: at most one request per second
            if line is None:
                continue
            (lon1, lat1), (lon2, lat2) = _coords(a), _coords(b)
            out[segment_key(a, b)] = {
                "line": line,
                "passes": 0,
                "good": 0,
                "length_m": round(path_length_m(line), 1),
                "straight_m": round(haversine_m(lon1, lat1, lon2, lat2), 1),
                "source": "osrm",
            }
            added += 1
        print(f"OSRM: {added} of {len(pairs)} segments without a GPS path")
    payload = {
        "format": SEGMENTS_FORMAT,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "splits": args.splits,
        "simplify_m": args.eps,
        "rule": "median-length good pass between detector passes of consecutive planned stops (GPS outliers "
        "dropped: > MAX_SPEED_MS from the last kept fix, out-and-back spikes); fallback: the track between "
        "visits (< NEAR_M) of the two stops; segments still without a path: OSRM driving route (source osrm)",
        "segments": out,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    points = sum(len(v["line"]) for v in out.values())
    by_source = Counter(v["source"] for v in out.values())
    print(f"{len(out)} segments {dict(by_source)}, {points} points after simplification → {path}")


MAP_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Foresight · маршруты</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>html, body, #map { margin: 0; height: 100%; } #legend { position: absolute; z-index: 1000; top: 8px;
right: 8px; background: #fff; padding: 6px 10px; font: 12px sans-serif; border-radius: 4px; }</style>
</head><body><div id="map"></div><div id="legend"></div>
<script>window.ROUTES = __ROUTES__;</script>
<script>
const q = new URLSearchParams(location.search), only = q.get("route");
const map = L.map("map");
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "© OpenStreetMap contributors" }).addTo(map);
const bounds = [], legend = [];
for (const r of window.ROUTES.routes) {
  if (only && r.route_id !== only) continue;
  for (const d of r.directions) {
    const color = only ? ["#1f5bd8", "#d8321f"][d.direction] : r.color;
    L.polyline(d.line.map((p) => [p[1], p[0]]), { color, weight: 4, opacity: 0.85,
      dashArray: d.direction ? "8 6" : null }).addTo(map);
    for (const s of d.stops) {
      const style = { radius: 3, color: "#000", weight: 1, fillColor: "#fff", fillOpacity: 1 };
      L.circleMarker([s.lat, s.lon], style)
        .bindTooltip(`${r.route_id} · ${s.seq} · ${s.name}`).addTo(map);
      bounds.push([s.lat, s.lon]);
    }
    legend.push(`${r.route_id} ${d.direction ? "обратно" : "прямо"}: ${d.stops.length} ост., ` +
      `по GPS ${d.gps_segments}/${d.segments}`);
  }
}
map.fitBounds(bounds, { padding: [20, 20] });
document.getElementById("legend").innerHTML = legend.join("<br>");
</script></body></html>
"""
"""Карта-проверка маршрутов (``python -m shared.routes map``): Leaflet + OSM, направления разными линиями,
``?route=R1`` — один маршрут (прямое — сплошная синяя, обратное — пунктир красный)."""


def _cmd_snap(args: argparse.Namespace) -> None:
    """Привязать геометрию кэша к дорогам (OSRM match), участок за участком; не привязался — как был."""
    src = Path(args.segments or SEGMENTS_FILE)
    out = Path(args.out or src)
    data = json.loads(src.read_text(encoding="utf-8"))
    if out.is_file() and out != src:  # продолжить прерванный прогон: привязанные участки уже в out
        data = json.loads(out.read_text(encoding="utf-8"))
    segments = data["segments"]
    data["rule"] = data.get("rule", "").split("; then snapped")[0] + (
        "; then snapped to roads: the OSRM route between the stops approached from the curb when its length "
        "matches the track (matched: route), else OSRM match of the track (matched: match), else the OSRM "
        "route between the stops if not much longer than the track (matched: route), else the GPS track"
    )

    def save() -> None:
        data["snapped_at"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    todo = [k for k in sorted(segments) if not segments[k].get("matched")]
    snapped = 0
    for i, key in enumerate(todo):
        seg = segments[key]
        line = None
        # на тяжёлых треках публичный сервер не отвечает — вторая попытка с меньшим числом точек
        a, b = key.split("|")
        ends = (_coords(a), _coords(b))
        # 1) путь по дорогам между остановками со стороны тротуара — через сами остановки, по их проезжей
        #    части; берётся, если по длине он как трек (ТС ехало этими улицами)
        via = "route"
        track = seg["line"] if seg.get("source") in ("detector", "proximity") else None
        try:
            line = osrm_route_line(
                seg["line"], args.osrm, ends=ends, curb=True, ratio=(0.75, 1.3), slack_m=150.0, track=track
            )
        except (OSError, ValueError) as exc:
            print(f"route {key}: {exc}", flush=True)
        time.sleep(args.pause)
        # 2) иначе — привязка самого трека к дорогам
        if line is None:
            via = "match"
            for points in (args.max_points, 5):
                try:
                    line = osrm_match(seg["line"], args.osrm, max_points=points, ends=ends)
                    break
                except (OSError, ValueError) as exc:
                    print(f"match {key} ({points} points): {exc}", flush=True)
                finally:
                    time.sleep(args.pause)  # the public OSRM server: at most one request per second
        # 3) иначе — путь по дорогам без стороны, если не длиннее трека в 1,4 раза; иначе трек как есть
        if line is None:
            via = "route"
            try:
                line = osrm_route_line(seg["line"], args.osrm, ends=ends, track=track)
            except (OSError, ValueError) as exc:
                print(f"route {key}: {exc}", flush=True)
            time.sleep(args.pause)
        if line is not None:
            seg["line"] = line
            seg["length_m"] = round(path_length_m(line), 1)
            seg["matched"] = via
            snapped += 1
        if (i + 1) % 50 == 0:
            save()
            print(f"{i + 1}/{len(todo)}: {snapped} snapped", flush=True)
    # участки, которые без геометрии рисовались бы прямыми (нет трека): путь по дорогам
    for split in [x.strip() for x in (args.fill_splits or "").split(",") if x.strip()]:
        from shared.data import load_schedule

        plan = load_schedule(split).drop(columns=["time_fact_begin", "manual_fill"], errors="ignore")
        lines = {k: v["line"] for k, v in segments.items()}
        for a, b in straight_pairs(plan, lines):
            try:
                path = osrm_path(a, b, args.osrm)
            except (OSError, ValueError) as exc:
                print(f"route {a} → {b}: {exc}", flush=True)
                continue
            if path is None:
                continue
            (lon1, lat1), (lon2, lat2) = _coords(a), _coords(b)
            segments[segment_key(a, b)] = {
                "line": path,
                "passes": 0,
                "good": 0,
                "length_m": round(path_length_m(path), 1),
                "straight_m": round(haversine_m(lon1, lat1, lon2, lat2), 1),
                "source": "osrm",
                "matched": "route",
            }
    save()
    by_way = Counter(str(v.get("matched") or "gps") for v in segments.values())
    print(f"{dict(by_way)} of {len(segments)} segments → {out}")


def _cmd_show(args: argparse.Namespace) -> None:
    from shared.data import load_schedule

    routes = build_routes(load_schedule(args.split), load_segments(args.segments) if args.segments else None)
    payload = {"routes": [r.to_dict() for r in routes.routes], "geometry": routes.geometry_source}
    text = json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2)
    if args.js:
        text = f"window.ROUTES = {text};"
    print(text)


def _cmd_map(args: argparse.Namespace) -> None:
    from shared.data import load_schedule

    routes = build_routes(load_schedule(args.split), load_segments(args.segments) if args.segments else None)
    payload = {"routes": [r.to_dict() for r in routes.routes], "geometry": routes.geometry_source}
    html = MAP_HTML.replace("__ROUTES__", json.dumps(payload, ensure_ascii=False))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{len(routes.routes)} routes, {routes.geometry_source} → {args.out}")


def main(argv: list[str] | None = None) -> None:
    """``python -m shared.routes build|snap|trace|show|map``: кэш геометрии участков, привязка к дорогам
    (участками или целыми направлениями), маршруты сплита, карта-проверка."""
    ap = argparse.ArgumentParser(description="Маршруты из плана и их геометрия по GPS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="геометрия участков по телеметрии и детектору → кэш")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--out", default=str(SEGMENTS_FILE))
    p.add_argument("--eps", type=float, default=SIMPLIFY_M, help="допуск Дугласа—Пекера, м")
    p.add_argument(
        "--osrm", default="", help=f"OSRM для участков без GPS-пути (например {OSRM_URL}); пусто — нет"
    )
    p = sub.add_parser("snap", help="привязать геометрию кэша к дорогам (OSRM match)")
    p.add_argument("--segments", default="", help="кэш геометрии (по умолчанию — файл репозитория)")
    p.add_argument("--out", default="", help="куда записать (по умолчанию — тот же файл)")
    p.add_argument("--osrm", default=OSRM_URL)
    p.add_argument(
        "--max-points", type=int, default=MATCH_MAX_POINTS, help="точек трека в запросе (свой OSRM — больше)"
    )
    p.add_argument("--pause", type=float, default=1.0, help="пауза между запросами, с (свой OSRM — 0)")
    p.add_argument("--fill-splits", default="", help="участки без геометрии этих сплитов — по дорогам")
    p = sub.add_parser("trace", help="привязать целые направления к дорогам (OSRM match трека направления)")
    p.add_argument("--gps", required=True, help="кэш build без привязки: треки GPS участков")
    p.add_argument("--segments", default="", help="кэш геометрии (по умолчанию — файл репозитория)")
    p.add_argument("--out", default="", help="куда записать (по умолчанию — тот же файл)")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--stops", action="store_true", help="остановки — точки трека с узким радиусом")
    p.add_argument("--join", action="store_true", help="куски привязки — одной линией через путь по дорогам")
    p.add_argument(
        "--osrm", required=True, help="свой OSRM с --max-matching-size (например http://127.0.0.1:5001)"
    )
    p = sub.add_parser("show", help="маршруты сплита в формате GET /api/routes")
    p.add_argument("--split", default="test")
    p.add_argument("--segments", default="", help="кэш геометрии (по умолчанию — файл репозитория)")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--js", action="store_true", help="window.ROUTES = … (для карты-проверки)")
    p = sub.add_parser("map", help="HTML-карта маршрутов поверх OSM (идут ли линии по дорогам)")
    p.add_argument("--split", default="test")
    p.add_argument("--segments", default="", help="кэш геометрии (по умолчанию — файл репозитория)")
    p.add_argument("--out", default="routes_map.html")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        _cmd_build(args)
    elif args.cmd == "snap":
        _cmd_snap(args)
    elif args.cmd == "trace":
        _cmd_trace(args)
    elif args.cmd == "map":
        _cmd_map(args)
    else:
        _cmd_show(args)


if __name__ == "__main__":
    main()
