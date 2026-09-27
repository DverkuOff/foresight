"""Признаки прогнозной точки ``(tr_id, T)`` — общий код для офлайна и онлайна.

Контекст (:class:`FeatureContext`) хранит телеметрию, плановое расписание и проходы остановок детектора
(:func:`shared.stops.detect_all`). :func:`build_features` для каждой точки использует только:

* телеметрию с ``event_time ≤ T``;
* проходы детектора с ``confirmed_at ≤ T`` (у любых ТС — это онлайн-информация);
* плановое расписание целиком (``time_begin``, координаты остановок);
* подсказку ``cur_dev_s`` из самой точки.

Факты расписания (``time_fact_begin``, ``manual_fill``) контекст отбрасывает при создании — они не могут
попасть в признаки даже случайно. Поэтому один и тот же контекст годится и для пачки точек сплита
(офлайн), и для состояния потока на момент ``T`` (онлайн): данные после ``T`` просто не читаются.

Время внутри модуля — секунды эпохи (float64) от «наивных» меток датасета, поэтому ``T % 86400`` — это
локальное время суток.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.stops import detect_all

FORBIDDEN_SCHEDULE_COLUMNS = ("time_fact_begin", "manual_fill")

STOP_SPEED_KMH = 2.0  # скорость ниже — стоянка
LAYOVER_S = 300.0  # плановый разрыв больше — отстой / конечная
NET_WINDOW_S = 3600.0  # окно онлайн-статистик участков по всем ТС
TREND_WINDOW_S = 1800.0  # окно тренда отклонения своего ТС
MOVE_WINDOWS_S = (60, 180, 300, 600)
N_LAST = 3  # число последних подтверждённых отклонений
POS_LOOKAHEAD = 12  # сколько плановых отрезков вперёд проверяем при привязке позиции
POS_TIE_M = 20.0  # допуск к минимуму расстояния: из близких отрезков берём самый ранний по плану
TERMINAL_M = 150.0  # ТС ближе к конечной — считается стоящим на ней
MAX_GPS_AGE_S = 900.0  # позиция старше — не используется для привязки

_M_PER_DEG = np.pi / 180.0 * 6_371_008.8
_SEG_MUL = 10_000_000  # ключ участка = код места A * _SEG_MUL + код места B

FEATURE_NAMES: list[str] = [
    # время
    "hour",
    "lead_s",
    # подсказка и отклонение по детектору
    "cur_dev_s",
    *[f"dev_{k}" for k in range(1, N_LAST + 1)],
    "dev_med5",
    "dev_slope",
    "age_1",
    "conf_age_1",
    "n_pass_30m",
    "n_skip_recent",
    "own_rate",
    "cur_minus_dev1",
    # позиция по GPS относительно плана
    "pos_delay",
    "pos_dist",
    "pos_u",
    "pos_on_layover",
    "cur_minus_pos",
    "n_unconfirmed",
    "cur_best",
    # движение
    *[f"spd_{w}" for w in MOVE_WINDOWS_S],
    *[f"stopfrac_{w}" for w in MOVE_WINDOWS_S],
    *[f"npts_{w}" for w in MOVE_WINDOWS_S],
    "stop_dur",
    "gps_age",
    "max_gap_10m",
    "dist_5m",
    "dist_next_stop",
    "dist_target",
    # маршрут до цели
    "n_to_target",
    "plan_to_target",
    "run_plan_ahead",
    "maxgap_ahead",
    "layover_ahead",
    "n_layover_ahead",
    "has_layover",
    "maxgap_anchor",
    "tgt_gap_prev",
    "tgt_gap_next",
    "tgt_trip_pos",
    "tgt_trip_left",
    # «физика»
    "phys",
    "own_extra",
    "phys_own",
    # участки и сеть (все ТС, confirmed_at ≤ T)
    "net_extra",
    "net_cover",
    "phys_net",
    "tgt_loc_dev",
    "tgt_loc_n",
    "net_rate_30m",
    "net_dev_30m",
]

# Сетевые признаки (по проходам других ТС) считаются, но в модель v1 не входят: в train почти все «соседи» —
# синтетические копии того же ТС со сдвигом по времени, и распределение этих признаков в train и
# test/validate разное (на test без них MAE лучше).
NETWORK_FEATURES = [
    "net_extra",
    "net_cover",
    "phys_net",
    "tgt_loc_dev",
    "tgt_loc_n",
    "net_rate_30m",
    "net_dev_30m",
]


def to_seconds(values: pd.Series | pd.Index | np.ndarray | list) -> np.ndarray:
    """Метки времени → секунды эпохи (float64), NaT → NaN; единицы приводятся к наносекундам явно."""
    s = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, format="mixed")
    s = s.astype("datetime64[ns]")
    na = s.isna().to_numpy()
    out = s.to_numpy().astype(np.int64).astype(np.float64) / 1e9
    out[na] = np.nan
    return out


def timestamp_seconds(ts: pd.Timestamp | str | np.datetime64) -> float:
    """Одна метка времени → секунды эпохи."""
    return pd.Timestamp(ts).as_unit("ns").value / 1e9


def _loc_key(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Ключ места остановки (округление ~1 м): одинаковые точки разных плановых прибытий совпадают."""
    lon_i = np.round(np.nan_to_num(lon) * 1e5).astype(np.int64)
    lat_i = np.round(np.nan_to_num(lat) * 1e5).astype(np.int64)
    return lon_i * 100_000_000 + lat_i


def _dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    coslat = np.cos(np.deg2rad(0.5 * (lat1 + lat2)))
    return float(np.hypot((lon2 - lon1) * coslat * _M_PER_DEG, (lat2 - lat1) * _M_PER_DEG))


@dataclass
class _Vehicle:
    """Данные одного ТС в виде numpy-массивов (телеметрия по времени, остановки в порядке плана)."""

    t: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    tb: np.ndarray
    stop_id: np.ndarray
    slon: np.ndarray
    slat: np.ndarray
    loc: np.ndarray
    pass_s: np.ndarray
    conf_s: np.ndarray
    index: dict[int, int]


class _WindowStore:
    """Наблюдения по ключу, упорядоченные по моменту подтверждения; медиана в окне (T − w, T]."""

    def __init__(self, keys: np.ndarray, conf: np.ndarray, values: np.ndarray) -> None:
        self._data: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        if len(keys) == 0:
            return
        order = np.lexsort((conf, keys))
        keys, conf, values = keys[order], conf[order], values[order]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(keys)], strict=True):
            self._data[int(keys[a])] = (conf[a:b], values[a:b])

    def median(self, key: int, t: float, window: float) -> tuple[float, int]:
        arr = self._data.get(int(key))
        if arr is None:
            return np.nan, 0
        conf, val = arr
        lo = int(np.searchsorted(conf, t - window, side="right"))
        hi = int(np.searchsorted(conf, t, side="right"))
        if hi <= lo:
            return np.nan, 0
        return float(np.median(val[lo:hi])), hi - lo


class FeatureContext:
    """Состояние, из которого считаются признаки: телеметрия, плановое расписание, проходы детектора.

    Контекст может содержать данные после ``T`` (офлайн — весь день): :func:`build_features` их не читает.

    Args:
        traffic: очищенная телеметрия (``shared.data.load_traffic``): tr_id, event_time, lon, lat, speed.
        schedule: плановое расписание: tr_id, tt_action_item_id, time_begin, stop_lon, stop_lat. Колонки
            факта (``time_fact_begin``, ``manual_fill``) отбрасываются.
        passages: проходы остановок (выход ``shared.stops.detect_all``); если ``None`` — считаются здесь.
        network_ids: ТС, чьи проходы идут в сетевые статистики (участки, место цели); ``None`` — все.
            В train сюда передаются только реальные ТС: синтетические — сдвинутые копии реальных, их
            проходы «подсказывают» будущее копии и дают признаки, которых нет в test/validate.
    """

    def __init__(
        self,
        traffic: pd.DataFrame,
        schedule: pd.DataFrame,
        passages: pd.DataFrame | None = None,
        network_ids: Iterable[int] | None = None,
    ) -> None:
        cols = ["tr_id", "tt_action_item_id", "time_begin", "stop_lon", "stop_lat"]
        schedule = schedule.drop(columns=[c for c in FORBIDDEN_SCHEDULE_COLUMNS if c in schedule.columns])
        schedule = schedule[cols].copy()
        if passages is None:
            passages = detect_all(traffic, schedule)
        self.vehicles: dict[int, _Vehicle] = {}
        self._build_vehicles(traffic, schedule, passages)
        self._build_network(None if network_ids is None else {int(x) for x in network_ids})

    # --- подготовка --------------------------------------------------------------------------------
    def _build_vehicles(self, traffic: pd.DataFrame, schedule: pd.DataFrame, passages: pd.DataFrame) -> None:
        sched = schedule.copy()
        sched["tb"] = to_seconds(sched["time_begin"])
        # компактный код места остановки (для ключей участков)
        sched["loc"] = pd.factorize(_loc_key(sched["stop_lon"].to_numpy(), sched["stop_lat"].to_numpy()))[0]
        sched = sched.sort_values(["tr_id", "tb", "tt_action_item_id"], kind="stable")
        pas = passages[["tr_id", "tt_action_item_id", "pass_time", "confirmed_at"]].copy()
        pas["pass_s"] = to_seconds(pas["pass_time"])
        pas["conf_s"] = to_seconds(pas["confirmed_at"])
        pas["tr_id"] = pas["tr_id"].astype(np.int64)
        pas["tt_action_item_id"] = pas["tt_action_item_id"].astype(np.int64)
        pas = pas.drop_duplicates(["tr_id", "tt_action_item_id"])
        sched = sched.merge(
            pas[["tr_id", "tt_action_item_id", "pass_s", "conf_s"]],
            on=["tr_id", "tt_action_item_id"],
            how="left",
        )
        tr = traffic.copy()
        if "speed" not in tr.columns:
            tr["speed"] = np.nan
        tr["ts"] = to_seconds(tr["event_time"])
        tr = tr.sort_values(["tr_id", "ts"], kind="stable")
        tracks = {int(k): g for k, g in tr.groupby("tr_id", sort=False)}
        empty = tr.iloc[:0]
        for tr_id, st in sched.groupby("tr_id", sort=False):
            g = tracks.get(int(tr_id), empty)
            slon = st["stop_lon"].to_numpy(dtype=np.float64)
            slat = st["stop_lat"].to_numpy(dtype=np.float64)
            stop_id = st["tt_action_item_id"].to_numpy(dtype=np.int64)
            self.vehicles[int(tr_id)] = _Vehicle(
                t=g["ts"].to_numpy(dtype=np.float64),
                lon=g["lon"].to_numpy(dtype=np.float64),
                lat=g["lat"].to_numpy(dtype=np.float64),
                speed=g["speed"].to_numpy(dtype=np.float64),
                tb=st["tb"].to_numpy(dtype=np.float64),
                stop_id=stop_id,
                slon=slon,
                slat=slat,
                loc=st["loc"].to_numpy(dtype=np.int64),
                pass_s=st["pass_s"].to_numpy(dtype=np.float64),
                conf_s=st["conf_s"].to_numpy(dtype=np.float64),
                index={int(s): i for i, s in enumerate(stop_id)},
            )

    def _build_network(self, network_ids: set[int] | None) -> None:
        """Наблюдения проходов всех ТС: отклонение на месте остановки и прирост отклонения на участке."""
        s_key, s_conf, s_dev = [], [], []
        g_key, g_conf, g_delta, g_plan = [], [], [], []
        for tr_id, v in self.vehicles.items():
            if network_ids is not None and tr_id not in network_ids:
                continue
            ok = np.isfinite(v.pass_s) & np.isfinite(v.conf_s)
            s_key.append(v.loc[ok])
            s_conf.append(v.conf_s[ok])
            s_dev.append(v.pass_s[ok] - v.tb[ok])
            if len(v.tb) < 2:
                continue
            gap = np.diff(v.tb)
            pair = ok[:-1] & ok[1:] & (gap <= LAYOVER_S)
            i = np.flatnonzero(pair)
            g_key.append(v.loc[i] * _SEG_MUL + v.loc[i + 1])
            g_conf.append(np.maximum(v.conf_s[i], v.conf_s[i + 1]))
            g_delta.append((v.pass_s[i + 1] - v.pass_s[i]) - gap[i])
            g_plan.append(gap[i])

        def cat(parts: list[np.ndarray], dtype: type) -> np.ndarray:
            return np.concatenate(parts).astype(dtype) if parts else np.array([], dtype=dtype)

        self._stop_store = _WindowStore(cat(s_key, np.int64), cat(s_conf, float), cat(s_dev, float))
        seg_key = cat(g_key, np.int64)
        seg_conf, seg_delta, seg_plan = cat(g_conf, float), cat(g_delta, float), cat(g_plan, float)
        self._seg_store = _WindowStore(seg_key, seg_conf, seg_delta)
        order = np.argsort(seg_conf, kind="stable")
        self._net_conf = seg_conf[order]
        self._net_delta_cum = np.r_[0.0, np.cumsum(seg_delta[order])]
        self._net_plan_cum = np.r_[0.0, np.cumsum(seg_plan[order])]
        dev_conf, dev = cat(s_conf, float), cat(s_dev, float)
        order = np.argsort(dev_conf, kind="stable")
        self._dev_conf, self._dev = dev_conf[order], dev[order]

    # --- признаки одной точки ----------------------------------------------------------------------
    def point_features(
        self, tr_id: int, t: float, target_stop_id: int, target_tb: float, cur_dev_s: float
    ) -> dict[str, float]:
        """Признаки одной точки; ``t`` и ``target_tb`` — секунды эпохи.

        Args:
            tr_id: ID ТС.
            t: момент прогноза T.
            target_stop_id: ``tt_action_item_id`` целевой остановки.
            target_tb: плановое время целевой остановки.
            cur_dev_s: подсказка ``cur_dev_s`` (NaN, если нет).

        Returns:
            Словарь ``{имя признака: значение}`` (NaN — признак не определён).
        """
        f: dict[str, float] = dict.fromkeys(FEATURE_NAMES, np.nan)
        f["hour"] = (t % 86400.0) / 3600.0
        f["lead_s"] = target_tb - t
        f["cur_dev_s"] = cur_dev_s
        self._network_features(f, t)
        v = self.vehicles.get(int(tr_id))
        if v is None:
            f["cur_best"] = cur_dev_s
            return f
        ti = v.index.get(int(target_stop_id))
        lm = self._deviation_features(f, v, t)
        g = int(np.searchsorted(v.t, t, side="right")) - 1
        self._movement_features(f, v, t, g)
        pos_seg = self._position_features(f, v, t, g, lm)
        if ti is None:
            f["cur_best"] = _first_finite(f["pos_delay"], f["dev_1"], cur_dev_s)
            return f
        self._route_features(f, v, t, g, ti, pos_seg, lm)
        return f

    def _network_features(self, f: dict[str, float], t: float) -> None:
        lo = int(np.searchsorted(self._net_conf, t - TREND_WINDOW_S, side="right"))
        hi = int(np.searchsorted(self._net_conf, t, side="right"))
        plan = self._net_plan_cum[hi] - self._net_plan_cum[lo]
        if plan > 0:
            f["net_rate_30m"] = (self._net_delta_cum[hi] - self._net_delta_cum[lo]) / plan
        lo = int(np.searchsorted(self._dev_conf, t - TREND_WINDOW_S, side="right"))
        hi = int(np.searchsorted(self._dev_conf, t, side="right"))
        if hi > lo:
            f["net_dev_30m"] = float(np.median(self._dev[lo:hi]))

    @staticmethod
    def _deviation_features(f: dict[str, float], v: _Vehicle, t: float) -> int | None:
        """Отклонения по подтверждённым проходам; возвращает индекс последней сопоставленной остановки."""
        decided = v.conf_s <= t  # NaN → False
        matched = decided & np.isfinite(v.pass_s)
        m_idx = np.flatnonzero(matched)
        dec_idx = np.flatnonzero(decided)
        if len(dec_idx):
            last = dec_idx[-5:]
            f["n_skip_recent"] = float(np.sum(~np.isfinite(v.pass_s[last])))
        if len(m_idx) == 0:
            return None
        dev = v.pass_s[m_idx] - v.tb[m_idx]
        for k in range(1, N_LAST + 1):
            if len(dev) >= k:
                f[f"dev_{k}"] = dev[-k]
        f["dev_med5"] = float(np.median(dev[-5:]))
        lm = int(m_idx[-1])
        f["age_1"] = t - v.pass_s[lm]
        f["conf_age_1"] = t - v.conf_s[lm]
        recent = v.pass_s[m_idx] >= t - TREND_WINDOW_S
        f["n_pass_30m"] = float(recent.sum())
        if recent.sum() >= 3:
            x = (v.pass_s[m_idx][recent] - t) / 60.0
            y = dev[recent]
            if np.ptp(x) > 0:
                f["dev_slope"] = float(np.polyfit(x, y, 1)[0])
        # темп набора опоздания на своих последних участках (без отстоев)
        mi = m_idx[recent]
        if len(mi) >= 2:
            adj = np.flatnonzero(np.diff(mi) == 1)
            if len(adj):
                a, b = mi[adj], mi[adj] + 1
                gap = v.tb[b] - v.tb[a]
                ok = gap <= LAYOVER_S
                if ok.any() and gap[ok].sum() > 0:
                    delta = (v.pass_s[b] - v.pass_s[a]) - gap
                    f["own_rate"] = float(delta[ok].sum() / gap[ok].sum())
        if np.isfinite(f["cur_dev_s"]):
            f["cur_minus_dev1"] = f["cur_dev_s"] - dev[-1]
        return lm

    @staticmethod
    def _movement_features(f: dict[str, float], v: _Vehicle, t: float, g: int) -> None:
        if g < 0:
            return
        f["gps_age"] = t - v.t[g]
        for w in MOVE_WINDOWS_S:
            lo = int(np.searchsorted(v.t, t - w, side="right"))
            spd = v.speed[lo : g + 1]
            f[f"npts_{w}"] = float(len(spd))
            spd = spd[np.isfinite(spd)]
            if len(spd):
                f[f"spd_{w}"] = float(spd.mean())
                f[f"stopfrac_{w}"] = float(np.mean(spd < STOP_SPEED_KMH))
        lo = int(np.searchsorted(v.t, t - 3600.0, side="right"))
        spd = v.speed[lo : g + 1]
        moving = np.flatnonzero(spd >= STOP_SPEED_KMH)
        f["stop_dur"] = t - (v.t[lo + moving[-1]] if len(moving) else v.t[lo] if lo <= g else t)
        lo = int(np.searchsorted(v.t, t - 600.0, side="right"))
        times = np.r_[v.t[max(lo - 1, 0) : g + 1], t]
        f["max_gap_10m"] = float(np.diff(times).max()) if len(times) > 1 else np.nan
        h = int(np.searchsorted(v.t, t - 300.0, side="right")) - 1
        if h >= 0:
            f["dist_5m"] = _dist_m(v.lon[h], v.lat[h], v.lon[g], v.lat[g])

    @staticmethod
    def _position_features(f: dict[str, float], v: _Vehicle, t: float, g: int, lm: int | None) -> int | None:
        """Привязка текущей GPS-позиции к плановой последовательности; возвращает индекс отрезка."""
        n = len(v.tb)
        if g < 0 or n < 2 or t - v.t[g] > MAX_GPS_AGE_S:
            return None  # нет свежей позиции (ТС выключено / потеря связи)
        if lm is not None and v.tb[lm] >= t - 5400.0:
            start = lm
        else:
            start = max(int(np.searchsorted(v.tb, t - 1800.0)) - 1, 0)
        end = min(start + POS_LOOKAHEAD, n - 1)
        if end <= start:
            return None
        k = np.arange(start, end)
        plon, plat = v.lon[g], v.lat[g]
        sx = np.cos(np.deg2rad(plat)) * _M_PER_DEG
        ax, ay = (v.slon[k] - plon) * sx, (v.slat[k] - plat) * _M_PER_DEG
        bx, by = (v.slon[k + 1] - plon) * sx, (v.slat[k + 1] - plat) * _M_PER_DEG
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(l2 > 0, -(ax * dx + ay * dy) / l2, 0.0)
        u = np.clip(np.nan_to_num(u), 0.0, 1.0)
        d = np.hypot(ax + u * dx, ay + u * dy)
        if not np.isfinite(d).any():
            return None
        j = int(np.flatnonzero(d <= np.nanmin(d) + POS_TIE_M)[0])
        kk = int(k[j])
        uu = float(u[j])
        tb = v.tb
        gap = tb[kk + 1] - tb[kk]
        if gap <= LAYOVER_S:
            # ТС стоит у конечной: считаем, что оно в окне отстоя, а не «опаздывает» к прибытию / раньше
            # отправления (иначе отклонение растёт всё время ожидания)
            if (
                kk + 2 < n
                and tb[kk + 2] - tb[kk + 1] > LAYOVER_S
                and _dist_m(plon, plat, v.slon[kk + 1], v.slat[kk + 1]) < TERMINAL_M
            ):
                kk, uu = kk + 1, 0.0
            elif (
                kk > 0
                and tb[kk] - tb[kk - 1] > LAYOVER_S
                and _dist_m(plon, plat, v.slon[kk], v.slat[kk]) < TERMINAL_M
            ):
                kk, uu = kk - 1, 1.0
            gap = tb[kk + 1] - tb[kk]
        # на отстое план — окно [прибытие, отправление]: ожидание в нём не считается опозданием
        plan_at = float(np.clip(t, tb[kk], tb[kk + 1])) if gap > LAYOVER_S else tb[kk] + uu * gap
        f["pos_delay"] = t - plan_at
        f["pos_dist"] = float(d[j])
        f["pos_u"] = uu
        f["pos_on_layover"] = float(gap > LAYOVER_S)
        if np.isfinite(f["cur_dev_s"]):
            f["cur_minus_pos"] = f["cur_dev_s"] - f["pos_delay"]
        decided = np.flatnonzero(v.conf_s <= t)
        f["n_unconfirmed"] = float(kk - decided[-1]) if len(decided) else np.nan
        f["dist_next_stop"] = _dist_m(plon, plat, v.slon[kk + 1], v.slat[kk + 1])
        return kk

    def _route_features(
        self, f: dict[str, float], v: _Vehicle, t: float, g: int, ti: int, pos_seg: int | None, lm: int | None
    ) -> None:
        tb = v.tb
        n = len(tb)
        f["tgt_gap_prev"] = tb[ti] - tb[ti - 1] if ti > 0 else np.nan
        f["tgt_gap_next"] = tb[ti + 1] - tb[ti] if ti + 1 < n else np.nan
        gaps_all = np.diff(tb)
        big = np.flatnonzero(gaps_all > LAYOVER_S)  # отрезок i → i+1 — отстой
        before = big[big < ti]
        f["tgt_trip_pos"] = float(ti - (before[-1] + 1 if len(before) else 0))
        after = big[big >= ti]
        f["tgt_trip_left"] = float((after[0] if len(after) else n - 1) - ti)
        if g >= 0:
            f["dist_target"] = _dist_m(v.lon[g], v.lat[g], v.slon[ti], v.slat[ti])
        med, cnt = self._stop_store.median(v.loc[ti], t, NET_WINDOW_S)
        f["tgt_loc_dev"], f["tgt_loc_n"] = med, float(cnt)
        if lm is not None and ti > lm:
            f["maxgap_anchor"] = float(gaps_all[lm:ti].max())

        pos_ok = pos_seg is not None and f["pos_dist"] < 300.0
        dev_fresh = f["dev_1"] if f["age_1"] < 2400.0 else np.nan
        cur = _first_finite(f["pos_delay"] if pos_ok else np.nan, dev_fresh, f["cur_dev_s"], f["dev_1"])
        f["cur_best"] = cur
        base = pos_seg if pos_seg is not None else lm
        if base is None:
            return
        f["n_to_target"] = float(ti - base)
        plan_at = t - f["pos_delay"] if pos_seg is not None else tb[base]
        f["plan_to_target"] = tb[ti] - plan_at
        if ti <= base:
            f["maxgap_ahead"] = f["layover_ahead"] = f["n_layover_ahead"] = f["run_plan_ahead"] = 0.0
            f["has_layover"] = 0.0
            f["phys"] = cur
            return
        ahead = gaps_all[base + 1 : ti]  # отрезки после текущего до цели
        lay = ahead > LAYOVER_S
        cur_gap = gaps_all[base]
        cur_u = f["pos_u"] if pos_seg is not None else 0.0
        cur_run = (1.0 - cur_u) * cur_gap if cur_gap <= LAYOVER_S else 0.0
        f["maxgap_ahead"] = float(ahead.max()) if len(ahead) else 0.0
        f["layover_ahead"] = float(ahead[lay].sum())
        f["n_layover_ahead"] = float(lay.sum())
        f["run_plan_ahead"] = float(ahead[~lay].sum() + cur_run)
        f["has_layover"] = float(lay.any() or (pos_seg is not None and cur_gap > LAYOVER_S))

        slack = f["layover_ahead"]
        f["phys"] = max(cur - slack, 0.0) if slack > 0 else cur
        if np.isfinite(f["own_rate"]):
            f["own_extra"] = f["own_rate"] * f["run_plan_ahead"]
            f["phys_own"] = _absorb(cur, f["own_extra"], slack)

        # онлайн-статистика участков по всем ТС: медианный прирост отклонения на каждом участке до цели
        extra, covered, total = 0.0, 0, 0
        for i in range(base + 1, ti):
            if gaps_all[i] > LAYOVER_S:
                continue
            total += 1
            med, cnt = self._seg_store.median(v.loc[i] * _SEG_MUL + v.loc[i + 1], t, NET_WINDOW_S)
            if cnt:
                extra += med
                covered += 1
        f["net_extra"] = extra
        f["net_cover"] = covered / total if total else np.nan
        f["phys_net"] = _absorb(cur, extra, slack)


def _first_finite(*values: float) -> float:
    for x in values:
        if x is not None and np.isfinite(x):
            return float(x)
    return np.nan


def _absorb(cur: float, extra: float, slack: float) -> float:
    """Текущее отклонение + прирост на пути, отстой на конечной поглощает опоздание (но не опережение)."""
    if not np.isfinite(cur):
        return np.nan
    x = cur + extra
    return max(x - slack, 0.0) if slack > 0 else x


def build_features(points: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Признаки для пачки прогнозных точек.

    Args:
        points: tr_id, T, target_stop_id, target_time_begin, cur_dev_s (формат labels / points.csv).
        ctx: контекст с телеметрией, расписанием и проходами детектора.

    Returns:
        DataFrame с колонками :data:`FEATURE_NAMES` и индексом ``points``.
    """
    t = to_seconds(points["T"])
    tgt = to_seconds(points["target_time_begin"])
    cur = (
        points["cur_dev_s"].to_numpy(dtype=np.float64)
        if "cur_dev_s" in points.columns
        else np.full(len(points), np.nan)
    )
    tr = points["tr_id"].to_numpy(dtype=np.int64)
    stop = points["target_stop_id"].to_numpy(dtype=np.int64)
    rows = [
        ctx.point_features(int(tr[i]), float(t[i]), int(stop[i]), float(tgt[i]), float(cur[i]))
        for i in range(len(points))
    ]
    return pd.DataFrame(rows, index=points.index, columns=FEATURE_NAMES, dtype=np.float64)
