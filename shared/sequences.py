"""Последовательности телеметрии точки ``(tr_id, T)`` — вход последовательной модели (GRU / Transformer).

Общий код офлайна и онлайна, как :mod:`shared.features`: вход — тот же
:class:`~shared.features.FeatureContext` (телеметрия, плановое расписание, проходы детектора).
Последовательность зависит только от ТС и момента ``T`` (не от целевой остановки), поэтому на потоке её
достаточно строить раз на ТС за тик.

Окно — :data:`SEQ_WINDOW_S` (20 мин) до ``T`` на сетке :data:`SEQ_STEP_S` (15 с): :data:`SEQ_LEN` = 80 шагов.
Шаг ``k`` описывает интервал ``(t_k − 15 с, t_k]``, ``t_k = T − (SEQ_LEN − 1 − k) · 15 с``; последний шаг
заканчивается ровно в ``T``. Для шага используются только:

* GPS-точки с ``event_time ≤ t_k``;
* проходы детектора с ``confirmed_at ≤ t_k``;
* плановое расписание (``time_begin``, координаты остановок).

Так как ``t_k ≤ T``, данные после ``T`` не читаются (проверяется ``tests/test_sequences.py``). Каналы
(:data:`SEQ_CHANNELS`) нормированы фиксированными масштабами и обрезаны, пропуски — нули плюс
канал-маска:

====================  ============================================================================
``obs``               1 — в интервале есть GPS-точка (маска пропусков GPS)
``speed``             средняя скорость точек интервала, км/ч / 50
``stopped``           доля точек интервала со скоростью < 2 км/ч
``move``              смещение между последними позициями на концах интервала, м / 200
``gps_age``           возраст последней точки на ``t_k``, с / 300 (≤ 3)
``has_dev``           1 — есть подтверждённый проход остановки
``dev``               отклонение по последнему подтверждённому проходу, с / 300 (±5)
``dev_age``           время с этого прохода, с / 600 (≤ 3)
``has_next``          1 — следующая плановая остановка известна
``next_dist``         расстояние от последней позиции до следующей плановой остановки, км (≤ 5)
``next_late``         ``t_k`` − план следующей остановки, с / 300 (±5): > 0 — план уже прошёл
``next_layover``      1 — перед следующей остановкой плановый отстой (ТС на конечной)
====================  ============================================================================

Следующая плановая остановка — следующая по плану после последней решённой детектором (проход подтверждён
или остановка пропущена, ``confirmed_at ≤ t_k``); если решённых нет в последние 90 мин — первая с планом
``≥ t_k``. Если ТС ближе 150 м к следующей остановке, а это конечная прибытия (после неё плановый отстой),
следующей считается конечная отправления: визит к конечной детектор подтверждает только при отъезде.
"""

from __future__ import annotations

import hashlib
import inspect
import sys

import numpy as np
import pandas as pd

from shared.features import LAYOVER_S, STOP_SPEED_KMH, TERMINAL_M, FeatureContext, to_seconds

SEQ_STEP_S = 15.0
SEQ_WINDOW_S = 1200.0
SEQ_LEN = int(SEQ_WINDOW_S // SEQ_STEP_S)
SEQ_CHANNELS: list[str] = [
    "obs",
    "speed",
    "stopped",
    "move",
    "gps_age",
    "has_dev",
    "dev",
    "dev_age",
    "has_next",
    "next_dist",
    "next_late",
    "next_layover",
]
N_CHANNELS = len(SEQ_CHANNELS)
ANCHOR_MAX_AGE_S = 5400.0  # решённая остановка старше (по плану) — не опора для «следующей остановки»

_M_PER_DEG = np.pi / 180.0 * 6_371_008.8
_CH = {name: i for i, name in enumerate(SEQ_CHANNELS)}


def sequence_version() -> str:
    """Хэш исходника модуля: версия формата последовательностей (пишется в манифест модели)."""
    src = inspect.getsource(sys.modules[__name__])
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def step_times(t: float) -> np.ndarray:
    """Концы интервалов сетки для момента ``t`` (секунды эпохи), по возрастанию; последний равен ``t``."""
    return t - SEQ_STEP_S * np.arange(SEQ_LEN - 1, -1, -1, dtype=np.float64)


def _dist_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    coslat = np.cos(np.deg2rad(0.5 * (lat1 + lat2)))
    return np.hypot((lon2 - lon1) * coslat * _M_PER_DEG, (lat2 - lat1) * _M_PER_DEG)


class _Prefix:
    """Префиксные суммы по телеметрии ТС и порядок решений детектора.

    Префиксная сумма в позиции ``i`` зависит только от точек ``< i``, поэтому разность по интервалу
    ``(a, b]`` не читает точки после ``b`` и совпадает бит в бит для обрезанного трека.
    """

    def __init__(self, v) -> None:
        spd = v.speed
        ok = np.isfinite(spd)
        self.speed_sum = np.r_[0.0, np.cumsum(np.where(ok, spd, 0.0))]
        self.speed_n = np.r_[0, np.cumsum(ok)]
        self.stop_n = np.r_[0, np.cumsum(ok & (spd < STOP_SPEED_KMH))]
        # решения детектора в порядке подтверждения; накопленный максимум индекса плана
        dec = np.flatnonzero(np.isfinite(v.conf_s))
        order = dec[np.argsort(v.conf_s[dec], kind="stable")]
        self.dec_conf = v.conf_s[order]
        self.dec_max = np.maximum.accumulate(order) if len(order) else order
        mat = dec[np.isfinite(v.pass_s[dec])]
        order = mat[np.argsort(v.conf_s[mat], kind="stable")]
        self.mat_conf = v.conf_s[order]
        self.mat_max = np.maximum.accumulate(order) if len(order) else order


def _last_by_plan(conf: np.ndarray, run_max: np.ndarray, tk: np.ndarray) -> np.ndarray:
    """Для каждого ``t_k``: максимальный индекс плана среди решений с ``confirmed_at ≤ t_k`` (−1 — нет)."""
    j = np.searchsorted(conf, tk, side="right") - 1
    out = np.full(len(tk), -1, dtype=np.int64)
    ok = j >= 0
    out[ok] = run_max[j[ok]]
    return out


class SequenceBuilder:
    """Построитель последовательностей по контексту признаков (кэширует префиксные суммы по ТС).

    Args:
        ctx: контекст :class:`~shared.features.FeatureContext` — тот же, что для :func:`build_features`.
    """

    def __init__(self, ctx: FeatureContext) -> None:
        self.ctx = ctx
        self._prefix: dict[int, _Prefix] = {}

    def _get_prefix(self, tr_id: int, v) -> _Prefix:
        p = self._prefix.get(tr_id)
        if p is None:
            p = self._prefix[tr_id] = _Prefix(v)
        return p

    def point(self, tr_id: int, t: float) -> np.ndarray:
        """Последовательность ``(SEQ_LEN, N_CHANNELS)`` float32 для ТС ``tr_id`` на момент ``t`` (с эпохи)."""
        out = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float32)
        v = self.ctx.vehicles.get(int(tr_id))
        if v is None:
            return out
        p = self._get_prefix(int(tr_id), v)
        tk = step_times(float(t))
        x = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float64)

        # --- GPS в интервале (t_k − step, t_k] ---
        hi = np.searchsorted(v.t, tk, side="right")
        lo = np.searchsorted(v.t, tk - SEQ_STEP_S, side="right")
        x[:, _CH["obs"]] = hi > lo
        n_spd = p.speed_n[hi] - p.speed_n[lo]
        has_spd = n_spd > 0
        denom = np.maximum(n_spd, 1)
        x[:, _CH["speed"]] = np.where(has_spd, (p.speed_sum[hi] - p.speed_sum[lo]) / denom, 0.0) / 50.0
        x[:, _CH["stopped"]] = np.where(has_spd, (p.stop_n[hi] - p.stop_n[lo]) / denom, 0.0)
        g = hi - 1  # последняя точка ≤ t_k
        gp = lo - 1  # последняя точка ≤ t_k − step
        has_g = g >= 0
        gi = np.maximum(g, 0)
        if len(v.t):
            x[:, _CH["gps_age"]] = np.where(has_g, np.clip((tk - v.t[gi]) / 300.0, 0.0, 3.0), 3.0)
            both = has_g & (gp >= 0)
            gpi = np.maximum(gp, 0)
            move = _dist_m(v.lon[gpi], v.lat[gpi], v.lon[gi], v.lat[gi])
            x[:, _CH["move"]] = np.where(both, np.clip(move / 200.0, 0.0, 5.0), 0.0)
        else:
            x[:, _CH["gps_age"]] = 3.0

        # --- отклонение по последнему подтверждённому проходу (порядок плана) ---
        lm = _last_by_plan(p.mat_conf, p.mat_max, tk)
        has_dev = lm >= 0
        lmi = np.maximum(lm, 0)
        if len(v.tb):
            dev = v.pass_s[lmi] - v.tb[lmi]
            x[:, _CH["has_dev"]] = has_dev
            x[:, _CH["dev"]] = np.where(has_dev, np.clip(dev / 300.0, -5.0, 5.0), 0.0)
            age = tk - v.pass_s[lmi]
            x[:, _CH["dev_age"]] = np.where(has_dev, np.clip(age / 600.0, 0.0, 3.0), 0.0)

            # --- следующая плановая остановка ---
            n = len(v.tb)
            d = _last_by_plan(p.dec_conf, p.dec_max, tk)
            di = np.maximum(d, 0)
            anchored = (d >= 0) & (v.tb[di] >= tk - ANCHOR_MAX_AGE_S)
            nxt = np.where(anchored, d + 1, np.searchsorted(v.tb, tk, side="left"))
            has_next = nxt < n
            ni = np.minimum(nxt, n - 1)
            x[:, _CH["has_next"]] = has_next
            ok = has_next & has_g
            if len(v.t):
                dist = _dist_m(v.lon[gi], v.lat[gi], v.slon[ni], v.slat[ni])
                # ТС стоит у конечной прибытия (визит ещё не подтверждён — детектор ждёт отъезда): следующая —
                # конечная отправления после отстоя, как в привязке позиции shared.features
                nj = np.minimum(ni + 1, n - 1)
                at_term = ok & (dist < TERMINAL_M) & (ni + 1 < n) & (v.tb[nj] - v.tb[ni] > LAYOVER_S)
                ni = np.where(at_term, nj, ni)
                dist = np.where(at_term, _dist_m(v.lon[gi], v.lat[gi], v.slon[ni], v.slat[ni]), dist)
                dist = np.where(np.isfinite(dist), dist, 0.0)
                x[:, _CH["next_dist"]] = np.where(ok, np.clip(dist / 1000.0, 0.0, 5.0), 0.0)
            x[:, _CH["next_late"]] = np.where(has_next, np.clip((tk - v.tb[ni]) / 300.0, -5.0, 5.0), 0.0)
            prev_gap = np.where(ni > 0, v.tb[ni] - v.tb[np.maximum(ni - 1, 0)], 0.0)
            x[:, _CH["next_layover"]] = has_next & (prev_gap > LAYOVER_S)
        out[:] = x
        return out

    def build(self, points: pd.DataFrame) -> np.ndarray:
        """Последовательности пачки точек (формат labels / points.csv: ``tr_id``, ``T``).

        Returns:
            float32 ``(len(points), SEQ_LEN, N_CHANNELS)`` в порядке ``points``; точки с одинаковыми
            ``(tr_id, T)`` получают одну и ту же последовательность.
        """
        out = np.zeros((len(points), SEQ_LEN, N_CHANNELS), dtype=np.float32)
        if len(points) == 0:
            return out
        t = to_seconds(points["T"])
        tr = points["tr_id"].to_numpy(dtype=np.int64)
        seen: dict[tuple[int, float], int] = {}
        for i in range(len(points)):
            key = (int(tr[i]), float(t[i]))
            j = seen.get(key)
            if j is None:
                seen[key] = i
                out[i] = self.point(key[0], key[1])
            else:
                out[i] = out[j]
        return out


def build_sequences(points: pd.DataFrame, ctx: FeatureContext) -> np.ndarray:
    """Последовательности для пачки прогнозных точек (как :func:`shared.features.build_features`).

    Args:
        points: ``tr_id``, ``T`` (прочие колонки не используются).
        ctx: контекст с телеметрией, расписанием и проходами детектора.

    Returns:
        float32 ``(len(points), SEQ_LEN, N_CHANNELS)``.
    """
    return SequenceBuilder(ctx).build(points)
