"""Дообучение на потоке («Переобучить» в админке): поправка прогноза активной версии по закрытым прогнозам.

Predictor закрывает прогноз фактом прохождения остановки (детектор на потоке), так что в ``predictions``
копится разметка: прогноз версии и факт. Дообучение подбирает поправку ``a + b · прогноз`` (медианная
регрессия, как MAE модели) на ранних 70 % закрытых прогнозов версии, меряет MAE до и после на поздних 30 % и
регистрирует результат новой версией ``<база>-online<N>``, если поправка снизила MAE: копия каталога базы
(модели не меняются) + ``online_calibration`` в манифесте (:mod:`ml.inference`). Версия не включается
сама — её активируют в админке.

Признаков в журнале нет, поэтому это не переобучение CatBoost/GRU, а калибровка их выхода: быстро (секунды на
CPU), без датасета в образе и без риска испортить модель. Демо-поток — тестовый день по кругу, поэтому выигрыш
на отложенной части оптимистичен (поправка видела тот же день на прошлых кругах).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.inference import MANIFEST_NAME, ModelBundle

MIN_ROWS = 300
"""Меньше закрытых прогнозов — не дообучаем: поправка по горстке точек шумная."""
FIT_SHARE = 0.7
B_RANGE = (0.5, 1.5)
A_LIMIT_S = 120.0


class RetrainError(ValueError):
    """Дообучение невозможно (мало данных, нет каталога моделей)."""


@dataclass(frozen=True)
class OnlineFit:
    """Поправка и её проверка на отложенной (более поздней) части прогнозов."""

    a: float
    b: float
    n_fit: int
    n_eval: int
    mae_before_s: float
    mae_after_s: float


def fit_l1(x: np.ndarray, y: np.ndarray, iters: int = 60) -> tuple[float, float]:
    """``y ≈ a + b·x`` по минимуму MAE (IRLS), ``b`` в :data:`B_RANGE`, ``|a|`` ≤ :data:`A_LIMIT_S`."""
    design = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - design @ coef), 1.0)
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
    b = float(np.clip(coef[1], *B_RANGE))
    a = float(np.clip(np.median(y - b * x), -A_LIMIT_S, A_LIMIT_S))
    return a, b


def fit_online(pred: np.ndarray, actual: np.ndarray) -> OnlineFit:
    """Поправка по закрытым прогнозам (в порядке закрытия): ранние :data:`FIT_SHARE` — обучение, поздние —
    проверка.

    Raises:
        RetrainError: закрытых прогнозов меньше :data:`MIN_ROWS`.
    """
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    if len(pred) < MIN_ROWS:
        raise RetrainError(f"мало закрытых прогнозов для дообучения: {len(pred)} < {MIN_ROWS}")
    cut = int(len(pred) * FIT_SHARE)
    a, b = fit_l1(pred[:cut], actual[:cut])
    x, y = pred[cut:], actual[cut:]
    return OnlineFit(
        a=a,
        b=b,
        n_fit=cut,
        n_eval=len(x),
        mae_before_s=float(np.mean(np.abs(y - x))),
        mae_after_s=float(np.mean(np.abs(y - (a + b * x)))),
    )


def next_version(root: Path, base: str) -> str:
    """Свободное имя ``<база>-online<N>``."""
    pattern = re.compile(rf"^{re.escape(base)}-online(\d+)$")
    taken = [int(m.group(1)) for d in root.iterdir() if (m := pattern.match(d.name))] if root.is_dir() else []
    return f"{base}-online{max(taken, default=0) + 1}"


def register_version(bundle: ModelBundle, fit: OnlineFit, root: Path, now: datetime | None = None) -> str:
    """Новая версия из ``bundle`` с поправкой ``fit`` (поверх поправки самой ``bundle``, если она есть).

    Файлы моделей копируются, манифест пишется последним: пока его нет, каталог не считается версией.

    Returns:
        Имя новой версии.
    """
    now = now or datetime.now(UTC)
    prev = bundle.manifest.get("online_calibration") or {}
    base = str(prev.get("base") or bundle.version)
    a0, b0 = bundle.online_shift or (0.0, 1.0)
    # поправка обучена на выходе bundle (уже с её поправкой): a + b·(a0 + b0·x)
    a, b = fit.a + fit.b * a0, fit.b * b0
    version = next_version(root, base)
    target = root / version
    shutil.copytree(bundle.path, target, ignore=shutil.ignore_patterns(MANIFEST_NAME))
    stamp = now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        **bundle.manifest,
        "version": version,
        "created_at": stamp,
        "description": (
            f"{base} + поправка по закрытым прогнозам потока: {fit.a:+.1f} с + {fit.b:.3f} × прогноз; "
            f"MAE на отложенных {fit.n_eval}: {fit.mae_before_s:.1f} → {fit.mae_after_s:.1f} с"
        ),
        # CV/test меряли базу, а не поправленную версию: не выдаём их за её метрики
        "metrics": {"online_holdout_mae_before": fit.mae_before_s, "online_holdout_mae": fit.mae_after_s},
        "online_calibration": {
            "a": a,
            "b": b,
            "base": base,
            "parent": bundle.version,
            "n_fit": fit.n_fit,
            "n_eval": fit.n_eval,
            "mae_before_s": round(fit.mae_before_s, 2),
            "mae_after_s": round(fit.mae_after_s, 2),
            "trained_at": stamp,
        },
    }
    tmp = target / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target / MANIFEST_NAME)
    return version
