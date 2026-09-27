"""Человекочитаемые подписи признаков для причин задержки и карточки инцидента.

Словарь покрывает все признаки :data:`shared.features.FEATURE_NAMES`. Для каждого признака — подпись
по-русски, единица измерения значения и подсказка «к какой причине относится» (коды причин —
``docs/api-contract.md`` §1). Подсказка причины — ориентир для правил predictor, а не классификатор: ``None``
значит, что признак описывает контекст прогноза (горизонт, маршрут до цели), а не причину.

Пример::

    >>> label("stop_dur")
    'Длительность текущей стоянки'
    >>> label("unknown_feature")
    'unknown_feature'
"""

from __future__ import annotations

from dataclasses import dataclass

# коды причин (docs/api-contract.md §1)
DWELL_LONG = "dwell_long"
SLOW_SEGMENT = "slow_segment"
LAYOVER = "layover"
ACCUMULATED_DELAY = "accumulated_delay"
GPS_LOST = "gps_lost"


@dataclass(frozen=True)
class FeatureInfo:
    """Описание признака для интерфейса.

    Attributes:
        label: подпись для диспетчера.
        unit: единица значения (``с``, ``м``, ``км/ч``, ``с/с``, ``шт``, ``доля``, ``ч``; пусто — флаг 0/1).
        cause: код причины из контракта API, к которой относится признак (``None`` — контекст).
    """

    label: str
    unit: str
    cause: str | None = None


_WIN = {60: "1 мин", 180: "3 мин", 300: "5 мин", 600: "10 мин"}

FEATURE_INFO: dict[str, FeatureInfo] = {
    # время
    "hour": FeatureInfo("Время суток", "ч"),
    "lead_s": FeatureInfo("Время до планового прибытия", "с"),
    # подсказка и отклонение по детектору
    "cur_dev_s": FeatureInfo("Текущее отклонение от графика", "с", ACCUMULATED_DELAY),
    "dev_1": FeatureInfo("Отклонение на последней пройденной остановке", "с", ACCUMULATED_DELAY),
    "dev_2": FeatureInfo("Отклонение на предпоследней остановке", "с", ACCUMULATED_DELAY),
    "dev_3": FeatureInfo("Отклонение три остановки назад", "с", ACCUMULATED_DELAY),
    "dev_med5": FeatureInfo("Медианное отклонение на 5 последних остановках", "с", ACCUMULATED_DELAY),
    "dev_slope": FeatureInfo("Рост отклонения за последние 30 мин", "с/мин", ACCUMULATED_DELAY),
    "age_1": FeatureInfo("Время с прохода последней остановки", "с"),
    "conf_age_1": FeatureInfo("Время с подтверждения последнего прохода", "с"),
    "n_pass_30m": FeatureInfo("Остановок пройдено за 30 мин", "шт"),
    "n_skip_recent": FeatureInfo("Остановки без отметки прохода (из 5 последних)", "шт", GPS_LOST),
    "own_rate": FeatureInfo("Темп набора опоздания на перегонах", "с/с", SLOW_SEGMENT),
    "cur_minus_dev1": FeatureInfo("Изменение отклонения после последней остановки", "с", ACCUMULATED_DELAY),
    # позиция по GPS относительно плана
    "pos_delay": FeatureInfo("Отклонение по текущей GPS-позиции", "с", ACCUMULATED_DELAY),
    "pos_dist": FeatureInfo("Удаление от линии маршрута", "м", GPS_LOST),
    "pos_u": FeatureInfo("Пройденная доля текущего перегона", "доля"),
    "pos_on_layover": FeatureInfo("ТС на отстое у конечной", "", LAYOVER),
    "cur_minus_pos": FeatureInfo("Расхождение отклонения и GPS-позиции", "с", ACCUMULATED_DELAY),
    "n_unconfirmed": FeatureInfo("Остановки позади без подтверждённого прохода", "шт", GPS_LOST),
    "cur_best": FeatureInfo("Оценка текущего отклонения", "с", ACCUMULATED_DELAY),
    # движение
    **{f"spd_{w}": FeatureInfo(f"Средняя скорость за {s}", "км/ч", SLOW_SEGMENT) for w, s in _WIN.items()},
    **{f"stopfrac_{w}": FeatureInfo(f"Доля стоянки за {s}", "доля", DWELL_LONG) for w, s in _WIN.items()},
    **{f"npts_{w}": FeatureInfo(f"GPS-отметок за {s}", "шт", GPS_LOST) for w, s in _WIN.items()},
    "stop_dur": FeatureInfo("Длительность текущей стоянки", "с", DWELL_LONG),
    "gps_age": FeatureInfo("Давность последней GPS-отметки", "с", GPS_LOST),
    "max_gap_10m": FeatureInfo("Наибольший пропуск GPS за 10 мин", "с", GPS_LOST),
    "dist_5m": FeatureInfo("Пройдено за 5 мин", "м", SLOW_SEGMENT),
    "dist_next_stop": FeatureInfo("Расстояние до следующей остановки", "м"),
    "dist_target": FeatureInfo("Расстояние до целевой остановки", "м"),
    # маршрут до цели
    "n_to_target": FeatureInfo("Остановок до цели", "шт"),
    "plan_to_target": FeatureInfo("Плановое время до цели", "с"),
    "run_plan_ahead": FeatureInfo("Плановое время движения до цели без отстоя", "с"),
    "maxgap_ahead": FeatureInfo("Наибольший плановый интервал до цели", "с", LAYOVER),
    "layover_ahead": FeatureInfo("Плановый отстой на конечной до цели", "с", LAYOVER),
    "n_layover_ahead": FeatureInfo("Отстоев на конечной до цели", "шт", LAYOVER),
    "has_layover": FeatureInfo("Отстой на конечной до цели", "", LAYOVER),
    "maxgap_anchor": FeatureInfo("Наибольший плановый интервал от последней остановки", "с", LAYOVER),
    "tgt_gap_prev": FeatureInfo("Плановый интервал перед целевой остановкой", "с", LAYOVER),
    "tgt_gap_next": FeatureInfo("Плановый интервал после целевой остановки", "с", LAYOVER),
    "tgt_trip_pos": FeatureInfo("Номер целевой остановки в рейсе", "шт"),
    "tgt_trip_left": FeatureInfo("Остановок рейса после цели", "шт"),
    # «физика»
    "phys": FeatureInfo("Отклонение с учётом отстоя до цели", "с", ACCUMULATED_DELAY),
    "own_extra": FeatureInfo("Ожидаемый прирост опоздания по своему темпу", "с", SLOW_SEGMENT),
    "phys_own": FeatureInfo("Оценка задержки по своему темпу с учётом отстоя", "с", SLOW_SEGMENT),
    # участки и сеть (все ТС)
    "net_extra": FeatureInfo("Ожидаемый прирост опоздания на участках по другим ТС", "с", SLOW_SEGMENT),
    "net_cover": FeatureInfo("Покрытие участков данными других ТС", "доля"),
    "phys_net": FeatureInfo("Оценка задержки по участкам с учётом отстоя", "с", SLOW_SEGMENT),
    "tgt_loc_dev": FeatureInfo("Отклонение других ТС на целевой остановке", "с", ACCUMULATED_DELAY),
    "tgt_loc_n": FeatureInfo("Проходов других ТС через цель за час", "шт"),
    "net_rate_30m": FeatureInfo("Темп набора опоздания по сети за 30 мин", "с/с", SLOW_SEGMENT),
    "net_dev_30m": FeatureInfo("Медианное отклонение по сети за 30 мин", "с", ACCUMULATED_DELAY),
}

FEATURE_LABELS: dict[str, str] = {name: info.label for name, info in FEATURE_INFO.items()}


def label(feature: str) -> str:
    """Подпись признака; для неизвестного признака — его имя."""
    info = FEATURE_INFO.get(feature)
    return info.label if info else feature


def cause_hint(feature: str) -> str | None:
    """Код причины, к которой относится признак (``None`` — контекст или неизвестный признак)."""
    info = FEATURE_INFO.get(feature)
    return info.cause if info else None
