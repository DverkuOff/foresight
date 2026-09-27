#!/usr/bin/env python3
"""Generate the Foresight Grafana dashboards (``deploy/grafana/dashboards/*.json``).

Dashboards are code: this script is the source of truth, the JSON files are its output, provisioned into
Grafana read-only (folder «Foresight»). After a change regenerate and commit both::

    python3 deploy/grafana/build_dashboards.py          # write the JSON files
    python3 deploy/grafana/build_dashboards.py --check  # exit 1 if the files are out of date

Metric names follow the exposition of the services (counters end with ``_total``); the contract is
``docs/observability.md``. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

Json = dict[str, Any]

DATASOURCE_UID = "foresight-prometheus"
DS: Json = {"type": "prometheus", "uid": DATASOURCE_UID}
OUT_DIR = Path(__file__).resolve().parent / "dashboards"
HOME_UID = "foresight-overview"
TAG = "foresight"

RATE = "$__rate_interval"
APP_JOBS = "ingest|predictor|api|replayer|ml-service"

# ---- thresholds and value mappings ----------------------------------------------------------------------


def steps(*pairs: tuple[str, float | None]) -> Json:
    """Absolute thresholds: ``("green", None), ("red", 1)`` → base green, red from 1."""
    return {"mode": "absolute", "steps": [{"color": color, "value": value} for color, value in pairs]}


def value_map(options: dict[str, tuple[str, str]]) -> list[Json]:
    """Value mappings ``{"1": ("UP", "green")}``."""
    return [
        {
            "type": "value",
            "options": {
                key: {"text": text, "color": color, "index": i}
                for i, (key, (text, color)) in enumerate(options.items())
            },
        }
    ]


UP_DOWN = value_map({"0": ("DOWN", "red"), "1": ("UP", "green")})
AVAILABLE = value_map({"0": ("недоступна", "red"), "1": ("доступна", "green")})
RED_AT_1 = steps(("green", None), ("red", 1))
LATENCY_S = steps(("green", None), ("yellow", 0.5), ("red", 1))  # event latency: target < 1 s
BATCH_S = steps(("green", None), ("yellow", 0.1), ("red", 1))  # tick / batch: target < 100 ms
LAG = steps(("green", None), ("yellow", 100), ("red", 500))  # consumer lag: target ≈ 0


def color(name: str, fixed: str) -> Json:
    """Override: a fixed color for the series ``name``."""
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": fixed}}],
    }


# ---- queries -----------------------------------------------------------------------------------------------


def q(expr: str, legend: str = "", *, fmt: str = "time_series", instant: bool = False) -> Json:
    """A Prometheus query target (``refId`` is assigned by :class:`Board`)."""
    return {
        "datasource": DS,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend or "__auto",
        "format": fmt,
        "range": not instant,
        "instant": instant,
    }


def sel(*matchers: str) -> str:
    """Label selector from matchers (empty string for none)."""
    parts = [m for m in matchers if m]
    return "{" + ", ".join(parts) + "}" if parts else ""


def rate(metric: str, matchers: str = "", by: str = "", window: str = RATE) -> str:
    """``sum [by (...)] (rate(metric{...}[window]))``."""
    agg = f"sum by ({by})" if by else "sum"
    return f"{agg} (rate({metric}{matchers}[{window}]))"


def per_minute(metric: str, matchers: str = "", by: str = "") -> str:
    """Events in the last minute: ``sum [by (...)] (increase(metric{...}[1m]))`` (rare events)."""
    agg = f"sum by ({by})" if by else "sum"
    return f"{agg} (increase({metric}{matchers}[1m]))"


def quantile(phi: float, histogram: str, matchers: str = "", by: str = "", window: str = RATE) -> str:
    """``histogram_quantile`` over ``<histogram>_bucket`` summed by ``le`` (and ``by``)."""
    group = f"{by}, le" if by else "le"
    return f"histogram_quantile({phi}, sum by ({group}) (rate({histogram}_bucket{matchers}[{window}])))"


def mean(histogram: str, matchers: str = "", window: str = RATE) -> str:
    """Mean of a histogram: ``rate(_sum) / rate(_count)``."""
    return (
        f"sum(rate({histogram}_sum{matchers}[{window}])) / sum(rate({histogram}_count{matchers}[{window}]))"
    )


def quantiles(histogram: str, matchers: str = "", *, with_mean: bool = True) -> list[Json]:
    """p50 / p95 / p99 (and the mean) of a histogram."""
    targets = [q(quantile(phi, histogram, matchers), f"p{round(phi * 100)}") for phi in (0.5, 0.95, 0.99)]
    if with_mean:
        targets.append(q(mean(histogram, matchers), "среднее"))
    return targets


# ---- panels ------------------------------------------------------------------------------------------------


def _field_defaults(
    unit: str,
    thresholds: Json | None,
    mappings: list[Json] | None,
    decimals: int | None,
    color_mode: str,
) -> Json:
    defaults: Json = {
        "unit": unit,
        "color": {"mode": color_mode},
        "thresholds": thresholds or steps(("green", None)),
        "mappings": mappings or [],
        "noValue": "нет данных",
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return defaults


def timeseries(
    title: str,
    targets: Sequence[Json],
    unit: str = "short",
    description: str = "",
    *,
    stack: bool = False,
    thresholds: Json | None = None,
    threshold_style: str = "off",
    overrides: Sequence[Json] = (),
    decimals: int | None = None,
    soft_min: float | None = 0,
    step: bool = False,
    axis_label: str = "",
) -> Json:
    """Time series panel. ``threshold_style`` ``dashed`` draws the threshold lines (targets)."""
    defaults = _field_defaults(unit, thresholds, None, decimals, "palette-classic")
    defaults["custom"] = {
        "drawStyle": "line",
        "lineInterpolation": "stepAfter" if step else "linear",
        "lineWidth": 1,
        "fillOpacity": 25 if stack else 10,
        "gradientMode": "none",
        "showPoints": "never",
        "pointSize": 4,
        "spanNulls": False,
        "insertNulls": False,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        "axisPlacement": "auto",
        "axisLabel": axis_label,
        "axisSoftMin": soft_min,
        "axisSoftMax": 1 if decimals == 0 else None,  # integer counts: all zeros show 0…1, not 0…100
        "thresholdsStyle": {"mode": threshold_style},
    }
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": list(targets),
        "fieldConfig": {"defaults": defaults, "overrides": list(overrides)},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True, "calcs": []},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(
    title: str,
    targets: Sequence[Json],
    unit: str = "short",
    description: str = "",
    *,
    thresholds: Json | None = None,
    mappings: list[Json] | None = None,
    color_mode: str = "value",
    graph: bool = True,
    text_mode: str = "auto",
    decimals: int | None = None,
) -> Json:
    """Stat panel (last non-null value)."""
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": list(targets),
        "fieldConfig": {
            "defaults": _field_defaults(unit, thresholds, mappings, decimals, "thresholds"),
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "auto",
            "textMode": text_mode,
            "colorMode": color_mode,
            "graphMode": "area" if graph else "none",
            "justifyMode": "auto",
            "wideLayout": True,
            "showPercentChange": False,
        },
    }


def state_timeline(
    title: str, targets: Sequence[Json], mappings: list[Json], description: str = "", *, thresholds: Json
) -> Json:
    """State timeline (up/down, normal/degraded)."""
    # colors come from the value mappings: the thresholds color mode would replace them with ranges («1+»)
    defaults = _field_defaults("short", thresholds, mappings, None, "fixed")
    defaults["custom"] = {"fillOpacity": 80, "lineWidth": 0}
    return {
        "type": "state-timeline",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": list(targets),
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "mergeValues": True,
            "showValue": "auto",
            "alignValue": "center",
            "rowHeight": 0.8,
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": False},
            "tooltip": {"mode": "single", "sort": "none"},
        },
    }


def heatmap(title: str, expr: str, unit: str, description: str = "") -> Json:
    """Heatmap of a Prometheus histogram (``expr`` sums ``_bucket`` rates or increases by ``le``)."""
    return {
        "type": "heatmap",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": [q(expr, "{{le}}", fmt="heatmap")],
        "fieldConfig": {"defaults": {"custom": {"scaleDistribution": {"type": "linear"}}}, "overrides": []},
        "options": {
            "calculate": False,
            "cellGap": 1,
            "color": {
                "mode": "scheme",
                "scheme": "Oranges",
                "scale": "exponential",
                "exponent": 0.5,
                "steps": 64,
                "reverse": False,
                "fill": "dark-orange",
            },
            "filterValues": {"le": 1e-9},
            "legend": {"show": True},
            "rowsFrame": {"layout": "auto"},
            "tooltip": {"mode": "single", "yHistogram": True, "showColorScale": False},
            "yAxis": {"axisPlacement": "left", "reverse": False, "unit": unit},
            "cellValues": {"unit": "short"},
            "exemplars": {"color": "rgba(255,0,255,0.7)"},
        },
    }


def distribution(title: str, expr: str, description: str = "", *, unit: str = "short") -> Json:
    """Bar gauge of histogram buckets (``expr`` sums ``_bucket`` increases by ``le``, one bar per bucket)."""
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": [q(expr, "{{le}}", fmt="heatmap", instant=True)],
        "fieldConfig": {
            "defaults": _field_defaults(unit, steps(("blue", None)), None, 0, "thresholds"),
            "overrides": [],
        },
        "options": {
            "displayMode": "gradient",
            "orientation": "vertical",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showUnfilled": True,
            "valueMode": "color",
            "namePlacement": "auto",
            "sizing": "auto",
            "minVizHeight": 16,
            "minVizWidth": 8,
            "maxVizHeight": 300,
        },
    }


def bars(
    title: str, targets: Sequence[Json], description: str = "", *, thresholds: Json | None = None
) -> Json:
    """Horizontal bar gauge (one bar per series)."""
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": list(targets),
        "fieldConfig": {
            "defaults": _field_defaults("short", thresholds or RED_AT_1, None, 0, "thresholds"),
            "overrides": [],
        },
        "options": {
            "displayMode": "basic",
            "orientation": "horizontal",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showUnfilled": True,
            "valueMode": "color",
            "namePlacement": "left",
            "sizing": "auto",
            "minVizHeight": 16,
            "minVizWidth": 8,
            "maxVizHeight": 300,
        },
    }


def alert_list(title: str, description: str = "") -> Json:
    """Firing and pending alerts, including the Prometheus rules (data source managed)."""
    return {
        "type": "alertlist",
        "title": title,
        "description": description,
        "options": {
            "viewMode": "list",
            "groupMode": "default",
            "groupBy": [],
            "maxItems": 20,
            "sortOrder": 1,
            "dashboardAlerts": False,
            "alertName": "",
            "alertInstanceLabelFilter": "",
            "showInstances": True,
            "stateFilter": {"firing": True, "pending": True, "noData": False, "normal": False, "error": True},
        },
    }


def text(title: str, content: str) -> Json:
    """Markdown text panel."""
    return {"type": "text", "title": title, "options": {"mode": "markdown", "content": content}}


# ---- dashboard -----------------------------------------------------------------------------------------


def query_variable(name: str, label: str, query: str) -> Json:
    """Multi-value query variable with «All» (``.*``)."""
    return {
        "type": "query",
        "name": name,
        "label": label,
        "datasource": DS,
        "query": query,
        "definition": query,
        "refresh": 2,
        "includeAll": True,
        "multi": True,
        "allValue": ".*",
        "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
        "options": [],
        "regex": "",
        "sort": 1,
        "hide": 0,
    }


ANNOTATIONS: Json = {
    "list": [
        {
            "builtIn": 1,
            "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True,
            "hide": True,
            "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts",
            "type": "dashboard",
        },
        {
            "datasource": DS,
            "enable": True,
            "iconColor": "red",
            "name": "Алерты Prometheus",
            "expr": 'ALERTS{alertstate="firing"}',
            "step": "5s",
            "titleFormat": "{{alertname}}",
            "textFormat": "{{service}} {{instance}}",
            "tagKeys": "severity,service",
            "useValueForTime": False,
        },
        {
            "datasource": DS,
            "enable": True,
            "iconColor": "purple",
            "name": "Сброс часов потока",
            "expr": "changes(foresight_predictor_clock_epoch[1m]) > 0",
            "step": "30s",
            "titleFormat": "Сброс часов потока",
            "textFormat": "{{instance}}",
        },
    ]
}


class Board:
    """A dashboard with automatic 24-column layout: panels fill a line left to right, rows start new lines.

    Args:
        uid: Dashboard uid.
        title: Title (``Foresight · ...``).
        description: Short description.
        variables: Template variables.
    """

    def __init__(self, uid: str, title: str, description: str, variables: Iterable[Json] = ()) -> None:
        self.uid = uid
        self.title = title
        self.description = description
        self.variables = list(variables)
        self.panels: list[Json] = []
        self._x = 0
        self._y = 0
        self._line_h = 0

    def _newline(self) -> None:
        self._y += self._line_h
        self._x = 0
        self._line_h = 0

    def row(self, title: str) -> Board:
        """Start a row (a titled section on a new line)."""
        self._newline()
        self.panels.append(
            {
                "type": "row",
                "title": title,
                "collapsed": False,
                "panels": [],
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": self._y},
            }
        )
        self._y += 1
        return self

    def add(self, panel: Json, w: int, h: int) -> Board:
        """Place a panel ``w`` columns wide and ``h`` units high."""
        if self._x + w > 24:
            self._newline()
        panel["gridPos"] = {"h": h, "w": w, "x": self._x, "y": self._y}
        self._x += w
        self._line_h = max(self._line_h, h)
        self.panels.append(panel)
        return self

    def build(self) -> Json:
        """Dashboard JSON (panel ids and target refIds assigned in order)."""
        for pid, panel in enumerate(self.panels, start=1):
            panel["id"] = pid
            for i, target in enumerate(panel.get("targets", [])):
                target["refId"] = chr(ord("A") + i)
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": [TAG],
            "timezone": "browser",
            "editable": False,
            "graphTooltip": 1,
            "liveNow": False,
            "refresh": "5s",
            "time": {"from": "now-15m", "to": "now"},
            "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
            "fiscalYearStartMonth": 0,
            "weekStart": "",
            "schemaVersion": 41,
            "version": 1,
            "links": [
                {
                    "type": "dashboards",
                    "title": "Foresight",
                    "tags": [TAG],
                    "asDropdown": False,
                    "includeVars": False,
                    "keepTime": True,
                    "targetBlank": False,
                    "icon": "external link",
                    "tooltip": "",
                    "url": "",
                }
            ],
            "annotations": ANNOTATIONS,
            "templating": {"list": self.variables},
            "panels": self.panels,
        }


# ---- «Обзор системы» -------------------------------------------------------------------------------------


def overview() -> Json:
    """Home dashboard: health, throughput, latency, lag, buffers, alerts, stream clock, resources."""
    b = Board(
        HOME_UID,
        "Foresight · Обзор системы",
        "Состояние сервисов и зависимостей, поток, задержки, лаг, буферы деградации и активные алерты.",
    )
    b.row("Состояние")
    b.add(
        stat(
            "Сервисы",
            [q("min by (service) (up)", "{{service}}")],
            description="up цели Prometheus (для реплик — минимум: DOWN, если не отвечает хотя бы одна).",
            mappings=UP_DOWN,
            thresholds=steps(("red", None), ("green", 1)),
            color_mode="background",
            graph=False,
            text_mode="value_and_name",
        ),
        10,
        5,
    )
    b.add(
        stat(
            "Зависимости",
            [
                q("min by (dependency) (foresight_dependency_up)", "{{dependency}} (сервисы)"),
                q("redis_up", "redis (экспортёр)"),
                q("pg_up", "postgres (экспортёр)"),
            ],
            description="foresight_dependency_up глазами сервисов (минимум по всем) и проверки экспортёров.",
            mappings=UP_DOWN,
            thresholds=steps(("red", None), ("green", 1)),
            color_mode="background",
            graph=False,
            text_mode="value_and_name",
        ),
        6,
        5,
    )
    b.add(
        stat(
            "Алерты firing",
            [q('count(ALERTS{alertstate="firing"}) or vector(0)', "firing")],
            description="Сработавшие правила Prometheus (deploy/prometheus/alerts.yml).",
            thresholds=RED_AT_1,
            color_mode="background",
            graph=False,
        ),
        4,
        5,
    )
    b.add(
        stat(
            "ТС на связи",
            [
                q('sum(foresight_vehicles{service="ingest", status="online"})', "онлайн"),
                q('sum(foresight_vehicles{service="ingest"})', "всего"),
            ],
            description="ТС по статусу связи (ingest).",
            graph=False,
            text_mode="value_and_name",
        ),
        4,
        5,
    )
    b.add(
        stat(
            "Пакеты NDTP/с",
            [q(rate("foresight_ndtp_realtime_packets_total"), "пакеты/с")],
            unit="pps",
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Задержка обработки p95",
            [q(quantile(0.95, "foresight_predictor_event_latency_seconds"), "p95")],
            unit="s",
            description="Приём пакета в ingest → обработка события в predictor. Цель < 1 с.",
            thresholds=LATENCY_S,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Лаг очереди",
            [q("max(foresight_consumer_lag)", "лаг")],
            description="Недоставленные события consumer group predictors. Цель ≈ 0.",
            thresholds=LAG,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Время тика",
            [q("max(foresight_predictor_tick_duration_seconds)", "тик")],
            unit="s",
            description="Длительность последнего тика прогнозов. Цель батча инференса < 100 мс.",
            thresholds=BATCH_S,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Буфер шины",
            [q("sum(foresight_bus_buffered)", "событий")],
            description="События ingest, ещё не записанные в Redis (растёт при отказе Redis).",
            thresholds=steps(("green", None), ("yellow", 100), ("red", 10000)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Буфер БД",
            [q("sum(foresight_db_buffered)", "строк")],
            description="Строки журнала, ещё не записанные в PostgreSQL (растёт при отказе PostgreSQL).",
            thresholds=steps(("green", None), ("yellow", 50), ("red", 10000)),
        ),
        4,
        4,
    )

    b.row("Поток")
    b.add(
        timeseries(
            "ТС по статусам связи",
            [q('sum by (status) (foresight_vehicles{service="ingest"})', "{{status}}")],
            stack=True,
            overrides=[color("online", "green"), color("stale", "yellow"), color("offline", "red")],
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Пропускная способность: приём → шина → predictor",
            [
                q(rate("foresight_ndtp_realtime_packets_total"), "пакеты NDTP (ingest)"),
                q(rate("foresight_bus_published_total"), "события в шину (ingest)"),
                q(rate("foresight_consumer_events_total"), "обработано (predictor)"),
            ],
            unit="ops",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Задержка обработки события",
            quantiles("foresight_predictor_event_latency_seconds"),
            unit="s",
            description="Приём пакета в ingest → обработка в predictor. Пунктир — цель 1 с.",
            thresholds=steps(("transparent", None), ("red", 1)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Лаг consumer group",
            [
                q("max(foresight_consumer_lag)", "лаг (недоставлено)"),
                q("max(foresight_consumer_pending)", "pending (не подтверждено)"),
            ],
            description="Пунктир — порог алерта ConsumerLagHigh (500).",
            thresholds=steps(("transparent", None), ("red", 500)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Время тика predictor",
            [q("foresight_predictor_tick_duration_seconds", "{{instance}}")],
            unit="s",
            description="Длительность последнего тика по репликам. Пунктир — 100 мс.",
            thresholds=steps(("transparent", None), ("orange", 0.1)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Буферы деградации",
            [
                q("sum(foresight_bus_buffered)", "шина ingest → Redis"),
                q("sum by (service) (foresight_db_buffered)", "журнал БД · {{service}}"),
            ],
            description="Растут, пока Redis или PostgreSQL недоступны; после восстановления дописываются.",
        ),
        8,
        8,
    )

    b.row("Часы потока")
    b.add(
        stat(
            "Часы потока (ingest)",
            [q("max(foresight_stream_time_seconds) * 1000", "время потока")],
            unit="dateTimeAsIso",
            description="Время данных, по которому считаются прогнозы (для replayer — историческое).",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Время данных replayer",
            [q("max(foresight_replayer_data_time_seconds) * 1000", "replayer")],
            unit="dateTimeAsIso",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Отставание часов потока от реального времени",
            [q("time() - max(foresight_stream_time_seconds)", "отставание")],
            unit="s",
            description="≈ 0 для живого потока (эмулятор); для исторического replayer — возраст данных.",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Скорость replayer",
            [q("max(foresight_replayer_speed)", "скорость")],
            unit="suffix:×",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        timeseries(
            "Скорость часов потока (сек данных за секунду)",
            [
                q("clamp_min(deriv(foresight_stream_time_seconds[1m]), 0)", "ingest"),
                q(
                    "clamp_min(deriv(foresight_predictor_stream_time_seconds[1m]), 0)",
                    "predictor {{instance}}",
                ),
                q("max(foresight_replayer_speed)", "replayer: заданная скорость"),
            ],
            unit="suffix:×",
            description="1× — живой поток; при воспроизведении истории — ускорение replayer.",
        ),
        24,
        7,
    )

    b.row("Алерты")
    b.add(alert_list("Активные алерты", "Правила Prometheus в состоянии firing / pending."), 12, 8)
    b.add(
        timeseries(
            "Алерты firing",
            [
                q('count by (alertname) (ALERTS{alertstate="firing"})', "{{alertname}}"),
                q('count(ALERTS{alertstate="firing"}) or vector(0)', "всего"),
            ],
            step=True,
            decimals=0,
        ),
        12,
        8,
    )

    b.row("Ресурсы процессов")
    b.add(
        timeseries(
            "CPU по сервисам",
            [q(rate("process_cpu_seconds_total", sel(f'job=~"{APP_JOBS}"'), by="service"), "{{service}}")],
            unit="percentunit",
            description="Доля одного ядра (100 % = одно ядро целиком), сумма по репликам.",
        ),
        12,
        7,
    )
    b.add(
        timeseries(
            "Память (RSS) по сервисам",
            [
                q(
                    f'sum by (service) (process_resident_memory_bytes{{job=~"{APP_JOBS}"}})',
                    "{{service}}",
                )
            ],
            unit="bytes",
        ),
        12,
        7,
    )
    return b.build()


# ---- «Приём NDTP» ----------------------------------------------------------------------------------------


def ingest() -> Json:
    """NDTP ingest and the replayer."""
    s = sel('instance=~"$instance"')
    b = Board(
        "foresight-ingest",
        "Foresight · Приём NDTP",
        "Соединения, кадры, ошибки протокола, анти-DoS, трафик ingest и исторический поток replayer.",
        [query_variable("instance", "ingest", 'label_values(up{job="ingest"}, instance)')],
    )
    b.row("Приём NDTP (ingest)")
    b.add(
        stat(
            "Соединения",
            [q(f"sum(foresight_ndtp_connections_active{s})", "открыто")],
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Приём соединений",
            [q(f"min(foresight_ndtp_listening{s})", "NDTP :9201")],
            mappings=value_map({"0": ("не слушает", "red"), "1": ("слушает", "green")}),
            thresholds=steps(("red", None), ("green", 1)),
            color_mode="background",
            graph=False,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Пакеты/с",
            [q(rate("foresight_ndtp_realtime_packets_total", s), "пакеты/с")],
            "pps",
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Трафик",
            [q(rate("foresight_ndtp_received_bytes_total", s), "байт/с")],
            "Bps",
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Ошибки CRC за период",
            [q(f"sum(increase(foresight_ndtp_crc_errors_total{s}[$__range]))", "CRC")],
            thresholds=steps(("green", None), ("yellow", 1), ("red", 50)),
            graph=False,
            decimals=0,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Отключено анти-DoS за период",
            [q(f"sum(increase(foresight_ndtp_abusive_disconnects_total{s}[$__range]))", "отключений")],
            thresholds=RED_AT_1,
            graph=False,
            decimals=0,
        ),
        4,
        4,
    )
    b.add(
        timeseries(
            "Активные соединения",
            [q(f"sum by (instance) (foresight_ndtp_connections_active{s})", "{{instance}}")],
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Подключения, отключения, таймауты (за минуту)",
            [
                q(per_minute("foresight_ndtp_connections_total", s), "подключения"),
                q(per_minute("foresight_ndtp_handshakes_total", s), "handshake"),
                q(per_minute("foresight_ndtp_disconnects_total", s), "отключения"),
                q(per_minute("foresight_ndtp_read_timeouts_total", s), "таймауты чтения"),
            ],
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Кадры и пакеты/с",
            [
                q(rate("foresight_ndtp_frames_total", s), "кадры с верной CRC"),
                q(rate("foresight_ndtp_realtime_packets_total", s), "realtime-пакеты"),
                q(rate("foresight_ndtp_nav_records_total", s), "записи Nav00"),
            ],
            unit="pps",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Ошибки протокола (за минуту)",
            [
                q(per_minute("foresight_ndtp_crc_errors_total", s), "CRC"),
                q(per_minute("foresight_ndtp_bad_headers_total", s), "мусорные заголовки NPL"),
                q(per_minute("foresight_ndtp_parse_errors_total", s), "ошибки разбора"),
                q(per_minute("foresight_ndtp_unknown_cells_total", s), "неизвестные ячейки"),
            ],
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Анти-DoS (за минуту)",
            [q(per_minute("foresight_ndtp_abusive_disconnects_total", s), "отключено за бюджет CRC")],
            description="Соединения, закрытые за превышение бюджета битых кадров.",
            decimals=0,
            thresholds=steps(("transparent", None), ("red", 1)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Трафик",
            [
                q(rate("foresight_ndtp_received_bytes_total", s), "принято"),
                q(rate("foresight_ndtp_discarded_bytes_total", s), "пропущено при ресинхронизации"),
            ],
            unit="Bps",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "ТС по статусам связи",
            [
                q(
                    'sum by (status) (foresight_vehicles{service="ingest", instance=~"$instance"})',
                    "{{status}}",
                )
            ],
            stack=True,
            overrides=[color("online", "green"), color("stale", "yellow"), color("offline", "red")],
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Часы потока ingest (за минуту)",
            [
                q(per_minute("foresight_stream_clock_resets_total", s), "сбросы (перезапуск источника)"),
                q(per_minute("foresight_stream_clock_garbage_total", s), "неправдоподобное время"),
                q(per_minute("foresight_stream_clock_off_timeline_total", s), "вне линии времени"),
            ],
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "События без tr_id/с",
            [q(rate("foresight_bus_unmapped_events_total", s), "unitId без tr_id")],
            unit="ops",
            description="Устройства, которых нет в справочнике unit_id → tr_id (в прогнозы не попадают).",
        ),
        8,
        8,
    )

    b.row("Replayer: исторический поток")
    b.add(
        stat(
            "Состояние",
            [q("max(foresight_replayer_running)", "replayer")],
            mappings=value_map({"0": ("остановлен", "blue"), "1": ("идёт", "green")}),
            thresholds=steps(("blue", None), ("green", 1)),
            color_mode="background",
            graph=False,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Скорость",
            [q("max(foresight_replayer_speed)", "скорость")],
            "suffix:×",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Время данных",
            [q("max(foresight_replayer_data_time_seconds) * 1000", "время данных")],
            "dateTimeAsIso",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Отставание от расписания",
            [q("max(foresight_replayer_lag_seconds)", "отставание")],
            "s",
            thresholds=steps(("green", None), ("yellow", 1), ("red", 5)),
        ),
        5,
        4,
    )
    b.add(
        stat(
            "Проходов",
            [q("max(foresight_replayer_cycles_total)", "проходов")],
            description="Запуски воспроизведения (старт, рестарт, новый круг петли).",
            graph=False,
            thresholds=steps(("blue", None)),
            decimals=0,
        ),
        5,
        4,
    )
    b.add(
        timeseries(
            "Отправлено пакетов/с",
            [
                q(rate("foresight_replayer_packets_sent_total"), "отправлено"),
                q(rate("foresight_replayer_packets_dispatched_total"), "выдано планировщиком"),
                q(rate("foresight_replayer_packets_dropped_total", by="reason"), "отброшено: {{reason}}"),
            ],
            unit="pps",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Отставание от расписания",
            [
                q("max(foresight_replayer_lag_seconds)", "старейший неотправленный пакет"),
                q("max(foresight_replayer_last_send_lag_seconds)", "последний пакет"),
            ],
            unit="s",
            description="Пунктир — порог алерта ReplayerLagging (5 с).",
            thresholds=steps(("transparent", None), ("red", 5)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Очередь отправки и соединения",
            [
                q("max(foresight_replayer_backlog_packets)", "пакетов в очередях"),
                q("max(foresight_replayer_connections_active)", "NDTP-соединений"),
            ],
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Переподключения (за минуту)",
            [
                q(per_minute("foresight_replayer_connects_total"), "подключения"),
                q(per_minute("foresight_replayer_reconnects_total"), "переподключения"),
                q(per_minute("foresight_replayer_disconnects_total"), "обрывы"),
                q(per_minute("foresight_replayer_connect_failures_total"), "неудачные попытки"),
            ],
            decimals=0,
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Мост эмулятора (за минуту)",
            [
                q(per_minute("foresight_replayer_bridge_posts_total"), "конфигураций принято"),
                q(per_minute("foresight_replayer_bridge_errors_total"), "ошибки"),
            ],
            description="Только в режиме emulator-bridge: треки гонятся через официальный эмулятор.",
            decimals=0,
        ),
        12,
        8,
    )
    return b.build()


# ---- «Поток и прогнозы» ----------------------------------------------------------------------------------


def stream() -> Json:
    """Redis Streams bus, consumer group and the predictor core."""
    s = sel('instance=~"$instance"')
    latency = "foresight_predictor_event_latency_seconds"
    b = Board(
        "foresight-stream",
        "Foresight · Поток и прогнозы",
        "Шина Redis Streams, consumer group predictors и ядро predictor: окна треков, тики, задержки.",
        [query_variable("instance", "predictor", 'label_values(up{job="predictor"}, instance)')],
    )
    b.row("Шина: ingest → Redis Streams")
    b.add(
        timeseries(
            "Запись в шину/с",
            [
                q(rate("foresight_bus_published_total"), "события в поток telemetry"),
                q(rate("foresight_bus_state_writes_total"), "горячее состояние ТС"),
                q(rate("foresight_bus_flushes_total"), "pipeline в Redis"),
            ],
            unit="ops",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Буфер и вытеснения",
            [
                q("sum(foresight_bus_buffered)", "в буфере"),
                q(per_minute("foresight_bus_evicted_total"), "вытеснено за минуту"),
            ],
            description="Буфер растёт, пока Redis недоступен; вытеснение — потеря данных (алерт DataLoss).",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Ошибки Redis (за минуту)",
            [
                q(per_minute("foresight_bus_flush_errors_total"), "неудачные pipeline"),
                q(per_minute("foresight_bus_command_errors_total"), "отклонённые команды"),
            ],
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Длительность pipeline",
            [q("max(foresight_bus_last_flush_seconds)", "последний pipeline")],
            unit="s",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Длина потока telemetry",
            [
                q("max(foresight_stream_length)", "по данным predictor"),
                q('max(redis_stream_length{stream="foresight:telemetry"})', "по данным redis_exporter"),
            ],
            description="Поток ограничен MAXLEN ≈ 200 000 записей.",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "События без tr_id/с",
            [
                q(rate("foresight_bus_unmapped_events_total"), "ingest"),
                q(rate("foresight_predictor_unmapped_events_total", s), "predictor"),
            ],
            unit="ops",
        ),
        8,
        8,
    )

    b.row("Consumer group predictors")
    b.add(
        stat(
            "Реплик predictor",
            [q('count(up{job="predictor"} == 1) or vector(0)', "реплик")],
            thresholds=steps(("red", None), ("green", 1)),
            graph=False,
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Потребителей в группе",
            [q('max(redis_stream_group_consumers{group="predictors"})', "потребителей")],
            description="По данным redis_exporter (включая неактивных, пока их не удалит GC).",
            graph=False,
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(stat("Лаг", [q("max(foresight_consumer_lag)", "лаг")], thresholds=LAG), 4, 4)
    b.add(
        stat(
            "Pending",
            [q("max(foresight_consumer_pending)", "pending")],
            thresholds=steps(("green", None), ("yellow", 1000)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "События/с",
            [q(rate("foresight_consumer_events_total", s), "событий/с")],
            "ops",
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        stat(
            "Подтверждено/с",
            [q(rate("foresight_consumer_acked_total", s), "ack/с")],
            "ops",
            thresholds=steps(("blue", None)),
        ),
        4,
        4,
    )
    b.add(
        timeseries(
            "Обработано событий/с по репликам",
            [q(rate("foresight_consumer_events_total", s, by="instance"), "{{instance}}")],
            unit="ops",
            stack=True,
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Лаг и pending",
            [
                q("max(foresight_consumer_lag)", "лаг (predictor)"),
                q("max(foresight_consumer_pending)", "pending (predictor)"),
                q('max(redis_stream_group_lag{group="predictors"})', "лаг (redis_exporter)"),
                q('max(redis_stream_group_messages_pending{group="predictors"})', "pending (redis_exporter)"),
            ],
            thresholds=steps(("transparent", None), ("red", 500)),
            threshold_style="dashed",
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Восстановление (за минуту)",
            [
                q(per_minute("foresight_consumer_claimed_total", s), "забрано у других (XCLAIM)"),
                q(per_minute("foresight_consumer_recovered_total", s), "свои pending перечитаны"),
                q(per_minute("foresight_consumer_history_total", s), "история при старте"),
                q(per_minute("foresight_consumer_removed_consumers_total", s), "удалено потребителей"),
            ],
            decimals=0,
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Ошибки consumer (за минуту)",
            [
                q(per_minute("foresight_consumer_malformed_total", s), "битые записи"),
                q(per_minute("foresight_consumer_handler_errors_total", s), "упавшие обработчики"),
                q(per_minute("foresight_consumer_read_errors_total", s), "ошибки чтения"),
            ],
            decimals=0,
        ),
        6,
        8,
    )

    b.row("Predictor")
    b.add(
        timeseries(
            "События predictor/с",
            [
                q(rate("foresight_predictor_events_total", s, by="instance"), "{{instance}}"),
                q(rate("foresight_predictor_unmapped_events_total", s), "без tr_id"),
            ],
            unit="ops",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Потерянные и переупорядоченные точки (за минуту)",
            [
                q(per_minute("foresight_predictor_late_points_total", s), "поздние (вставлены по времени)"),
                q(per_minute("foresight_predictor_duplicate_points_total", s), "дубли (отброшены)"),
                q(per_minute("foresight_predictor_expired_points_total", s), "старше окна (отброшены)"),
                q(per_minute("foresight_predictor_ahead_points_total", s), "из будущего (отложены)"),
            ],
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Треки",
            [
                q(f"sum(foresight_predictor_tracks{s})", "ТС с окном трека"),
                q(f"sum(foresight_predictor_held_points{s})", "отложенные точки"),
            ],
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Точек в окнах треков",
            [q(f"sum by (instance) (foresight_predictor_window_points{s})", "{{instance}}")],
            description="Окно трека — 30 минут потокового времени на ТС.",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Тики прогнозов (за минуту)",
            [
                q(per_minute("foresight_predictor_ticks_total", s), "тики"),
                q(per_minute("foresight_predictor_ticks_skipped_total", s), "пропущенные границы"),
                q(per_minute("foresight_predictor_tick_errors_total", s), "ошибки"),
            ],
            description="Тик — каждые 30 с потокового времени (при x30 — раз в секунду).",
            decimals=0,
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Длительность тика",
            [
                q(f"max(foresight_predictor_tick_duration_seconds{s})", "последний"),
                q(
                    f"max(quantile_over_time(0.5, foresight_predictor_tick_duration_seconds{s}[5m]))",
                    "p50 за 5 мин",
                ),
                q(
                    f"max(quantile_over_time(0.95, foresight_predictor_tick_duration_seconds{s}[5m]))",
                    "p95 за 5 мин",
                ),
            ],
            unit="s",
            description="Квантили по снимкам длительности последнего тика (гейдж). Пунктир — 100 мс.",
            thresholds=steps(("transparent", None), ("orange", 0.1)),
            threshold_style="dashed",
        ),
        8,
        8,
    )
    b.add(
        heatmap(
            "Задержка события: распределение",
            f"sum by (le) (increase({latency}_bucket{s}[{RATE}]))",
            "s",
            "Приём пакета в ingest → обработка события в predictor.",
        ),
        12,
        9,
    )
    b.add(
        timeseries(
            "Задержка события: квантили",
            quantiles(latency, s),
            unit="s",
            description="Пунктир — цель 1 с (алерт EventLatencyHigh по p95).",
            thresholds=steps(("transparent", None), ("red", 1)),
            threshold_style="dashed",
        ),
        12,
        9,
    )
    b.add(
        timeseries(
            "Часы потока predictor (за минуту)",
            [
                q(per_minute("foresight_predictor_clock_resets_total", s), "сбросы часов"),
                q(per_minute("foresight_predictor_clock_garbage_total", s), "неправдоподобное время"),
                q(per_minute("foresight_predictor_off_timeline_total", s), "вне линии времени"),
            ],
            decimals=0,
        ),
        12,
        7,
    )
    b.add(
        timeseries(
            "Время потока predictor",
            [q(f"foresight_predictor_stream_time_seconds{s} * 1000", "{{instance}}")],
            unit="dateTimeAsIso",
            soft_min=None,
        ),
        12,
        7,
    )
    return b.build()


# ---- «Хранилище и деградация» ----------------------------------------------------------------------------


def storage() -> Json:
    """Dependencies, degradation, the PostgreSQL journal writer and the Redis / PostgreSQL exporters."""
    svc = sel('service=~"$service"')
    b = Board(
        "foresight-storage",
        "Foresight · Хранилище и деградация",
        "Доступность Redis и PostgreSQL глазами сервисов, буферы деградации, журнал PostgreSQL, экспортёры.",
        [query_variable("service", "сервис", "label_values(foresight_dependency_up, service)")],
    )
    b.row("Зависимости и деградация")
    b.add(
        state_timeline(
            "Доступность зависимостей по сервисам",
            [
                q(
                    f"min by (service, dependency) (foresight_dependency_up{svc})",
                    "{{service}} → {{dependency}}",
                )
            ],
            AVAILABLE,
            "foresight_dependency_up: 1 — зависимость отвечает, 0 — отказ (или не проверена после старта).",
            thresholds=steps(("red", None), ("green", 1)),
        ),
        24,
        7,
    )
    b.add(
        state_timeline(
            "Redis и PostgreSQL (экспортёры)",
            [q("redis_up", "Redis"), q("pg_up", "PostgreSQL")],
            AVAILABLE,
            thresholds=steps(("red", None), ("green", 1)),
        ),
        12,
        5,
    )
    b.add(
        state_timeline(
            "api: режим",
            [q("max(foresight_api_degraded)", "api")],
            value_map({"0": ("норма", "green"), "1": ("последнее известное состояние", "orange")}),
            "1 — Redis недоступен, api отдаёт последнее известное состояние.",
            thresholds=steps(("green", None), ("orange", 1)),
        ),
        12,
        5,
    )
    b.add(
        bars(
            "Сбои зависимостей с запуска сервиса",
            [
                q(
                    f"max by (service, dependency) (foresight_dependency_outages_total{svc})",
                    "{{service}} → {{dependency}}",
                )
            ],
            "Переходы «доступна → недоступна» (foresight_dependency_outages_total).",
        ),
        12,
        7,
    )
    b.add(
        timeseries(
            "WebSocket-клиенты api",
            [q("sum(foresight_api_ws_clients)", "клиентов")],
            decimals=0,
        ),
        12,
        7,
    )

    b.row("Журнал PostgreSQL (буферизованная запись)")
    b.add(
        timeseries(
            "Записано строк/с",
            [q(rate("foresight_db_written_total", svc, by="service"), "{{service}}")],
            unit="ops",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Буфер записи",
            [q(f"sum by (service) (foresight_db_buffered{svc})", "{{service}}")],
            description="Строки в памяти, ожидающие PostgreSQL. При переполнении старые отбрасываются.",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Ошибки, отброшено, отклонено (за минуту)",
            [
                q(per_minute("foresight_db_errors_total", svc, by="service"), "ошибки · {{service}}"),
                q(per_minute("foresight_db_dropped_total", svc, by="service"), "отброшено · {{service}}"),
                q(per_minute("foresight_db_rejected_total", svc, by="service"), "отклонено · {{service}}"),
            ],
            decimals=0,
        ),
        8,
        8,
    )

    b.row("PostgreSQL (postgres_exporter)")
    db = sel('datname="foresight"')
    b.add(
        timeseries(
            "Соединения",
            [
                q(f"sum(pg_stat_database_numbackends{db})", "backend'ов"),
                q(f"sum by (state) (pg_stat_activity_count{db})", "{{state}}"),
            ],
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Транзакции/с",
            [
                q(rate("pg_stat_database_xact_commit", db), "commit"),
                q(rate("pg_stat_database_xact_rollback", db), "rollback"),
            ],
            unit="ops",
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Строки/с",
            [
                q(rate("pg_stat_database_tup_inserted", db), "вставлено"),
                q(rate("pg_stat_database_tup_updated", db), "обновлено"),
                q(rate("pg_stat_database_tup_deleted", db), "удалено"),
            ],
            unit="ops",
        ),
        6,
        8,
    )
    b.add(
        timeseries("Размер БД", [q(f"max(pg_database_size_bytes{db})", "foresight")], unit="bytes"),
        6,
        8,
    )

    b.row("Redis (redis_exporter)")
    b.add(
        timeseries(
            "Память",
            [q("redis_memory_used_bytes", "used"), q("redis_memory_used_rss_bytes", "rss")],
            unit="bytes",
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Клиенты",
            [
                q("redis_connected_clients", "подключено"),
                q("redis_blocked_clients", "заблокировано (XREAD BLOCK)"),
            ],
        ),
        6,
        8,
    )
    b.add(
        timeseries("Команд/с", [q(rate("redis_commands_processed_total"), "команд/с")], unit="ops"),
        6,
        8,
    )
    b.add(
        timeseries(
            "Сеть",
            [
                q(rate("redis_net_input_bytes_total"), "вход"),
                q(rate("redis_net_output_bytes_total"), "выход"),
            ],
            unit="Bps",
        ),
        6,
        8,
    )
    b.add(
        timeseries(
            "Ключи db0",
            [
                q('sum(redis_db_keys{db="db0"})', "ключей"),
                q('sum(redis_db_keys_expiring{db="db0"})', "с TTL"),
            ],
        ),
        12,
        7,
    )
    b.add(
        timeseries(
            "Поток telemetry",
            [
                q('max(redis_stream_length{stream="foresight:telemetry"})', "длина"),
                q('max(redis_stream_group_lag{group="predictors"})', "лаг группы"),
            ],
        ),
        12,
        7,
    )

    b.row("api: HTTP (контракт следующего блока)")
    http = "foresight_http_request_duration_seconds"
    b.add(
        timeseries(
            "Запросы/с по маршрутам",
            [q(rate(f"{http}_count", by="route"), "{{route}}")],
            unit="reqps",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Латентность p95 по маршрутам",
            [q(quantile(0.95, http, by="route"), "{{route}}")],
            unit="s",
        ),
        8,
        8,
    )
    b.add(
        timeseries(
            "Ответы 5xx/с",
            [q(rate(f"{http}_count", sel('status=~"5.."'), by="route"), "{{route}}")],
            unit="reqps",
        ),
        8,
        8,
    )
    return b.build()


# ---- «Модель» --------------------------------------------------------------------------------------------


def model() -> Json:
    """ML contract: inference latency, batch size, fallback, online MAE, horizon, retroactive alerts."""
    ml = "foresight_ml_request_duration_seconds"
    predict = sel('endpoint="/predict"')
    fallback_share = (
        'sum(rate(foresight_predictions_total{source="fallback"}[$__rate_interval]))'
        " / sum(rate(foresight_predictions_total[$__rate_interval]))"
    )
    b = Board(
        "foresight-model",
        "Foresight · Модель",
        "ml-service и качество прогнозов на потоке: инференс, батчи, fallback, онлайн-MAE, горизонт.",
    )
    b.add(
        text(
            "О дашборде",
            "Метрики этого дашборда — **контракт следующего блока** (ml-service, онлайн-сверка прогнозов "
            "в predictor): `docs/observability.md`, раздел «Метрики следующего блока». До их реализации "
            "панели показывают *No data*. Цели: инференс batch < 100 мс, алертов задним числом — 0, "
            "онлайн-MAE ниже baseline «прогноз = cur_dev_s».",
        ),
        24,
        3,
    )
    b.row("ml-service")
    b.add(
        stat(
            "Версия модели",
            [q("max by (version) (foresight_ml_model_info)", "{{version}}")],
            graph=False,
            text_mode="name",
            thresholds=steps(("blue", None)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Инференс p95",
            [q(quantile(0.95, ml, predict), "p95")],
            "s",
            "POST /predict. Цель < 100 мс, таймаут predictor — 500 мс.",
            thresholds=steps(("green", None), ("yellow", 0.05), ("red", 0.1)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Доля fallback",
            [q(fallback_share, "fallback")],
            "percentunit",
            thresholds=steps(("green", None), ("yellow", 0.05), ("red", 0.5)),
        ),
        6,
        4,
    )
    b.add(
        stat(
            "Алерты задним числом",
            [q("sum(increase(foresight_alerts_retroactive_total[$__range]))", "задним числом")],
            description="Алерты, выданные в момент фактического события или позже, за выбранный период. "
            "Должно быть 0.",
            thresholds=RED_AT_1,
            color_mode="background",
            graph=False,
            decimals=0,
        ),
        6,
        4,
    )
    b.add(
        timeseries(
            "Латентность инференса POST /predict",
            quantiles(ml, predict),
            unit="s",
            description="Пунктир — цель 100 мс (алерт MLInferenceSlow по p95).",
            thresholds=steps(("transparent", None), ("red", 0.1)),
            threshold_style="dashed",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Латентность p95 по endpoint",
            [q(quantile(0.95, ml, by="endpoint"), "{{endpoint}}")],
            unit="s",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Размер батча",
            [
                q(quantile(0.5, "foresight_ml_batch_size"), "p50"),
                q(quantile(0.95, "foresight_ml_batch_size"), "p95"),
                q(mean("foresight_ml_batch_size"), "среднее"),
            ],
            description="Прогнозных точек в одном POST /predict.",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "GPU-память ml-service",
            [q("sum by (device) (foresight_ml_gpu_memory_bytes)", "GPU {{device}}")],
            unit="bytes",
        ),
        12,
        8,
    )

    b.row("Прогнозы и качество на потоке")
    b.add(
        timeseries(
            "Прогнозы/с по источнику",
            [q(rate("foresight_predictions_total", by="source"), "{{source}}")],
            unit="ops",
            stack=True,
            overrides=[color("model", "green"), color("fallback", "orange")],
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Доля fallback",
            [q(fallback_share, "fallback")],
            unit="percentunit",
            description="Пунктир — порог алерта MLFallbackActive (50 %).",
            thresholds=steps(("transparent", None), ("red", 0.5)),
            threshold_style="dashed",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Онлайн-MAE против baseline",
            [
                q("max(foresight_online_mae_seconds)", "модель"),
                q("max(foresight_online_baseline_mae_seconds)", "baseline «прогноз = cur_dev_s»"),
                q(
                    "sum(rate(foresight_prediction_abs_error_seconds_sum[15m]))"
                    " / sum(rate(foresight_prediction_abs_error_seconds_count[15m]))",
                    "MAE закрытых прогнозов за 15 мин (гистограмма)",
                ),
            ],
            unit="s",
            description="MAE по прогнозам, закрытым фактом детектора прохождения остановок.",
        ),
        12,
        8,
    )
    b.add(
        timeseries(
            "Открытые прогнозы",
            [q("sum(foresight_predictions_open)", "ждут факта")],
            description="Прогнозы, по которым ТС ещё не прошло целевую остановку.",
        ),
        12,
        8,
    )

    b.row("Горизонт и алерты")
    b.add(
        distribution(
            "Заблаговременность алертов за период",
            'sum by (le) (increase(foresight_prediction_lead_seconds_bucket{kind="alert"}[$__range]))',
            "Факт прохождения остановки − момент выдачи алерта, с (верхние границы бакетов). "
            "Честный горизонт — 600…900 с; бакет 0 — алерты задним числом.",
        ),
        12,
        9,
    )
    b.add(
        heatmap(
            "Заблаговременность прогнозов во времени",
            f'sum by (le) (increase(foresight_prediction_lead_seconds_bucket{{kind="prediction"}}[{RATE}]))',
            "s",
            "Все закрытые прогнозы: факт − выдача, с.",
        ),
        12,
        9,
    )
    b.add(
        timeseries(
            "Алерты по уровню и причине (за минуту)",
            [q(per_minute("foresight_alerts_total", by="level, cause"), "{{level}} · {{cause}}")],
            decimals=0,
        ),
        12,
        8,
    )
    b.add(
        distribution(
            "Ошибка закрытых прогнозов за период",
            "sum by (le) (increase(foresight_prediction_abs_error_seconds_bucket[$__range]))",
            "|прогноз − факт|, с (верхние границы бакетов).",
        ),
        12,
        8,
    )
    return b.build()


# ---- output ----------------------------------------------------------------------------------------------

DASHBOARDS = {
    "foresight-overview.json": overview,
    "foresight-ingest.json": ingest,
    "foresight-stream.json": stream,
    "foresight-storage.json": storage,
    "foresight-model.json": model,
}


def render(dashboard: Json) -> str:
    """Serialise a dashboard exactly as it is stored in the repository."""
    return json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n"


def build_all() -> dict[str, str]:
    """File name → JSON text of every dashboard."""
    return {name: render(factory()) for name, factory in DASHBOARDS.items()}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="only verify that the files are up to date")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = parser.parse_args(argv)

    stale = []
    for name, content in build_all().items():
        path = args.out / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        stale.append(name)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if args.check and stale:
        print(f"out of date: {', '.join(stale)} (run deploy/grafana/build_dashboards.py)", file=sys.stderr)
        return 1
    print(f"{'stale' if args.check else 'written'}: {len(stale)}, up to date: {len(DASHBOARDS) - len(stale)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
