"""Observability: Grafana dashboards, Prometheus config and alert rules against the metrics contract.

Static checks (no running stack): dashboards are valid and generated, every panel uses the provisioned
data source, every ``foresight_*`` metric in dashboards and rules is either exported by ``backend/`` /
``replayer/`` / ``ml/service.py`` code or declared in the next-block contract of ``docs/observability.md``,
the delivered part of the contract is exported with the contracted type, alert names are unique.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
DASHBOARDS = REPO / "deploy" / "grafana" / "dashboards"
PROVISIONING = REPO / "deploy" / "grafana" / "provisioning"
PROMETHEUS = REPO / "deploy" / "prometheus"
COMPOSE = REPO / "docker-compose.yml"
CODE_PATHS = (REPO / "backend", REPO / "replayer", REPO / "ml" / "service.py")

REQUIRED_CONTRACT = {
    "foresight_ml_gpu_memory_bytes",  # ml-service runs on CPU: no series (the contract allows it)
    "foresight_http_request_duration_seconds",  # api
}
"""Contracted metrics not delivered yet: declared in the next-block section, exported by nobody."""

DELIVERED = {
    # ml-service (ml/service.py)
    "foresight_ml_request_duration_seconds": "histogram",
    "foresight_ml_batch_size": "histogram",
    "foresight_ml_model_info": "gauge",
    # predictor (backend/forecast.py, backend/predictor.py)
    "foresight_predictions_total": "counter",
    "foresight_prediction_lead_seconds": "histogram",
    "foresight_prediction_abs_error_seconds": "histogram",
    "foresight_online_mae_seconds": "gauge",
    "foresight_online_baseline_mae_seconds": "gauge",
    "foresight_alerts_total": "counter",
    "foresight_alerts_retroactive_total": "counter",
    "foresight_predictions_open": "gauge",
    "foresight_predictor_tick_seconds": "histogram",  # «Доработки сервисов», п. 1
}
"""The delivered part of the contract (docs/observability.md §3.2, §3.7) and its metric types."""

_FACTORIES = {
    "CounterMetricFamily": "counter",
    "_counter": "counter",
    "Counter": "counter",
    "GaugeMetricFamily": "gauge",
    "_gauge": "gauge",
    "Gauge": "gauge",
    "Histogram": "histogram",
    "HistogramMetricFamily": "histogram",
    "Summary": "summary",
    "SummaryMetricFamily": "summary",
}
_METRIC_LITERAL = re.compile(r"^foresight_[a-z0-9_]+$")

Json = dict[str, Any]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


builder = _load("build_dashboards", REPO / "deploy" / "grafana" / "build_dashboards.py")
checker = _load("check_dashboards", REPO / "scripts" / "check_dashboards.py")


def _dashboards() -> dict[str, Json]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(DASHBOARDS.glob("*.json"))}


def _panels(dashboard: Json) -> list[Json]:
    return list(checker.iter_panels(dashboard["panels"]))


def _rules() -> list[Json]:
    groups = yaml.safe_load((PROMETHEUS / "alerts.yml").read_text(encoding="utf-8"))["groups"]
    return [rule for group in groups for rule in group["rules"]]


def _exposition(name: str, kind: str | None) -> set[str]:
    """Names a metric family appears under in the text exposition."""
    base = name.removesuffix("_total")
    series = {
        "counter": {f"{base}_total"},
        "gauge": {name},
        "histogram": {f"{name}_bucket", f"{name}_sum", f"{name}_count", f"{name}_created"},
        "summary": {name, f"{name}_sum", f"{name}_count", f"{name}_created"},
    }
    if kind is None:  # a literal whose metric type the scan could not infer: accept every form
        return set().union(*series.values())
    return series[kind]


def exported_metrics() -> set[str]:
    """Exposition names of the ``foresight_*`` metrics defined in :data:`CODE_PATHS`.

    The type comes from the metric factory the literal is passed to (``CounterMetricFamily``, ``_gauge``,
    ``Histogram``...) or from the name of the table it is declared in (``_COUNTERS``, ``gauges``).
    """
    kinds: dict[str, str | None] = {}
    for code in CODE_PATHS:
        for path in [code] if code.is_file() else sorted(code.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.args:
                    func = node.func
                    fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    first = node.args[0]
                    literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
                    if fname in _FACTORIES and literal:
                        kinds[first.value] = _FACTORIES[fname]
                elif isinstance(node, ast.Assign | ast.AnnAssign):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    label = " ".join(ast.unparse(t) for t in targets).lower()
                    kind = "counter" if "counter" in label else "gauge" if "gauge" in label else None
                    if kind is None or node.value is None:
                        continue
                    for const in ast.walk(node.value):
                        if isinstance(const, ast.Constant) and _METRIC_LITERAL.match(str(const.value)):
                            kinds.setdefault(const.value, kind)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and _METRIC_LITERAL.match(str(node.value)):
                    kinds.setdefault(node.value, None)
    return set().union(*(_exposition(name, kind) for name, kind in kinds.items()))


def referenced_metrics() -> dict[str, set[str]]:
    """``foresight_*`` metric names in dashboard queries, annotations and alert rules → their uses."""
    uses: dict[str, set[str]] = {}

    def add(expr: str, where: str) -> None:
        for name in checker.metric_names(expr):
            if name.startswith("foresight_"):
                uses.setdefault(name, set()).add(where)

    for file, dashboard in _dashboards().items():
        for panel in _panels(dashboard):
            for target in panel.get("targets", []):
                add(target.get("expr", ""), f"{file}: {panel['title']}")
        for annotation in dashboard["annotations"]["list"]:
            add(annotation.get("expr", ""), f"{file}: annotation {annotation['name']}")
    for rule in _rules():
        add(rule["expr"], f"alerts.yml: {rule['alert']}")
    return uses


# ---- dashboards --------------------------------------------------------------------------------------------


def test_dashboards_match_the_generator() -> None:
    generated = builder.build_all()
    assert sorted(generated) == sorted(p.name for p in DASHBOARDS.glob("*.json"))
    for name, text in generated.items():
        assert (DASHBOARDS / name).read_text(encoding="utf-8") == text, (
            f"{name} is stale: run python3 deploy/grafana/build_dashboards.py"
        )


def test_dashboards_are_valid_and_unique() -> None:
    dashboards = _dashboards()
    assert len(dashboards) == 5
    uids = [d["uid"] for d in dashboards.values()]
    assert len(set(uids)) == len(uids)
    titles = {d["title"] for d in dashboards.values()}
    assert titles == {
        "Foresight · Обзор системы",
        "Foresight · Приём NDTP",
        "Foresight · Поток и прогнозы",
        "Foresight · Хранилище и деградация",
        "Foresight · Модель",
    }
    for dashboard in dashboards.values():
        assert dashboard["refresh"] == "5s"
        assert dashboard["time"] == {"from": "now-15m", "to": "now"}
        panels = _panels(dashboard)
        ids = [p["id"] for p in panels]
        assert len(set(ids)) == len(ids), dashboard["title"]
        for panel in panels:
            grid = panel["gridPos"]
            assert grid["x"] >= 0 and grid["x"] + grid["w"] <= 24, (dashboard["title"], panel["title"])
            refs = [t["refId"] for t in panel.get("targets", [])]
            assert len(set(refs)) == len(refs)


def test_panels_use_the_provisioned_datasource() -> None:
    config = yaml.safe_load((PROVISIONING / "datasources" / "prometheus.yml").read_text(encoding="utf-8"))
    (source,) = config["datasources"]
    assert source["isDefault"] is True and source["type"] == "prometheus"
    uid = source["uid"]
    assert uid == builder.DATASOURCE_UID
    for dashboard in _dashboards().values():
        for panel in _panels(dashboard):
            targets = panel.get("targets", [])
            if panel["type"] in ("row", "text", "alertlist"):
                assert not targets
                continue
            assert targets, (dashboard["title"], panel["title"])
            assert panel["datasource"] == {"type": "prometheus", "uid": uid}, panel["title"]
            for target in targets:
                assert target["datasource"]["uid"] == uid
                assert target["expr"].strip()
        for variable in dashboard["templating"]["list"]:
            assert variable["datasource"]["uid"] == uid


def test_counters_are_queried_through_rate_or_increase() -> None:
    counters = {m for m in exported_metrics() if m.endswith("_total")}
    shown_as_totals = {"foresight_replayer_cycles_total", "foresight_dependency_outages_total"}  # «с запуска»
    for dashboard in _dashboards().values():
        for panel in _panels(dashboard):
            for target in panel.get("targets", []):
                expr = target["expr"]
                for name in checker.metric_names(expr) & counters - shown_as_totals:
                    assert re.search(rf"(rate|increase)\(\s*{name}\b", expr), (panel["title"], expr)


def test_home_dashboard_and_grafana_access() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    env = compose["services"]["grafana"]["environment"]
    home = Path(env["GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"])
    assert home.parent == Path("/etc/grafana/dashboards")
    assert json.loads((DASHBOARDS / home.name).read_text(encoding="utf-8"))["uid"] == builder.HOME_UID
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "true"
    assert env["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Viewer"
    assert env["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GF_SECURITY_ADMIN_PASSWORD:?")
    providers = yaml.safe_load((PROVISIONING / "dashboards" / "foresight.yml").read_text(encoding="utf-8"))
    (provider,) = providers["providers"]
    assert provider["folder"] == "Foresight"
    assert provider["options"]["path"] == str(home.parent)


# ---- metrics contract ------------------------------------------------------------------------------------


def test_contract_declares_the_next_block_metrics() -> None:
    missing = {n for n in REQUIRED_CONTRACT if checker.base_name(n) not in checker.contract_metrics()}
    assert not missing, f"docs/observability.md, «{checker.CONTRACT_HEADING}»: {sorted(missing)}"
    assert not REQUIRED_CONTRACT & exported_metrics(), "contract metrics are exported now: move them to §3"


def test_delivered_contract_metrics_are_exported_with_their_type() -> None:
    exported = exported_metrics()
    contract = checker.contract_metrics()
    for name, kind in DELIVERED.items():
        assert _exposition(name, kind) <= exported, (name, kind)
        # a delivered metric left the next-block section (§7) for the exposition tables (§3)
        assert checker.base_name(name) not in contract, f"{name}: move it from §7 to §3 of observability.md"
    documented = (REPO / "docs" / "observability.md").read_text(encoding="utf-8")
    for name in DELIVERED:
        assert f"`{name}`" in documented, f"{name} is not documented in docs/observability.md"


def test_used_metrics_are_exported_or_contracted() -> None:
    exported = exported_metrics()
    contract = checker.contract_metrics()
    assert "foresight_ndtp_frames_total" in exported and "foresight_ndtp_frames" not in exported
    assert "foresight_predictor_event_latency_seconds_bucket" in exported
    assert "foresight_replayer_lag_seconds" in exported
    unknown = {
        name: sorted(where)
        for name, where in referenced_metrics().items()
        if name not in exported and checker.base_name(name) not in contract
    }
    assert not unknown, f"not exported and not in docs/observability.md contract: {unknown}"


def test_model_dashboard_uses_only_the_contract() -> None:
    # the model dashboard was built on the contract before the code: now every metric of it is a delivered
    # contract metric or still in the next-block section
    contract = checker.contract_metrics()
    delivered = {checker.base_name(n) for n in DELIVERED}
    model = _dashboards()["foresight-model.json"]
    for panel in _panels(model):
        for target in panel.get("targets", []):
            names = checker.metric_names(target["expr"])
            used = {checker.base_name(n) for n in names}
            assert names and used <= contract | delivered, (panel["title"], names)


# ---- Prometheus --------------------------------------------------------------------------------------------


def test_prometheus_config() -> None:
    config = yaml.safe_load((PROMETHEUS / "prometheus.yml").read_text(encoding="utf-8"))
    assert config["global"]["scrape_interval"] == "5s"
    assert config["global"]["evaluation_interval"] == "5s"
    assert config["rule_files"] == ["alerts.yml"]
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    ports = {"ingest": 8001, "predictor": 8002, "api": 8000, "replayer": 8010, "ml-service": 8003}
    for name, port in ports.items():
        (sd,) = jobs[name]["dns_sd_configs"]
        assert sd == {**sd, "names": [name], "type": "A", "port": port}
        assert {"target_label": "service", "replacement": name} in jobs[name]["relabel_configs"]
    for name in ("redis-exporter", "postgres-exporter", "prometheus"):
        (static,) = jobs[name]["static_configs"]
        assert static["labels"]["service"] == name


def test_alert_rules() -> None:
    rules = _rules()
    names = [rule["alert"] for rule in rules]
    assert len(set(names)) == len(names), "alert names must be unique"
    required = {
        "ServiceDown",
        "DependencyDown",
        "NoTelemetry",
        "ConsumerLagHigh",
        "EventLatencyHigh",
        "CRCErrorSpike",
        "AbusiveClientDisconnected",
        "BusBuffering",
        "DbBuffering",
        "DataLoss",
        "ReplayerLagging",
        "RedisDown",
        "PostgresDown",
        "MLInferenceSlow",
        "MLFallbackActive",
        "OnlineAccuracyDegraded",
        "RetroactiveAlerts",
    }
    assert required <= set(names)
    cyrillic = re.compile("[а-яА-ЯёЁ]")
    for rule in rules:
        assert rule["labels"]["severity"] in {"critical", "warning"}, rule["alert"]
        for key in ("summary", "description"):
            assert cyrillic.search(rule["annotations"][key]), (rule["alert"], key)


# ---- the PromQL parser of scripts/check_dashboards.py ----------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "names"),
    [
        ("sum(rate(foresight_ndtp_frames_total[$__rate_interval]))", {"foresight_ndtp_frames_total"}),
        (
            'histogram_quantile(0.95, sum by (service, le) (rate(x_bucket{instance=~"$instance"}[1m]))) > 1',
            {"x_bucket"},
        ),
        ('count(up{job="predictor"} == 1) or vector(0)', {"up"}),
        ("foo > 5 and on (job, instance) bar == 1", {"foo", "bar"}),
        ("time() - max(foresight_stream_time_seconds) * 1e3", {"foresight_stream_time_seconds"}),
        ('sum by (le) (increase(a_bucket{kind="alert"}[$__range])) offset 5m', {"a_bucket"}),
    ],
)
def test_metric_names(expr: str, names: set[str]) -> None:
    assert checker.metric_names(expr) == names


def test_substitute_and_base_name() -> None:
    expr = 'rate(x{instance=~"$instance"}[$__rate_interval]) / ${__range_s}'
    assert checker.substitute(expr, {"instance"}) == 'rate(x{instance=~".*"}[20s]) / 900'
    assert checker.base_name("a_seconds_bucket") == "a_seconds"
    assert checker.base_name("a_total") == "a"
