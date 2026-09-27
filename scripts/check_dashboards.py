#!/usr/bin/env python3
"""Check the Foresight Grafana dashboards against a live Prometheus and the Prometheus rules with promtool.

Standard library only, so it runs on the host without the project environment::

    python3 scripts/check_dashboards.py --prometheus http://localhost:9090  # every panel has data
    python3 scripts/check_dashboards.py --allow-empty      # next-block contract panels may be empty
    python3 scripts/check_dashboards.py --promtool --no-panels  # promtool check config / rules, test rules

Every panel query (``deploy/grafana/dashboards/*.json``) runs as a range query over the dashboard default
range (15 min, step 15 s) with the Grafana variables substituted (``$__rate_interval`` → 20s, ``$__range`` →
15m, template variables → ``.*``). A panel is empty when none of its queries returns a sample. The exit code
is 1 on a PromQL error or an empty panel; with ``--allow-empty`` panels whose metrics all belong to the
next-block contract (``docs/observability.md``, section «Метрики следующего блока») may be empty.
``--promtool`` also runs ``promtool check config``, ``check rules`` and ``test rules`` in the Prometheus
image pinned in ``docker-compose.yml`` (needs Docker).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO / "deploy" / "grafana" / "dashboards"
PROMETHEUS_DIR = REPO / "deploy" / "prometheus"
DOCS = REPO / "docs" / "observability.md"
COMPOSE = REPO / "docker-compose.yml"
CONTRACT_HEADING = "Метрики следующего блока"

RANGE_S = 15 * 60
STEP_S = 15
BUILTIN_VARIABLES = {
    "__rate_interval": "20s",  # max($__interval + scrape, 4 × scrape) at 5 s scrape
    "__interval": "5s",
    "__range": "15m",
    "__range_s": str(RANGE_S),
    "__range_ms": str(RANGE_S * 1000),
}

_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`[^`]*`')
_BRACES = re.compile(r"\{[^{}]*\}")
_RANGE = re.compile(r"\[[^\]]*\]")
_GROUPING = re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")
_IDENT = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_KEYWORDS = {"and", "or", "unless", "bool", "offset", "by", "without", "on", "ignoring", "inf", "nan"}
_SUFFIXES = ("_bucket", "_sum", "_count", "_created", "_total")
EMPTY_WHEN_HEALTHY = {"ALERTS"}  # firing alerts: a query of them alone is empty on a healthy stack
_VARIABLE = re.compile(r"\$\{(\w+)(?::\w+)?\}|\$(\w+)")
_LABEL_VALUES = re.compile(r"^\s*label_values\((.*),\s*(\w+)\s*\)\s*$", re.S)

Json = dict[str, Any]


def metric_names(expr: str) -> set[str]:
    """Metric names used in a PromQL expression (functions, keywords and labels excluded).

    Args:
        expr: PromQL expression.

    Returns:
        Names of the selected metrics.
    """
    text = _STRING.sub('""', expr)
    text = _BRACES.sub(" ", text)
    text = _RANGE.sub(" ", text)
    text = _GROUPING.sub(" ", text)
    names = set()
    for match in _IDENT.finditer(text):
        token = match.group()
        before = text[match.start() - 1] if match.start() else " "
        if before.isdigit() or before == ".":
            continue  # a number or a duration: 1e9, 5m
        if text[match.end() :].lstrip().startswith("(") or token.lower() in _KEYWORDS:
            continue  # a function, an aggregation or an operator
        names.add(token)
    return names


def base_name(metric: str) -> str:
    """Metric family name: ``x_bucket`` / ``x_sum`` / ``x_count`` / ``x_created`` / ``x_total`` → ``x``."""
    for suffix in _SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


def contract_metrics(docs: Path = DOCS) -> set[str]:
    """Family names declared in the next-block contract section of ``docs/observability.md``."""
    lines = docs.read_text(encoding="utf-8").splitlines()
    names: set[str] = set()
    level = None
    for line in lines:
        heading = re.match(r"^(#+)\s", line)
        if heading:
            if level is not None and len(heading.group(1)) <= level:
                break
            if CONTRACT_HEADING in line:
                level = len(heading.group(1))
            continue
        if level is not None:
            names.update(base_name(n) for n in re.findall(r"`(foresight_[a-z0-9_]+)", line))
    return names


def substitute(expr: str, template_variables: set[str]) -> str:
    """Replace Grafana variables with the values Grafana would use for the default time range."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in BUILTIN_VARIABLES:
            return BUILTIN_VARIABLES[name]
        if name in template_variables:
            return ".*"
        return match.group(0)

    return _VARIABLE.sub(repl, expr)


def iter_panels(panels: list[Json]) -> Iterator[Json]:
    """All panels, including the ones nested in collapsed rows."""
    for panel in panels:
        yield panel
        yield from iter_panels(panel.get("panels", []))


class Prometheus:
    """Minimal Prometheus HTTP API client.

    Args:
        url: Base URL, e.g. ``http://localhost:9090``.
        timeout: Request timeout, seconds.
    """

    def __init__(self, url: str, timeout: float = 20.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path: str, params: dict[str, str] | list[tuple[str, str]], *, post: bool = True) -> Json:
        query = urllib.parse.urlencode(params)
        if post:  # long PromQL fits into a form body
            request = urllib.request.Request(
                f"{self.url}{path}",
                data=query.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        else:
            request = urllib.request.Request(f"{self.url}{path}?{query}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:  # 400/422: bad query, the body says why
            try:
                return json.load(exc)
            except ValueError:
                return {"status": "error", "errorType": f"HTTP {exc.code}", "error": exc.reason}

    def query_range(self, expr: str, end: float) -> tuple[list[Json] | None, str | None]:
        """Run a range query over the last 15 minutes before ``end``.

        Returns:
            ``(series, None)`` or ``(None, error)``.
        """
        reply = self._call(
            "/api/v1/query_range",
            {"query": expr, "start": str(end - RANGE_S), "end": str(end), "step": str(STEP_S)},
        )
        if reply.get("status") != "success":
            return None, f"{reply.get('errorType')}: {reply.get('error')}"
        return reply["data"]["result"], None

    def label_values(self, label: str, selector: str) -> tuple[list[str] | None, str | None]:
        """Values of ``label`` on series matching ``selector``."""
        params = [("match[]", selector), ("start", str(time.time() - RANGE_S))]
        reply = self._call(f"/api/v1/label/{label}/values", params, post=False)
        if reply.get("status") != "success":
            return None, f"{reply.get('errorType')}: {reply.get('error')}"
        return reply["data"], None


def has_samples(series: list[Json]) -> bool:
    """Whether any series has a non-NaN sample."""
    for item in series:
        for _, value in item.get("values", []):
            if not math.isnan(float(value)):
                return True
    return False


@dataclass
class Report:
    """Outcome of the panel check."""

    errors: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: int = 0

    @property
    def failed(self) -> bool:
        """Errors or empty panels outside the allowed contract."""
        return bool(self.errors or self.empty)


def check_panels(prom: Prometheus, dashboards: list[Json], contract: set[str] | None, strict: bool) -> Report:
    """Run every panel, annotation and variable query of the dashboards.

    Args:
        prom: Prometheus client.
        dashboards: Dashboard JSON documents.
        contract: Family names that may be empty (``None``: nothing may be empty).
        strict: Report an empty query of a non-empty panel as an error, not a warning.

    Returns:
        The report.
    """
    report = Report()
    end = time.time()
    for dashboard in dashboards:
        title = dashboard["title"]
        variables = {v["name"] for v in dashboard.get("templating", {}).get("list", [])}
        for variable in dashboard.get("templating", {}).get("list", []):
            match = _LABEL_VALUES.match(str(variable.get("query", "")))
            if not match:
                continue
            values, error = prom.label_values(match.group(2), match.group(1))
            where = f"{title} / переменная ${variable['name']}"
            if error:
                report.errors.append(f"{where}: {error}")
            elif not values:
                report.warnings.append(f"{where}: нет значений")
        for annotation in dashboard.get("annotations", {}).get("list", []):
            if "expr" in annotation:
                _, error = prom.query_range(substitute(annotation["expr"], variables), end)
                if error:
                    report.errors.append(f"{title} / аннотация «{annotation['name']}»: {error}")
        for panel in iter_panels(dashboard.get("panels", [])):
            targets = [t for t in panel.get("targets", []) if t.get("expr")]
            if not targets:
                continue
            where = f"{title} / {panel.get('title')}"
            data = False
            names: set[str] = set()
            empty_targets = []
            for target in targets:
                names |= metric_names(target["expr"])
                series, error = prom.query_range(substitute(target["expr"], variables), end)
                if error:
                    report.errors.append(f"{where} [{target.get('refId')}]: {error}\n    {target['expr']}")
                    continue
                if series and has_samples(series):
                    data = True
                elif not metric_names(target["expr"]) <= EMPTY_WHEN_HEALTHY:
                    empty_targets.append(f"{where} [{target.get('refId')}]: пусто: {target['expr']}")
            if data:
                report.ok += 1
                (report.errors if strict else report.warnings).extend(empty_targets)
            elif contract is not None and names and {base_name(n) for n in names} <= contract:
                report.allowed.append(where)
            else:
                report.empty.append(f"{where}: {', '.join(sorted(names))}")
    return report


def load_dashboards(directory: Path = DASHBOARD_DIR) -> list[Json]:
    """Dashboard JSON documents of the repository, sorted by file name."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def prometheus_image(compose: Path = COMPOSE) -> str:
    """The Prometheus image pinned in the observability compose file."""
    match = re.search(r"image:\s*(prom/prometheus:\S+)", compose.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no prom/prometheus image in {compose}")
    return match.group(1)


def run_promtool(image: str, rules_dir: Path = PROMETHEUS_DIR) -> bool:
    """``promtool check config``, ``check rules`` and ``test rules`` in the Prometheus image.

    Returns:
        ``True`` if every check passed.
    """
    base = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{rules_dir}:/etc/prometheus:ro",
        "--entrypoint",
        "promtool",
        image,
    ]
    commands = [
        ["check", "config", "/etc/prometheus/prometheus.yml"],
        ["check", "rules", "/etc/prometheus/alerts.yml"],
    ]
    commands += [["test", "rules", f"/etc/prometheus/{p.name}"] for p in sorted(rules_dir.glob("*_test.yml"))]
    ok = True
    for command in commands:
        print(f"$ promtool {' '.join(command)}", flush=True)
        result = subprocess.run([*base, *command], check=False)
        ok &= result.returncode == 0
    return ok


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prometheus",
        default=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus base URL (default: $PROMETHEUS_URL or http://localhost:9090)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="panels that use only next-block contract metrics (docs/observability.md) may be empty",
    )
    parser.add_argument(
        "--strict", action="store_true", help="an empty query of a panel with data is an error"
    )
    parser.add_argument("--promtool", action="store_true", help="also run promtool checks (Docker)")
    parser.add_argument("--no-panels", action="store_true", help="skip the live panel check")
    args = parser.parse_args(argv)

    failed = False
    if args.promtool:
        failed |= not run_promtool(prometheus_image())
    if not args.no_panels:
        contract = contract_metrics() if args.allow_empty else None
        report = check_panels(Prometheus(args.prometheus), load_dashboards(), contract, args.strict)
        for line in report.warnings:
            print(f"[warn ] {line}")
        for line in report.allowed:
            print(f"[contr] {line}: пусто, метрики контракта следующего блока")
        for line in report.empty:
            print(f"[EMPTY] {line}")
        for line in report.errors:
            print(f"[ERROR] {line}")
        print(
            f"panels with data: {report.ok}, contract (allowed empty): {len(report.allowed)}, "
            f"empty: {len(report.empty)}, errors: {len(report.errors)}, warnings: {len(report.warnings)}"
        )
        failed |= report.failed
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
