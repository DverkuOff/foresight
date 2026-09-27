"""Performance figures for the dashboard (``GET /api/metrics/perf``) without Prometheus.

The api scrapes the ``/metrics`` of the predictor and ml-service every few seconds and keeps the cumulative
histograms; a quantile is taken over the increase in the last ``window_s`` (as ``histogram_quantile`` over
``rate(...[5m])`` does). A target that does not answer is ``down`` and its figures are ``None``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from prometheus_client.parser import text_string_to_metric_families

from backend.mlclient import HttpClient

log = logging.getLogger(__name__)


def parse_metrics(text: str) -> tuple[dict[str, dict[float, float]], dict[str, float]]:
    """Histogram buckets (``le`` → cumulative count, summed over the other labels) and gauges of a
    Prometheus exposition; histogram names get a ``{label=value}`` suffix for the labels in :data:`KEEP`."""
    hist: dict[str, dict[float, float]] = {}
    gauges: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name.endswith("_bucket") and "le" in s.labels:
                name = s.name[: -len("_bucket")]
                for label in KEEP:
                    if label in s.labels:
                        name += f"{{{label}={s.labels[label]}}}"
                le = math.inf if s.labels["le"] in ("+Inf", "inf") else float(s.labels["le"])
                buckets = hist.setdefault(name, {})
                buckets[le] = buckets.get(le, 0.0) + s.value
            elif family.type in ("gauge", "untyped", "unknown"):
                key = s.name
                if "dependency" in s.labels:
                    key += f"{{dependency={s.labels['dependency']}}}"
                gauges[key] = s.value
    return hist, gauges


KEEP = ("endpoint",)
"""Labels kept apart in histogram names (ml-service's endpoint: only ``/predict`` is the inference)."""


def quantile(q: float, buckets: Mapping[float, float]) -> float | None:
    """``histogram_quantile``: linear interpolation inside the bucket (``None`` without observations)."""
    items = sorted(buckets.items())
    if not items or items[-1][1] <= 0:
        return None
    total = items[-1][1]
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in items:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            if count == prev_count:
                return le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return prev_le


@dataclass
class Target:
    """One scraped service."""

    name: str
    client: HttpClient
    ok: bool | None = None
    error: str | None = None
    history: deque[tuple[float, dict[str, dict[float, float]]]] = field(default_factory=deque)
    gauges: dict[str, float] = field(default_factory=dict)
    scraped_at: float | None = None


class PerfScraper:
    """Scrapes the targets, keeps ``window_s`` of histogram snapshots.

    Args:
        urls: Base URLs by name (``predictor``, ``ml``); an empty URL is skipped.
        interval_s: Scrape period.
        window_s: Window of the quantiles.
        timeout_s: Timeout of one scrape.
        clock: Monotonic clock (tests).
    """

    def __init__(
        self,
        urls: Mapping[str, str],
        *,
        interval_s: float = 5.0,
        window_s: float = 300.0,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.targets: dict[str, Target] = {}
        for name, url in urls.items():
            if not url:
                continue
            try:
                self.targets[name] = Target(name, HttpClient(url))
            except ValueError as exc:
                log.warning("perf: %s not scraped: %s", name, exc)
        self.interval_s = interval_s
        self.window_s = window_s
        self.timeout_s = timeout_s
        self.clock = clock

    def feed(self, name: str, text: str) -> None:
        """Take one exposition of a target (also the test entry point)."""
        target = self.targets[name]
        hist, gauges = parse_metrics(text)
        now = self.clock()
        target.history.append((now, hist))
        while len(target.history) > 2 and target.history[1][0] <= now - self.window_s:
            target.history.popleft()
        target.gauges = gauges
        target.scraped_at = now
        target.ok = True
        target.error = None

    async def scrape(self, name: str) -> None:
        target = self.targets[name]
        try:
            status, body = await asyncio.wait_for(target.client.request("GET", "/metrics"), self.timeout_s)
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            self.feed(name, body.decode("utf-8", "replace"))
        except Exception as exc:
            target.client.close()
            target.ok = False
            target.error = str(exc) or type(exc).__name__

    async def run(self) -> None:
        """Scrape loop (runs until cancelled)."""
        while True:
            await asyncio.gather(*(self.scrape(n) for n in self.targets), return_exceptions=True)
            await asyncio.sleep(self.interval_s)

    def close(self) -> None:
        for target in self.targets.values():
            with contextlib.suppress(Exception):
                target.client.close()

    def state(self, name: str) -> str:
        """``up`` / ``down`` / ``unknown`` / ``disabled`` of a target."""
        target = self.targets.get(name)
        if target is None:
            return "disabled"
        if target.ok is None:
            return "unknown"
        return "up" if target.ok else "down"

    def p(self, name: str, metric: str, q: float = 0.95) -> float | None:
        """Quantile of a histogram over the window (``None``: target down, no observations)."""
        target = self.targets.get(name)
        if target is None or not target.ok or not target.history:
            return None
        newest = target.history[-1][1].get(metric)
        if newest is None:
            return None
        oldest = target.history[0][1].get(metric, {}) if len(target.history) > 1 else {}
        delta = {le: c - oldest.get(le, 0.0) for le, c in newest.items()}
        if any(v < 0 for v in delta.values()):  # the target restarted: counters began again
            delta = dict(newest)
        return quantile(q, delta)

    def gauge(self, name: str, metric: str) -> float | None:
        """Last value of a gauge (``None``: target down or no such series)."""
        target = self.targets.get(name)
        if target is None or not target.ok:
            return None
        return target.gauges.get(metric)
