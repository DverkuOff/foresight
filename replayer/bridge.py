"""Emulator bridge: drive the official NDTP emulator with the replayed positions.

The official emulator (``ndtp-telemetry-emulator:1.0``) takes cell values only as
``{"type": "G6CellNav00", "fields": {...}}`` (flat keys are silently ignored), stamps every Nav00 with the
current time and reconnects all its devices on every ``POST /api/config`` (see ``docs/facts.md`` §3). So the
bridge periodically (every ``interval_s``, 5 s by default) posts one config with the current position of every
device, and skips the post when nothing changed. The historical timestamps are lost in this mode; it exists to
show that the same tracks flow through the reference emulator.

Only the standard library is used for HTTP, so the service needs no extra dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from http.client import HTTPException
from typing import Any, Protocol

from replayer.clock import Clock, SystemClock
from replayer.stats import ReplayStats
from shared.ndtp import NavRecord

log = logging.getLogger(__name__)

_COORD_SCALE = 10_000_000
_MIN_INTERVAL_S = 1.0


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class HttpClient(Protocol):
    """Minimal async HTTP client used by the bridge (replaced by a fake in tests)."""

    async def get(self, url: str) -> int:
        """Return the status code of ``GET url``."""
        ...

    async def post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, str]:
        """``POST`` JSON and return status code and body."""
        ...


class UrllibClient:
    """:class:`HttpClient` on top of :mod:`urllib` run in a worker thread.

    Args:
        timeout_s: Request timeout.
    """

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s

    def _request(self, url: str, payload: dict[str, Any] | None) -> tuple[int, str]:
        """Send the request; a broken HTTP exchange is reported as :class:`OSError`.

        Raises:
            OSError: On network errors and on malformed or truncated responses (``BadStatusLine``,
                ``IncompleteRead``: an emulator that is restarting), which :mod:`http.client` raises as
                :class:`http.client.HTTPException` rather than :class:`OSError`.
        """
        try:
            return self._exchange(url, payload)
        except HTTPException as exc:
            raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc

    def _exchange(self, url: str, payload: dict[str, Any] | None) -> tuple[int, str]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    async def get(self, url: str) -> int:
        """Return the status code of ``GET url``."""
        return (await asyncio.to_thread(self._request, url, None))[0]

    async def post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, str]:
        """``POST`` JSON and return status code and body."""
        return await asyncio.to_thread(self._request, url, payload)


def nav_fields(nav: NavRecord) -> dict[str, int | bool]:
    """Nav00 ``fields`` for the emulator config.

    Args:
        nav: Position to send.

    Returns:
        ``longitude``/``latitude`` as unsigned ``|deg| * 1e7``, hemisphere bits ``extraDopBit5`` (N) and
        ``extraDopBit6`` (E), validity ``extraDopBit7``, ``speedAvg``, ``course`` and ``altitude``.
    """
    return {
        "longitude": round(abs(nav.lon) * _COORD_SCALE),
        "latitude": round(abs(nav.lat) * _COORD_SCALE),
        "extraDopBit5": nav.lat >= 0,
        "extraDopBit6": nav.lon >= 0,
        "extraDopBit7": nav.valid,
        "speedAvg": nav.speed_avg,
        "course": nav.course,
        "altitude": nav.altitude,
    }


def build_emulator_config(
    positions: Mapping[int, NavRecord], target_host: str, target_port: int, interval_ms: int
) -> dict[str, Any]:
    """Build a ``POST /api/config`` body with one fixed-position device per entry.

    Args:
        positions: ``unit_id -> current position``.
        target_host: NDTP receiver host as seen from the emulator.
        target_port: NDTP receiver port.
        interval_ms: Send period of every device.

    Returns:
        JSON-serialisable config; ``autoGenerate`` is off so the emulator sends the values as given.
    """
    return {
        "targetHost": target_host,
        "targetPort": target_port,
        "units": [
            {
                "unitId": unit,
                "intervalMs": interval_ms,
                "autoGenerate": False,
                "cells": [{"type": "G6CellNav00", "fields": nav_fields(nav)}],
            }
            for unit, nav in sorted(positions.items())
        ],
    }


class EmulatorBridge:
    """Pushes replayed positions into the official emulator.

    Args:
        emulator_url: Emulator REST base URL, e.g. ``http://emulator:18080``.
        target_host: NDTP receiver host as seen from the emulator.
        target_port: NDTP receiver port.
        interval_s: Period of config updates, at least 1 s (each update reconnects every device).
        http: HTTP client; :class:`UrllibClient` by default.
        stats: Shared counters.
        ready_retry_s: Delay between readiness probes.
        clock: Clock for the update period (virtual in tests).

    Attributes:
        posts: Configs accepted by the emulator through this bridge.

    Raises:
        ValueError: If ``interval_s`` is below 1 s.
    """

    def __init__(
        self,
        emulator_url: str,
        target_host: str,
        target_port: int,
        interval_s: float = 5.0,
        *,
        http: HttpClient | None = None,
        stats: ReplayStats | None = None,
        ready_retry_s: float = 2.0,
        clock: Clock | None = None,
    ) -> None:
        if interval_s < _MIN_INTERVAL_S:
            raise ValueError(f"bridge interval must be at least {_MIN_INTERVAL_S} s, got {interval_s}")
        self.base = emulator_url.rstrip("/")
        self.target_host = target_host
        self.target_port = target_port
        self.interval_s = interval_s
        self.http: HttpClient = http or UrllibClient()
        self.stats = stats if stats is not None else ReplayStats()
        self.ready_retry_s = ready_retry_s
        self.clock: Clock = clock or SystemClock()
        self.posts = 0
        self._last: dict[str, Any] | None = None

    @property
    def interval_ms(self) -> int:
        """Send period configured for every emulated device."""
        return round(self.interval_s * 1000)

    async def wait_ready(self) -> None:
        """Wait until ``GET /api/cells`` answers 200 (the emulator starts in about 3 s)."""
        next_warning = time.monotonic() + 30
        while True:
            with contextlib.suppress(OSError):
                if await self.http.get(f"{self.base}/api/cells") == 200:
                    return
            if time.monotonic() >= next_warning:
                log.warning("still waiting for the emulator API at %s", self.base)
                next_warning = time.monotonic() + 30
            await asyncio.sleep(self.ready_retry_s)

    async def push(self, positions: Mapping[int, NavRecord]) -> bool:
        """Post the positions unless they equal the last posted ones.

        Args:
            positions: ``unit_id -> current position``.

        Returns:
            Whether a config was accepted by the emulator. Any failure of the request (network, malformed
            response, client error) is counted in ``bridge_errors`` and logged; the next update retries.
        """
        if not positions:
            return False
        config = build_emulator_config(positions, self.target_host, self.target_port, self.interval_ms)
        if config == self._last:
            return False
        try:
            status, body = await self.http.post_json(f"{self.base}/api/config", config)
        except Exception as exc:
            self.stats.bridge_errors += 1
            log.warning("emulator request failed: %s", _describe(exc))
            return False
        if status != 200:
            self.stats.bridge_errors += 1
            log.warning("emulator rejected the config: HTTP %d: %s", status, body[:300])
            return False
        self.stats.bridge_posts += 1
        self.posts += 1
        self._last = config
        return True

    async def run(self, snapshot: Callable[[], Mapping[int, NavRecord]]) -> None:
        """Push ``snapshot()`` every ``interval_s`` until cancelled.

        A failed update never ends the loop: it is logged, counted in ``bridge_errors`` and retried on the
        next period, so a restarting or misbehaving emulator degrades to retries instead of silently
        stopping the bridge.

        Args:
            snapshot: Returns the current positions.
        """
        while True:
            try:
                await self.push(snapshot())
            except Exception:
                self.stats.bridge_errors += 1
                log.exception("emulator bridge update failed")
            await self.clock.sleep(self.interval_s)

    async def stop_emulation(self) -> None:
        """Post an empty device list so the emulator stops sending (errors are only logged)."""
        config = {"targetHost": self.target_host, "targetPort": self.target_port, "units": []}
        try:
            status, body = await self.http.post_json(f"{self.base}/api/config", config)
        except Exception as exc:
            log.warning("could not stop the emulator: %s", _describe(exc))
            return
        if status != 200:
            log.warning("emulator refused to stop: HTTP %d: %s", status, body[:300])
        self._last = None
