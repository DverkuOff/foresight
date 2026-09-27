"""Replay control for the HTTP API: start/stop/speed, a cache of loaded splits and an event journal.

The controller owns at most one :class:`~replayer.engine.Replayer` session at a time; starting a new one stops
the current one first. Counters (:class:`~replayer.stats.ReplayStats`) live here so Prometheus counters stay
monotonic across sessions. ``epoch`` grows on every ``start``/``restart`` event: whoever follows the stream
clock (the predictor) must reset it when the epoch changes, because timestamps jump back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any

from replayer.bridge import HttpClient
from replayer.clock import Clock
from replayer.engine import RUNNING, WAITING, ReplayConfig, Replayer, check_speed, select_replay
from replayer.source import ReplayData, load_replay_data
from replayer.stats import ReplayStats

log = logging.getLogger(__name__)

IDLE = "idle"
LOADING = "loading"

Loader = Callable[[str], ReplayData]
"""Loads a split, e.g. :func:`replayer.source.load_replay_data`."""

_EPOCH_EVENTS = ("start", "restart")


class ReplayController:
    """Runs replay sessions on request.

    Args:
        config: Default parameters; each :meth:`start` may override some of them and they become the new
            defaults.
        clock: Clock for the sessions (tests).
        http: HTTP client for the bridge mode (tests).
        loader: Split loader; results are cached per split.
        max_events: Size of the event journal.

    Attributes:
        config: Parameters of the current (or next) session.
        stats: Counters since process start.
        events: Recent lifecycle events, oldest first.
        epoch: Number of ``start``/``restart`` events so far.
        error: Last start failure (autostart), if any.
    """

    def __init__(
        self,
        config: ReplayConfig | None = None,
        *,
        clock: Clock | None = None,
        http: HttpClient | None = None,
        loader: Loader = load_replay_data,
        max_events: int = 100,
    ) -> None:
        self.config = config or ReplayConfig()
        self.config.validate()
        self.stats = ReplayStats()
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.epoch = 0
        self.error: str | None = None
        self.session: Replayer | None = None
        self._clock = clock
        self._http = http
        self._loader = loader
        self._cache: dict[str, ReplayData] = {}
        self._task: asyncio.Task[None] | None = None
        self._autostart: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._loading = False

    # ---- control ------------------------------------------------------------------------------------

    async def start(
        self,
        *,
        split: str | None = None,
        speed: float | None = None,
        start: str | None = None,
        until: str | None = None,
        units: list[int] | tuple[int, ...] | None = None,
        loop: bool | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Start a session, replacing the running one.

        ``None`` keeps the current value; an empty ``start``/``until`` clears the bound, empty ``units``
        selects all devices.

        Returns:
            Status right after the start.

        Raises:
            ValueError: On invalid parameters or an empty selection.
            FileNotFoundError: If the split's ``traffic.csv`` is missing.
        """
        changes: dict[str, Any] = {}
        if split is not None:
            changes["split"] = split
        if speed is not None:
            changes["speed"] = speed
        if start is not None:
            changes["start"] = start or None
        if until is not None:
            changes["until"] = until or None
        if units is not None:
            changes["units"] = tuple(int(u) for u in units)
        if loop is not None:
            changes["loop"] = loop
        if mode is not None:
            changes["mode"] = mode
        config = replace(self.config, **changes)
        config.validate()
        async with self._lock:
            data = await self._load(config.split)
            selected = select_replay(data, config)
            await self._stop_session()
            session = Replayer(
                selected, config, clock=self._clock, stats=self.stats, http=self._http, on_event=self._record
            )
            self.config = config
            self.error = None
            self.session = session
            self._task = asyncio.create_task(session.run(), name="replay")
        await asyncio.sleep(0)  # let the session enter its first state
        return self.status()

    async def stop(self) -> dict[str, Any]:
        """Stop the running session (no-op when idle).

        Returns:
            Status after the stop.
        """
        async with self._lock:
            await self._stop_session()
        return self.status()

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Change the speed of the running session and the default for the next one.

        Args:
            speed: New speed.

        Returns:
            Status after the change.

        Raises:
            ValueError: If the speed is out of range.
        """
        check_speed(speed)
        self.config = replace(self.config, speed=speed)
        session = self.session
        if session is not None and not session.done:
            session.set_speed(speed)
        else:
            self._record("speed", {"speed": speed})
        return self.status()

    def start_in_background(self) -> None:
        """Start with the default parameters without blocking (autostart); failures go to ``error``."""

        async def autostart() -> None:
            try:
                await self.start()
            except (ValueError, OSError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                log.error("autostart failed: %s", self.error)

        self._autostart = asyncio.create_task(autostart(), name="replay-autostart")

    async def shutdown(self) -> None:
        """Stop everything (service shutdown)."""
        if self._autostart is not None and not self._autostart.done():
            self._autostart.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._autostart
        await self.stop()

    # ---- observation --------------------------------------------------------------------------------

    @property
    def state(self) -> str:
        """``idle``, ``loading`` or the session state."""
        if self._loading:
            return LOADING
        return self.session.state if self.session is not None else IDLE

    @property
    def running(self) -> bool:
        """Whether a session is waiting for the receiver or replaying."""
        return self.session is not None and self.session.state in (WAITING, RUNNING)

    def lag_s(self) -> float:
        """Lag of the current session, 0 when idle."""
        return self.session.lag_s() if self.session is not None and not self.session.done else 0.0

    def backlog(self) -> int:
        """Queued packets of the current session."""
        return self.session.backlog() if self.session is not None else 0

    def data_time(self) -> float | None:
        """Data clock of the current session."""
        return self.session.data_time if self.session is not None else None

    def status(self) -> dict[str, Any]:
        """Full status: configuration, session progress, epoch and recent events."""
        cfg = self.config
        out: dict[str, Any] = {
            "state": IDLE,
            "mode": cfg.mode,
            "split": cfg.split,
            "speed": cfg.speed,
            "loop": cfg.loop,
            "start": cfg.start,
            "until": cfg.until,
            "units_filter": list(cfg.units),
            "target": f"{cfg.host}:{cfg.port}",
        }
        if self.session is not None:
            out.update(self.session.status())
        out["state"] = self.state
        out["epoch"] = self.epoch
        out["error"] = self.error or out.get("error")
        out["events"] = list(self.events)[-20:]
        return out

    def config_dict(self) -> dict[str, Any]:
        """Current parameters as a dict."""
        return asdict(self.config)

    # ---- internals ----------------------------------------------------------------------------------

    async def _load(self, split: str) -> ReplayData:
        cached = self._cache.get(split)
        if cached is not None:
            return cached
        self._loading = True
        try:
            data = await asyncio.to_thread(self._loader, split)
        finally:
            self._loading = False
        self._cache[split] = data
        log.info("loaded split %s: %d packets, %d devices", split, len(data), len(data.units))
        return data

    async def _stop_session(self) -> None:
        session = self.session
        if session is not None and not session.done:
            await session.stop()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})

    def _record(self, event: str, details: dict[str, Any]) -> None:
        if event in _EPOCH_EVENTS:
            self.epoch += 1
        self.events.append({"at": datetime.now(UTC), "event": event, "epoch": self.epoch, "details": details})
