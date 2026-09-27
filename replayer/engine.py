"""Replay engine: one schedule for all devices, delivered over NDTP or through the emulator bridge.

The scheduler walks the packets in ``receive_time`` order and hands each one out when the data clock
reaches it (``data = data0 + (mono - mono0) * speed``, see :class:`~replayer.clock.ReplaySchedule`). Sleeps
are capped at :attr:`Replayer.MAX_IDLE_S` so speed changes and stops apply quickly; due times always come from
the anchor, so there is no drift. Handing out is a queue append, so the scheduler itself does not fall behind:
lag builds up only in the per-device queues (slow or absent receiver), is bounded by ``max_queue`` and is
reported by :meth:`Replayer.lag_s`.

Lifecycle: ``waiting`` (for the receiver or the emulator API) → ``running`` → ``finished`` (``--once``) or
``stopped``/``failed``. Every pass over the data emits a ``start`` (first) or ``restart`` (``--loop``) event:
consumers that follow the stream clock must reset it then, because timestamps jump back to the first packet.

In ``--loop`` the next pass starts :attr:`Replayer.LOOP_GAP_S` wall seconds after the last packet of the
previous one was due. The gap lets consumers see the end of a pass before timestamps jump back, and it keeps a
window of zero length (every packet at the same ``receive_time``) from restarting in a tight loop that never
yields to the event loop. Without ``--until`` a loop pass also leaves out stragglers received long after the
last ``event_time`` (see :func:`select_replay`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from replayer.bridge import EmulatorBridge, HttpClient
from replayer.clock import Clock, ReplaySchedule, SystemClock
from replayer.link import LinkSettings, UnitLink
from replayer.source import ReplayData
from replayer.stats import ReplayStats
from shared.data import SPLITS
from shared.ndtp import NavRecord

log = logging.getLogger(__name__)

MIN_SPEED = 0.1
MAX_SPEED = 1000.0
MODES = ("ndtp", "bridge")
ON_DISCONNECT = ("buffer", "skip")

WAITING = "waiting"
RUNNING = "running"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"

_EPS_S = 1e-6

#: In ``--loop`` without ``until``, a pass ends this many data seconds after the last ``event_time``.
LOOP_TAIL_S = 600.0

EventHandler = Callable[[str, dict[str, Any]], None]
"""Callback ``(event, details)`` for lifecycle events: start, restart, speed, finish, stop, fail."""


def check_speed(speed: float) -> float:
    """Validate a replay speed.

    Args:
        speed: Data seconds per wall second.

    Returns:
        The speed.

    Raises:
        ValueError: If it is outside ``[MIN_SPEED, MAX_SPEED]``.
    """
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be within [{MIN_SPEED}, {MAX_SPEED}], got {speed}")
    return speed


def _utc(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(float(ts), UTC)


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Replay parameters.

    Attributes:
        split: Dataset split to replay.
        speed: Data seconds per wall second (x1…x120 in practice).
        start: Start of the ``receive_time`` window (``HH:MM[:SS]`` of the data day or ISO datetime).
        until: End of the window, exclusive.
        units: Devices to replay (``unit_id`` or ``tr_id``); empty means all.
        loop: Start over after the last packet instead of finishing.
        mode: ``ndtp`` — own NDTP connections; ``bridge`` — through the official emulator.
        host: NDTP receiver host.
        port: NDTP receiver port.
        on_disconnect: ``buffer`` queues packets while a connection is down and sends them after the new
            handshake; ``skip`` drops them.
        max_queue: Per-device queue bound (the oldest packets are dropped beyond it).
        backoff_initial_s: First reconnect delay.
        backoff_max_s: Largest reconnect delay.
        connect_timeout_s: TCP connect timeout.
        write_timeout_s: Unflushable write timeout (dead receiver).
        drain_timeout_s: How long ``--once`` keeps sending queued packets after the last one is due.
        emulator_url: Emulator REST base URL (bridge mode).
        bridge_interval_s: Period of emulator config updates, at least 1 s.
        bridge_target_host: Receiver host as seen from the emulator; defaults to ``host``.
        bridge_target_port: Receiver port as seen from the emulator; defaults to ``port``.
        bridge_stale_s: Devices silent for longer (data seconds) are left out of the emulator config.
    """

    split: str = "test"
    speed: float = 30.0
    start: str | None = None
    until: str | None = None
    units: tuple[int, ...] = ()
    loop: bool = False
    mode: str = "ndtp"
    host: str = "127.0.0.1"
    port: int = 9201
    on_disconnect: str = "buffer"
    max_queue: int = 5000
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 5.0
    connect_timeout_s: float = 5.0
    write_timeout_s: float = 10.0
    drain_timeout_s: float = 5.0
    emulator_url: str = "http://localhost:18080"
    bridge_interval_s: float = 5.0
    bridge_target_host: str | None = None
    bridge_target_port: int | None = None
    bridge_stale_s: float = 600.0

    def validate(self) -> None:
        """Check the values.

        Raises:
            ValueError: On the first invalid value.
        """
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}, expected one of {SPLITS}")
        check_speed(self.speed)
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {MODES}")
        if self.on_disconnect not in ON_DISCONNECT:
            raise ValueError(f"unknown on_disconnect {self.on_disconnect!r}, expected one of {ON_DISCONNECT}")
        if not 0 < self.port <= 0xFFFF:
            raise ValueError(f"bad port {self.port}")
        if self.max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        if self.bridge_interval_s < 1:
            raise ValueError("bridge_interval_s must be at least 1 s (every update reconnects all devices)")

    def link_settings(self) -> LinkSettings:
        """Connection parameters for :class:`~replayer.link.UnitLink`."""
        return LinkSettings(
            host=self.host,
            port=self.port,
            max_queue=self.max_queue,
            buffer_while_down=self.on_disconnect == "buffer",
            backoff_initial_s=self.backoff_initial_s,
            backoff_max_s=self.backoff_max_s,
            connect_timeout_s=self.connect_timeout_s,
            write_timeout_s=self.write_timeout_s,
        )


def select_replay(data: ReplayData, config: ReplayConfig) -> ReplayData:
    """Select the packets a session replays: devices, the ``[start, until)`` window and loop stragglers.

    In ``--loop`` without ``until`` packets received more than :data:`LOOP_TAIL_S` after the last
    ``event_time`` are left out: a handful of stragglers arrived hours late (in ``test`` one packet at 04:24
    of the next day) and would stretch every pass by hours of silence before the restart. ``--once`` and an
    explicit ``until`` replay the window exactly.

    Args:
        data: A loaded split.
        config: Session parameters.

    Returns:
        The packets to replay.

    Raises:
        ValueError: On unknown devices, a bad time or an empty selection.
    """
    selected = data.select(config.units, config.start, config.until)
    if config.loop and not config.until:
        trimmed = selected.trim_tail(LOOP_TAIL_S)
        if len(trimmed) < len(selected):
            log.info(
                "loop pass ends %g s after the last event_time: %d late packets left out",
                LOOP_TAIL_S,
                len(selected) - len(trimmed),
            )
        selected = trimmed
    return selected


class Replayer:
    """One replay session over prepared data.

    Args:
        data: Packets to replay (already filtered), sorted by ``receive_time``.
        config: Parameters; ``split``/``start``/``until``/``units`` are informational here.
        clock: Clock for the schedule; virtual in tests.
        stats: Counters shared with the process (Prometheus).
        http: HTTP client for the bridge mode.
        on_event: Lifecycle event callback.
        on_dispatch: Observer called as ``(row, monotonic_time)`` for every packet handed out.

    Raises:
        ValueError: On an invalid config or empty data.
    """

    #: Longest single sleep of the scheduler, wall seconds.
    MAX_IDLE_S = 0.5
    #: Pause between ``--loop`` passes after the last packet was due, wall seconds.
    LOOP_GAP_S = 1.0
    #: Packets handed out before yielding to the event loop during a catch-up burst.
    YIELD_EVERY = 500
    #: Period of "still waiting for the receiver" warnings.
    WAIT_LOG_S = 30.0

    def __init__(
        self,
        data: ReplayData,
        config: ReplayConfig,
        *,
        clock: Clock | None = None,
        stats: ReplayStats | None = None,
        http: HttpClient | None = None,
        on_event: EventHandler | None = None,
        on_dispatch: Callable[[int, float], None] | None = None,
    ) -> None:
        config.validate()
        if not len(data):
            raise ValueError("nothing to replay")
        self.data = data
        self.config = config
        self.clock = clock or SystemClock()
        self.stats = stats if stats is not None else ReplayStats()
        self.speed = config.speed
        self.state = "created"
        self.error: str | None = None
        self.cycle = 0
        self.started_at: datetime | None = None
        self._on_event = on_event
        self._on_dispatch = on_dispatch
        self._schedule: ReplaySchedule | None = None
        self._idx = 0
        self._final_time: float | None = None
        self._task: asyncio.Task[Any] | None = None
        self._played = False
        self._any_connected = asyncio.Event()
        self._latest: dict[int, int] = {}
        self._unit_count = len(data.units)
        self._links: dict[int, UnitLink] = {}
        self._bridge: EmulatorBridge | None = None
        self._bridge_task: asyncio.Task[None] | None = None
        if config.mode == "bridge":
            self._bridge = EmulatorBridge(
                config.emulator_url,
                config.bridge_target_host or config.host,
                config.bridge_target_port or config.port,
                config.bridge_interval_s,
                http=http,
                stats=self.stats,
            )
        else:
            settings = config.link_settings()
            self._links = {
                int(unit): UnitLink(
                    int(unit), settings, stats=self.stats, clock=self.clock, on_connected=self._connected
                )
                for unit in data.units
            }

    # ---- control ------------------------------------------------------------------------------------

    async def run(self) -> None:
        """Replay until the data ends (``loop=False``), :meth:`stop` or cancellation.

        Failures are caught and reported through :attr:`state`/:attr:`error`; cancellation propagates after a
        graceful shutdown.

        Raises:
            RuntimeError: If the session was already started.
        """
        if self._task is not None:
            raise RuntimeError("a Replayer runs only once")
        self._task = asyncio.current_task()
        self.started_at = datetime.now(UTC)
        try:
            self.state = WAITING
            await self._connect()
            self.state = RUNNING
            self._played = True
            if self._bridge is not None:
                self._bridge_task = asyncio.create_task(
                    self._bridge.run(self.positions), name="emulator-bridge"
                )
            await self._play()
            self._final_time = float(self.data.recv[-1])
            await self._finish()
            self.state = FINISHED
            self._emit("finish", cycles=self.cycle, packets_sent=self._sent())
        except asyncio.CancelledError:
            self._final_time = self._current_data_time()
            self.state = STOPPED
            self._emit("stop", data_time=_utc(self._final_time), packets_sent=self._sent())
            raise
        except Exception as exc:
            self._final_time = self._current_data_time()
            self.state = FAILED
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("replay failed")
            self._emit("fail", error=self.error)
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Stop the running session gracefully and wait until it is down."""
        task = self._task
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait({task})

    def set_speed(self, speed: float) -> None:
        """Change the speed on the fly without a jump of the data clock.

        Args:
            speed: New speed.

        Raises:
            ValueError: If the speed is out of range.
        """
        check_speed(speed)
        if self._schedule is not None:
            self._schedule.set_speed(self.clock.monotonic(), speed)
        previous, self.speed = self.speed, speed
        self._emit("speed", speed=speed, previous=previous)

    # ---- observation --------------------------------------------------------------------------------

    @property
    def links(self) -> Mapping[int, UnitLink]:
        """NDTP connections by ``unit_id`` (empty in bridge mode)."""
        return self._links

    @property
    def done(self) -> bool:
        """Whether the session has ended (finished, stopped or failed)."""
        return self.state in (FINISHED, STOPPED, FAILED)

    @property
    def data_time(self) -> float | None:
        """Current position of the data clock (Unix seconds of ``receive_time``) or ``None`` before start."""
        if self.done:
            return self._final_time
        return self._current_data_time()

    def lag_s(self) -> float:
        """How far behind schedule delivery is: age (wall seconds) of the oldest packet not yet sent."""
        now = self.clock.monotonic()
        dues = [due for link in self._links.values() if (due := link.oldest_due()) is not None]
        lag = now - min(dues) if dues else 0.0
        schedule = self._schedule
        if self.state == RUNNING and schedule is not None and self._idx < len(self.data):
            lag = max(lag, now - schedule.mono_at(float(self.data.recv[self._idx])))
        return max(lag, 0.0)

    def backlog(self) -> int:
        """Packets waiting in the per-device queues."""
        return sum(link.pending for link in self._links.values())

    def connections_active(self) -> int:
        """Open NDTP connections of this session."""
        return sum(link.connected for link in self._links.values())

    @property
    def target(self) -> str:
        """Human-readable destination."""
        cfg = self.config
        if self._bridge is not None:
            return f"{self._bridge.base} -> {self._bridge.target_host}:{self._bridge.target_port}"
        return f"{cfg.host}:{cfg.port}"

    def positions(self) -> dict[int, NavRecord]:
        """Latest fix of every device heard recently (bridge mode): the dispatched packet with the newest
        ``event_time``; devices silent for more than ``bridge_stale_s`` data seconds are left out."""
        horizon = self.data_time
        if horizon is None:
            return {}
        stale = self.config.bridge_stale_s
        recv = self.data.recv
        return {unit: self.data.nav(i) for unit, i in self._latest.items() if horizon - recv[i] <= stale}

    def status(self) -> dict[str, Any]:
        """Snapshot for ``GET /replay/status`` and CLI progress lines."""
        cfg = self.config
        links = self._links.values()
        total = len(self.data)
        return {
            "state": self.state,
            "mode": cfg.mode,
            "split": self.data.split,
            "speed": self.speed,
            "loop": cfg.loop,
            "start": cfg.start,
            "until": cfg.until,
            "units_filter": list(cfg.units),
            "target": self.target,
            "cycle": self.cycle,
            "units": self._unit_count,
            "connections_active": self.connections_active(),
            "packets_total": total,
            "packets_dispatched": self._idx,
            "progress": round(self._idx / total, 4),
            "packets_sent": self._sent(),
            "packets_dropped": sum(link.dropped for link in links),
            "reconnects": sum(max(link.connections - 1, 0) for link in links),
            "bridge_posts": self._bridge.posts if self._bridge is not None else 0,
            "backlog": self.backlog(),
            "lag_s": round(self.lag_s(), 3),
            "data_time": _utc(self.data_time),
            "data_first": _utc(self.data.recv[0]),
            "data_last": _utc(self.data.recv[-1]),
            "started_at": self.started_at,
            "error": self.error,
        }

    # ---- internals ----------------------------------------------------------------------------------

    def _sent(self) -> int:
        return sum(link.sent for link in self._links.values())

    def _emit(self, event: str, **details: Any) -> None:
        log.info("replay %s: %s", event, ", ".join(f"{k}={v}" for k, v in details.items()))
        if self._on_event is not None:
            self._on_event(event, details)

    def _connected(self, _unit_id: int) -> None:
        self._any_connected.set()

    def _current_data_time(self) -> float | None:
        if self._schedule is None:
            return None
        recv = self.data.recv
        return min(max(self._schedule.data_at(self.clock.monotonic()), float(recv[0])), float(recv[-1]))

    async def _connect(self) -> None:
        """Wait for the receiver (a first successful handshake) or the emulator API."""
        if self._bridge is not None:
            log.info("waiting for the emulator API at %s", self._bridge.base)
            await self._bridge.wait_ready()
            return
        for link in self._links.values():
            link.start()
        cfg = self.config
        while not self._any_connected.is_set():
            try:
                await asyncio.wait_for(self._any_connected.wait(), self.WAIT_LOG_S)
            except TimeoutError:
                errors = sorted({link.last_error for link in self._links.values() if link.last_error})
                log.warning(
                    "waiting for the NDTP receiver at %s:%d (%s)",
                    cfg.host,
                    cfg.port,
                    "; ".join(errors) or "no answer",
                )
        log.info("NDTP receiver at %s:%d is up", cfg.host, cfg.port)

    def _begin_cycle(self) -> ReplaySchedule:
        self.cycle += 1
        self.stats.cycles += 1
        self._latest.clear()
        self._idx = 0
        first = float(self.data.recv[0])
        self._schedule = ReplaySchedule(first, self.clock.monotonic(), self.speed)
        self._emit(
            "start" if self.cycle == 1 else "restart",
            cycle=self.cycle,
            split=self.data.split,
            data_time=_utc(first),
            speed=self.speed,
        )
        return self._schedule

    async def _play(self) -> None:
        recv = self.data.recv
        n = len(recv)
        last = float(recv[-1])
        while True:
            schedule = self._begin_cycle()
            idx = 0
            while idx < n:
                self._check_bridge()
                now = self.clock.monotonic()
                burst = 0
                while idx < n and burst < self.YIELD_EVERY:
                    due = schedule.mono_at(float(recv[idx]))
                    if due > now + _EPS_S:
                        break
                    self._dispatch(idx, due)
                    idx += 1
                    burst += 1
                self._idx = idx
                if burst >= self.YIELD_EVERY:
                    await asyncio.sleep(0)  # catching up: let the connections write
                    continue
                if idx >= n:
                    break
                delay = schedule.mono_at(float(recv[idx])) - self.clock.monotonic()
                await self.clock.sleep(min(max(delay, 0.0), self.MAX_IDLE_S))
            if not self.config.loop:
                return
            # always await before the next pass (a zero-length window would otherwise spin forever); the
            # schedule is re-read on every wake-up because the speed may change during the gap
            await asyncio.sleep(0)
            while (delay := schedule.mono_at(last) + self.LOOP_GAP_S - self.clock.monotonic()) > _EPS_S:
                self._check_bridge()
                await self.clock.sleep(min(delay, self.MAX_IDLE_S))

    def _check_bridge(self) -> None:
        """Fail the session if the bridge task ended while replaying (it only stops when cancelled).

        Raises:
            RuntimeError: If the emulator bridge is no longer running.
        """
        task = self._bridge_task
        if task is None or not task.done():
            return
        exc = None if task.cancelled() else task.exception()
        raise RuntimeError(f"emulator bridge stopped: {exc!r}") from exc

    def _dispatch(self, i: int, due: float) -> None:
        self.stats.packets_dispatched += 1
        unit = int(self.data.unit_id[i])
        if self._bridge is not None:
            prev = self._latest.get(unit)
            if prev is None or self.data.event_s[i] >= self.data.event_s[prev]:
                self._latest[unit] = i
        else:
            self._links[unit].enqueue(self.data.nav(i), due)
        if self._on_dispatch is not None:
            self._on_dispatch(i, self.clock.monotonic())

    async def _finish(self) -> None:
        """Finish a ``--once`` pass.

        NDTP: flush the queues within ``drain_timeout_s``. Bridge: post the last positions and keep them for
        one bridge interval so the emulator sends them at least once; :meth:`_shutdown` then clears the
        emulator (``units: []``) as after a stop, so devices do not keep reporting a parked position forever.
        """
        if self._bridge is not None:
            await self._stop_bridge_task()
            await self._bridge.push(self.positions())
            if self._bridge.posts:
                await self._sleep_wall(self._bridge.interval_s)
            return
        drain = self.config.drain_timeout_s
        await asyncio.gather(*(link.close(drain) for link in self._links.values()))

    async def _sleep_wall(self, seconds: float) -> None:
        """Sleep ``seconds`` on the session clock in slices of :attr:`MAX_IDLE_S`."""
        deadline = self.clock.monotonic() + seconds
        while (delay := deadline - self.clock.monotonic()) > _EPS_S:
            await self.clock.sleep(min(delay, self.MAX_IDLE_S))

    async def _stop_bridge_task(self) -> None:
        task = self._bridge_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})

    async def _shutdown(self) -> None:
        await self._stop_bridge_task()
        if self._bridge is not None and self._played:
            await self._bridge.stop_emulation()
        if self._links:
            await asyncio.gather(*(link.close() for link in self._links.values()))
