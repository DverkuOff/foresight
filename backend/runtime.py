"""Building blocks shared by the services: backoff, dependency status, buffers, logging, the HTTP runner."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class BoundedBuffer[T]:
    """FIFO buffer with a size limit that evicts the oldest items (and counts them).

    Consumers :meth:`peek` a batch, try to deliver it and :meth:`commit` it only on success, so a failed
    delivery needs no re-queueing and keeps the order. Sequence numbers make :meth:`commit` correct even if
    items were evicted while the delivery was in flight.

    Args:
        maxlen: Largest number of items kept.
    """

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.evicted = 0
        self._items: deque[tuple[int, T]] = deque()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: T) -> None:
        """Add an item, evicting the oldest ones beyond ``maxlen``."""
        self._seq += 1
        self._items.append((self._seq, item))
        while len(self._items) > self.maxlen:
            self._items.popleft()
            self.evicted += 1

    def peek(self, count: int) -> tuple[int, list[T]]:
        """Return up to ``count`` oldest items without removing them.

        Returns:
            The sequence number of the last returned item (0 if empty) and the items.
        """
        head = list(itertools.islice(self._items, count))
        return (head[-1][0] if head else 0), [item for _, item in head]

    def commit(self, upto_seq: int) -> None:
        """Remove delivered items (sequence numbers up to ``upto_seq``)."""
        while self._items and self._items[0][0] <= upto_seq:
            self._items.popleft()

    def clear(self) -> int:
        """Drop everything; returns the number of dropped items."""
        count = len(self._items)
        self._items.clear()
        return count


class TaskSet:
    """Background tasks of a service: started together, cancelled together on shutdown."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[Any]] = []

    def spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        """Start a background task; an unexpected crash is logged."""
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(_log_crash)
        self._tasks.append(task)
        return task

    async def cancel(self) -> None:
        """Cancel all tasks and wait for them."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(BaseException):
                await task
        self._tasks.clear()


def _log_crash(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("background task %s crashed", task.get_name(), exc_info=task.exception())


class Backoff:
    """Exponential reconnect delay: ``initial``, ``2 * initial``, ... capped at ``maximum``.

    Args:
        initial: First delay, seconds.
        maximum: Largest delay, seconds.
    """

    def __init__(self, initial: float = 0.5, maximum: float = 5.0) -> None:
        self.initial = initial
        self.maximum = max(maximum, initial)
        self._next = initial

    def next(self) -> float:
        """Return the delay to wait now and grow the next one."""
        delay = self._next
        self._next = min(self._next * 2, self.maximum)
        return delay

    def reset(self) -> None:
        """Start over after a success."""
        self._next = self.initial


TransitionHook = Callable[["DependencyStatus", bool, str | None], None]
"""Callback ``(status, ok, error)`` fired when a dependency goes down (``ok=False``) or recovers."""


@dataclass(slots=True)
class DependencyStatus:
    """Availability of one external dependency (Redis, PostgreSQL) as seen by a service.

    The state starts unknown (``ok is None``). :meth:`mark_ok` / :meth:`mark_down` report every attempt;
    hooks fire only on transitions, so a flapping error is logged once per outage.

    Attributes:
        name: Dependency name (``redis``, ``postgres``).
        ok: ``True`` up, ``False`` down, ``None`` not checked yet (or disabled).
        enabled: ``False`` when the dependency is switched off in the settings.
        since: Time of the last transition (UTC).
        last_error: Text of the last error.
        failures: Failed attempts in total.
        outages: Number of up -> down transitions.
        hooks: Transition callbacks.
    """

    name: str
    ok: bool | None = None
    enabled: bool = True
    since: datetime | None = None
    last_error: str | None = None
    failures: int = 0
    outages: int = 0
    hooks: list[TransitionHook] = field(default_factory=list)
    _down_monotonic: float | None = None

    def mark_ok(self) -> bool:
        """Report a successful operation.

        Returns:
            ``True`` if this is a transition (first success or recovery).
        """
        if self.ok is True:
            return False
        was_down = self.ok is False
        self.ok = True
        self.since = datetime.now(UTC)
        downtime = None
        if was_down and self._down_monotonic is not None:
            downtime = time.monotonic() - self._down_monotonic
        self._down_monotonic = None
        if was_down:
            log.warning("%s recovered after %.1f s", self.name, downtime or 0.0)
        else:
            log.info("%s is available", self.name)
        self._fire(True, f"{downtime:.1f}" if downtime is not None else None)
        return True

    def mark_down(self, error: BaseException | str) -> bool:
        """Report a failed operation.

        Args:
            error: The exception or its description.

        Returns:
            ``True`` if this is a transition to ``down``.
        """
        self.failures += 1
        self.last_error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        if self.ok is False:
            return False
        self.ok = False
        self.outages += 1
        self.since = datetime.now(UTC)
        self._down_monotonic = time.monotonic()
        log.warning("%s is unavailable: %s", self.name, self.last_error)
        self._fire(False, self.last_error)
        return True

    def _fire(self, ok: bool, detail: str | None) -> None:
        for hook in self.hooks:
            try:
                hook(self, ok, detail)
            except Exception:
                log.exception("dependency hook failed")

    @property
    def state(self) -> str:
        """``up``, ``down``, ``unknown`` or ``disabled``."""
        if not self.enabled:
            return "disabled"
        return {True: "up", False: "down", None: "unknown"}[self.ok]


def dependencies_ok(statuses: Iterable[DependencyStatus]) -> bool:
    """Whether every enabled dependency is confirmed up (``unknown`` right after start is not yet ok)."""
    return all(s.ok is True or not s.enabled for s in statuses)


def setup_logging(level: str) -> None:
    """Configure root logging once per process.

    Args:
        level: Level name (``info``, ``debug``...).
    """
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run_http(app_factory: str, host: str, port: int, log_level: str) -> None:
    """Serve an application factory with uvicorn (blocking).

    Args:
        app_factory: Import path of the factory, e.g. ``backend.api:create_app``.
        host: Bind address.
        port: TCP port.
        log_level: Log level name.
    """
    import uvicorn

    setup_logging(log_level)
    uvicorn.run(
        app_factory,
        factory=True,
        host=host,
        port=port,
        log_level=log_level.lower(),
        proxy_headers=True,
    )
