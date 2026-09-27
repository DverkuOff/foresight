"""Clocks and the replay schedule that maps the data clock onto the monotonic clock.

The schedule is one linear map ``data = data0 + (mono - mono0) * speed``, re-anchored on every speed change.
Due times are always computed from the anchor, never by adding up sleeps, so the pace does not drift however
late individual wake-ups are.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """Source of monotonic time and sleeping; tests substitute a virtual clock."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``."""
        ...


class SystemClock:
    """The real clock: :func:`time.monotonic` and :func:`asyncio.sleep`."""

    def monotonic(self) -> float:
        """Return :func:`time.monotonic`."""
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        """Sleep with :func:`asyncio.sleep`."""
        await asyncio.sleep(seconds)


class ReplaySchedule:
    """Linear mapping between data time and monotonic time.

    Args:
        data0: Data time (Unix seconds) at the anchor.
        mono0: Monotonic time at the anchor.
        speed: Data seconds per wall second.

    Raises:
        ValueError: If ``speed`` is not positive.
    """

    def __init__(self, data0: float, mono0: float, speed: float) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self.data0 = data0
        self.mono0 = mono0
        self.speed = speed

    def data_at(self, mono: float) -> float:
        """Data time at monotonic time ``mono``."""
        return self.data0 + (mono - self.mono0) * self.speed

    def mono_at(self, data: float) -> float:
        """Monotonic time when data time ``data`` is due."""
        return self.mono0 + (data - self.data0) / self.speed

    def set_speed(self, mono_now: float, speed: float) -> None:
        """Change the speed without a jump of the data clock.

        Args:
            mono_now: Current monotonic time (the new anchor).
            speed: New speed.

        Raises:
            ValueError: If ``speed`` is not positive.
        """
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self.data0 = self.data_at(mono_now)
        self.mono0 = mono_now
        self.speed = speed
