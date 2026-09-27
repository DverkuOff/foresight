"""Stream clock: the time of the telemetry stream («часы потока», docs/architecture.md §5).

The stream time is the largest plausible fix time of the current *timeline*. Replayed data carries historical
times, the emulator stamps the current time, and a replayer restart sends every device back to the start of
its range. The clock tells four cases apart:

* **garbage** — a fix time before ``min_ts`` (a device without RTC or GPS fix: 1970, 2000, GPS week
  rollover) or more than ``max_future_s`` ahead of the wall clock. Ignored;
* **on the timeline** — within ``jump_s`` of the clock. Moves the clock forward;
* **off the timeline** — further than ``jump_s`` from the clock: a late historical (black-box) point, a device
  with a skewed clock or the first points after a source restart. Ignored by the clock (late points still
  belong to the tracks);
* **jump** — the clock moves to a new timeline when a *quorum* of the active devices (``quorum`` of those
  heard in the last ``active_s`` wall seconds) have each sent ``confirm`` consecutive off-timeline points that
  agree with each other (within ``jump_s``). A replayer restart moves every device at once; one device dumping
  its black box or running with a wrong clock never does. Every jump starts a new *epoch*.

A device only counts as moved *back* when it goes back on its own track: further than ``jump_s`` behind its
own latest fix on the timeline (or behind the clock it rejoins after a restore or a jump). A device that
reconnects after a break in its link (an ingest restart, a network outage) sends the backlog it queued,
from where it stopped; the devices that reconnected first have already moved the clock past it. Such a
backlog is off the timeline (late) but never a restart, however long the break and whatever the speed.
After :meth:`StreamClock.restore` the devices that were active when the clock was saved count as active
for ``active_s`` while they reconnect one by one, so the first few of them cannot move the clock alone.

Checked on the dataset in replay order (test and train days, x1 and x60): no jump with the defaults, and a
restart 30 minutes back is detected after 31 s of stream time.

The ingest owns the clock (:meth:`StreamClock.observe`) and stamps every stream event with its epoch and clock
value; the predictor follows it (:meth:`StreamClock.follow`), so the dashboard (through the api) and the
predictor always agree on "now" and reset together.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

log = logging.getLogger(__name__)

PLAN_DAY_START_H = 3
"""The plan day runs from 03:00: a realtime fix after midnight (before 03:00) is the end of the plan day."""


def to_plan_day(ts: datetime, day: date, tz_offset_s: float, near_s: float = 2 * 86_400.0) -> datetime:
    """A realtime fix time on the plan day of the schedule.

    The schedule is one day (the dataset's), in naive local time read as UTC; replayed telemetry carries
    times of that day. A realtime source (the NDTP emulator, a live feed) stamps the current UTC time: its
    local time of day (``ts + tz_offset_s``) goes onto the plan day, so vehicles meet their stops by the time
    of day (after midnight — the end of the plan day). A time within ``near_s`` of the plan day is kept.
    """
    anchor = datetime(day.year, day.month, day.day, tzinfo=UTC)
    if abs((ts - anchor).total_seconds() - 43_200.0) <= near_s:
        return ts
    local = ts.astimezone(UTC) + timedelta(seconds=tz_offset_s)
    tod = timedelta(
        hours=local.hour, minutes=local.minute, seconds=local.second, microseconds=local.microsecond
    )
    out = anchor + tod
    if local.hour < PLAN_DAY_START_H:
        out += timedelta(days=1)
    return out


MIN_PLAUSIBLE_TS = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
"""Fix times before 2020-01-01 are garbage (no RTC / no fix / GPS week rollover)."""

MAX_FUTURE_S = 86_400.0
"""Fix times this far beyond the wall clock are garbage (the replayer sends past dates only)."""


class Fix(StrEnum):
    """How a fix time relates to the stream clock."""

    GARBAGE = "garbage"
    """Implausible time: ignored."""
    ON = "on"
    """On the timeline (within ``jump_s`` of the clock)."""
    OFF = "off"
    """Off the timeline: ignored by the clock."""
    JUMP = "jump"
    """The clock moved to a new timeline (new epoch) with this point."""


@dataclass(frozen=True, slots=True)
class ClockJump:
    """A move of the clock to a new timeline.

    Attributes:
        before: Clock before the jump (Unix seconds), ``None`` if it was not set.
        after: Clock after the jump.
        epoch: The new epoch.
        devices: Devices that moved together (0 when following another clock).
        reason: ``quorum`` (detected here) or ``follow`` (the ingest's jump seen by the predictor).
    """

    before: float | None
    after: float
    epoch: int
    devices: int
    reason: str

    @property
    def back(self) -> bool:
        """Whether the clock went back (a source restart) rather than forward (a resume after a pause)."""
        return self.before is not None and self.after < self.before


JumpHook = Callable[[ClockJump], None]
"""Called after every jump of the clock."""


@dataclass(slots=True)
class _Device:
    seen: float  # wall time of the last fix
    off: int = 0  # consecutive off-timeline fixes
    off_ts: float = 0.0  # latest fix time among them
    last: float | None = None  # latest fix time of the device on the current timeline (None: not yet)


class StreamClock:
    """Stream time with garbage rejection and restart (jump) detection; see the module docstring.

    All methods are synchronous; call them from one event loop only.

    Args:
        jump_s: Distance from the clock beyond which a fix is off the timeline.
        confirm: Consecutive off-timeline fixes before a device counts as moved.
        quorum: Share of the active devices that must move together for a jump.
        active_s: A device is active if heard within this many wall seconds.
        min_ts: Earliest plausible fix time, Unix seconds.
        max_future_s: Latest plausible fix time relative to the wall clock.
        on_jump: Hook called after every jump.
    """

    def __init__(
        self,
        *,
        jump_s: float = 300.0,
        confirm: int = 3,
        quorum: float = 0.5,
        active_s: float = 60.0,
        min_ts: float = MIN_PLAUSIBLE_TS,
        max_future_s: float = MAX_FUTURE_S,
        on_jump: JumpHook | None = None,
    ) -> None:
        if jump_s <= 0 or confirm < 1 or not 0 < quorum <= 1 or active_s <= 0:
            raise ValueError("jump_s and active_s must be positive, confirm >= 1, 0 < quorum <= 1")
        self.jump_s = jump_s
        self.confirm = confirm
        self.quorum = quorum
        self.active_s = active_s
        self.min_ts = min_ts
        self.max_future_s = max_future_s
        self.on_jump = on_jump
        self.now: float | None = None
        self.epoch = 0
        self.resets = 0
        self.garbage = 0
        self.off_timeline = 0
        self.last_jump: ClockJump | None = None
        self._devices: dict[int, _Device] = {}
        self._floor: float | None = None  # the clock a device (re)joins at: seed, restore or the last jump
        self._expected: tuple[int, float] | None = None  # (devices, wall time) of the last restore

    @property
    def time(self) -> datetime | None:
        """The clock as an aware UTC datetime."""
        return datetime.fromtimestamp(self.now, UTC) if self.now is not None else None

    def plausible(self, ts: float, wall: float | None = None) -> bool:
        """Whether a fix time is plausible at wall-clock time ``wall`` (default: now); NaN is not."""
        limit = (time.time() if wall is None else wall) + self.max_future_s
        return self.min_ts <= ts <= limit

    def on_timeline(self, ts: float) -> bool:
        """Whether a fix time is within ``jump_s`` of the clock."""
        return self.now is not None and abs(ts - self.now) <= self.jump_s

    def active_devices(self, wall: float | None = None) -> int:
        """Devices the quorum is taken of at wall-clock time ``wall`` (default: now): those that sent a
        plausible fix within ``active_s``, and for ``active_s`` after a restore at least the devices expected
        back. The ingest saves it with the clock for the next :meth:`restore`."""
        wall = time.time() if wall is None else wall
        active = sum(1 for d in self._devices.values() if wall - d.seen <= self.active_s)
        return self._fleet(active, wall)

    # ---- authority (ingest) ----------------------------------------------------------------------

    def observe(self, ts: float, device: int, wall: float | None = None) -> Fix:
        """Feed a fix of one device and move the clock.

        Args:
            ts: Fix time, Unix seconds.
            device: Device id (``unitId``).
            wall: Wall-clock time of receipt (default: ``time.time()``).

        Returns:
            How the fix relates to the clock; :attr:`Fix.JUMP` if it completed a quorum.
        """
        wall = time.time() if wall is None else wall
        if not self.plausible(ts, wall):
            self.garbage += 1
            return Fix.GARBAGE
        dev = self._devices.get(device)
        if dev is None:
            dev = self._devices[device] = _Device(wall)
        dev.seen = wall
        if self.now is None:
            self.now = self._floor = ts
            self.epoch = self._next_epoch()
            dev.off = 0
            dev.last = ts
            return Fix.ON
        if abs(ts - self.now) <= self.jump_s:
            dev.off = 0
            dev.last = ts if dev.last is None else max(dev.last, ts)
            if ts > self.now:
                self.now = ts
            return Fix.ON
        self.off_timeline += 1
        if ts < self.now and ts >= self._track_ref(dev) - self.jump_s:
            # behind the clock, but the device goes on from where it stopped: the backlog it sends after a
            # break in its link, while the devices that reconnected first moved the clock on. Late only.
            dev.off = 0
            dev.last = ts if dev.last is None else max(dev.last, ts)
            return Fix.OFF
        if dev.off and abs(ts - dev.off_ts) <= self.jump_s:  # the run of off-timeline fixes goes on
            dev.off += 1
            dev.off_ts = max(dev.off_ts, ts)
        else:  # the first one, or one on yet another timeline
            dev.off = 1
            dev.off_ts = ts
        if dev.off >= self.confirm and self._quorum_jump(dev, wall):
            return Fix.JUMP
        return Fix.OFF

    def _track_ref(self, dev: _Device) -> float:
        """Where the device's own track stands: its latest fix on the timeline, else the clock it joined."""
        if dev.last is not None:
            return dev.last
        if self._floor is not None:
            return self._floor
        return self.now if self.now is not None else -math.inf

    def _fleet(self, active: int, wall: float) -> int:
        """Devices the quorum is taken of: the active ones, and for ``active_s`` after a restore at least the
        devices that were active when the clock was saved (they reconnect one by one)."""
        if self._expected is not None:
            devices, since = self._expected
            if wall - since <= self.active_s:
                return max(active, devices)
            self._expected = None
        return active

    def _quorum_jump(self, current: _Device, wall: float) -> bool:
        active = [d for d in self._devices.values() if wall - d.seen <= self.active_s]
        need = max(1, math.ceil(self.quorum * self._fleet(len(active), wall) - 1e-9))
        moved = [d.off_ts for d in active if d.off >= self.confirm]
        if len(moved) < need:
            return False
        anchor = statistics.median_low(moved)
        group = [t for t in moved if abs(t - anchor) <= self.jump_s]
        # the current fix must be on the new timeline, so a JUMP always means "this point is on it"
        if len(group) < need or abs(current.off_ts - anchor) > self.jump_s:
            return False
        self._jump(max(group), self._next_epoch(), devices=len(group), reason="quorum")
        self._expected = None
        for dev in self._devices.values():
            # the devices that moved stand on the new timeline; the others rejoin it at the new clock
            moved_here = dev.off >= self.confirm and abs(dev.off_ts - anchor) <= self.jump_s
            dev.last = dev.off_ts if moved_here else None
            dev.off = 0
        return True

    # ---- follower (predictor) --------------------------------------------------------------------

    def follow(self, ts: float, clock_ts: float, epoch: int, wall: float | None = None) -> Fix:
        """Follow the clock of the ingest carried by a stream event.

        A new epoch whose clock is more than ``jump_s`` away is a jump. A new epoch close to the current clock
        (the ingest restarted without its saved clock) is adopted silently. An older epoch never moves the
        clock: epochs only grow, so such an event is a redelivery (own pending entries after a restart or an
        outage, entries claimed from another consumer) and its point is classified against the current clock.

        Args:
            ts: Fix time of the event.
            clock_ts: The ingest's clock after this event.
            epoch: The ingest's epoch.
            wall: Wall-clock time for the plausibility check (the ingest's receive time).

        Returns:
            How the fix relates to the clock; :attr:`Fix.JUMP` if the epoch jumped (classify the point with
            :meth:`on_timeline` then).
        """
        if not self.plausible(ts, wall):
            self.garbage += 1
            return Fix.GARBAGE
        if epoch < self.epoch:
            pass  # a redelivered event of an older epoch: the clock never goes back to it
        elif epoch != self.epoch:
            if self.now is not None and self.epoch and abs(clock_ts - self.now) > self.jump_s:
                self._jump(clock_ts, epoch, devices=0, reason="follow")
                return Fix.JUMP
            self.epoch = epoch
            self.now = clock_ts if self.now is None else max(self.now, clock_ts)
        elif self.now is None or clock_ts > self.now:
            self.now = clock_ts
        if self.now is not None and abs(ts - self.now) <= self.jump_s:
            return Fix.ON
        self.off_timeline += 1
        return Fix.OFF

    # ---- state -----------------------------------------------------------------------------------

    def restore(self, now: float, epoch: int, *, devices: int = 0, wall: float | None = None) -> None:
        """Continue a saved clock (after a restart) without counting a jump.

        Args:
            now: The saved clock, Unix seconds.
            epoch: Its epoch.
            devices: Devices that were active when the clock was saved. For ``active_s`` wall seconds they
                count as active in the quorum, so the first devices to reconnect cannot move the clock alone.
            wall: Wall-clock time of the restore (default: ``time.time()``).
        """
        self.now = self._floor = now
        self.epoch = epoch
        since = time.time() if wall is None else wall
        self._expected = (devices, since) if devices > 0 else None
        for dev in self._devices.values():
            dev.off = 0
            dev.last = None

    def reset(self) -> None:
        """Forget the clock: the next fix starts a new epoch."""
        self.now = self._floor = None
        self._expected = None
        for dev in self._devices.values():
            dev.off = 0
            dev.last = None

    def _next_epoch(self) -> int:
        # wall-clock milliseconds: unique across restarts of the ingest, and increasing
        return max(self.epoch + 1, int(time.time() * 1000))

    def _jump(self, after: float, epoch: int, *, devices: int, reason: str) -> None:
        jump = ClockJump(self.now, after, epoch, devices, reason)
        self.now = self._floor = after
        self.epoch = epoch
        self.resets += 1
        self.last_jump = jump
        log.warning(
            "stream clock jumped %s: %s -> %s (epoch %d, %s, %d devices)",
            "back" if jump.back else "forward",
            _iso(jump.before),
            _iso(after),
            epoch,
            reason,
            devices,
        )
        if self.on_jump is not None:
            try:
                self.on_jump(jump)
            except Exception:
                log.exception("stream clock jump hook failed")


def _iso(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else "-"
