"""Stop detector on the stream: the offline detector (:mod:`shared.stops`) run incrementally, per vehicle.

The offline detector is causal: run on the points with ``time ≤ T`` it makes the same decisions as on the
whole day for every stop with ``confirmed_at ≤ T``. On the stream it cannot re-run over the whole day on
every tick (tens of ms per vehicle and growing), so :class:`VehicleDetector` keeps its decisions and
resumes where the day-long run would be:

* a decision (a stop matched with its pass time, or skipped) is **frozen** the moment it is made — a
  confirmed passage is a fact at ``confirmed_at`` and is never revised;
* every tick the matcher runs only on the undecided stops (from the first undecided one to an hour ahead)
  and on the track from the pass of the last matched stop (the detector's cursor) to the tick; the stop
  modes (regular / terminal arrival / terminal departure) come from the whole plan, as offline;
* the decision time never goes back: ``confirmed_at`` is at least the previous decision's.

Only points with ``ts ≤ t`` are given to the matcher, so a passage returned at tick ``t`` has
``confirmed_at ≤ t`` (it is decided at a real GPS point). How close this is to the day-long run is measured on
the test day by ``scripts/validate_online.py`` (docs/online-validation.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.schedule import VehiclePlan
from shared.stops import DEFAULT_PARAMS, MODE_MIN, DetectorParams, _Matcher

LOOKAHEAD_S = 3600.0
"""Undecided stops up to this far after the tick take part in a run (lookahead of the matcher)."""
MIN_EXTRA_STOPS = 6
"""... and at least this many more, so a terminal is never the artificial last stop of a run."""


@dataclass(frozen=True, slots=True)
class Passage:
    """A detector decision about one planned stop.

    Attributes:
        index: Position of the stop in the vehicle plan.
        stop_id: ``tt_action_item_id``.
        time_begin: Plan time, Unix seconds.
        pass_s: Pass time, Unix seconds (NaN: the stop was skipped / not matched).
        dist_m: Closest distance to the stop, m (NaN if skipped).
        confirmed_at: When the decision became known, Unix seconds (≤ the tick).
        covered: The track covered the stop's window (an unmatched stop is a real skip, not a gap in data).
    """

    index: int
    stop_id: int
    time_begin: float
    pass_s: float
    dist_m: float
    confirmed_at: float
    covered: bool = True

    @property
    def matched(self) -> bool:
        """Whether the vehicle passed the stop."""
        return not math.isnan(self.pass_s)

    @property
    def delay_s(self) -> float:
        """Pass time − plan (NaN if skipped)."""
        return self.pass_s - self.time_begin


class VehicleDetector:
    """Incremental stop detector of one vehicle (see the module docstring).

    Args:
        plan: Planned stops of the vehicle.
        params: Detector parameters (the offline defaults).
    """

    def __init__(self, plan: VehiclePlan, params: DetectorParams = DEFAULT_PARAMS) -> None:
        self.plan = plan
        self.params = params
        n = len(plan)
        self.pass_s = np.full(n, np.nan)
        self.dist_m = np.full(n, np.nan)
        self.conf_s = np.full(n, np.nan)
        self.decided = np.zeros(n, dtype=bool)
        self.first_open = 0
        self.anchor: int | None = None
        self.last_conf = -math.inf
        self.first_ts: float | None = None
        self.matched: list[int] = []  # matched stops in decision order (non-decreasing confirmed_at)

    def _advance(self) -> None:
        plan, n = self.plan, len(self.plan)
        while self.first_open < n and (self.decided[self.first_open] or not plan.valid[self.first_open]):
            self.first_open += 1

    def update(self, t: float, ts: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> list[Passage]:
        """Run the matcher on the points known at tick ``t``; return the new decisions.

        Args:
            t: Tick time, Unix seconds.
            ts: Cleaned fix times (ascending, unique), Unix seconds; only ``ts ≤ t`` are used.
            lon: Longitudes.
            lat: Latitudes.

        Returns:
            New passages in decision order (all with ``confirmed_at ≤ t``).
        """
        plan, p = self.plan, self.params
        k = int(np.searchsorted(ts, t, side="right"))
        ts, lon, lat = ts[:k], lon[:k], lat[:k]
        if len(ts) and (self.first_ts is None or ts[0] < self.first_ts):
            self.first_ts = float(ts[0])
        self._advance()
        n = len(plan)
        if self.first_open >= n or len(ts) < 2:
            return []
        end = int(np.searchsorted(plan.tb, t + LOOKAHEAD_S, side="right"))
        end = min(max(end, self.first_open + 1) + MIN_EXTRA_STOPS, n)
        cand = np.arange(self.first_open, end)
        cand = cand[plan.valid[cand] & ~self.decided[cand]]
        if len(cand) == 0:
            return []
        start = 0
        if self.anchor is not None:
            start = max(int(np.searchsorted(ts, self.pass_s[self.anchor], side="left")) - 1, 0)
        tt = ts[start:]
        if len(tt) < 2:
            return []
        t0 = float(tt[0])
        matcher = _Matcher(
            tt - t0, lon[start:], lat[start:], plan.tb[cand] - t0, plan.slon[cand], plan.slat[cand], p
        )
        matcher.modes = plan.modes[cand]  # modes of the whole plan, not of this slice
        pass_t, dist, conf = matcher.run()
        decided = np.flatnonzero(np.isfinite(conf))
        if len(decided) == 0:
            return []
        out: list[Passage] = []
        for j in decided[np.argsort(conf[decided], kind="stable")]:
            i = int(cand[j])
            confirmed = max(float(conf[j]) + t0, self.last_conf)
            self.last_conf = confirmed
            passed = float(pass_t[j]) + t0 if np.isfinite(pass_t[j]) else math.nan
            self.decided[i] = True
            self.pass_s[i] = passed
            self.dist_m[i] = dist[j]
            self.conf_s[i] = confirmed
            if not math.isnan(passed):
                self.matched.append(i)
                if self.anchor is None or passed >= self.pass_s[self.anchor]:
                    self.anchor = i
            covered = self.first_ts is not None and plan.tb[i] - p.window_before_s >= self.first_ts
            out.append(
                Passage(
                    index=i,
                    stop_id=int(plan.stop_ids[i]),
                    time_begin=float(plan.tb[i]),
                    pass_s=passed,
                    dist_m=float(dist[j]),
                    confirmed_at=confirmed,
                    covered=covered or not math.isnan(passed),
                )
            )
        self._advance()
        return out

    def last_deviation(self, t: float, lag_s: float = 0.0, *, regular_only: bool = False) -> float:
        """Delay at the last matched stop confirmed by ``t − lag_s`` (NaN if none): online ``cur_dev_s``.

        Args:
            t: Tick time.
            lag_s: Only decisions confirmed this long before ``t`` count.
            regular_only: Skip terminals (arrival / departure after a layover): their delay is absorbed by
                the layover and says little about the next trip.
        """
        cutoff = t - lag_s
        for i in reversed(self.matched):
            if self.conf_s[i] <= cutoff and (not regular_only or self.plan.modes[i] == MODE_MIN):
                return float(self.pass_s[i] - self.plan.tb[i])
        return math.nan

    def median_deviation(self, t: float, n: int = 3) -> float:
        """Median delay at the last ``n`` matched stops confirmed by ``t`` (NaN if none).

        The online ``cur_dev_s`` by default: the dataset's hint is the fact at the last stop *planned* by
        ``T``, which is after ``T`` when the vehicle is late — the stream never has it; of the causal
        analogues the median of the last three passes serves the model best (docs/online-validation.md).
        """
        dev = [float(self.pass_s[i] - self.plan.tb[i]) for i in self.matched if self.conf_s[i] <= t][-n:]
        return float(np.median(dev)) if dev else math.nan

    def decided_indices(self) -> np.ndarray:
        """Positions of the decided stops."""
        return np.flatnonzero(self.decided)
