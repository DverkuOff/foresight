"""Loading a split's raw ``traffic.csv`` for replay.

Unlike :func:`shared.data.load_traffic` nothing is cleaned: invalid fixes, speed outliers and late
(``is_hist_data``) packets are replayed exactly as the organisers' server received them. Rows are ordered by
``receive_time`` (the moment a packet really arrived), while the NDTP timestamp carries ``event_time``: at any
moment ``T`` of the replay the receiver has seen exactly the packets that had really arrived by ``T``.

Naive dataset times are put on the wire as UTC, as everywhere else in the project (see
:func:`shared.ndtp.encode_nav`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data import SPLITS, dataset_dir, to_ns
from shared.ndtp import NavRecord

#: Columns of ``traffic.csv`` the replayer needs.
REPLAY_COLUMNS = (
    "tr_id",
    "unit_id",
    "event_time",
    "location_valid",
    "lon",
    "lat",
    "alt",
    "speed",
    "heading",
    "receive_time",
    "is_hist_data",
)

_TIME_OF_DAY = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_U16_MAX = 0xFFFF
_NS = 1_000_000_000


def replay_dataset_dir() -> Path:
    """Return the dataset root.

    ``FORESIGHT_DATASET_DIR`` wins; otherwise :func:`shared.data.dataset_dir` is used (``MT_DATASET_DIR`` or
    ``<repo>/dataset``).

    Returns:
        Directory that contains ``<split>/traffic.csv``.
    """
    env = os.environ.get("FORESIGHT_DATASET_DIR")
    return Path(env) if env else dataset_dir()


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"]).to_numpy()


def _as_float(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _as_u16(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.nan_to_num(values, nan=0.0)), 0, _U16_MAX).astype(np.int64)


def _is_time_of_day(text: str) -> bool:
    return bool(_TIME_OF_DAY.match(text.strip()))


def parse_data_time(value: str, day: date) -> float:
    """Parse a replay boundary into Unix seconds of the data clock.

    Args:
        value: ``HH:MM`` or ``HH:MM:SS`` on ``day`` (hours up to 47 reach into the next day) or an ISO
            datetime; naive values are UTC like the rest of the data.
        day: Day of the data.

    Returns:
        Unix seconds.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    text = value.strip()
    match = _TIME_OF_DAY.match(text)
    if match:
        hours, minutes, seconds = int(match[1]), int(match[2]), int(match[3] or 0)
        if hours > 47 or minutes > 59 or seconds > 59:
            raise ValueError(f"bad time of day {value!r}")
        start_of_day = datetime.combine(day, time(0), tzinfo=UTC)
        return (start_of_day + timedelta(hours=hours, minutes=minutes, seconds=seconds)).timestamp()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"bad time {value!r}: expected HH:MM, HH:MM:SS or an ISO datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


@dataclass(frozen=True, eq=False)
class ReplayData:
    """Telemetry of one split as column arrays sorted by ``receive_time``.

    Attributes:
        split: Split name (or ``"custom"``).
        recv: ``receive_time`` as Unix seconds (float), ascending; drives the replay clock.
        event_s: ``event_time`` as whole Unix seconds; goes into the Nav00 timestamp.
        unit_id: NDTP ``unitId``.
        tr_id: Vehicle id (one-to-one with ``unit_id``).
        valid: Whether the fix is valid (``location_valid`` and coordinates present).
        lon: Longitude in degrees, 0 when missing.
        lat: Latitude in degrees, 0 when missing.
        speed: Speed, km/h, rounded and clipped to u16.
        course: Heading, degrees 0..359.
        altitude: Altitude, m, clipped to u16.
        hist: ``is_hist_data`` flag (for statistics; the protocol has no field for it).
    """

    split: str
    recv: np.ndarray
    event_s: np.ndarray
    unit_id: np.ndarray
    tr_id: np.ndarray
    valid: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    speed: np.ndarray
    course: np.ndarray
    altitude: np.ndarray
    hist: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, split: str = "custom") -> ReplayData:
        """Build replay arrays from a raw ``traffic.csv`` frame.

        Rows without ``event_time`` or ``unit_id`` cannot be encoded and are skipped; a missing
        ``receive_time`` falls back to ``event_time``.

        Args:
            frame: Raw telemetry with :data:`REPLAY_COLUMNS` (``tr_id`` and ``is_hist_data`` are optional).
            split: Split name for reporting.

        Returns:
            Arrays sorted by ``receive_time`` (then ``event_time``, then file order).

        Raises:
            ValueError: If no row can be replayed.
        """
        event = to_ns(frame["event_time"])
        unit = pd.to_numeric(frame["unit_id"], errors="coerce")
        keep = (event.notna() & unit.notna()).to_numpy()
        if not keep.any():
            raise ValueError("no replayable telemetry rows")
        frame = frame.loc[keep]
        event = event[keep]
        event_ns = event.to_numpy().astype(np.int64)
        recv_ts = to_ns(frame["receive_time"]) if "receive_time" in frame else event
        recv_ns = np.where(recv_ts.isna().to_numpy(), event_ns, recv_ts.to_numpy().astype(np.int64))

        lon = _as_float(frame["lon"])
        lat = _as_float(frame["lat"])
        valid = _as_bool(frame["location_valid"]) & np.isfinite(lon) & np.isfinite(lat)
        n = len(frame)
        tr = (
            pd.to_numeric(frame["tr_id"], errors="coerce").fillna(-1).to_numpy().astype(np.int64)
            if "tr_id" in frame
            else np.full(n, -1, dtype=np.int64)
        )
        hist = _as_bool(frame["is_hist_data"]) if "is_hist_data" in frame else np.zeros(n, dtype=bool)
        course = (np.rint(np.nan_to_num(_as_float(frame["heading"]), nan=0.0)) % 360).astype(np.int64)
        order = np.lexsort((np.arange(n), event_ns, recv_ns))
        return cls(
            split=split,
            recv=(recv_ns / _NS)[order],
            event_s=(event_ns // _NS)[order],
            unit_id=unit[keep].to_numpy().astype(np.int64)[order],
            tr_id=tr[order],
            valid=valid[order],
            lon=np.nan_to_num(lon, nan=0.0)[order],
            lat=np.nan_to_num(lat, nan=0.0)[order],
            speed=_as_u16(_as_float(frame["speed"]))[order],
            course=course[order],
            altitude=_as_u16(_as_float(frame["alt"]))[order],
            hist=hist[order],
        )

    def __len__(self) -> int:
        return len(self.recv)

    def take(self, index: np.ndarray) -> ReplayData:
        """Return the rows at ``index`` (kept in the given order).

        Args:
            index: Integer positions.

        Returns:
            A new :class:`ReplayData`.
        """
        arrays = {f.name: getattr(self, f.name)[index] for f in fields(self) if f.name != "split"}
        return ReplayData(split=self.split, **arrays)

    @property
    def units(self) -> np.ndarray:
        """Sorted unique ``unit_id`` values."""
        return np.unique(self.unit_id)

    def unit_map(self) -> dict[int, int]:
        """Return ``unit_id -> tr_id``."""
        pairs = np.unique(np.stack([self.unit_id, self.tr_id], axis=1), axis=0)
        return {int(u): int(t) for u, t in pairs}

    def day(self) -> date:
        """Day of the data: the date of the median ``event_time`` (robust to stray rows around midnight)."""
        return datetime.fromtimestamp(int(np.median(self.event_s)), UTC).date()

    def nav(self, i: int) -> NavRecord:
        """Build the Nav00 record of row ``i``.

        Args:
            i: Row position.

        Returns:
            Navigation record with ``timestamp = event_time``; satellites and PDOP are unknown (0).
        """
        speed = int(self.speed[i])
        return NavRecord(
            timestamp=datetime.fromtimestamp(int(self.event_s[i]), UTC),
            lon=float(self.lon[i]),
            lat=float(self.lat[i]),
            valid=bool(self.valid[i]),
            speed_avg=speed,
            speed_max=speed,
            course=int(self.course[i]),
            altitude=int(self.altitude[i]),
        )

    def select(
        self, units: Iterable[int] | None = None, start: str | None = None, until: str | None = None
    ) -> ReplayData:
        """Filter by devices and by the ``receive_time`` window ``[start, until)``.

        Args:
            units: ``unit_id`` or ``tr_id`` values (both are accepted); ``None`` or empty keeps all.
            start: Start of the window, see :func:`parse_data_time`.
            until: End of the window (exclusive). A time of day not after ``start`` means the next day.

        Returns:
            Filtered data.

        Raises:
            ValueError: On unknown devices, a bad time or an empty selection.
        """
        mask = np.ones(len(self), dtype=bool)
        wanted = sorted({int(u) for u in units or ()})
        if wanted:
            ids = np.asarray(wanted, dtype=np.int64)
            known = set(self.unit_id.tolist()) | set(self.tr_id.tolist())
            unknown = [u for u in wanted if u not in known]
            if unknown:
                raise ValueError(f"unknown units (neither unit_id nor tr_id): {unknown}")
            mask &= np.isin(self.unit_id, ids) | np.isin(self.tr_id, ids)
        day = self.day()
        t0 = parse_data_time(start, day) if start else None
        t1 = parse_data_time(until, day) if until else None
        if t0 is not None and t1 is not None and t1 <= t0 and until and _is_time_of_day(until):
            t1 += 86_400
        if t0 is not None:
            mask &= self.recv >= t0
        if t1 is not None:
            mask &= self.recv < t1
        if not mask.any():
            raise ValueError(
                f"no packets in split {self.split!r} for units={wanted or 'all'} [{start}, {until})"
            )
        return self.take(np.flatnonzero(mask))

    def trim_tail(self, tail_s: float) -> ReplayData:
        """Leave out stragglers received more than ``tail_s`` after the last ``event_time``.

        The first packet always stays, so the result is never empty.

        Args:
            tail_s: Allowed delivery tail, data seconds.

        Returns:
            ``self`` if nothing is cut, otherwise the kept rows.
        """
        cutoff = max(float(self.event_s.max()) + tail_s, float(self.recv[0]))
        keep = self.recv <= cutoff
        return self if keep.all() else self.take(np.flatnonzero(keep))


def load_replay_data(split: str, root: Path | None = None) -> ReplayData:
    """Read ``<root>/<split>/traffic.csv`` without cleaning.

    Args:
        split: ``train``, ``test`` or ``validate``.
        root: Dataset root; defaults to :func:`replay_dataset_dir`.

    Returns:
        Replay arrays sorted by ``receive_time``.

    Raises:
        ValueError: On an unknown split.
        FileNotFoundError: If the file does not exist.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    path = (root or replay_dataset_dir()) / split / "traffic.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (set FORESIGHT_DATASET_DIR or MT_DATASET_DIR)")
    frame = pd.read_csv(path, usecols=lambda c: c in REPLAY_COLUMNS)
    return ReplayData.from_frame(frame, split)
