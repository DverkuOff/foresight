"""Plan schedule of the predictor: per-vehicle stop arrays, routes and stop names.

The schedule is loaded once at start from ``<dataset>/<split>/schedule.csv`` (``validate`` —
``schedule_plan.csv``). Only the plan is kept: the fact columns (``time_fact_begin``, ``manual_fill``) are
dropped right after reading, so no fact can reach the features, the detector or the forecasts
(docs/architecture.md §2). Times are naive dataset times read as UTC, like the stream (the replayer puts them
on the wire as UTC), so ``t % 86400`` is the local time of day everywhere.

Stops of a vehicle are ordered as the detector (:func:`shared.stops.detect_passages`) and the features
(:class:`shared.features.FeatureContext`) order them: by ``time_begin``, then ``tt_action_item_id``; the
index of a stop is the same in all three.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data import parse_point, to_ns
from shared.features import FORBIDDEN_SCHEDULE_COLUMNS
from shared.routes import RouteMap, build_routes, stop_keys
from shared.stops import DEFAULT_PARAMS, DetectorParams, _stop_modes

log = logging.getLogger(__name__)

PLAN_COLUMNS = ["tr_id", "tt_action_item_id", "time_begin", "stop_lon", "stop_lat"]
"""Columns of :attr:`Schedule.frame` (the input of :class:`shared.features.FeatureContext`)."""


def schedule_path(dataset_dir: Path, split: str) -> Path:
    """Plan schedule file of a split (``validate`` has no fact: ``schedule_plan.csv``)."""
    name = "schedule_plan.csv" if split == "validate" else "schedule.csv"
    return Path(dataset_dir) / split / name


def load_plan(dataset_dir: Path, split: str) -> pd.DataFrame:
    """Read the plan schedule of a split without the fact columns.

    Args:
        dataset_dir: Dataset root with ``<split>/schedule*.csv``.
        split: ``test``, ``train`` or ``validate``.

    Returns:
        ``tr_id``, ``tt_action_item_id``, ``time_begin`` (datetime64[ns]), ``stop_lon``, ``stop_lat``,
        ``building_address`` — sorted by vehicle and plan.

    Raises:
        FileNotFoundError: No schedule file.
    """
    path = schedule_path(dataset_dir, split)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in FORBIDDEN_SCHEDULE_COLUMNS if c in df.columns])
    return plan_frame(df)


def plan_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw plan table (``geom`` or ``stop_lon`` / ``stop_lat``) to the plan columns."""
    df = df.drop(columns=[c for c in FORBIDDEN_SCHEDULE_COLUMNS if c in df.columns]).copy()
    df["tr_id"] = df["tr_id"].astype(np.int64)
    df["tt_action_item_id"] = df["tt_action_item_id"].astype(np.int64)
    df["time_begin"] = to_ns(df["time_begin"])
    if "stop_lon" not in df.columns:
        df["stop_lon"], df["stop_lat"] = parse_point(df["geom"])
    if "building_address" not in df.columns:
        df["building_address"] = ""
    df["building_address"] = df["building_address"].astype("string").fillna("").str.strip().astype(object)
    df = df[[*PLAN_COLUMNS, "building_address"]]
    df = df.sort_values(["tr_id", "time_begin", "tt_action_item_id"], kind="stable")
    return df.reset_index(drop=True)


@dataclass
class VehiclePlan:
    """Planned stops of one vehicle, in detector order.

    Attributes:
        tr_id: Vehicle.
        route_id: Route (``shared.routes``), ``None`` if unknown.
        stop_ids: ``tt_action_item_id`` of the stops.
        tb: Plan times, Unix seconds.
        slon: Stop longitudes.
        slat: Stop latitudes.
        modes: Detector stop modes on the whole plan (regular / terminal arrival / terminal departure).
        valid: Stops with coordinates (the detector sees only these).
        names: Stop names (address).
        keys: Stop places (``stop_key``).
        index: ``tt_action_item_id → position``.
    """

    tr_id: int
    route_id: str | None
    stop_ids: np.ndarray
    tb: np.ndarray
    slon: np.ndarray
    slat: np.ndarray
    modes: np.ndarray
    valid: np.ndarray
    names: tuple[str, ...]
    keys: tuple[str, ...]
    index: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tb)

    def window(self, lo: float, hi: float) -> np.ndarray:
        """Positions of the stops with plan in ``(lo, hi]``."""
        a = int(np.searchsorted(self.tb, lo, side="right"))
        b = int(np.searchsorted(self.tb, hi, side="right"))
        return np.arange(a, b)


class Schedule:
    """Plan schedule: the frame for the features, per-vehicle arrays for the detector, routes.

    Args:
        frame: Plan table (:func:`plan_frame`).
        params: Detector parameters (terminal detection of the stop modes).
        segments: Route geometry by GPS (:func:`shared.routes.load_segments`); ``None`` — the cache of the
            repository.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        params: DetectorParams = DEFAULT_PARAMS,
        segments: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> None:
        frame = plan_frame(frame)
        self.frame = frame[PLAN_COLUMNS]
        self.routes: RouteMap = build_routes(frame, segments)
        self.vehicles: dict[int, VehiclePlan] = {}
        tb_all = self.frame["time_begin"].to_numpy().astype(np.int64) / 1e9
        lon_all = self.frame["stop_lon"].to_numpy(dtype=np.float64)
        lat_all = self.frame["stop_lat"].to_numpy(dtype=np.float64)
        ids_all = self.frame["tt_action_item_id"].to_numpy(dtype=np.int64)
        keys_all = stop_keys(np.nan_to_num(lon_all), np.nan_to_num(lat_all))
        names_all = frame["building_address"].to_numpy(dtype=object)
        tr_all = self.frame["tr_id"].to_numpy()
        # a stop without an address takes the name the route map gives it («Остановка №n» of its route)
        route_names = {s.stop_key: s.name for r in self.routes.routes for s in r.stops}
        bounds = np.flatnonzero(np.diff(tr_all)) + 1
        for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(tr_all)], strict=True):
            if a == b:
                continue
            tr_id = int(tr_all[a])
            tb = tb_all[a:b]
            slon, slat = lon_all[a:b], lat_all[a:b]
            valid = np.isfinite(slon) & np.isfinite(slat)
            modes = np.zeros(b - a, dtype=np.int8)
            if valid.any():
                modes[valid] = _stop_modes(tb[valid], params.terminal_gap_s)
            stop_ids = ids_all[a:b]
            keys = tuple(keys_all[a:b])
            names = tuple(
                str(n) if n else self.routes.names.get(k) or route_names.get(k) or "Остановка без адреса"
                for n, k in zip(names_all[a:b], keys, strict=True)
            )
            self.vehicles[tr_id] = VehiclePlan(
                tr_id=tr_id,
                route_id=self.routes.by_tr.get(tr_id),
                stop_ids=stop_ids,
                tb=tb,
                slon=slon,
                slat=slat,
                modes=modes,
                valid=valid,
                names=names,
                keys=keys,
                index={int(s): i for i, s in enumerate(stop_ids)},
            )
        log.info(
            "schedule: %d vehicles, %d stops, %d routes",
            len(self.vehicles),
            len(self.frame),
            len(self.routes.routes),
        )

    @classmethod
    def load(
        cls, dataset_dir: Path, split: str, segments: Mapping[str, Sequence[Sequence[float]]] | None = None
    ) -> Schedule:
        """Load the plan of a split (see :func:`load_plan`) with the route geometry ``segments``."""
        return cls(load_plan(dataset_dir, split), segments=segments)

    def get(self, tr_id: int) -> VehiclePlan | None:
        """Plan of a vehicle (``None``: not in the schedule)."""
        return self.vehicles.get(int(tr_id))

    def frame_of(self, tr_ids: set[int]) -> pd.DataFrame:
        """Plan rows of the given vehicles (for :class:`shared.features.FeatureContext`)."""
        return self.frame[self.frame["tr_id"].isin(tr_ids)]
