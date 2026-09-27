"""In-memory store of the latest state of every vehicle (keyed by NDTP ``unitId``).

All methods are synchronous and never await, so when they are called from the event loop (the NDTP server,
REST handlers, the WebSocket loop) each call is atomic with respect to other coroutines and no lock is needed.
Do not call them from other threads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from backend.clock import StreamClock
from shared.ndtp import IrmaRecord, NavRecord


class LinkStatus(StrEnum):
    """Connection status of a vehicle as shown to dispatchers."""

    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


@dataclass(slots=True)
class VehicleState:
    """Latest known state of one vehicle.

    Attributes:
        unit_id: NDTP device id.
        nav: Last navigation record (``None`` until the first Nav00 arrives).
        valid_nav: Last navigation record with a valid fix and non-zero coordinates: the position shown on the
            map while the current fix is invalid (``None`` until the first valid fix).
        doors: Last Irma04 record, if the device reports doors.
        first_seen: Server time of the first frame from this device.
        last_packet_at: Server time of the last frame (handshake or realtime).
        last_nav_at: Server time of the last Nav00.
        packets: Number of realtime packets received.
        connections: Number of currently open TCP connections of this device.
        reconnects: Number of handshakes after the first one.
        handshakes: Number of handshakes received.
        version: Store version of the last change (for WebSocket deltas).
    """

    unit_id: int
    nav: NavRecord | None = None
    valid_nav: NavRecord | None = None
    doors: IrmaRecord | None = None
    first_seen: datetime | None = None
    last_packet_at: datetime | None = None
    last_nav_at: datetime | None = None
    packets: int = 0
    connections: int = 0
    reconnects: int = 0
    handshakes: int = 0
    version: int = 0


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StateStore:
    """Latest-state store with change versioning and change notification.

    Args:
        stale_after_s: Silence after which a connected vehicle is ``stale``.
        offline_after_s: Silence after which a vehicle is ``offline``.
        clock: Returns the current timezone-aware UTC time (injectable for tests).
        stream_clock: The stream clock fed with every fix (the ingest is its authority, see
            :mod:`backend.clock`); a default one if omitted.
    """

    def __init__(
        self,
        stale_after_s: float = 30.0,
        offline_after_s: float = 120.0,
        clock: Callable[[], datetime] = _utcnow,
        stream_clock: StreamClock | None = None,
    ) -> None:
        self.stale_after_s = stale_after_s
        self.offline_after_s = offline_after_s
        self.clock = clock
        self.stream_clock = stream_clock or StreamClock()
        self._vehicles: dict[int, VehicleState] = {}
        self._version = 0
        self._changed = asyncio.Event()

    @property
    def stream_time(self) -> datetime | None:
        """Stream clock: the largest plausible fix time of the current timeline (UTC)."""
        return self.stream_clock.time

    # ---- mutation --------------------------------------------------------------------------------

    def _touch(self, unit_id: int, now: datetime) -> VehicleState:
        vehicle = self._vehicles.get(unit_id)
        if vehicle is None:
            vehicle = VehicleState(unit_id=unit_id, first_seen=now)
            self._vehicles[unit_id] = vehicle
        vehicle.last_packet_at = now
        return vehicle

    def _bump(self, vehicle: VehicleState) -> None:
        self._version += 1
        vehicle.version = self._version
        # wake every waiter; waiters hold the old (now set) event, new waiters get a fresh one
        self._changed.set()
        self._changed = asyncio.Event()

    def on_connect(self, unit_id: int) -> None:
        """Register a handshake / new connection of a device.

        Args:
            unit_id: Device id.
        """
        vehicle = self._touch(unit_id, self.clock())
        vehicle.connections += 1
        vehicle.handshakes += 1
        if vehicle.handshakes > 1:
            vehicle.reconnects += 1
        self._bump(vehicle)

    def on_disconnect(self, unit_id: int) -> None:
        """Register a closed connection of a device.

        Args:
            unit_id: Device id.
        """
        vehicle = self._vehicles.get(unit_id)
        if vehicle is None:
            return
        vehicle.connections = max(0, vehicle.connections - 1)
        self._bump(vehicle)

    def on_nav(self, unit_id: int, nav: NavRecord, doors: IrmaRecord | None = None) -> VehicleState:
        """Store a navigation record (and optional door state) from a realtime packet.

        Args:
            unit_id: Device id.
            nav: Decoded Nav00 record.
            doors: Decoded Irma04 record, if present in the same packet.

        Returns:
            The updated vehicle state.
        """
        now = self.clock()
        vehicle = self._touch(unit_id, now)
        vehicle.nav = nav
        if nav.valid and not (nav.lat == 0 and nav.lon == 0):
            vehicle.valid_nav = nav
        vehicle.last_nav_at = now
        vehicle.packets += 1
        if doors is not None:
            vehicle.doors = doors
        # garbage times are ignored, a source restart (most devices jump together) moves the clock back
        self.stream_clock.observe(nav.timestamp.timestamp(), unit_id, now.timestamp())
        self._bump(vehicle)
        return vehicle

    def reset_stream_time(self) -> None:
        """Forget the stream clock: the next fix starts a new epoch (e.g. an explicit replayer restart)."""
        self.stream_clock.reset()

    def on_packet_without_nav(self, unit_id: int) -> None:
        """Register a realtime packet that carried no Nav00 (keeps the link alive).

        Args:
            unit_id: Device id.
        """
        vehicle = self._touch(unit_id, self.clock())
        vehicle.packets += 1
        self._bump(vehicle)

    # ---- queries ---------------------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Monotonic version of the last change."""
        return self._version

    def __len__(self) -> int:
        return len(self._vehicles)

    def get(self, unit_id: int) -> VehicleState | None:
        """Return the state of one vehicle or ``None``."""
        return self._vehicles.get(unit_id)

    def all(self) -> list[VehicleState]:
        """Return all vehicles sorted by ``unit_id``."""
        return [self._vehicles[k] for k in sorted(self._vehicles)]

    def changed_since(self, version: int) -> list[VehicleState]:
        """Return vehicles changed after ``version``, sorted by ``unit_id``."""
        return [v for v in self.all() if v.version > version]

    def age_s(self, vehicle: VehicleState, now: datetime | None = None) -> float | None:
        """Seconds since the last frame from the vehicle."""
        if vehicle.last_packet_at is None:
            return None
        return ((now or self.clock()) - vehicle.last_packet_at).total_seconds()

    def status(self, vehicle: VehicleState, now: datetime | None = None) -> LinkStatus:
        """Link status: ``offline`` without an open connection or after ``offline_after_s`` of silence,
        ``stale`` after ``stale_after_s``, otherwise ``online``."""
        age = self.age_s(vehicle, now)
        if vehicle.connections == 0 or age is None or age >= self.offline_after_s:
            return LinkStatus.OFFLINE
        if age >= self.stale_after_s:
            return LinkStatus.STALE
        return LinkStatus.ONLINE

    def status_counts(self, now: datetime | None = None) -> dict[LinkStatus, int]:
        """Number of vehicles per link status."""
        now = now or self.clock()
        counts = dict.fromkeys(LinkStatus, 0)
        for vehicle in self._vehicles.values():
            counts[self.status(vehicle, now)] += 1
        return counts

    async def wait_for_change(self, timeout: float) -> bool:
        """Wait until any vehicle changes or ``timeout`` expires.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            ``True`` if a change happened, ``False`` on timeout.
        """
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return False
        return True
