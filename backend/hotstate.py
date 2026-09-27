"""API-side view of the hot vehicle state: an in-memory cache kept in sync with Redis.

:class:`HotStateSync` subscribes to the ``foresight:vehicles`` channel, applies deltas as they come and
re-reads the full state (ZSET index + hashes) on (re)connect and periodically. While Redis is down the
:class:`VehicleCache` keeps serving the last known state, and the API marks responses as degraded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from backend.bus import (
    CHANNEL_ALERTS,
    CHANNEL_INCIDENTS,
    CHANNEL_PREDICTIONS,
    CHANNEL_VEHICLES,
    FORECAST_KEY,
    FORECAST_STATUS_KEY,
    INGEST_STATS_KEY,
    REDIS_ERRORS,
    ROUTES_KEY,
    STREAM_EPOCH_KEY,
    STREAM_TIME_KEY,
    VEHICLE_INDEX,
    VehicleRecord,
    vehicle_key,
)
from backend.runtime import Backoff, DependencyStatus
from backend.state import LinkStatus

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _json(value: str | bytes | None) -> Any:
    """Parsed JSON or ``None`` (missing or malformed)."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


class VehicleCache:
    """Last known state of all vehicles with change versioning (same query API as the ingest store).

    Args:
        stale_after_s: Silence after which a connected vehicle is ``stale``.
        offline_after_s: Silence after which a vehicle is ``offline``.
        clock: Current UTC time (injectable for tests).
    """

    def __init__(
        self,
        stale_after_s: float = 30.0,
        offline_after_s: float = 120.0,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.stale_after_s = stale_after_s
        self.offline_after_s = offline_after_s
        self.clock = clock
        self.stream_time: datetime | None = None
        self.stream_epoch = 0
        self.synced_at: datetime | None = None
        self.forecasts: dict[int, dict[str, Any]] = {}
        """Hot forecast state of the predictor by ``tr_id`` (``foresight:forecast``)."""
        self.forecast_status: dict[str, Any] = {}
        self.scheduled: frozenset[int] | None = None
        """``tr_id`` of the predictor's plan schedule (``None`` — not published yet)."""
        self.routes: list[dict[str, Any]] | None = None
        self._records: dict[int, VehicleRecord] = {}
        self._version = 0
        self._changed = asyncio.Event()

    def _bump(self) -> int:
        self._version += 1
        self._changed.set()
        self._changed = asyncio.Event()
        return self._version

    def apply(self, records: Iterable[VehicleRecord], *, full: bool = False) -> int:
        """Merge vehicle records into the cache.

        A record older than the cached one (by ``updated_ms``) is ignored, so a delta that raced with a
        snapshot cannot roll the state back.

        Args:
            records: Parsed hot-state records.
            full: The records are a complete snapshot: vehicles not in it are removed.

        Returns:
            Number of vehicles that changed.
        """
        changed = 0
        seen: set[int] = set()
        for record in records:
            seen.add(record.unit_id)
            old = self._records.get(record.unit_id)
            if old is not None and (record.updated_ms < old.updated_ms or record.same_state(old)):
                continue
            self._records[record.unit_id] = record
            record.version = self._bump()
            changed += 1
        if full:
            for unit_id in set(self._records) - seen:
                del self._records[unit_id]
                self._bump()
                changed += 1
        self.synced_at = self.clock()
        return changed

    def set_forecasts(self, forecasts: dict[int, dict[str, Any]], status: dict[str, Any] | None) -> int:
        """Take the predictor's forecast snapshot; vehicles whose forecast changed get a new version.

        Returns:
            Number of vehicles that changed.
        """
        changed_tr = {
            tr for tr in set(forecasts) | set(self.forecasts) if forecasts.get(tr) != self.forecasts.get(tr)
        }
        self.forecasts = forecasts
        if status is not None:
            self.forecast_status = status
            scheduled = status.get("scheduled_tr_ids")
            if isinstance(scheduled, list):
                self.scheduled = frozenset(int(x) for x in scheduled)
        changed = 0
        for record in self._records.values():
            if record.tr_id is not None and record.tr_id in changed_tr:
                record.version = self._bump()
                changed += 1
        return changed

    def set_routes(self, payload: dict[str, Any] | None) -> None:
        """Take the predictor's route network (``foresight:routes``)."""
        if not payload:
            return
        routes = payload.get("routes")
        if isinstance(routes, list):
            self.routes = routes
        scheduled = payload.get("scheduled_tr_ids")
        if isinstance(scheduled, list) and self.scheduled is None:
            self.scheduled = frozenset(int(x) for x in scheduled)

    def forecast_of(self, tr_id: int | None) -> dict[str, Any] | None:
        """Forecast state of a vehicle, ``None`` without one or when the predictor's snapshot is stale.

        The snapshot is stale when its stream time lags the ingest's clock by more than
        :attr:`forecast_stale_s` (the predictor stopped or is far behind): risk then is unknown, not old.
        """
        if tr_id is None:
            return None
        fc = self.forecasts.get(tr_id)
        if fc is None:
            return None
        at = _parse_iso(self.forecast_status.get("stream_time")) if self.forecast_status else None
        if (
            at is not None
            and self.stream_time is not None
            and (self.stream_time - at).total_seconds() > self.forecast_stale_s
        ):
            return None
        return fc

    forecast_stale_s: float = 300.0
    """Stream seconds the forecast snapshot may lag the stream clock before it is ignored."""

    @property
    def version(self) -> int:
        """Monotonic version of the last change."""
        return self._version

    def __len__(self) -> int:
        return len(self._records)

    def get(self, unit_id: int) -> VehicleRecord | None:
        """Return one vehicle or ``None``."""
        return self._records.get(unit_id)

    def all(self) -> list[VehicleRecord]:
        """All vehicles sorted by ``unit_id``."""
        return [self._records[k] for k in sorted(self._records)]

    def changed_since(self, version: int) -> list[VehicleRecord]:
        """Vehicles changed after ``version``, sorted by ``unit_id``."""
        return [r for r in self.all() if r.version > version]

    def set_stream_time(self, stream_time: datetime | None, epoch: int) -> None:
        """Take the ingest's stream clock (from a delta or a full read).

        Within an epoch the clock only moves forward (a message read late must not roll it back); a newer
        epoch (the clock jumped, e.g. the replayer restarted) replaces it even if it is earlier. Epochs grow
        (see :class:`~backend.clock.StreamClock`), so a late message of an older epoch is ignored.

        Args:
            stream_time: The clock, ``None`` if unknown.
            epoch: Its epoch (0 if unknown).
        """
        if stream_time is None or (epoch and epoch < self.stream_epoch):
            return
        if self.stream_time is None or epoch > self.stream_epoch or stream_time > self.stream_time:
            self.stream_time = stream_time
            self.stream_epoch = max(epoch, self.stream_epoch)

    def age_s(self, record: VehicleRecord, now: datetime | None = None) -> float | None:
        """Seconds since the last frame from the vehicle."""
        if record.last_packet_at is None:
            return None
        return ((now or self.clock()) - record.last_packet_at).total_seconds()

    def status(self, record: VehicleRecord, now: datetime | None = None) -> LinkStatus:
        """Link status with the same rules as :meth:`backend.state.StateStore.status`."""
        age = self.age_s(record, now)
        if not record.connected or age is None or age >= self.offline_after_s:
            return LinkStatus.OFFLINE
        if age >= self.stale_after_s:
            return LinkStatus.STALE
        return LinkStatus.ONLINE

    def status_counts(self, now: datetime | None = None) -> dict[LinkStatus, int]:
        """Number of vehicles per link status."""
        now = now or self.clock()
        counts = dict.fromkeys(LinkStatus, 0)
        for record in self._records.values():
            counts[self.status(record, now)] += 1
        return counts

    def notify(self) -> None:
        """Wake the waiters of :meth:`wait_for_change` without a state change (a predictor event came)."""
        self._changed.set()
        self._changed = asyncio.Event()

    async def wait_for_change(self, timeout: float) -> bool:
        """Wait until any vehicle changes or ``timeout`` expires; ``True`` on change."""
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return False
        return True


def _records(hashes: Iterable[dict[str, str] | None]) -> list[VehicleRecord]:
    out = []
    for fields in hashes:
        if not fields:
            continue  # the hash expired between ZRANGE and HGETALL
        try:
            out.append(VehicleRecord.from_fields(fields))
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("malformed vehicle hash %r: %s", fields.get("unit_id"), exc)
    return out


class HotStateSync:
    """Keeps a :class:`VehicleCache` in sync with Redis (pub/sub deltas + periodic full reads).

    Args:
        client: Redis client (reconnects on its own after errors).
        cache: Cache to fill.
        status: Redis dependency status to report to.
        resync_s: Period of full re-reads (catches expired hashes and missed deltas).
        stats_s: Period of ingest stats re-reads (their age tells whether the ingest is alive).
        backoff: Reconnect delays.
        on_event: Receives the predictor's messages (alerts, incidents, closed forecasts: parsed JSON);
            ``None`` — those channels are not subscribed.
    """

    def __init__(
        self,
        client: aioredis.Redis,
        cache: VehicleCache,
        status: DependencyStatus,
        *,
        resync_s: float = 10.0,
        stats_s: float = 2.0,
        backoff: Backoff | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.on_event = on_event
        self.cache = cache
        self.status = status
        self.resync_s = resync_s
        self.stats_s = stats_s
        self.backoff = backoff or Backoff()
        self.ingest_stats: dict[str, str] | None = None
        self.deltas = 0
        self.full_syncs = 0
        self.malformed = 0

    async def full_sync(self) -> None:
        """Read the index, all vehicle hashes, the stream clock, the ingest stats, the forecasts and the
        routes in one round trip."""
        members = await self.client.zrange(VEHICLE_INDEX, 0, -1)
        pipe = self.client.pipeline(transaction=False)
        for member in members:
            pipe.hgetall(vehicle_key(member))
        pipe.get(STREAM_TIME_KEY)
        pipe.get(STREAM_EPOCH_KEY)
        pipe.hgetall(INGEST_STATS_KEY)
        pipe.hgetall(FORECAST_KEY)
        pipe.get(FORECAST_STATUS_KEY)
        pipe.get(ROUTES_KEY)
        results = await pipe.execute()
        self.cache.apply(_records(results[: len(members)]), full=True)
        self.cache.set_stream_time(_parse_iso(results[-6]), _parse_int(results[-5]))
        self.ingest_stats = results[-4] or self.ingest_stats
        self.apply_forecasts(results[-3], results[-2])
        self.cache.set_routes(_json(results[-1]))
        self.full_syncs += 1

    async def read_live(self) -> None:
        """Read the ingest stats and the forecast snapshot (every ``stats_s``)."""
        pipe = self.client.pipeline(transaction=False)
        pipe.hgetall(INGEST_STATS_KEY)
        pipe.hgetall(FORECAST_KEY)
        pipe.get(FORECAST_STATUS_KEY)
        stats, forecasts, status = await pipe.execute()
        self.ingest_stats = stats or self.ingest_stats
        self.apply_forecasts(forecasts, status)

    def apply_forecasts(self, fields: dict[str, str] | None, status: str | None) -> None:
        """Parse the ``foresight:forecast`` hash (JSON per ``tr_id``) and its status into the cache."""
        forecasts: dict[int, dict[str, Any]] = {}
        for key, value in (fields or {}).items():
            parsed = _json(value)
            if parsed is not None:
                with contextlib.suppress(ValueError):
                    forecasts[int(key)] = parsed
        self.cache.set_forecasts(forecasts, _json(status))

    def on_message(self, data: Any) -> None:
        """Apply one pub/sub delta message."""
        try:
            payload = json.loads(data)
            records = _records(payload.get("vehicles") or [])
            stream_time = _parse_iso(payload.get("stream_time"))
            epoch = _parse_int(payload.get("epoch"))
        except (ValueError, TypeError, AttributeError) as exc:
            self.malformed += 1
            log.warning("malformed vehicles delta: %s", exc)
            return
        self.cache.set_stream_time(stream_time, epoch)
        self.cache.apply(records)
        self.deltas += 1

    def relay(self, data: Any) -> None:
        """Pass one predictor message (alert, incident, closed forecast) to :attr:`on_event`."""
        payload = _json(data)
        if not isinstance(payload, dict) or self.on_event is None:
            self.malformed += payload is None
            return
        try:
            self.on_event(payload)
        except Exception:
            log.exception("event relay failed")

    async def run(self) -> None:
        """Sync loop: runs until cancelled; on errors keeps the cache and reconnects with backoff."""
        while True:
            pubsub = self.client.pubsub()
            try:
                channels = [CHANNEL_VEHICLES]
                if self.on_event is not None:
                    channels += [CHANNEL_ALERTS, CHANNEL_INCIDENTS, CHANNEL_PREDICTIONS]
                await pubsub.subscribe(*channels)
                await self.full_sync()  # after subscribing, so no delta falls in between
                self.status.mark_ok()
                self.backoff.reset()
                next_full = time.monotonic() + self.resync_s
                next_stats = time.monotonic() + self.stats_s
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message is not None and message.get("type") == "message":
                        if message.get("channel") in (CHANNEL_VEHICLES, None):
                            self.on_message(message["data"])
                        else:
                            self.relay(message["data"])
                    if time.monotonic() >= next_full:
                        await self.full_sync()
                        self.status.mark_ok()
                        next_full = time.monotonic() + self.resync_s
                        next_stats = time.monotonic() + self.stats_s
                    elif time.monotonic() >= next_stats:
                        await self.read_live()
                        self.status.mark_ok()
                        next_stats = time.monotonic() + self.stats_s
            except REDIS_ERRORS as exc:
                self.status.mark_down(exc)
            except Exception:
                log.exception("hot state sync failed")
            finally:
                try:
                    await asyncio.wait_for(pubsub.aclose(), 2)
                except Exception as exc:
                    log.debug("closing pubsub: %s", exc)
            await asyncio.sleep(self.backoff.next())
