"""Prometheus metrics: collectors that read live service objects at scrape time.

Reading live objects instead of keeping duplicate Prometheus counters means there is one source of truth and
several app instances (tests) never clash in a global registry: each app owns its own
:class:`prometheus_client.CollectorRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, PlatformCollector, ProcessCollector
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from backend.clock import StreamClock
from backend.ndtp_server import NdtpServer
from backend.runtime import DependencyStatus
from backend.state import LinkStatus, StateStore

if TYPE_CHECKING:
    from backend.bus import StreamConsumer, TelemetryPublisher
    from backend.db import BufferedWriter

_NDTP_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("connections_total", "foresight_ndtp_connections", "Accepted NDTP connections."),
    ("disconnects", "foresight_ndtp_disconnects", "Closed NDTP connections."),
    ("read_timeouts", "foresight_ndtp_read_timeouts", "Connections closed because of silence."),
    ("bytes_received", "foresight_ndtp_received_bytes", "Bytes received on NDTP connections."),
    ("bytes_discarded", "foresight_ndtp_discarded_bytes", "Bytes skipped while resynchronising."),
    ("frames", "foresight_ndtp_frames", "Frames with a valid CRC."),
    ("handshakes", "foresight_ndtp_handshakes", "Handshake frames."),
    ("realtime_packets", "foresight_ndtp_realtime_packets", "Realtime packets."),
    ("nav_records", "foresight_ndtp_nav_records", "Nav00 records stored."),
    ("crc_errors", "foresight_ndtp_crc_errors", "Frames dropped because of a CRC mismatch."),
    ("bad_headers", "foresight_ndtp_bad_headers", "Rejected implausible NPL headers."),
    ("parse_errors", "foresight_ndtp_parse_errors", "Frames with a malformed body."),
    ("unknown_cells", "foresight_ndtp_unknown_cells", "Packets stopped at an unknown cell type."),
    ("abusive_disconnects", "foresight_ndtp_abusive_disconnects", "Closed: CRC error or garbage budget."),
    ("first_frame_timeouts", "foresight_ndtp_first_frame_timeouts", "Closed: no valid frame in time."),
    ("rejected_connections", "foresight_ndtp_rejected_connections", "Closed: connection limit reached."),
)


def _counter(name: str, doc: str, value: float) -> CounterMetricFamily:
    return CounterMetricFamily(name, doc, value=value)


def _gauge(name: str, doc: str, value: float) -> GaugeMetricFamily:
    return GaugeMetricFamily(name, doc, value=value)


def vehicle_status_metric(counts: dict[LinkStatus, int]) -> GaugeMetricFamily:
    """``foresight_vehicles{status=...}``."""
    vehicles = GaugeMetricFamily("foresight_vehicles", "Vehicles by link status.", labels=["status"])
    for status, count in counts.items():
        vehicles.add_metric([status.value], count)
    return vehicles


class NdtpCollector(Collector):
    """Exports :class:`~backend.ndtp_server.IngestStats` and vehicle status counts of the ingest.

    Args:
        server: NDTP server whose counters are exported.
        store: Vehicle state store.
    """

    def __init__(self, server: NdtpServer, store: StateStore) -> None:
        self.server = server
        self.store = store

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        stats = self.server.stats
        for attr, name, doc in _NDTP_COUNTERS:
            yield _counter(name, doc, getattr(stats, attr))
        yield _gauge("foresight_ndtp_connections_active", "Open NDTP connections.", stats.connections_active)
        yield _gauge(
            "foresight_ndtp_packets_per_second",
            "Realtime packets per second over the last 10 s.",
            stats.packets_rate.rate(),
        )
        yield _gauge("foresight_ndtp_listening", "1 if NDTP accepts connections.", int(self.server.listening))
        yield vehicle_status_metric(self.store.status_counts())


class ClockCollector(Collector):
    """The ingest's stream clock: time, epoch, jumps, rejected fix times."""

    def __init__(self, clock: StreamClock) -> None:
        self.clock = clock

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        c = self.clock
        if c.now is not None:
            yield _gauge("foresight_stream_time_seconds", "Stream clock, Unix seconds.", c.now)
        yield _gauge("foresight_stream_clock_epoch", "Epoch of the stream clock.", c.epoch)
        yield _counter("foresight_stream_clock_resets", "Stream clock jumps (source restarts).", c.resets)
        yield _counter("foresight_stream_clock_garbage", "Fixes with an implausible time.", c.garbage)
        yield _counter(
            "foresight_stream_clock_off_timeline", "Fixes off the stream timeline.", c.off_timeline
        )


class PublisherCollector(Collector):
    """Redis publisher of the ingest: buffer, evictions, throughput."""

    def __init__(self, publisher: TelemetryPublisher) -> None:
        self.publisher = publisher

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        p = self.publisher
        yield _counter("foresight_bus_published", "Events written to the telemetry stream.", p.published)
        yield _gauge("foresight_bus_buffered", "Events waiting in memory for Redis.", p.buffered)
        yield _counter("foresight_bus_evicted", "Events dropped because the buffer overflowed.", p.evicted)
        yield _counter("foresight_bus_flushes", "Successful Redis pipelines.", p.flushes)
        yield _counter("foresight_bus_flush_errors", "Failed Redis pipelines.", p.flush_errors)
        yield _counter("foresight_bus_command_errors", "Commands rejected by Redis.", p.command_errors)
        yield _counter("foresight_bus_state_writes", "Vehicle hot-state hash writes.", p.state_writes)
        yield _counter("foresight_bus_unmapped_events", "Events from devices without a tr_id.", p.unmapped)
        yield _gauge(
            "foresight_bus_last_flush_seconds", "Duration of the last Redis pipeline.", p.last_flush_s
        )


class ConsumerCollector(Collector):
    """Telemetry consumer of the predictor: throughput, lag, recovery."""

    def __init__(self, consumer: StreamConsumer) -> None:
        self.consumer = consumer

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        c = self.consumer
        yield _counter("foresight_consumer_events", "Stream events processed.", c.processed)
        yield _counter("foresight_consumer_acked", "Stream entries acknowledged.", c.acked)
        yield _gauge("foresight_consumer_events_per_second", "Events per second over 10 s.", c.rate.rate())
        yield _counter("foresight_consumer_malformed", "Malformed entries skipped.", c.malformed)
        yield _counter("foresight_consumer_handler_errors", "Failed batch handlers.", c.handler_errors)
        yield _counter("foresight_consumer_recovered", "Own pending entries re-read.", c.recovered)
        yield _counter("foresight_consumer_claimed", "Pending entries claimed from others.", c.claimed)
        yield _counter("foresight_consumer_read_errors", "Failed stream reads.", c.read_errors)
        yield _counter(
            "foresight_consumer_history", "Processed entries read back at start.", c.history_entries
        )
        yield _counter(
            "foresight_consumer_removed_consumers",
            "Idle consumers removed from the group.",
            c.consumers_removed,
        )
        if c.lag is not None:
            yield _gauge("foresight_consumer_lag", "Consumer group lag (undelivered entries).", c.lag)
        if c.pending is not None:
            yield _gauge("foresight_consumer_pending", "Delivered but unacknowledged entries.", c.pending)
        if c.stream_length is not None:
            yield _gauge("foresight_stream_length", "Entries in the telemetry stream.", c.stream_length)


class WriterCollector(Collector):
    """Buffered PostgreSQL writer: buffer, drops, throughput."""

    def __init__(self, writer: BufferedWriter) -> None:
        self.writer = writer

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        w = self.writer
        yield _gauge("foresight_db_buffered", "Rows waiting in memory for PostgreSQL.", w.buffered)
        yield _counter("foresight_db_dropped", "Rows lost (buffer overflow or database disabled).", w.dropped)
        yield _counter("foresight_db_written", "Rows written to PostgreSQL.", w.written)
        yield _counter("foresight_db_rejected", "Rows rejected by PostgreSQL (bad data).", w.rejected)
        yield _counter("foresight_db_errors", "Failed database operations.", w.errors)


class DependencyCollector(Collector):
    """``foresight_dependency_up{dependency=...}`` and outage counters."""

    def __init__(self, statuses: Iterable[DependencyStatus]) -> None:
        self.statuses = list(statuses)

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        up = GaugeMetricFamily(
            "foresight_dependency_up", "1 if the dependency is reachable.", labels=["dependency"]
        )
        outages = CounterMetricFamily(
            "foresight_dependency_outages", "Dependency outages since start.", labels=["dependency"]
        )
        for status in self.statuses:
            if not status.enabled:
                continue
            up.add_metric([status.name], 1 if status.ok else 0)
            outages.add_metric([status.name], status.outages)
        yield up
        yield outages


class FunctionCollector(Collector):
    """Metrics computed by a function at scrape time (service-specific gauges)."""

    def __init__(self, fn: Callable[[], Iterable[Metric]]) -> None:
        self.fn = fn

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        yield from self.fn()


def build_registry(*collectors: Collector) -> CollectorRegistry:
    """Create a registry with process, platform and the given collectors.

    Args:
        collectors: Service collectors.

    Returns:
        A new registry for ``/metrics``.
    """
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    for collector in collectors:
        registry.register(collector)
    return registry
