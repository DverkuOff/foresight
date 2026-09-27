"""Prometheus metrics of the replayer, read from the controller at scrape time.

Like ``backend.metrics`` the collector reads live objects instead of duplicating counters, and every app owns
its own :class:`~prometheus_client.CollectorRegistry`, so several instances (tests) never clash.
"""

from __future__ import annotations

from collections.abc import Iterator

from prometheus_client import CollectorRegistry, PlatformCollector, ProcessCollector
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from replayer.controller import ReplayController

_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("packets_dispatched", "foresight_replayer_packets_dispatched", "Packets handed out by the scheduler."),
    (
        "packets_sent",
        "foresight_replayer_packets_sent",
        "Realtime NDTP frames written to the socket (handed to the kernel; delivery is not acknowledged).",
    ),
    ("handshakes", "foresight_replayer_handshakes", "NDTP handshakes written."),
    ("connects", "foresight_replayer_connects", "Successful TCP connects to the receiver."),
    ("reconnects", "foresight_replayer_reconnects", "Connects after a lost connection."),
    ("disconnects", "foresight_replayer_disconnects", "Connections lost unexpectedly."),
    ("connect_failures", "foresight_replayer_connect_failures", "Failed connect attempts."),
    ("cycles", "foresight_replayer_cycles", "Replay passes started (start, restart, loop)."),
    ("bridge_posts", "foresight_replayer_bridge_posts", "Emulator configs accepted."),
    ("bridge_errors", "foresight_replayer_bridge_errors", "Failed emulator requests."),
)


class ReplayerCollector(Collector):
    """Exports :class:`~replayer.stats.ReplayStats` and the state of the current session.

    Args:
        controller: Replay controller.
    """

    def __init__(self, controller: ReplayController) -> None:
        self.controller = controller

    def collect(self) -> Iterator[Metric]:
        """Yield metric families for one scrape."""
        ctl = self.controller
        stats = ctl.stats
        for attr, name, doc in _COUNTERS:
            yield CounterMetricFamily(name, doc, value=getattr(stats, attr))
        dropped = CounterMetricFamily(
            "foresight_replayer_packets_dropped", "Packets not sent, by reason.", labels=["reason"]
        )
        for reason in sorted({"overflow", "disconnected", "stopped", *stats.dropped}):
            dropped.add_metric([reason], stats.dropped.get(reason, 0))
        yield dropped
        gauges = (
            ("foresight_replayer_connections_active", "Open NDTP connections.", stats.connections_active),
            ("foresight_replayer_running", "1 while a replay is waiting or running.", int(ctl.running)),
            ("foresight_replayer_speed", "Replay speed, data seconds per wall second.", ctl.config.speed),
            ("foresight_replayer_lag_seconds", "Age of the oldest packet not yet sent.", ctl.lag_s()),
            ("foresight_replayer_backlog_packets", "Packets queued for sending.", ctl.backlog()),
            (
                "foresight_replayer_last_send_lag_seconds",
                "How late the last packet was written relative to its schedule.",
                stats.last_send_lag_s,
            ),
            (
                "foresight_replayer_epoch",
                "Replay passes started; a change resets the stream clock.",
                ctl.epoch,
            ),
        )
        for name, doc, value in gauges:
            yield GaugeMetricFamily(name, doc, value=value)
        data_time = ctl.data_time()
        if data_time is not None:
            yield GaugeMetricFamily(
                "foresight_replayer_data_time_seconds",
                "Current position of the data clock (receive_time), Unix seconds.",
                value=data_time,
            )


def build_registry(controller: ReplayController) -> CollectorRegistry:
    """Create a registry with process, platform and replayer metrics.

    Args:
        controller: Replay controller.

    Returns:
        A new registry for ``/metrics``.
    """
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    registry.register(ReplayerCollector(controller))
    return registry
