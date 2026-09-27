"""Cumulative replayer counters shared by all replay sessions of a process (exported to Prometheus)."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Reasons a packet is not sent, used as the ``reason`` label.
DROP_OVERFLOW = "overflow"  #: the per-device queue was full, the oldest packet was dropped
DROP_DISCONNECTED = "disconnected"  #: no connection and ``on_disconnect=skip``
DROP_STOPPED = "stopped"  #: still queued when the replay was stopped


@dataclass(slots=True)
class ReplayStats:
    """Counters since process start.

    Attributes:
        packets_dispatched: Packets handed out by the scheduler (in bridge mode: positions updated).
        packets_sent: Realtime frames written to NDTP sockets (handed to the kernel). NDTP here has no
            acknowledgements, so this is not confirmed delivery: frames still in the socket buffers when a
            receiver dies are lost but counted (see :mod:`replayer.link`).
        handshakes: Handshake frames written.
        connects: Successful TCP connects.
        reconnects: Connects after the first one of a device.
        disconnects: Connections lost unexpectedly.
        connect_failures: Failed connect attempts.
        connections_active: Open connections right now.
        dropped: Packets not sent, by reason.
        last_send_lag_s: How late (wall seconds) the last packet was written relative to its due time.
        cycles: Replay passes started (first start, restarts and ``--loop`` wrap-arounds).
        bridge_posts: Successful ``POST /api/config`` to the emulator.
        bridge_errors: Failed emulator requests.
    """

    packets_dispatched: int = 0
    packets_sent: int = 0
    handshakes: int = 0
    connects: int = 0
    reconnects: int = 0
    disconnects: int = 0
    connect_failures: int = 0
    connections_active: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    last_send_lag_s: float = 0.0
    cycles: int = 0
    bridge_posts: int = 0
    bridge_errors: int = 0

    def drop(self, reason: str, count: int = 1) -> None:
        """Count ``count`` dropped packets.

        Args:
            reason: One of ``DROP_*``.
            count: Number of packets.
        """
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + count

    @property
    def dropped_total(self) -> int:
        """All dropped packets."""
        return sum(self.dropped.values())
