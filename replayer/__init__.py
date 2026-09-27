"""Foresight replayer: historical ``traffic.csv`` replayed as real NDTP over TCP.

The main demo source of the system (``docs/architecture.md`` §3, §5): every device of a split gets its own
TCP connection to the ingest service (handshake, then realtime Nav00 packets with CRC); packets go out in the
order and at the pace of ``receive_time`` while carrying ``event_time`` as their timestamp, at a configurable
speed. The ``bridge`` mode drives the official emulator with the same tracks instead.

Modules:

* :mod:`replayer.source` — loading a split without cleaning, ``--units``/``--start``/``--until`` selection;
* :mod:`replayer.clock` — clock abstraction and the drift-free schedule;
* :mod:`replayer.link` — one NDTP connection per device with reconnects and a bounded queue;
* :mod:`replayer.bridge` — emulator bridge (``POST /api/config``);
* :mod:`replayer.engine` — the replay session;
* :mod:`replayer.controller`, :mod:`replayer.api`, :mod:`replayer.metrics` — HTTP control and Prometheus;
* ``python -m replayer run|serve`` — command line (:mod:`replayer.__main__`).
"""

__version__ = "0.1.0"
