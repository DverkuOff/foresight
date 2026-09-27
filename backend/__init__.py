"""Foresight backend: the ``ingest``, ``predictor`` and ``api`` services (one package, one image).

* :mod:`backend.ingest` — NDTP TCP server, publishes telemetry and hot vehicle state to Redis;
* :mod:`backend.predictor` — consumer group on the telemetry stream, track windows, prediction ticks;
* :mod:`backend.api` — REST / WebSocket / Swagger over the hot state;
* :mod:`backend.bus`, :mod:`backend.hotstate`, :mod:`backend.db` — Redis and PostgreSQL plumbing with
  degradation (bounded in-memory buffers, reconnect with backoff).
"""

__version__ = "0.2.0"
