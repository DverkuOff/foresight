"""Synthetic NDTP load for the ingest: many devices, each on its own TCP connection, sending real-time
navigation packets at a fixed rate (the path «NDTP → parse → Redis Stream → predictor consumer»).

    python scripts/loadtest.py --host ingest --port 9201 --devices 500 --interval 2 --duration 60

The devices have unit ids that are not in the dataset (from 9 000 000): the ingest parses, stores and
publishes them, the predictor consumes and counts them as unmapped (they get no forecasts). Uses only the
standard library and ``shared.ndtp`` (runs in the replayer or backend image, see docs/performance.md).
Prints the packets sent per second; the service side is read from ``/api/metrics/perf`` and ``make status``.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from datetime import UTC, datetime

from shared.ndtp import NavRecord, encode_handshake, encode_realtime

BASE_UNIT = 9_000_000
CENTER = (37.62, 55.75)


async def device(i: int, args: argparse.Namespace, sent: list[int], stop: float) -> None:
    unit = BASE_UNIT + i
    reader, writer = await asyncio.open_connection(args.host, args.port)

    async def drain() -> None:  # the ingest answers every packet: read the replies so the socket never fills
        while await reader.read(65536):
            pass

    drainer = asyncio.create_task(drain())
    writer.write(encode_handshake(unit, 1))
    await writer.drain()
    seq = 2
    phase = (i * 0.618) % 1.0
    await asyncio.sleep(phase * args.interval)  # spread the devices over the interval
    while time.monotonic() < stop:
        angle = seq / 50 + i
        lon = CENTER[0] + 0.1 * math.cos(angle) * (0.3 + (i % 7) / 10)
        lat = CENTER[1] + 0.06 * math.sin(angle) * (0.3 + (i % 5) / 10)
        nav = NavRecord(timestamp=datetime.now(UTC), lon=lon, lat=lat, valid=True, speed_avg=25, course=90)
        writer.write(encode_realtime(unit, seq & 0xFFFF, nav))
        seq += 1
        sent[0] += 1
        await writer.drain()
        await asyncio.sleep(args.interval)
    writer.close()
    drainer.cancel()


async def main(args: argparse.Namespace) -> None:
    sent = [0]
    stop = time.monotonic() + args.duration
    tasks = [asyncio.create_task(device(i, args, sent, stop)) for i in range(args.devices)]
    last, t0 = 0, time.monotonic()
    while any(not t.done() for t in tasks):
        await asyncio.sleep(5)
        now = sent[0]
        print(f"{time.monotonic() - t0:5.0f} s  {(now - last) / 5:7.1f} packets/s  total {now}", flush=True)
        last = now
    errors = [t.exception() for t in tasks if t.exception() is not None]
    print(f"done: {sent[0]} packets from {args.devices} devices, {len(errors)} connection errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="ingest")
    ap.add_argument("--port", type=int, default=9201)
    ap.add_argument("--devices", type=int, default=500)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between packets of one device")
    ap.add_argument("--duration", type=float, default=60.0)
    asyncio.run(main(ap.parse_args()))
