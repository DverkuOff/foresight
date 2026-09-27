#!/usr/bin/env python3
"""Configure the official NDTP emulator to stream live telemetry into the Foresight ingest service.

Standard library only, so it runs on the host without the project environment, or inside any container of the
stack: ``make emulator`` pipes it to the python of the replayer container (``python - --emulator
http://emulator:18080 … < scripts/emulator_demo.py``), so the host needs no python::

    make emulator                                       # dataset devices 664030,794446,663271 -> ingest:9201
    python3 scripts/emulator_demo.py                    # 5 devices, 1-2 s interval, target ingest:9201
    python3 scripts/emulator_demo.py --units 3 --doors  # also send Irma04 (doors) cells
    python3 scripts/emulator_demo.py --unit-ids 664030,794446  # devices from the dataset (known tr_id)
    python3 scripts/emulator_demo.py --stop             # stop streaming

Every ``POST /api/config`` makes the emulator drop its connections and reconnect with a new handshake.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def build_config(
    target_host: str,
    target_port: int,
    units: int,
    first_unit: int,
    interval_min_ms: int,
    interval_max_ms: int,
    doors: bool,
    unit_ids: list[int] | None = None,
) -> dict:
    """Build an emulator config with ``units`` auto-generated devices.

    Args:
        target_host: NDTP server host as seen from the emulator container.
        target_port: NDTP server port.
        units: Number of devices (ignored when ``unit_ids`` is given).
        first_unit: ``unitId`` of the first device; the rest follow sequentially.
        interval_min_ms: Send interval of the first device.
        interval_max_ms: Send interval of the last device (spread evenly in between).
        doors: Add a ``G6CellIrma04`` cell with door states.
        unit_ids: Explicit device ids (e.g. from the dataset, so the ingest knows their ``tr_id``).

    Returns:
        JSON-serialisable config for ``POST /api/config``.
    """
    ids = list(unit_ids) if unit_ids else [first_unit + i for i in range(units)]
    units = len(ids)
    step = (interval_max_ms - interval_min_ms) / max(units - 1, 1)
    config_units = []
    for i in range(units):
        cells: list[dict] = []
        if doors:
            # explicit cells: Nav00 is still randomised by autoGenerate, Irma04 goes as given
            cells = [
                {"type": "G6CellNav00", "fields": {}},
                {
                    "type": "G6CellIrma04",
                    "fields": {
                        "irma_present_door1": True,
                        "irma_present_door2": True,
                        "irma_closed_door1": i % 2 == 0,
                        "irma_closed_door2": True,
                        "irma_door_in1": i,
                        "irma_door_out1": 0,
                    },
                },
            ]
        config_units.append(
            {
                "unitId": ids[i],
                "intervalMs": round(interval_min_ms + i * step),
                "autoGenerate": True,
                "cells": cells,
            }
        )
    return {"targetHost": target_host, "targetPort": target_port, "units": config_units}


def request(url: str, payload: dict | None = None, timeout: float = 5.0) -> tuple[int, str]:
    """Send a GET (or a JSON POST when ``payload`` is given) and return status and body."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def wait_for_api(base: str, timeout_s: float) -> None:
    """Wait until the emulator REST API answers ``GET /api/cells``.

    Raises:
        SystemExit: If the API is not up within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if request(f"{base}/api/cells", timeout=2)[0] == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"emulator API at {base} is not reachable")
        time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=fmt)
    parser.add_argument("--emulator", default="http://localhost:18080", help="emulator REST API base URL")
    parser.add_argument("--target-host", default="ingest", help="NDTP host as seen from the emulator")
    parser.add_argument("--target-port", type=int, default=9201)
    parser.add_argument("--units", type=int, default=5)
    parser.add_argument("--first-unit", type=int, default=100001)
    parser.add_argument(
        "--unit-ids",
        type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=None,
        help="comma-separated unitIds (overrides --units/--first-unit), e.g. from dataset/test/traffic.csv",
    )
    parser.add_argument("--interval-min-ms", type=int, default=1000)
    parser.add_argument("--interval-max-ms", type=int, default=2000)
    parser.add_argument("--doors", action="store_true", help="also send Irma04 (doors) cells")
    parser.add_argument("--stop", action="store_true", help="stop streaming (empty unit list)")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds to wait for the emulator API")
    args = parser.parse_args(argv)

    base = args.emulator.rstrip("/")
    wait_for_api(base, args.wait)
    if args.stop:
        config = {"targetHost": args.target_host, "targetPort": args.target_port, "units": []}
    else:
        config = build_config(
            args.target_host,
            args.target_port,
            args.units,
            args.first_unit,
            args.interval_min_ms,
            args.interval_max_ms,
            args.doors,
            args.unit_ids,
        )
    status, body = request(f"{base}/api/config", config)
    if status != 200:
        print(f"emulator rejected the config: HTTP {status}: {body}", file=sys.stderr)
        return 1
    if args.stop:
        print("emulator stopped")
    else:
        units = ", ".join(f"{u['unitId']}@{u['intervalMs']}ms" for u in config["units"])
        print(f"emulator streaming to {args.target_host}:{args.target_port}: {units}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
