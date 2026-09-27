"""Command line: ``python -m replayer run`` (replay without API) and ``python -m replayer serve`` (with API).

Every option has an environment default with the ``FORESIGHT_REPLAY_`` prefix (``--speed`` ↔
``FORESIGHT_REPLAY_SPEED``, ``--host`` ↔ ``FORESIGHT_REPLAY_TARGET_HOST``, ``--autostart`` ↔
``FORESIGHT_REPLAY_AUTOSTART`` …), so the Docker image is configured by the environment alone. The dataset
root comes from ``--dataset-dir``, ``FORESIGHT_DATASET_DIR`` or ``MT_DATASET_DIR`` (see
:func:`replayer.source.replay_dataset_dir`).

Examples::

    python -m replayer run --split test --speed 60 --start 07:00 --until 08:00 --port 9201
    python -m replayer run --units 115106,116057 --loop
    python -m replayer run --mode bridge --emulator-url http://localhost:18080 --host host.docker.internal
    python -m replayer serve --autostart --loop --speed 30 --host ingest --start 07:00 --until 10:00

For a demo loop pick a window: the whole ``test`` day at x30 is a 48-minute pass, while
``07:00``-``10:00`` (the busiest hours) wraps every 6 minutes and shows closed forecasts within the first 2-3
minutes. Without ``--until`` a loop pass ends 10 minutes (data time) after the last ``event_time``, leaving
out a few stragglers that arrived hours late.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from replayer.engine import FINISHED, MODES, ON_DISCONNECT, STOPPED, ReplayConfig, Replayer, select_replay
from replayer.source import load_replay_data
from shared.data import SPLITS

log = logging.getLogger("replayer")

ENV_PREFIX = "FORESIGHT_REPLAY_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _units(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in re.split(r"[,\s]+", text.strip()) if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc


def _add_replay_options(parser: argparse.ArgumentParser) -> None:
    # string defaults go through ``type`` like command-line values, so environment values are validated too
    replay = parser.add_argument_group("replay")
    replay.add_argument("--split", choices=SPLITS, default=_env("SPLIT", "test"), help="dataset split")
    replay.add_argument(
        "--speed", type=float, default=_env("SPEED", "30"), help="data seconds per wall second (x1…x120)"
    )
    replay.add_argument(
        "--start", default=_env("START"), help="start of the receive_time window: HH:MM[:SS] or ISO datetime"
    )
    replay.add_argument("--until", default=_env("UNTIL"), help="end of the window (exclusive)")
    replay.add_argument(
        "--units",
        type=_units,
        default=_env("UNITS", ""),
        help="comma-separated unit_id or tr_id (default: all)",
    )
    once = replay.add_mutually_exclusive_group()
    once.add_argument("--loop", dest="loop", action="store_true", help="start over after the last packet")
    once.add_argument("--once", dest="loop", action="store_false", help="finish after the last packet")
    replay.set_defaults(loop=_env_bool("LOOP"))
    replay.add_argument(
        "--mode", choices=MODES, default=_env("MODE", "ndtp"), help="own NDTP or the official emulator"
    )
    replay.add_argument("--dataset-dir", type=Path, default=None, help="dataset root")

    ndtp = parser.add_argument_group("NDTP receiver")
    ndtp.add_argument("--host", default=_env("TARGET_HOST", "127.0.0.1"), help="receiver host")
    ndtp.add_argument("--port", type=int, default=_env("TARGET_PORT", "9201"), help="receiver NDTP port")
    ndtp.add_argument(
        "--on-disconnect",
        choices=ON_DISCONNECT,
        default=_env("ON_DISCONNECT", "buffer"),
        help="queue packets while a connection is down and send them after reconnecting, or drop them",
    )
    ndtp.add_argument(
        "--max-queue", type=int, default=_env("MAX_QUEUE", "5000"), help="per-device queue bound"
    )

    bridge = parser.add_argument_group("emulator bridge")
    bridge.add_argument("--emulator-url", default=_env("EMULATOR_URL", "http://localhost:18080"))
    bridge.add_argument(
        "--bridge-interval",
        type=float,
        default=_env("BRIDGE_INTERVAL", "5"),
        help="seconds between emulator config updates (each one reconnects all devices)",
    )
    bridge.add_argument(
        "--bridge-target-host", default=_env("BRIDGE_TARGET_HOST"), help="receiver host seen by the emulator"
    )
    bridge.add_argument(
        "--bridge-target-port", type=int, default=_env("BRIDGE_TARGET_PORT"), help="receiver port"
    )
    parser.add_argument("--log-level", default=_env("LOG_LEVEL", "info"), help="logging level")


def _config(args: argparse.Namespace) -> ReplayConfig:
    return ReplayConfig(
        split=args.split,
        speed=args.speed,
        start=args.start or None,
        until=args.until or None,
        units=args.units,
        loop=args.loop,
        mode=args.mode,
        host=args.host,
        port=args.port,
        on_disconnect=args.on_disconnect,
        max_queue=args.max_queue,
        emulator_url=args.emulator_url,
        bridge_interval_s=args.bridge_interval,
        bridge_target_host=args.bridge_target_host,
        bridge_target_port=args.bridge_target_port,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``run`` and ``serve`` commands."""
    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="python -m replayer", description=__doc__, formatter_class=fmt)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="replay without the HTTP API", formatter_class=fmt)
    _add_replay_options(run)
    run.add_argument("--report-every", type=float, default=10.0, help="progress log period, s (0: off)")
    run.set_defaults(func=_cmd_run)

    serve = commands.add_parser("serve", help="HTTP control API (+ optional autostart)", formatter_class=fmt)
    _add_replay_options(serve)
    serve.add_argument("--http-host", default=_env("HTTP_HOST", "0.0.0.0"), help="API bind address")
    serve.add_argument("--http-port", type=int, default=_env("HTTP_PORT", "8010"), help="API port")
    serve.add_argument(
        "--autostart",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("AUTOSTART"),
        help="start replaying on startup with the given options",
    )
    serve.set_defaults(func=_cmd_serve)
    return parser


def _fmt_status(st: dict[str, Any], rate: float) -> str:
    data_time = st["data_time"].strftime("%Y-%m-%d %H:%M:%S") if st["data_time"] else "-"
    return (
        f"{st['state']} | data {data_time} | sent {st['packets_sent']} ({rate:.1f}/s) | "
        f"dispatched {st['packets_dispatched']}/{st['packets_total']} | "
        f"conn {st['connections_active']}/{st['units']} | lag {st['lag_s']:.2f} s | "
        f"backlog {st['backlog']} | dropped {st['packets_dropped']} | reconnects {st['reconnects']}"
    )


async def _report(replayer: Replayer, every: float) -> None:
    if every <= 0:
        return
    last_sent, last_t = 0, time.monotonic()
    while True:
        await asyncio.sleep(every)
        st = replayer.status()
        now = time.monotonic()
        rate = (st["packets_sent"] - last_sent) / max(now - last_t, 1e-9)
        last_sent, last_t = st["packets_sent"], now
        log.info("%s", _fmt_status(st, rate))


async def _run(config: ReplayConfig, root: Path | None, report_every: float) -> int:
    try:
        data = await asyncio.to_thread(load_replay_data, config.split, root)
        data = select_replay(data, config)
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        return 2
    log.info(
        "replaying %d packets of %d devices from split %s to %s at x%g (%s)",
        len(data),
        len(data.units),
        config.split,
        f"{config.host}:{config.port}" if config.mode == "ndtp" else config.emulator_url,
        config.speed,
        "loop" if config.loop else "once",
    )
    replayer = Replayer(data, config)
    task = asyncio.create_task(replayer.run(), name="replay")
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, task.cancel)
    reporter = asyncio.create_task(_report(replayer, report_every), name="report")
    started = time.monotonic()
    await asyncio.wait({task})
    reporter.cancel()
    elapsed = max(time.monotonic() - started, 1e-9)
    st = replayer.status()
    log.info("done in %.1f s: %s", elapsed, _fmt_status(st, st["packets_sent"] / elapsed))
    return 0 if replayer.state in (FINISHED, STOPPED) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    try:
        config.validate()
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    return asyncio.run(_run(config, args.dataset_dir, args.report_every))


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from replayer.api import create_app

    config = _config(args)
    try:
        config.validate()
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    loader = load_replay_data
    if args.dataset_dir is not None:
        loader = functools.partial(load_replay_data, root=args.dataset_dir)
    app = create_app(config, autostart=args.autostart, loader=loader)
    uvicorn.run(app, host=args.http_host, port=args.http_port, log_level=args.log_level.lower())
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns:
        Exit code: 0 on success, 1 if the replay failed, 2 on invalid input.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
