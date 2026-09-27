"""Replayer tests: raw CSV loading, NDTP over TCP, pacing, reconnects, emulator bridge, API and CLI."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import socket
import socketserver
import threading
import time
from collections.abc import Callable, Coroutine
from datetime import UTC, date, datetime
from http.client import BadStatusLine
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from replayer.__main__ import main as cli_main
from replayer.bridge import EmulatorBridge, UrllibClient, build_emulator_config
from replayer.clock import ReplaySchedule
from replayer.controller import ReplayController
from replayer.engine import ReplayConfig, Replayer, select_replay
from replayer.link import LinkSettings, UnitLink
from replayer.source import ReplayData, load_replay_data, parse_data_time
from replayer.stats import ReplayStats
from shared.ndtp import Frame, FrameDecoder, NavRecord, RealtimePacket, decode_handshake, decode_realtime

DAY = "2026-01-06"
COLUMNS = [
    "packet_id",
    "tr_id",
    "unit_id",
    "event_time",
    "device_event_id",
    "location_valid",
    "gps_time",
    "lon",
    "lat",
    "alt",
    "speed",
    "heading",
    "receive_time",
    "is_hist_data",
]


def _row(
    unit: int,
    tr: int,
    event: str,
    recv: str,
    *,
    lon: float = 37.61,
    lat: float = 55.75,
    valid: bool = True,
    coords: bool = True,
    speed: float = 25.4,
    heading: float = 90.0,
    hist: bool = False,
) -> dict[str, Any]:
    return {
        "packet_id": 0,
        "tr_id": tr,
        "unit_id": unit,
        "event_time": f"{DAY} {event}",
        "device_event_id": 0,
        "location_valid": valid,
        "gps_time": f"{DAY} {event}" if valid else None,
        "lon": lon if coords else None,
        "lat": lat if coords else None,
        "alt": 150.0 if coords else None,
        "speed": speed if coords else None,
        "heading": heading if coords else None,
        "receive_time": f"{DAY} {recv}",
        "is_hist_data": hist,
    }


# receive_time order: 1002@06:59:52, 1001@07:00:01.25, 1002@07:00:06, 1001@07:00:21, 1002@07:00:26,
# 1001@07:00:31, 1001@07:00:40 (a late packet from 07:00:10)
ROWS = [
    _row(1001, 501, "07:00:00", "07:00:01.250000", lon=37.6100001, lat=55.7500002, heading=359.6),
    _row(1001, 501, "07:00:10", "07:00:40", lon=37.62, lat=55.76, hist=True),
    _row(1001, 501, "07:00:20", "07:00:21", lon=37.63, lat=55.77, speed=368.0),
    _row(1001, 501, "07:00:30.500000", "07:00:31", valid=False, coords=False),
    _row(1002, 502, "06:59:50", "06:59:52", lon=37.50, lat=55.70),
    _row(1002, 502, "07:00:05", "07:00:06", lon=37.51, lat=55.71),
    _row(1002, 502, "07:00:25", "07:00:26", lon=37.52, lat=55.72, valid=False),
]


def _dataset(tmp_path: Path, rows: list[dict[str, Any]] | None = None, split: str = "test") -> Path:
    folder = tmp_path / split
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(reversed(rows or ROWS)), columns=COLUMNS)  # file order must not matter
    frame.to_csv(folder / "traffic.csv", index=False)
    return tmp_path


def _dt(hms: str) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hms}").replace(tzinfo=UTC)


def _ts(hms: str) -> int:
    return int(_dt(hms).timestamp())


def _run(coro: Coroutine[Any, Any, Any], timeout: float = 30.0) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout))


def _run_guarded(coro: Coroutine[Any, Any, Any], timeout: float = 20.0) -> Any:
    """Run ``coro`` in a daemon thread: a blocked event loop fails the test instead of hanging the suite."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = _run(coro, timeout)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout + 5)
    if thread.is_alive():
        raise AssertionError("the event loop is blocked")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _wait_sync(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeClock:
    """Virtual monotonic clock: ``sleep`` yields to the event loop, then advances time instantly."""

    def __init__(self, t0: float = 1000.0, real_step: float = 0.0) -> None:
        self.t0 = t0
        self.t = t0
        self.real_step = real_step
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(self.real_step)
        self.t += max(seconds, 0.0)


class StubReceiver:
    """Asyncio NDTP receiver: decodes every connection with ``FrameDecoder`` and records the frames."""

    def __init__(self) -> None:
        self.frames: list[tuple[int, Frame]] = []
        self.decoders: list[FrameDecoder] = []
        self.writers: dict[int, asyncio.StreamWriter] = {}
        self.on_frame: Callable[[int, Frame], None] | None = None
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self, port: int = 0) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", port)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = len(self.decoders)
        decoder = FrameDecoder()
        self.decoders.append(decoder)
        self.writers[conn] = writer
        try:
            with contextlib.suppress(OSError):
                while data := await reader.read(65536):
                    for frame in decoder.feed(data):
                        self.frames.append((conn, frame))
                        if self.on_frame is not None:
                            self.on_frame(conn, frame)
        finally:
            self.writers.pop(conn, None)
            writer.close()

    def drop(self, conn: int) -> None:
        writer = self.writers.get(conn)
        if writer is not None:
            writer.close()

    async def stop(self) -> None:
        for writer in list(self.writers.values()):
            writer.close()
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    @property
    def crc_errors(self) -> int:
        return sum(d.crc_errors + d.bad_headers for d in self.decoders)

    def by_connection(self) -> dict[int, list[Frame]]:
        out: dict[int, list[Frame]] = {}
        for conn, frame in self.frames:
            out.setdefault(conn, []).append(frame)
        return out

    def handshakes(self, unit: int) -> list[Frame]:
        return [f for _, f in self.frames if f.is_handshake and f.unit_id == unit]

    def packets(self, unit: int) -> list[RealtimePacket]:
        return [decode_realtime(f) for _, f in self.frames if f.is_realtime and f.unit_id == unit]


class ThreadedReceiver:
    """Blocking NDTP receiver in a background thread (for the API and CLI, which run their own loops)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: list[Frame] = []
        self.decoders: list[FrameDecoder] = []
        receiver = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                decoder = FrameDecoder()
                with receiver.lock:
                    receiver.decoders.append(decoder)
                while True:
                    try:
                        data = self.request.recv(65536)
                    except OSError:
                        return
                    if not data:
                        return
                    frames = decoder.feed(data)
                    with receiver.lock:
                        receiver.frames.extend(frames)

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def count(self, unit: int) -> int:
        with self.lock:
            return sum(f.unit_id == unit for f in self.frames)

    def crc_errors(self) -> int:
        with self.lock:
            return sum(d.crc_errors + d.bad_headers for d in self.decoders)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class SilentReceiver:
    """Accepts NDTP connections and never reads them: a hung receiver whose socket buffers fill up."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        # a small receive buffer (inherited by accepted sockets) keeps the TCP window tiny
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._accept())

    async def _accept(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            conn, _ = await loop.sock_accept(self.sock)
            self.conns.append(conn)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait({self._task})
        for conn in self.conns:
            conn.close()
        self.sock.close()


class FakeHttp:
    """Records emulator requests; ``GET`` fails ``ready_after`` times before the emulator is up."""

    def __init__(self, ready_after: int = 0) -> None:
        self.ready_after = ready_after
        self.gets = 0
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.times: list[float] = []

    async def get(self, url: str) -> int:
        self.gets += 1
        if self.gets <= self.ready_after:
            raise ConnectionRefusedError("emulator is starting")
        return 200

    async def post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, str]:
        self.posts.append((url, json.loads(json.dumps(payload))))
        self.times.append(time.monotonic())
        return 200, "{}"


class FlakyHttp(FakeHttp):
    """Fails the first posts with the given exceptions (none of them an ``OSError``), then works."""

    def __init__(self, *errors: Exception) -> None:
        super().__init__()
        self.errors = list(errors)

    async def post_json(self, url: str, payload: dict[str, Any]) -> tuple[int, str]:
        if self.errors:
            raise self.errors.pop(0)
        return await super().post_json(url, payload)


# ---- loading and selection -----------------------------------------------------------------------------


def test_load_keeps_raw_rows_sorted_by_receive_time(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path))
    assert len(data) == 7
    assert (np.diff(data.recv) >= 0).all()
    assert data.unit_id.tolist() == [1002, 1001, 1002, 1001, 1002, 1001, 1001]
    assert data.recv[1] == pytest.approx(_ts("07:00:01") + 0.25)
    # timestamp = event_time (whole seconds), so the late packet keeps its original 07:00:10
    assert data.event_s.tolist() == [
        _ts(t) for t in ("06:59:50", "07:00:00", "07:00:05", "07:00:20", "07:00:25", "07:00:30", "07:00:10")
    ]
    assert data.valid.tolist() == [True, True, True, True, False, False, True]
    assert data.hist.tolist() == [False] * 6 + [True]
    # invalid fix without coordinates -> zeros; invalid fix with coordinates -> sent as is
    assert (data.lon[5], data.lat[5]) == (0.0, 0.0)
    assert (data.lon[4], data.lat[4]) == (pytest.approx(37.52), pytest.approx(55.72))
    assert data.speed[3] == 368 and data.speed[5] == 0  # outliers are not cleaned
    assert data.course[1] == 0 and data.speed[1] == 25 and data.altitude[1] == 150
    assert data.unit_map() == {1001: 501, 1002: 502}
    assert data.day() == date(2026, 1, 6)
    nav = data.nav(1)
    assert nav.timestamp == _dt("07:00:00") and nav.valid and nav.lon == pytest.approx(37.6100001)


def test_select_by_units_and_time_window(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path))
    assert set(data.select(units=[502]).unit_id.tolist()) == {1002}  # tr_id
    assert set(data.select(units=[1001]).unit_id.tolist()) == {1001}  # unit_id
    after = data.select(start="07:00")
    assert len(after) == 6 and after.recv[0] >= _ts("07:00:00")
    window = data.select(start="07:00:05", until="07:00:30")
    assert window.event_s.tolist() == [_ts("07:00:05"), _ts("07:00:20"), _ts("07:00:25")]
    assert len(data.select(start=f"{DAY}T07:00:21")) == 4
    assert len(data.select(start="07:00:30", until="01:00")) == 2  # an earlier until means the next day
    assert parse_data_time("24:00", date(2026, 1, 6)) == _ts("00:00:00") + 86_400
    with pytest.raises(ValueError, match="unknown units"):
        data.select(units=[42])
    with pytest.raises(ValueError, match="no packets"):
        data.select(start="08:00")
    with pytest.raises(ValueError, match="bad time"):
        data.select(start="7h")
    with pytest.raises(ValueError, match="bad time of day"):
        parse_data_time("23:61", date(2026, 1, 6))


def test_loop_pass_leaves_out_stragglers(tmp_path: Path) -> None:
    # a packet from 07:00:27 that reached the server 2.5 h later, long after the last event_time (07:00:30)
    rows = [*ROWS, _row(1002, 502, "07:00:27", "09:30:00")]
    data = load_replay_data("test", _dataset(tmp_path, rows))
    assert len(select_replay(data, ReplayConfig())) == 8  # --once replays the window exactly
    looped = select_replay(data, ReplayConfig(loop=True))
    assert len(looped) == 7 and looped.recv[-1] == pytest.approx(_ts("07:00:40"))
    assert len(select_replay(data, ReplayConfig(loop=True, until="10:00"))) == 8  # an explicit until wins
    assert len(select_replay(data, ReplayConfig(loop=True, start="07:00:41"))) == 1  # never empty
    assert len(select_replay(data, ReplayConfig(loop=True, units=(1001,)))) == 4


def test_schedule_mapping_and_config_validation() -> None:
    schedule = ReplaySchedule(data0=1000.0, mono0=10.0, speed=30)
    assert schedule.mono_at(1060) == pytest.approx(12.0) and schedule.data_at(12.0) == pytest.approx(1060)
    schedule.set_speed(12.0, 60)  # no jump of the data clock
    assert schedule.data_at(12.0) == pytest.approx(1060) and schedule.mono_at(1120) == pytest.approx(13.0)
    with pytest.raises(ValueError):
        schedule.set_speed(13.0, 0)
    for bad in (
        {"speed": 5000},
        {"split": "prod"},
        {"mode": "udp"},
        {"on_disconnect": "retry"},
        {"max_queue": 0},
        {"bridge_interval_s": 0.5},
    ):
        with pytest.raises(ValueError):
            ReplayConfig(**bad).validate()


# ---- NDTP replay ----------------------------------------------------------------------------------------


def test_replay_sends_valid_ndtp_per_device(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00")

    async def scenario() -> tuple[StubReceiver, Replayer]:
        stub = StubReceiver()
        await stub.start()
        replayer = Replayer(data, ReplayConfig(port=stub.port, speed=100), clock=FakeClock())
        await replayer.run()
        await _until(lambda: len(stub.frames) == 2 + 6)
        await stub.stop()
        return stub, replayer

    stub, replayer = _run(scenario())
    assert replayer.state == "finished"
    assert stub.crc_errors == 0
    connections = stub.by_connection()
    assert len(connections) == 2  # one TCP connection per device
    for frames in connections.values():
        unit = frames[0].unit_id
        assert frames[0].is_handshake and decode_handshake(frames[0]).peer_address == unit
        assert all(f.is_realtime and f.unit_id == unit for f in frames[1:])
        assert [f.nph.request_id for f in frames] == list(range(1, len(frames) + 1))

    first = stub.packets(1001)
    assert [p.nav.timestamp for p in first if p.nav] == [
        _dt(t) for t in ("07:00:00", "07:00:20", "07:00:30", "07:00:10")
    ]
    navs = [p.nav for p in first]
    assert all(n is not None for n in navs)
    assert [n.valid for n in navs if n] == [True, True, False, True]
    n0 = navs[0]
    assert n0 is not None
    assert n0.lon == pytest.approx(37.6100001, abs=1e-7) and n0.lat == pytest.approx(55.7500002, abs=1e-7)
    assert (n0.speed_avg, n0.course, n0.altitude) == (25, 0, 150)
    assert navs[1] is not None and navs[1].speed_avg == 368
    assert navs[2] is not None and (navs[2].lon, navs[2].lat) == (0.0, 0.0)

    second = [p.nav for p in stub.packets(1002)]
    assert [n.timestamp for n in second if n] == [_dt("07:00:05"), _dt("07:00:25")]
    assert second[1] is not None and second[1].valid is False and second[1].lon == pytest.approx(37.52)

    status = replayer.status()
    assert (status["packets_sent"], status["packets_dropped"], status["backlog"]) == (6, 0, 0)
    assert status["packets_total"] == 6 and status["progress"] == 1.0
    assert status["data_time"] == _dt("07:00:40")


def _pace_rows() -> list[dict[str, Any]]:
    return [
        _row(1001, 501, "07:00:00", "07:00:00"),
        _row(1002, 502, "07:00:25", "07:00:30"),
        _row(1001, 501, "07:01:30", "07:01:30"),
        _row(1002, 502, "07:01:30", "07:01:30.500000"),
        _row(1001, 501, "07:03:00", "07:03:00"),
    ]


def test_pace_follows_receive_time_without_drift(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path, _pace_rows()))
    dispatched: list[tuple[int, float]] = []
    clock = FakeClock()

    async def scenario() -> None:
        stub = StubReceiver()
        await stub.start()
        replayer = Replayer(
            data,
            ReplayConfig(port=stub.port, speed=30),
            clock=clock,
            on_dispatch=lambda i, t: dispatched.append((i, t)),
        )
        await replayer.run()
        await _until(lambda: len(stub.frames) == 2 + 5)
        await stub.stop()

    _run(scenario())
    assert [i for i, _ in dispatched] == [0, 1, 2, 3, 4]  # global receive_time order
    assert [t - clock.t0 for _, t in dispatched] == pytest.approx([0, 1, 3, 3 + 0.5 / 30, 6], abs=1e-6)
    assert clock.t - clock.t0 == pytest.approx(6, abs=1e-6)  # slept exactly the schedule, no drift
    assert max(clock.sleeps) <= Replayer.MAX_IDLE_S + 1e-9


def test_speed_change_reanchors_the_schedule(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path, _pace_rows()))
    dispatched: list[tuple[int, float]] = []
    events: list[tuple[str, dict[str, Any]]] = []
    clock = FakeClock()
    holder: dict[str, Replayer] = {}

    def on_dispatch(i: int, t: float) -> None:
        dispatched.append((i, t))
        if i == 1:
            holder["r"].set_speed(60)

    async def scenario() -> None:
        stub = StubReceiver()
        await stub.start()
        holder["r"] = Replayer(
            data,
            ReplayConfig(port=stub.port, speed=30),
            clock=clock,
            on_dispatch=on_dispatch,
            on_event=lambda e, d: events.append((e, d)),
        )
        await holder["r"].run()
        await stub.stop()

    _run(scenario())
    assert [t - clock.t0 for _, t in dispatched] == pytest.approx([0, 1, 2, 2 + 0.5 / 60, 3.5], abs=1e-6)
    assert ("speed", {"speed": 60, "previous": 30}) in events
    assert [e for e, _ in events] == ["start", "speed", "finish"]


def test_waits_for_the_receiver_at_start(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00", units=[1002])
    port = _free_port()

    async def scenario() -> tuple[Replayer, StubReceiver]:
        config = ReplayConfig(port=port, speed=1000, backoff_initial_s=0.02, backoff_max_s=0.05)
        replayer = Replayer(data, config)
        task = asyncio.create_task(replayer.run())
        await asyncio.sleep(0.3)
        assert replayer.state == "waiting"
        assert replayer.stats.packets_dispatched == 0 and replayer.stats.connect_failures >= 1
        stub = StubReceiver()
        await stub.start(port)
        await asyncio.wait_for(task, 5)
        await _until(lambda: len(stub.frames) == 3)
        await stub.stop()
        return replayer, stub

    replayer, stub = _run(scenario())
    assert replayer.state == "finished"
    assert [p.nav.timestamp for p in stub.packets(1002) if p.nav] == [_dt("07:00:05"), _dt("07:00:25")]


def test_reconnects_after_a_drop_and_sends_the_backlog(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00", units=[1001])

    async def scenario() -> tuple[Replayer, StubReceiver]:
        stub = StubReceiver()
        await stub.start()
        dropped: list[int] = []

        def on_frame(conn: int, frame: Frame) -> None:
            if frame.is_realtime and not dropped:  # the receiver closes the link after the first packet
                dropped.append(conn)
                stub.drop(conn)

        stub.on_frame = on_frame
        config = ReplayConfig(port=stub.port, speed=5, backoff_initial_s=0.01, backoff_max_s=0.02)
        replayer = Replayer(data, config, clock=FakeClock(real_step=0.02))
        await replayer.run()
        await _until(lambda: len(stub.packets(1001)) == 4)
        await stub.stop()
        return replayer, stub

    replayer, stub = _run(scenario())
    assert replayer.state == "finished"
    assert len(stub.handshakes(1001)) == 2  # a new handshake after reconnecting
    stamps = [p.nav.timestamp for p in stub.packets(1001) if p.nav]
    assert stamps == [_dt(t) for t in ("07:00:00", "07:00:20", "07:00:30", "07:00:10")]  # nothing lost
    connections = list(stub.by_connection().values())
    assert [[f.nph.request_id for f in c] for c in connections] == [[1, 2], [3, 4, 5, 6]]
    assert replayer.stats.reconnects == 1 and replayer.stats.disconnects == 1
    assert replayer.status()["reconnects"] == 1
    assert stub.crc_errors == 0


def test_loop_restarts_over_the_same_connections(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00", units=[1002])
    events: list[str] = []

    async def scenario() -> tuple[Replayer, StubReceiver]:
        stub = StubReceiver()
        await stub.start()
        replayer = Replayer(
            data,
            ReplayConfig(port=stub.port, speed=100, loop=True),
            clock=FakeClock(real_step=0.005),
            on_event=lambda e, d: events.append(e),
        )
        task = asyncio.create_task(replayer.run())
        await _until(lambda: events.count("restart") >= 2 and len(stub.packets(1002)) >= 4)
        await replayer.stop()
        assert task.done()
        await stub.stop()
        return replayer, stub

    replayer, stub = _run(scenario())
    assert replayer.state == "stopped" and replayer.cycle >= 3
    assert events[0] == "start" and events[-1] == "stop" and "finish" not in events
    assert len(stub.handshakes(1002)) == 1
    stamps = [p.nav.timestamp for p in stub.packets(1002) if p.nav][:4]
    assert stamps == [_dt("07:00:05"), _dt("07:00:25")] * 2


def _single_packet(tmp_path: Path) -> ReplayData:
    # a window of zero length: one packet (1001, event 07:00:20, received 07:00:21)
    data = load_replay_data("test", _dataset(tmp_path))
    data = data.select(units=[1001], start="07:00:21", until="07:00:22")
    assert len(data) == 1
    return data


def test_loop_over_a_zero_length_window_is_paced_and_stops(tmp_path: Path) -> None:
    data = _single_packet(tmp_path)
    clock = FakeClock(real_step=0.001)
    restarts: list[int] = []

    def on_event(event: str, details: dict[str, Any]) -> None:
        if event == "restart":
            restarts.append(details["cycle"])
            if len(restarts) > 1000:  # a regression spins here without ever awaiting: fail instead of hanging
                raise RuntimeError("loop restarts without a pause")

    async def scenario() -> tuple[Replayer, StubReceiver, float]:
        stub = StubReceiver()
        await stub.start()
        config = ReplayConfig(port=stub.port, speed=30, loop=True)
        replayer = Replayer(data, config, clock=clock, on_event=on_event)
        task = asyncio.create_task(replayer.run())
        await _until(lambda: (replayer.cycle >= 5 and len(stub.packets(1001)) >= 4) or task.done())
        started = time.monotonic()
        await replayer.stop()
        stop_s = time.monotonic() - started
        await stub.stop()
        return replayer, stub, stop_s

    replayer, stub, stop_s = _run(scenario())
    assert replayer.state == "stopped" and replayer.error is None
    assert 5 <= replayer.cycle < 1000 and stop_s < 2.0
    # every pass waits LOOP_GAP_S after its only packet, so passes are bounded by (virtual) time
    assert (replayer.cycle - 1) * Replayer.LOOP_GAP_S <= clock.t - clock.t0 + 1e-6
    assert max(clock.sleeps) <= Replayer.MAX_IDLE_S + 1e-9
    assert {p.nav.timestamp for p in stub.packets(1001) if p.nav} == {_dt("07:00:20")}
    assert len(stub.handshakes(1001)) == 1


def test_controller_stays_responsive_in_a_zero_length_loop(tmp_path: Path) -> None:
    root = _dataset(tmp_path)

    async def scenario() -> tuple[dict[str, Any], float, int]:
        stub = StubReceiver()
        await stub.start()
        loader = functools.partial(load_replay_data, root=root)
        ctl = ReplayController(ReplayConfig(port=stub.port, speed=30), loader=loader)
        await ctl.start(units=[1001], start="07:00:21", until="07:00:22", loop=True)
        await _until(lambda: ctl.epoch >= 1)
        started = time.monotonic()
        await _until(lambda: ctl.epoch >= 3, timeout=10)  # this polling needs the event loop to be free
        spent = time.monotonic() - started
        status = await ctl.stop()
        await stub.stop()
        return status, spent, len(stub.packets(1001))

    status, spent, packets = _run_guarded(scenario())
    assert status["state"] == "stopped" and status["packets_total"] == 1
    assert spent >= 2 * Replayer.LOOP_GAP_S - 0.05  # two restarts, one gap each, in real time
    assert status["epoch"] <= 5 and packets >= 2


def test_skip_mode_drops_the_packets_of_an_outage(tmp_path: Path) -> None:
    rows = [
        _row(1001, 501, f"07:00:{s:02d}", f"07:00:{s:02d}", lon=37.6 + s / 1000) for s in (0, 10, 20, 30, 40)
    ]
    data = load_replay_data("test", _dataset(tmp_path, rows))

    async def scenario() -> tuple[Replayer, StubReceiver]:
        stub = StubReceiver()
        await stub.start()
        port = stub.port
        down: list[bool] = []
        restarted: list[asyncio.Task[None]] = []

        def on_frame(conn: int, frame: Frame) -> None:
            if frame.is_realtime and not down:  # the receiver goes down after the first packet
                down.append(True)
                stub.drop(conn)
                assert stub.server is not None
                stub.server.close()

        def on_dispatch(i: int, _t: float) -> None:
            if i == 1:  # dispatched during the outage (dropped by the skip policy): bring the receiver back
                restarted.append(asyncio.get_running_loop().create_task(stub.start(port)))

        stub.on_frame = on_frame
        config = ReplayConfig(
            port=port, speed=2, on_disconnect="skip", backoff_initial_s=0.01, backoff_max_s=0.02
        )
        replayer = Replayer(data, config, clock=FakeClock(real_step=0.02), on_dispatch=on_dispatch)
        await replayer.run()
        await _until(lambda: len(stub.packets(1001)) == 4)
        await stub.stop()
        return replayer, stub

    replayer, stub = _run(scenario())
    assert replayer.state == "finished"
    stamps = [p.nav.timestamp for p in stub.packets(1001) if p.nav]
    assert stamps == [_dt(t) for t in ("07:00:00", "07:00:20", "07:00:30", "07:00:40")]  # 07:00:10 skipped
    assert replayer.stats.dropped == {"disconnected": 1}
    assert replayer.stats.connect_failures >= 1 and replayer.stats.reconnects == 1
    assert len(stub.handshakes(1001)) == 2
    status = replayer.status()
    assert (status["packets_sent"], status["packets_dropped"]) == (4, 1)
    assert stub.crc_errors == 0


def test_flapping_receiver_gets_no_frames_and_is_not_taken_as_up() -> None:
    async def scenario() -> tuple[UnitLink, ReplayStats, int, int]:
        accepted: list[int] = []
        up: list[int] = []

        async def flap(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            accepted.append(1)
            writer.close()  # accepts and closes at once, like a receiver that rejects the device

        server = await asyncio.start_server(flap, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        stats = ReplayStats()
        settings = LinkSettings("127.0.0.1", port, backoff_initial_s=0.01, backoff_max_s=0.02)
        link = UnitLink(7, settings, stats=stats, on_connected=up.append)
        for k in range(200):
            link.enqueue(NavRecord(timestamp=_dt("07:00:00"), lon=37.6, lat=55.7 + k / 1e4), 0.0)
        link.start()
        await _until(lambda: link.connections >= 5)
        await link.close()
        server.close()
        await server.wait_closed()
        return link, stats, len(accepted), len(up)

    link, stats, accepted, up = _run(scenario())
    assert link.connections >= 5 and accepted >= 5 and stats.handshakes >= 5
    # the close arrives during the settle pause after the handshake: no frame is written into the dying
    # connection, the backlog is kept, and the receiver never counts as up
    assert link.sent == 0 and stats.packets_sent == 0 and up == 0
    assert stats.dropped == {"stopped": 200}  # still queued when the link was closed
    assert link.last_error is not None and link.last_error.startswith("ConnectionResetError")


def test_hung_receiver_times_out_reconnects_and_keeps_the_queue_bounded() -> None:
    max_queue = 50

    async def scenario() -> tuple[UnitLink, ReplayStats, int]:
        receiver = SilentReceiver()
        receiver.start()
        stats = ReplayStats()
        settings = LinkSettings(
            "127.0.0.1",
            receiver.port,
            max_queue=max_queue,
            write_timeout_s=0.2,
            backoff_initial_s=0.01,
            backoff_max_s=0.02,
        )
        link = UnitLink(7, settings, stats=stats)
        link.start()
        nav = NavRecord(timestamp=_dt("07:00:00"), lon=37.6, lat=55.7)
        peak = 0
        deadline = time.monotonic() + 20
        while link.connections < 2 and time.monotonic() < deadline:
            for _ in range(10):  # faster than the link can write: the queue overflows
                link.enqueue(nav, 0.0)
            peak = max(peak, link.pending)
            await asyncio.sleep(0)
        await link.close()
        await receiver.stop()
        return link, stats, peak

    link, stats, peak = _run(scenario())
    assert link.connections >= 2 and stats.disconnects >= 1  # the write timeout dropped the connection
    assert link.last_error is not None and "did not take a frame" in link.last_error
    assert link.sent > 0 and stats.handshakes >= 2
    assert peak <= max_queue + 1  # the queue plus one frame in flight
    assert stats.dropped["overflow"] > 0
    assert stats.connections_active == 0 and not link.connected


def test_link_queue_is_bounded_and_lag_is_measured(tmp_path: Path) -> None:
    stats = ReplayStats()
    clock = FakeClock(t0=100.0)
    link = UnitLink(7, LinkSettings("127.0.0.1", 1, max_queue=3), stats=stats, clock=clock)
    navs = [NavRecord(timestamp=_dt(f"07:00:0{k}"), lon=37.6, lat=55.7) for k in range(5)]
    for k, nav in enumerate(navs):
        link.enqueue(nav, due=float(k))
    assert link.pending == 3 and link.dropped == 2 and stats.dropped == {"overflow": 2}
    assert link.oldest_due() == 2.0  # the oldest packets were dropped

    skip = UnitLink(8, LinkSettings("127.0.0.1", 1, buffer_while_down=False), stats=stats)
    skip.enqueue(navs[0], 0.0)
    assert skip.pending == 0 and stats.dropped["disconnected"] == 1

    data = load_replay_data("test", _dataset(tmp_path))
    replayer = Replayer(data, ReplayConfig(port=1), clock=clock)
    replayer.links[1001].enqueue(data.nav(0), due=clock.t - 5.0)
    assert replayer.lag_s() == pytest.approx(5.0) and replayer.backlog() == 1


# ---- emulator bridge ------------------------------------------------------------------------------------


def test_emulator_config_shape() -> None:
    ts = _dt("07:00:00")
    east = NavRecord(timestamp=ts, lon=37.617321, lat=55.7551234, speed_avg=33, course=270, altitude=150)
    west = NavRecord(timestamp=ts, lon=-0.1276, lat=-33.87, valid=False)
    config = build_emulator_config({2004: east, 7: west}, "ingest", 9201, 5000)
    assert config["targetHost"] == "ingest" and config["targetPort"] == 9201
    assert [u["unitId"] for u in config["units"]] == [7, 2004]
    unit = config["units"][1]
    assert unit["intervalMs"] == 5000 and unit["autoGenerate"] is False
    # values only inside "fields": flat keys next to "type" are ignored by the emulator
    assert unit["cells"] == [
        {
            "type": "G6CellNav00",
            "fields": {
                "longitude": 376173210,
                "latitude": 557551234,
                "extraDopBit5": True,
                "extraDopBit6": True,
                "extraDopBit7": True,
                "speedAvg": 33,
                "course": 270,
                "altitude": 150,
            },
        }
    ]
    fields = config["units"][0]["cells"][0]["fields"]
    assert (fields["longitude"], fields["latitude"]) == (1_276_000, 338_700_000)  # unsigned
    assert (fields["extraDopBit5"], fields["extraDopBit6"], fields["extraDopBit7"]) == (False, False, False)
    json.dumps(config)


def test_bridge_waits_for_emulator_and_skips_unchanged_configs() -> None:
    http = FakeHttp(ready_after=2)
    bridge = EmulatorBridge("http://emu:18080/", "ingest", 9201, 1.0, http=http, ready_retry_s=0.01)
    here = NavRecord(timestamp=_dt("07:00:00"), lon=37.6, lat=55.7)
    there = NavRecord(timestamp=_dt("07:00:10"), lon=37.7, lat=55.7)

    async def scenario() -> list[bool]:
        await bridge.wait_ready()
        results = [await bridge.push({1: here}), await bridge.push({1: here}), await bridge.push({1: there})]
        await bridge.stop_emulation()
        return results

    assert _run(scenario()) == [True, False, True]
    assert http.gets == 3
    assert [url for url, _ in http.posts] == ["http://emu:18080/api/config"] * 3
    assert http.posts[-1][1] == {"targetHost": "ingest", "targetPort": 9201, "units": []}
    with pytest.raises(ValueError):
        EmulatorBridge("http://emu", "ingest", 9201, 0.5)


def test_bridge_mode_posts_latest_positions(tmp_path: Path) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00")
    http = FakeHttp()
    config = ReplayConfig(mode="bridge", speed=1000, host="ingest", port=9201, bridge_interval_s=1)

    async def scenario() -> Replayer:
        replayer = Replayer(data, config, http=http)
        await replayer.run()
        return replayer

    replayer = _run(scenario())
    assert replayer.state == "finished" and replayer.status()["bridge_posts"] >= 1
    configs = [payload for _, payload in http.posts]
    assert configs[-1]["units"] == []  # emulation is stopped at the end...
    # ...but only after the last positions were kept for one interval, so the emulator sent them
    assert http.times[-1] - http.times[-2] >= config.bridge_interval_s - 0.05
    final = configs[-2]
    assert (final["targetHost"], final["targetPort"]) == ("ingest", 9201)
    fields = {u["unitId"]: u["cells"][0]["fields"] for u in final["units"]}
    # 1001: the newest fix by event_time is 07:00:30 (invalid) although the late 07:00:10 arrived after it
    assert fields[1001]["extraDopBit7"] is False and fields[1001]["longitude"] == 0
    assert fields[1002]["longitude"] == 375_200_000 and fields[1002]["extraDopBit7"] is False


def test_bridge_survives_emulator_errors() -> None:
    # a broken response while the emulator restarts, a client bug, then a snapshot failure: none is an OSError
    http = FlakyHttp(BadStatusLine("garbage"), RuntimeError("client bug"))
    stats = ReplayStats()
    clock = FakeClock()
    bridge = EmulatorBridge("http://emu:18080", "ingest", 9201, 5.0, http=http, stats=stats, clock=clock)
    here = NavRecord(timestamp=_dt("07:00:00"), lon=37.6, lat=55.7)
    calls: list[int] = []

    def snapshot() -> dict[int, NavRecord]:
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("snapshot bug")
        return {1: here}

    async def scenario() -> bool:
        task = asyncio.create_task(bridge.run(snapshot))
        await _until(lambda: len(http.posts) == 1 and len(calls) >= 5)
        alive = not task.done()
        task.cancel()
        await asyncio.wait({task})
        return alive

    assert _run(scenario()) is True  # the bridge keeps running after every failure
    assert stats.bridge_errors == 3 and stats.bridge_posts == 1 and bridge.posts == 1
    assert clock.sleeps[:4] == [5.0] * 4  # retried on the next period


def test_urllib_client_reports_a_broken_response_as_oserror() -> None:
    class Garbage(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.recv(65536)
            self.request.sendall(b"NOT-HTTP\r\n\r\n")

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Garbage)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        client = UrllibClient(timeout_s=2.0)
        with pytest.raises(OSError, match="BadStatusLine"):
            _run(client.post_json(f"{base}/api/config", {"units": []}))
        stats = ReplayStats()
        bridge = EmulatorBridge(base, "ingest", 9201, 1.0, http=client, stats=stats)
        here = NavRecord(timestamp=_dt("07:00:00"), lon=37.6, lat=55.7)
        assert _run(bridge.push({1: here})) is False and stats.bridge_errors == 1
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_mode_fails_when_the_bridge_task_dies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = load_replay_data("test", _dataset(tmp_path)).select(start="07:00")

    async def broken_run(self: EmulatorBridge, snapshot: Callable[[], Any]) -> None:
        raise RuntimeError("bridge bug")

    monkeypatch.setattr(EmulatorBridge, "run", broken_run)
    http = FakeHttp()

    async def scenario() -> Replayer:
        config = ReplayConfig(mode="bridge", speed=1, bridge_interval_s=1)
        replayer = Replayer(data, config, http=http, clock=FakeClock(real_step=0.001))
        await replayer.run()
        return replayer

    replayer = _run(scenario())
    assert replayer.state == "failed" and "bridge bug" in (replayer.error or "")
    assert http.posts[-1][1]["units"] == []  # the emulator is still cleared


# ---- control API and CLI --------------------------------------------------------------------------------


def test_control_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("prometheus_client")
    from fastapi.testclient import TestClient

    from replayer.api import create_app

    root = _dataset(tmp_path)
    receiver = ThreadedReceiver()
    try:
        app = create_app(
            ReplayConfig(port=receiver.port, speed=1000),
            loader=functools.partial(load_replay_data, root=root),
        )
        with TestClient(app) as client:

            def status() -> dict[str, Any]:
                return client.get("/replay/status").json()

            assert client.get("/health").json() == {"status": "ok", "state": "idle", "epoch": 0}
            assert status()["state"] == "idle" and status()["speed"] == 1000

            resp = client.post(
                "/replay/start", json={"split": "test", "speed": 500, "start": "07:00", "units": [501]}
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["units_filter"] == [501] and resp.json()["speed"] == 500
            _wait_sync(lambda: status()["state"] == "finished")
            st = status()
            assert (st["packets_total"], st["packets_sent"], st["epoch"]) == (4, 4, 1)
            assert [e["event"] for e in st["events"]] == ["start", "finish"]
            _wait_sync(lambda: receiver.count(1001) == 5)  # handshake + 4 packets
            assert receiver.crc_errors() == 0

            st = client.post(
                "/replay/start", json={"loop": True, "speed": 1, "units": [], "start": ""}
            ).json()
            assert st["loop"] is True and st["units_filter"] == [] and st["start"] is None
            _wait_sync(lambda: status()["state"] == "running")
            assert client.post("/replay/speed", json={"speed": 120}).json()["speed"] == 120
            st = client.post("/replay/stop").json()
            assert st["state"] == "stopped" and st["epoch"] == 2
            assert [e["event"] for e in st["events"]][-3:] == ["start", "speed", "stop"]

            metrics = client.get("/metrics").text
            assert "foresight_replayer_packets_sent_total" in metrics
            assert "foresight_replayer_epoch 2.0" in metrics
            assert 'foresight_replayer_packets_dropped_total{reason="overflow"} 0.0' in metrics

            assert client.post("/replay/start", json={"split": "train"}).status_code == 404
            assert client.post("/replay/start", json={"start": "99:99"}).status_code == 400
            assert client.post("/replay/start", json={"units": [42]}).status_code == 400
            assert client.post("/replay/speed", json={"speed": 0}).status_code == 422
            assert status()["state"] == "stopped"  # failed starts leave the last session alone
            assert client.get("/openapi.json").json()["info"]["title"] == "Foresight · Replayer"
    finally:
        receiver.stop()


def test_cli_run_once(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    receiver = ThreadedReceiver()
    try:
        args = ["run", "--dataset-dir", str(root), "--split", "test", "--speed", "1000"]
        args += [
            "--port",
            str(receiver.port),
            "--units",
            "1002",
            "--start",
            "07:00",
            "--once",
            "--report-every",
            "0",
        ]
        assert cli_main(args) == 0
        _wait_sync(lambda: receiver.count(1002) == 3)
        assert receiver.crc_errors() == 0
        assert cli_main(["run", "--dataset-dir", str(root), "--start", "23:99", "--port", "1"]) == 2
    finally:
        receiver.stop()
