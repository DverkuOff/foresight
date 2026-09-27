"""Tests for the NDTP codec (shared.ndtp), including frames captured from the official emulator."""

from datetime import UTC, datetime

import pytest

from shared import ndtp
from shared.ndtp import (
    Cell,
    FrameDecoder,
    IrmaRecord,
    NavRecord,
    NdtpError,
    crc16_modbus,
    decode_frame,
    decode_handshake,
    decode_realtime,
    encode_handshake,
    encode_irma,
    encode_realtime,
)

# Captured from ndtp-telemetry-emulator:1.0 (unitId 1, autoGenerate): handshake and two realtime packets
# with Nav00 + Usi08 + Termo16 + IntSensor02 + Can10.
EMU_HANDSHAKE = bytes.fromhex(
    "7e7e1c0000002d62020100000000000000640001000100000006000200000001000000ffff000000000000"
)
EMU_REALTIME_1 = bytes.fromhex(
    "7e7e7b00000081b502010000000000010065000100020000000000b04bb66a050f5a16242a3321e07510001a0047014c1896"
    "000e020800018702e9000f100000000000090000000200c601be00c4010601030506001500050008004c1800001401030d0a"
    "00b0d2010054680000b07d0900a90100005700e4084700107102de02ec02dd02fa0200000000"
)
EMU_REALTIME_2 = bytes.fromhex(
    "7e7e7b000000587002010000000000010065000100030000000000b14bb66a730e5a162a2b3321e07618001b004b01f41899"
    "000e020800019402e3000e10000000000006000000020053025d010402eb000e040800160007000900f41800001901000d0a"
    "00e3a101005868000050bf0900ab0100001e005308470018b8027f0225039e02ab0200000000"
)
# unitId 2004: Nav00 (lon 376173210, lat 557551234, N/E/valid, speedAvg 33, course 270) + Irma04 with
# odometer 0x01020304, zone 0x1234, in 200,2,3,4, out 5,6,7,250, present 1,1,0,1, closed 1,0,0,0.
EMU_IRMA = bytes.fromhex(
    "7e7e37000000804702d40700000000010065000100020000000000f24bb66a9af26b16828e3b21e000210000000e0100000000"
    "00000400040302013412c8020304050607fa1b"
)


def test_crc16_modbus_check_vector() -> None:
    assert crc16_modbus(b"123456789") == 0x4B37
    assert crc16_modbus(b"") == 0xFFFF


def test_emulator_handshake() -> None:
    frame = decode_frame(EMU_HANDSHAKE)
    assert frame.is_handshake and not frame.is_realtime
    assert frame.unit_id == 1
    assert frame.nph.request_id == 1
    hs = decode_handshake(frame)
    assert (hs.proto_version_high, hs.proto_version_low) == (6, 2)
    assert hs.peer_address == 1
    assert hs.max_packet_size == 65535


def test_emulator_realtime_autogenerate() -> None:
    frame = decode_frame(EMU_REALTIME_1)
    assert frame.is_realtime
    packet = decode_realtime(frame)
    assert packet.unit_id == 1
    assert packet.request_id == 2
    assert packet.cell_types == (0, 8, 16, 2, 10)
    assert packet.unknown_cell_type is None and not packet.truncated
    nav = packet.nav
    assert nav is not None
    assert nav.valid
    assert 55.6 < nav.lat < 55.8 and 37.4 < nav.lon < 37.6
    assert nav.timestamp.tzinfo is not None and nav.timestamp.year >= 2026


def test_emulator_irma_layout() -> None:
    packet = decode_realtime(decode_frame(EMU_IRMA))
    assert packet.unit_id == 2004
    nav = packet.nav
    assert nav is not None
    assert nav.lon == pytest.approx(37.617321) and nav.lat == pytest.approx(55.7551234)
    assert (nav.speed_avg, nav.course, nav.valid) == (33, 270, True)
    (irma,) = packet.irma
    assert irma.odometer == 0x01020304 and irma.zone == 0x1234
    assert irma.door_in == (200, 2, 3, 4)
    assert irma.door_out == (5, 6, 7, 250)
    assert irma.door_present == (True, True, False, True)
    assert irma.door_closed == (True, False, False, False)
    # the encoder reproduces the emulator's bytes exactly
    assert encode_irma(irma).encode() == EMU_IRMA[-17:]


def _nav(**kw: object) -> NavRecord:
    base = dict(
        timestamp=datetime(2026, 9, 25, 7, 30, 0, tzinfo=UTC),
        lon=37.6173210,
        lat=55.7551234,
        valid=True,
        speed_avg=42,
        speed_max=55,
        course=181,
        track=1234,
        altitude=150,
        nsat=12,
        pdop=2,
        battery_mv=4000,
        aux_flags=0b00010,
    )
    base.update(kw)
    return NavRecord(**base)  # type: ignore[arg-type]


def test_roundtrip_handshake() -> None:
    frame = decode_frame(encode_handshake(123456, request_id=7))
    assert frame.unit_id == 123456 and frame.nph.request_id == 7
    assert decode_handshake(frame).peer_address == 123456


def test_roundtrip_realtime_with_extra_cells() -> None:
    nav = _nav()
    present, closed = (True, True, False, False), (False, True, False, True)
    irma = IrmaRecord(1, 99, 3, (1, 2, 3, 4), (4, 3, 2, 1), present, closed)
    usi = (8, 0, bytes(6))
    data = encode_realtime(42, 5, nav, [usi, encode_irma(irma), Cell(16, bytes(8))])
    packet = decode_realtime(decode_frame(data))
    assert packet.unit_id == 42 and packet.request_id == 5
    assert packet.nav == nav
    assert packet.irma == (irma,)
    assert packet.cell_types == (0, 8, 4, 16)


@pytest.mark.parametrize(
    ("lat", "lon", "flags"),
    [(55.75, 37.61, 0xE0), (-33.86, 151.2, 0xC0), (40.7, -74.0, 0xA0), (-22.9, -43.2, 0x80)],
)
def test_coordinate_sign_flags(lat: float, lon: float, flags: int) -> None:
    data = encode_realtime(1, 1, _nav(lat=lat, lon=lon, aux_flags=0))
    body = decode_frame(data).body
    assert body[2 + 12] == flags  # cell header (2) + offset of the flag byte in Nav00 (12)
    nav = decode_realtime(decode_frame(data)).nav
    assert nav is not None
    assert nav.lat == pytest.approx(lat) and nav.lon == pytest.approx(lon)


def test_invalid_fix_flag() -> None:
    nav = decode_realtime(decode_frame(encode_realtime(1, 1, _nav(valid=False)))).nav
    assert nav is not None and not nav.valid


def test_decode_frame_rejects_bad_crc() -> None:
    bad = bytearray(EMU_REALTIME_1)
    bad[-1] ^= 0xFF
    with pytest.raises(NdtpError):
        decode_frame(bytes(bad))


def test_unknown_cell_stops_parsing_but_keeps_nav() -> None:
    data = encode_realtime(7, 1, _nav(), [(8, 0, bytes(6)), (99, 0, b"\x01\x02\x03"), (16, 0, bytes(8))])
    packet = decode_realtime(decode_frame(data))
    assert packet.nav is not None
    assert packet.cell_types == (0, 8)
    assert packet.unknown_cell_type == 99


def test_truncated_cell() -> None:
    data = ndtp.encode_frame(7, ndtp.SERVICE_NAVDATA, ndtp.NPH_SND_REALTIME, 1, bytes((0, 0)) + bytes(10))
    packet = decode_realtime(decode_frame(data))
    assert packet.nav is None and packet.truncated


# ---- streaming decoder ----

STREAM = EMU_HANDSHAKE + EMU_REALTIME_1 + EMU_REALTIME_2 + EMU_IRMA


def test_decoder_whole_stream() -> None:
    dec = FrameDecoder()
    frames = dec.feed(STREAM)
    assert [f.nph.request_id for f in frames] == [1, 2, 3, 2]
    assert dec.buffered == 0 and dec.crc_errors == 0 and dec.bytes_discarded == 0


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 16, 25, 64])
def test_decoder_chunked(chunk: int) -> None:
    dec = FrameDecoder()
    frames = []
    for i in range(0, len(STREAM), chunk):
        frames += dec.feed(STREAM[i : i + chunk])
    expected = [decode_frame(x).body for x in (EMU_HANDSHAKE, EMU_REALTIME_1, EMU_REALTIME_2, EMU_IRMA)]
    assert [f.body for f in frames] == expected
    assert dec.buffered == 0


def test_decoder_garbage_and_bad_crc() -> None:
    bad = bytearray(EMU_REALTIME_1)
    bad[40] ^= 0x55
    garbage = b"\x00\x01\x7e\xffhello\x7e"  # includes lone 0x7E bytes
    false_sig = b"\x7e\x7e\xff\xff\x00\x00"  # signature with an implausible dataSize
    stream = garbage + EMU_HANDSHAKE + false_sig + bytes(bad) + b"junk" + EMU_REALTIME_2 + garbage + EMU_IRMA
    dec = FrameDecoder()
    frames = []
    for i in range(0, len(stream), 5):
        frames += dec.feed(stream[i : i + 5])
    assert [(f.unit_id, f.nph.request_id) for f in frames] == [(1, 1), (1, 3), (2004, 2)]
    assert dec.crc_errors == 1
    assert dec.bad_headers >= 1
    assert dec.bytes_discarded > 0


def test_decoder_waits_for_partial_frame() -> None:
    dec = FrameDecoder()
    assert dec.feed(EMU_REALTIME_1[:30]) == []
    assert dec.buffered == 30
    (frame,) = dec.feed(EMU_REALTIME_1[30:])
    assert frame.nph.request_id == 2


def _false_header(size: int = 8000) -> bytes:
    """A plausible NPL header (signature, sane dataSize, type=NPH) with a wrong CRC, padded with zeros."""
    header = bytearray(15)
    header[0:2] = b"\x7e\x7e"
    header[2:4] = size.to_bytes(2, "little")
    header[8] = 2
    return bytes(header) + bytes(size)


def test_decoder_flood_of_false_headers_is_cut_off_quickly() -> None:
    import time

    good = encode_realtime(1, 1, _nav())
    # false headers every 64 bytes: without a budget each one costs a CRC over 8 KB
    flood = b"".join(_false_header(8000)[:64] for _ in range(4000)) + bytes(8000)
    dec = FrameDecoder(max_data_size=8192, max_crc_errors=32)
    started = time.perf_counter()
    frames = dec.feed(good + flood + good)
    elapsed = time.perf_counter() - started
    assert len(frames) == 1  # the frame before the flood survives
    assert dec.abusive
    assert dec.crc_errors == 33
    assert elapsed < 1.0
    assert dec.feed(good) == []  # an abusive connection is ignored until closed
    assert dec.buffered == 0


def test_decoder_sporadic_crc_errors_do_not_trip_budget() -> None:
    good = encode_realtime(1, 1, _nav())
    bad = bytearray(good)
    bad[-1] ^= 0xFF
    dec = FrameDecoder(max_crc_errors=2)
    frames = []
    for _ in range(50):  # two bad frames between good ones, many times over
        frames += dec.feed(bytes(bad) + bytes(bad) + good)
    assert len(frames) == 50
    assert not dec.abusive
    assert dec.crc_errors == 100
