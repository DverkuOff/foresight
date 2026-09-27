"""NDTP protocol codec: frame parsing and building without third-party dependencies.

A frame on the wire is ``[NPL 15 bytes][NPH 10 bytes][body]``, all little-endian. ``NPL.dataSize`` is the
length of NPH + body; ``NPL.crc`` is CRC-16/Modbus over NPH + body stored with swapped bytes.

The module provides:

* dataclasses for headers and decoded payloads (:class:`Frame`, :class:`Handshake`, :class:`NavRecord`,
  :class:`IrmaRecord`, :class:`RealtimePacket`);
* :func:`crc16_modbus`;
* :class:`FrameDecoder` — a streaming decoder that accepts arbitrary byte chunks, yields whole frames,
  resynchronises on ``0x7E7E`` after garbage and drops (and counts) frames with a bad CRC;
* :func:`decode_handshake`, :func:`decode_realtime` and cell decoders;
* encoders :func:`encode_handshake` and :func:`encode_realtime` used by the replayer and tests.

Cell payload sizes were measured on the official emulator ``ndtp-telemetry-emulator:1.0`` (see
``CELL_SIZES``); the Irma04 bit layout was verified the same way.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

SIGNATURE = 0x7E7E
SIGNATURE_BYTES = b"\x7e\x7e"
NPL_SIZE = 15
NPH_SIZE = 10
NPL_TYPE_NPH = 0x02

SERVICE_GENERIC_CONTROLS = 0
SERVICE_NAVDATA = 1
NPH_SGC_CONN_REQUEST = 100
NPH_SND_REALTIME = 101
NPH_FLAG_REQUEST = 0x0001

PROTO_VERSION_HIGH = 6
PROTO_VERSION_LOW = 2
DEFAULT_MAX_PACKET_SIZE = 65535

CELL_NAV00 = 0
CELL_INT_SENSOR02 = 2
CELL_IRMA04 = 4

#: Payload size (without the 2-byte ``[type][number]`` header) for every cell type the emulator knows.
#: Measured on the official emulator; matches the specification where the specification gives a layout.
CELL_SIZES: dict[int, int] = {
    0: 26,  # G6CellNav00
    2: 26,  # G6CellIntSensor02
    3: 14,  # G6CellCrown03
    4: 15,  # G6CellIrma04
    5: 6,  # G6CellKdm05
    6: 9,  # G6CellIdn06
    7: 1,  # G6CellIdn07
    8: 6,  # G6CellUsi08
    9: 40,  # G6CellReg09
    10: 37,  # G6CellCan10
    12: 5,  # G6CellRfid12
    13: 13,  # G6CellPlo13
    14: 15,  # G6CellBms14
    15: 50,  # G6CellLls15
    16: 8,  # G6CellTermo16
    17: 46,  # G6CellAlcohol1st17
    18: 50,  # G6CellCAN18
    19: 40,  # G6CellGSMstations19
    20: 8,  # G6CellM333CAN20
    21: 180,  # G6CellAlcohol2nd21
    22: 24,  # G6CellServerStatistics22
    23: 16,  # G6CellTrackerStatistics23
    100: 44,  # G6CellZipSensorData100
}

_NPL = struct.Struct("<HHHHBIH")
_NPH = struct.Struct("<HHHI")
_HANDSHAKE = struct.Struct("<HHHIII")
_NAV00 = struct.Struct("<IIIBBHHHHHBB")
_IRMA04 = struct.Struct("<IH4B4BB")

_FLAG_NORTH = 1 << 5
_FLAG_EAST = 1 << 6
_FLAG_VALID = 1 << 7
_AUX_FLAGS_MASK = 0x1F
_COORD_SCALE = 10_000_000
_BATTERY_MV_PER_UNIT = 20


class NdtpError(ValueError):
    """Raised when bytes cannot be interpreted as a valid NDTP structure."""


def crc16_modbus(data: bytes | bytearray | memoryview) -> int:
    """Compute CRC-16/Modbus (poly 0xA001 reflected, init 0xFFFF).

    Args:
        data: Bytes to checksum.

    Returns:
        The CRC as an unsigned 16-bit integer (not byte-swapped).
    """
    crc = 0xFFFF
    table = _CRC_TABLE
    for byte in bytes(data):
        crc = (crc >> 8) ^ table[(crc ^ byte) & 0xFF]
    return crc


def _crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _crc_table()


def _swap16(value: int) -> int:
    return ((value & 0xFF) << 8) | (value >> 8)


# --------------------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NplHeader:
    """Network-layer header (15 bytes).

    Attributes:
        signature: Always ``0x7E7E``.
        data_size: Length of NPH header + body.
        flags: NPL flags (encryption, crc, delay); the emulator sends 0.
        crc: CRC-16/Modbus of NPH + body as a plain integer (already un-swapped).
        type: Payload type, ``0x02`` for NPH.
        peer_address: Device ``unitId``.
        request_id: NPL request id (0 from the emulator).
    """

    signature: int
    data_size: int
    flags: int
    crc: int
    type: int
    peer_address: int
    request_id: int


@dataclass(frozen=True, slots=True)
class NphHeader:
    """Service-layer header (10 bytes).

    Attributes:
        service_id: ``0`` generic controls, ``1`` navdata.
        type: ``100`` connection request (handshake), ``101`` realtime.
        flags: Bit 0 is the ``request`` flag.
        request_id: Packet counter starting at 1.
    """

    service_id: int
    type: int
    flags: int
    request_id: int


@dataclass(frozen=True, slots=True)
class Frame:
    """A complete NDTP frame with a verified CRC.

    Attributes:
        npl: Network-layer header.
        nph: Service-layer header.
        body: NPH body bytes (cells for realtime, 18 bytes for a handshake).
    """

    npl: NplHeader
    nph: NphHeader
    body: bytes

    @property
    def unit_id(self) -> int:
        """Device id taken from ``NPL.peerAddress``."""
        return self.npl.peer_address

    @property
    def is_handshake(self) -> bool:
        """Whether this frame is ``NPH_SGC_CONN_REQUEST``."""
        return self.nph.service_id == SERVICE_GENERIC_CONTROLS and self.nph.type == NPH_SGC_CONN_REQUEST

    @property
    def is_realtime(self) -> bool:
        """Whether this frame is ``NPH_SND_REALTIME``."""
        return self.nph.service_id == SERVICE_NAVDATA and self.nph.type == NPH_SND_REALTIME


@dataclass(frozen=True, slots=True)
class Handshake:
    """Decoded ``NPH_SGC_CONN_REQUEST`` body.

    Attributes:
        proto_version_high: Protocol major version (6).
        proto_version_low: Protocol minor version (2).
        flags: Connection flags.
        peer_address: Device ``unitId``.
        max_packet_size: Maximum packet size the device accepts.
    """

    proto_version_high: int
    proto_version_low: int
    flags: int
    peer_address: int
    max_packet_size: int


@dataclass(frozen=True, slots=True)
class NavRecord:
    """Decoded ``G6CellNav00`` navigation cell.

    Attributes:
        timestamp: Fix time, timezone-aware UTC (Unix seconds precision).
        lon: Longitude in degrees, negative for W.
        lat: Latitude in degrees, negative for S.
        valid: Whether coordinates are reliable (``extraDopBit7``).
        speed_avg: Average speed, km/h.
        speed_max: Maximum speed, km/h.
        course: Course, degrees 0..360.
        track: Distance travelled, m (mod 65535).
        altitude: Altitude, m.
        nsat: Number of satellites.
        pdop: PDOP.
        battery_mv: Battery voltage, mV (protocol unit is 20 mV).
        aux_flags: ``extraDopBit0..4`` as bits 0..4 (voice, alarm, SOS, first power-on, battery).
    """

    timestamp: datetime
    lon: float
    lat: float
    valid: bool = True
    speed_avg: int = 0
    speed_max: int = 0
    course: int = 0
    track: int = 0
    altitude: int = 0
    nsat: int = 0
    pdop: int = 0
    battery_mv: int = 0
    aux_flags: int = 0


@dataclass(frozen=True, slots=True)
class IrmaRecord:
    """Decoded ``G6CellIrma04`` (IRMA passenger counter and doors).

    Byte layout (15 bytes, verified on the emulator): ``odometer u32, zone u16, door_in1..4 u8,
    door_out1..4 u8`` and one bit byte, LSB first: ``present_door1..4``, ``closed_door1..4``.

    Attributes:
        number: Cell index among Irma04 cells of the packet.
        odometer: Odometer value.
        zone: Zone id.
        door_in: Passengers entered through doors 1..4.
        door_out: Passengers exited through doors 1..4.
        door_present: Whether door sensors 1..4 are present.
        door_closed: Whether doors 1..4 are closed.
    """

    number: int
    odometer: int
    zone: int
    door_in: tuple[int, int, int, int]
    door_out: tuple[int, int, int, int]
    door_present: tuple[bool, bool, bool, bool]
    door_closed: tuple[bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class RealtimePacket:
    """Decoded ``NPH_SND_REALTIME`` packet.

    Attributes:
        unit_id: Device id.
        request_id: NPH request id.
        nav: The first Nav00 cell, if present.
        irma: Irma04 cells.
        cell_types: Types of all cells decoded (or skipped by length) in order.
        unknown_cell_type: Type of the first unknown cell; parsing stopped there.
        truncated: Whether the body ended in the middle of a cell.
    """

    unit_id: int
    request_id: int
    nav: NavRecord | None = None
    irma: tuple[IrmaRecord, ...] = ()
    cell_types: tuple[int, ...] = ()
    unknown_cell_type: int | None = None
    truncated: bool = False


# --------------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------------


def _parse_headers(data: bytes | bytearray | memoryview, offset: int = 0) -> tuple[NplHeader, NphHeader]:
    sig, size, flags, crc_swapped, ptype, peer, req = _NPL.unpack_from(data, offset)
    npl = NplHeader(sig, size, flags, _swap16(crc_swapped), ptype, peer, req)
    nph = NphHeader(*_NPH.unpack_from(data, offset + NPL_SIZE))
    return npl, nph


def decode_frame(data: bytes) -> Frame:
    """Decode exactly one complete frame.

    Args:
        data: Frame bytes (NPL + NPH + body); trailing bytes are not allowed.

    Returns:
        The decoded frame.

    Raises:
        NdtpError: On bad signature, size mismatch or CRC mismatch.
    """
    if len(data) < NPL_SIZE + NPH_SIZE:
        raise NdtpError(f"frame too short: {len(data)} bytes")
    if data[:2] != SIGNATURE_BYTES:
        raise NdtpError("bad signature")
    size = struct.unpack_from("<H", data, 2)[0]
    if len(data) != NPL_SIZE + size or size < NPH_SIZE:
        raise NdtpError(f"size mismatch: dataSize={size}, got {len(data) - NPL_SIZE}")
    npl, nph = _parse_headers(data)
    if crc16_modbus(data[NPL_SIZE:]) != npl.crc:
        raise NdtpError("CRC mismatch")
    return Frame(npl, nph, bytes(data[NPL_SIZE + NPH_SIZE :]))


class FrameDecoder:
    """Streaming frame decoder for one TCP connection.

    Feed it arbitrary chunks; it returns every complete frame with a valid CRC. Garbage before a signature
    is skipped; frames with a bad CRC or an implausible header are dropped and decoding resumes from the
    next ``0x7E7E``.

    Args:
        max_data_size: Upper bound for ``NPL.dataSize``. Larger values are treated as a false signature,
            so garbage cannot make the decoder wait for megabytes.
        max_crc_errors: CRC mismatches allowed since the last valid frame. Each mismatch costs a CRC over up
            to ``max_data_size`` bytes, so a stream of plausible false headers could otherwise stall the event
            loop. When the budget is exhausted the decoder stops, drops its buffer and sets ``abusive``; the
            caller should close the connection.

    Attributes:
        frames: Number of valid frames produced.
        crc_errors: Number of frames dropped because of a CRC mismatch.
        bad_headers: Number of signatures rejected because of an implausible NPL header.
        bytes_discarded: Number of bytes skipped while resynchronising.
        abusive: True once ``max_crc_errors`` was exceeded; the decoder then ignores further input.
    """

    def __init__(self, max_data_size: int = 8192, max_crc_errors: int = 32) -> None:
        self.max_data_size = max_data_size
        self.max_crc_errors = max_crc_errors
        self.abusive = False
        self._errors_since_frame = 0
        self._buf = bytearray()
        self.frames = 0
        self.crc_errors = 0
        self.bad_headers = 0
        self.bytes_discarded = 0

    @property
    def buffered(self) -> int:
        """Number of bytes waiting for the rest of a frame."""
        return len(self._buf)

    def _discard(self, count: int) -> None:
        del self._buf[:count]
        self.bytes_discarded += count

    def feed(self, data: bytes | bytearray | memoryview) -> list[Frame]:
        """Append bytes and extract all complete frames.

        Args:
            data: Next chunk from the socket (may be empty).

        Returns:
            Frames completed by this chunk, in order.
        """
        if self.abusive:
            return []
        self._buf += data
        out: list[Frame] = []
        buf = self._buf
        while True:
            start = buf.find(SIGNATURE_BYTES)
            if start < 0:
                # keep a trailing 0x7E: it may be the first half of the next signature
                keep = 1 if buf.endswith(b"\x7e") else 0
                self._discard(len(buf) - keep)
                return out
            if start:
                self._discard(start)
            if len(buf) < NPL_SIZE:
                return out
            size = struct.unpack_from("<H", buf, 2)[0]
            if size < NPH_SIZE or size > self.max_data_size or buf[8] != NPL_TYPE_NPH:
                self.bad_headers += 1
                self._discard(1)
                continue
            total = NPL_SIZE + size
            if len(buf) < total:
                return out
            npl, nph = _parse_headers(buf)
            if crc16_modbus(memoryview(buf)[NPL_SIZE:total]) != npl.crc:
                self.crc_errors += 1
                self._errors_since_frame += 1
                if self._errors_since_frame > self.max_crc_errors:
                    self.abusive = True
                    self._discard(len(buf))
                    return out
                self._discard(1)
                continue
            out.append(Frame(npl, nph, bytes(buf[NPL_SIZE + NPH_SIZE : total])))
            del buf[:total]
            self.frames += 1
            self._errors_since_frame = 0


def decode_handshake(frame: Frame) -> Handshake:
    """Decode the body of a handshake frame.

    Args:
        frame: A frame with ``is_handshake`` true.

    Returns:
        The decoded handshake.

    Raises:
        NdtpError: If the frame is not a handshake or the body is shorter than 18 bytes.
    """
    if not frame.is_handshake:
        raise NdtpError(f"not a handshake: service={frame.nph.service_id} type={frame.nph.type}")
    if len(frame.body) < _HANDSHAKE.size:
        raise NdtpError(f"handshake body too short: {len(frame.body)}")
    high, low, flags, peer, max_size, _reserved = _HANDSHAKE.unpack_from(frame.body)
    return Handshake(high, low, flags, peer, max_size)


def decode_nav(payload: bytes | memoryview) -> NavRecord:
    """Decode a ``G6CellNav00`` payload (26 bytes, without the cell header).

    Args:
        payload: Cell payload.

    Returns:
        Navigation record with signed coordinates in degrees and a UTC timestamp.
    """
    ts, lon_raw, lat_raw, flags, bat, spd_avg, spd_max, course, track, alt, nsat, pdop = _NAV00.unpack_from(
        payload
    )
    lon = lon_raw / _COORD_SCALE
    lat = lat_raw / _COORD_SCALE
    if not flags & _FLAG_EAST:
        lon = -lon
    if not flags & _FLAG_NORTH:
        lat = -lat
    return NavRecord(
        timestamp=datetime.fromtimestamp(ts, UTC),
        lon=lon,
        lat=lat,
        valid=bool(flags & _FLAG_VALID),
        speed_avg=spd_avg,
        speed_max=spd_max,
        course=course,
        track=track,
        altitude=alt,
        nsat=nsat,
        pdop=pdop,
        battery_mv=bat * _BATTERY_MV_PER_UNIT,
        aux_flags=flags & _AUX_FLAGS_MASK,
    )


def decode_irma(payload: bytes | memoryview, number: int = 0) -> IrmaRecord:
    """Decode a ``G6CellIrma04`` payload (15 bytes, without the cell header).

    Args:
        payload: Cell payload.
        number: Cell index from the cell header.

    Returns:
        Door counters and states.
    """
    odo, zone, i1, i2, i3, i4, o1, o2, o3, o4, bits = _IRMA04.unpack_from(payload)
    present = (bool(bits & 1), bool(bits & 2), bool(bits & 4), bool(bits & 8))
    closed = (bool(bits & 16), bool(bits & 32), bool(bits & 64), bool(bits & 128))
    return IrmaRecord(number, odo, zone, (i1, i2, i3, i4), (o1, o2, o3, o4), present, closed)


def decode_realtime(frame: Frame) -> RealtimePacket:
    """Decode the cells of a realtime frame.

    Nav00 and Irma04 are decoded; other known cells are skipped by length. An unknown cell type stops
    parsing (its length is unknown) but whatever was decoded before it is returned.

    Args:
        frame: A frame with ``is_realtime`` true.

    Returns:
        The decoded packet.

    Raises:
        NdtpError: If the frame is not a realtime packet.
    """
    if not frame.is_realtime:
        raise NdtpError(f"not a realtime packet: service={frame.nph.service_id} type={frame.nph.type}")
    body = memoryview(frame.body)
    pos = 0
    nav: NavRecord | None = None
    irma: list[IrmaRecord] = []
    types: list[int] = []
    unknown: int | None = None
    truncated = False
    while pos < len(body):
        if len(body) - pos < 2:
            truncated = True
            break
        ctype, number = body[pos], body[pos + 1]
        size = CELL_SIZES.get(ctype)
        if size is None:
            unknown = ctype
            break
        payload = body[pos + 2 : pos + 2 + size]
        if len(payload) < size:
            truncated = True
            break
        if ctype == CELL_NAV00:
            if nav is None:
                nav = decode_nav(payload)
        elif ctype == CELL_IRMA04:
            irma.append(decode_irma(payload, number))
        types.append(ctype)
        pos += 2 + size
    return RealtimePacket(
        unit_id=frame.unit_id,
        request_id=frame.nph.request_id,
        nav=nav,
        irma=tuple(irma),
        cell_types=tuple(types),
        unknown_cell_type=unknown,
        truncated=truncated,
    )


# --------------------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """A raw cell for :func:`encode_realtime`.

    Attributes:
        type: Cell type id.
        payload: Cell payload bytes (without header).
        number: Index among cells of the same type.
    """

    type: int
    payload: bytes
    number: int = 0

    def encode(self) -> bytes:
        """Return ``[type][number][payload]``."""
        return bytes((self.type, self.number)) + self.payload


def encode_frame(
    unit_id: int,
    service_id: int,
    nph_type: int,
    request_id: int,
    body: bytes,
    *,
    nph_flags: int = NPH_FLAG_REQUEST,
    npl_request_id: int = 0,
) -> bytes:
    """Build a complete frame (NPL + NPH + body) with a CRC.

    Args:
        unit_id: Device id written to ``NPL.peerAddress``.
        service_id: NPH service id.
        nph_type: NPH packet type.
        request_id: NPH request id (wrapped to u32).
        body: NPH body.
        nph_flags: NPH flags; bit 0 ``request`` is set by default like the emulator does.
        npl_request_id: NPL request id (the emulator sends 0).

    Returns:
        Frame bytes ready to be written to a socket.
    """
    nph_and_body = _NPH.pack(service_id, nph_type, nph_flags, request_id & 0xFFFFFFFF) + body
    crc = crc16_modbus(nph_and_body)
    npl = _NPL.pack(SIGNATURE, len(nph_and_body), 0, _swap16(crc), NPL_TYPE_NPH, unit_id, npl_request_id)
    return npl + nph_and_body


def encode_handshake(
    unit_id: int, request_id: int = 1, max_packet_size: int = DEFAULT_MAX_PACKET_SIZE
) -> bytes:
    """Build an ``NPH_SGC_CONN_REQUEST`` frame identical in layout to the emulator's.

    Args:
        unit_id: Device id.
        request_id: NPH request id.
        max_packet_size: Advertised maximum packet size.

    Returns:
        Frame bytes.
    """
    body = _HANDSHAKE.pack(PROTO_VERSION_HIGH, PROTO_VERSION_LOW, 0, unit_id, max_packet_size, 0)
    return encode_frame(unit_id, SERVICE_GENERIC_CONTROLS, NPH_SGC_CONN_REQUEST, request_id, body)


def encode_nav(nav: NavRecord) -> bytes:
    """Encode a navigation record into a 26-byte ``G6CellNav00`` payload.

    Args:
        nav: Navigation record; a naive ``timestamp`` is interpreted as UTC.

    Returns:
        Payload bytes (without the cell header).
    """
    ts = nav.timestamp if nav.timestamp.tzinfo else nav.timestamp.replace(tzinfo=UTC)
    flags = nav.aux_flags & _AUX_FLAGS_MASK
    if nav.lat >= 0:
        flags |= _FLAG_NORTH
    if nav.lon >= 0:
        flags |= _FLAG_EAST
    if nav.valid:
        flags |= _FLAG_VALID
    return _NAV00.pack(
        int(ts.timestamp()) & 0xFFFFFFFF,
        round(abs(nav.lon) * _COORD_SCALE),
        round(abs(nav.lat) * _COORD_SCALE),
        flags,
        min(nav.battery_mv // _BATTERY_MV_PER_UNIT, 0xFF),
        nav.speed_avg & 0xFFFF,
        nav.speed_max & 0xFFFF,
        nav.course & 0xFFFF,
        nav.track & 0xFFFF,
        nav.altitude & 0xFFFF,
        nav.nsat & 0xFF,
        nav.pdop & 0xFF,
    )


def encode_irma(irma: IrmaRecord) -> Cell:
    """Encode an Irma04 record into a :class:`Cell`.

    Args:
        irma: Door record.

    Returns:
        Cell with type 4 and the record's ``number``.
    """
    bits = 0
    for i in range(4):
        bits |= int(irma.door_present[i]) << i
        bits |= int(irma.door_closed[i]) << (4 + i)
    payload = _IRMA04.pack(irma.odometer, irma.zone, *irma.door_in, *irma.door_out, bits)
    return Cell(CELL_IRMA04, payload, irma.number)


def encode_realtime(
    unit_id: int,
    request_id: int,
    nav: NavRecord,
    extra_cells: Iterable[Cell | tuple[int, int, bytes]] = (),
) -> bytes:
    """Build an ``NPH_SND_REALTIME`` frame: Nav00 first, then ``extra_cells`` in order.

    Args:
        unit_id: Device id.
        request_id: NPH request id.
        nav: Navigation record for the Nav00 cell.
        extra_cells: Additional cells as :class:`Cell` or ``(type, number, payload)`` tuples.

    Returns:
        Frame bytes.
    """
    parts = [Cell(CELL_NAV00, encode_nav(nav)).encode()]
    for cell in extra_cells:
        if isinstance(cell, Cell):
            parts.append(cell.encode())
        else:
            ctype, number, payload = cell
            parts.append(Cell(ctype, bytes(payload), number).encode())
    return encode_frame(unit_id, SERVICE_NAVDATA, NPH_SND_REALTIME, request_id, b"".join(parts))
