"""Versioned, framed Python client for the Arduino LowLevelServer.

Wire details are specified in protocol/v1.md. All transactions are serialized;
no motion command is retransmitted after a timeout.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass

import serial


PROTOCOL_VERSION = 1
CMD_HELLO = 0x01
CMD_GET_STATE = 0x02
CMD_SET_ACCEL = 0x03
CMD_ARM = 0x04
CMD_DISARM = 0x05
CMD_TARE = 0x06
CMD_ZERO = 0x08
RESP_ACK = 0x80
RESP_HELLO_ACK = 0x81
RESP_STATE = 0x82

STATUS_OK = 0
STATUS_BAD_VERSION = 1
STATUS_BAD_LENGTH = 2
STATUS_BAD_CRC = 3
STATUS_BAD_SEQUENCE = 4
STATUS_NOT_DISARMED = 5
STATUS_INVALID_VALUE = 6
STATUS_FAULT_LATCHED = 7
STATUS_NOT_ARMED = 8
STATUS_UNKNOWN_COMMAND = 9
STATUS_STATE_UNAVAILABLE = 10

MAX_PAYLOAD = 24
MAX_ACCEL_RAD_S2 = 150.0  # protocol v1 command envelope; matches firmware and deployed config
_HEADER = struct.Struct("<BBHB")
_STATE = struct.Struct("<IffffHH")
_ACK = struct.Struct("<BBH")
_HELLO_ACK = struct.Struct("<BBHH")
MAX_DECODED = _HEADER.size + MAX_PAYLOAD + 2
MAX_ENCODED = MAX_DECODED + 1  # COBS overhead, excluding delimiter


class ProtocolError(RuntimeError):
    """Malformed, mismatched or stale serial-protocol data."""


class TransactionTimeout(TimeoutError):
    """No matching response arrived before the bounded transaction deadline."""


@dataclass(frozen=True)
class Ack:
    request_type: int
    status: int
    fault_flags: int
    sequence: int
    rtt_ns: int

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class CommandRejected(ProtocolError):
    def __init__(self, ack: Ack):
        self.ack = ack
        super().__init__(
            f"command 0x{ack.request_type:02x} rejected: status={ack.status}, "
            f"fault_flags=0x{ack.fault_flags:04x}"
        )


@dataclass(frozen=True)
class State:
    time_us: int
    motor_pos_rad: float
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float
    health_flags: int
    fault_flags: int
    rx_monotonic_ns: int
    rtt_ns: int


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly=0x1021, init=0xffff, xorout=0."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    """Encode one non-empty record; caller appends the 0 delimiter."""
    out = bytearray([0])
    code_index = 0
    code = 1
    for byte in data:
        if byte == 0:
            out[code_index] = code
            code_index = len(out)
            out.append(0)
            code = 1
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index = len(out)
                out.append(0)
                code = 1
    out[code_index] = code
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    if not data:
        raise ProtocolError("empty COBS record")
    out = bytearray()
    i = 0
    while i < len(data):
        code = data[i]
        if code == 0:
            raise ProtocolError("zero byte inside COBS record")
        i += 1
        end = i + code - 1
        if end > len(data):
            raise ProtocolError("truncated COBS block")
        out.extend(data[i:end])
        i = end
        if code != 0xFF and i < len(data):
            out.append(0)
    return bytes(out)


def encode_frame(frame_type: int, sequence: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    body = _HEADER.pack(PROTOCOL_VERSION, frame_type, sequence & 0xFFFF, len(payload)) + payload
    decoded = body + struct.pack("<H", crc16_ccitt(body))
    return cobs_encode(decoded) + b"\x00"


def decode_frame(encoded: bytes) -> tuple[int, int, bytes]:
    decoded = cobs_decode(encoded)
    if len(decoded) < _HEADER.size + 2 or len(decoded) > MAX_DECODED:
        raise ProtocolError("invalid decoded frame length")
    version, frame_type, sequence, payload_len = _HEADER.unpack_from(decoded)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if payload_len > MAX_PAYLOAD or len(decoded) != _HEADER.size + payload_len + 2:
        raise ProtocolError("payload length does not match frame")
    expected_crc = struct.unpack_from("<H", decoded, len(decoded) - 2)[0]
    if crc16_ccitt(decoded[:-2]) != expected_crc:
        raise ProtocolError("CRC mismatch")
    return frame_type, sequence, decoded[_HEADER.size:-2]


class LowLevelClient:
    """Serialized request/response client; ARM is always explicit."""

    def __init__(self, port: str, baud: int = 2_000_000, timeout: float = 0.020):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._seq = 0
        self._last_state_time_us: int | None = None

    def __enter__(self) -> "LowLevelClient":
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout, write_timeout=self.timeout)
        # Opening the USB serial port resets many Nano bootloaders.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            try:
                self.disarm()
            except Exception:
                pass
            self._ser.close()
            self._ser = None

    @property
    def ser(self) -> serial.Serial:
        if self._ser is None:
            raise RuntimeError("LowLevelClient must be used as a context manager")
        return self._ser

    def _next_sequence(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    def _read_packet(self, deadline: float) -> tuple[int, int, bytes]:
        encoded = bytearray()
        while time.monotonic() < deadline:
            self.ser.timeout = max(0.0, min(0.005, deadline - time.monotonic()))
            byte = self.ser.read(1)
            if not byte:
                continue
            if byte == b"\x00":
                if not encoded:
                    continue
                record = bytes(encoded)
                return decode_frame(record)
            encoded.extend(byte)
            if len(encoded) > MAX_ENCODED:
                # Drain to the next delimiter, within this transaction's deadline.
                while time.monotonic() < deadline:
                    self.ser.timeout = max(0.0, min(0.005, deadline - time.monotonic()))
                    if self.ser.read(1) == b"\x00":
                        break
                raise ProtocolError("encoded frame exceeds fixed maximum")
        raise TransactionTimeout("serial response deadline expired")

    def _transact(self, frame_type: int, payload: bytes, response_type: int,
                  *, timeout: float | None = None) -> tuple[int, bytes, int]:
        with self._lock:
            seq = self._next_sequence()
            request = encode_frame(frame_type, seq, payload)
            started = time.monotonic_ns()
            self.ser.write(request)
            self.ser.flush()
            deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
            while True:
                got_type, got_seq, got_payload = self._read_packet(deadline)
                if got_seq != seq:
                    # A delayed response is stale; never let it satisfy this transaction.
                    continue
                if got_type != response_type:
                    raise ProtocolError(
                        f"sequence {seq}: expected response 0x{response_type:02x}, "
                        f"got 0x{got_type:02x}"
                    )
                ended = time.monotonic_ns()
                return seq, got_payload, ended - started

    @staticmethod
    def _parse_ack(sequence: int, payload: bytes, request_type: int,
                   rtt_ns: int) -> Ack:
        if len(payload) != _ACK.size:
            raise ProtocolError(f"ACK payload length {len(payload)} != {_ACK.size}")
        ack_type, status, faults = _ACK.unpack(payload)
        if ack_type != request_type:
            raise ProtocolError("ACK request type does not match request")
        ack = Ack(ack_type, status, faults, sequence, rtt_ns)
        if not ack.ok:
            raise CommandRejected(ack)
        return ack

    def wait_until_ready(self, *, retries: int = 5, retry_delay_s: float = 0.1) -> bool:
        """Perform HELLO; safe to retry because it never arms the motor."""
        for attempt in range(retries):
            try:
                seq, payload, _ = self._transact(CMD_HELLO, b"", RESP_HELLO_ACK)
                if len(payload) != _HELLO_ACK.size:
                    raise ProtocolError("invalid HELLO_ACK length")
                version, max_payload, _health, _faults = _HELLO_ACK.unpack(payload)
                if version != PROTOCOL_VERSION or max_payload != MAX_PAYLOAD:
                    raise ProtocolError("board protocol capabilities do not match client")
                return True
            except TransactionTimeout:
                if attempt + 1 == retries:
                    return False
                time.sleep(retry_delay_s)
        return False

    def get_state(self, *, retries: int = 0) -> State:
        """Read state; optional retries issue a new GET_STATE sequence."""
        for attempt in range(retries + 1):
            try:
                seq, payload, rtt_ns = self._transact(CMD_GET_STATE, b"", RESP_STATE)
                if len(payload) != _STATE.size:
                    raise ProtocolError(f"STATE payload length {len(payload)} != {_STATE.size}")
                time_us, motor_pos, pend_pos, motor_vel, pend_vel, health, faults = _STATE.unpack(payload)
                values = (motor_pos, pend_pos, motor_vel, pend_vel)
                if not all(math.isfinite(value) for value in values):
                    raise ProtocolError("STATE contains non-finite sensor value")
                if self._last_state_time_us is not None:
                    delta = (time_us - self._last_state_time_us) & 0xFFFFFFFF
                    if delta == 0 or delta >= 0x80000000:
                        raise ProtocolError("STATE timestamp did not advance")
                self._last_state_time_us = time_us
                return State(int(time_us), *map(float, values), int(health), int(faults),
                             time.monotonic_ns(), rtt_ns)
            except TransactionTimeout:
                if attempt == retries:
                    raise
        raise AssertionError("unreachable")

    def _command(self, command: int, *, timeout: float | None = None) -> Ack:
        seq, payload, rtt_ns = self._transact(command, b"", RESP_ACK, timeout=timeout)
        return self._parse_ack(seq, payload, command, rtt_ns)

    def set_acceleration(self, accel_rad_s2: float) -> Ack:
        value = float(accel_rad_s2)
        if not math.isfinite(value):
            raise ValueError("acceleration must be finite")
        if abs(value) > MAX_ACCEL_RAD_S2:
            raise ValueError(
                f"acceleration exceeds protocol limit ±{MAX_ACCEL_RAD_S2:g} rad/s²"
            )
        payload = struct.pack("<f", value)
        seq, response, rtt_ns = self._transact(CMD_SET_ACCEL, payload, RESP_ACK)
        return self._parse_ack(seq, response, CMD_SET_ACCEL, rtt_ns)

    def arm(self) -> Ack:
        return self._command(CMD_ARM)

    def disarm(self) -> Ack:
        return self._command(CMD_DISARM)

    # Backward-compatible names, still requiring explicit caller action.
    def engage_motor(self) -> Ack:
        return self.arm()

    def disengage_motor(self) -> Ack:
        return self.disarm()

    def tare_pendulum(self, *, timeout_s: float = 0.5) -> bool:
        self._command(CMD_TARE, timeout=timeout_s)
        return True

    def zero_motor(self, *, timeout_s: float = 0.5) -> bool:
        self._command(CMD_ZERO, timeout=timeout_s)
        return True
