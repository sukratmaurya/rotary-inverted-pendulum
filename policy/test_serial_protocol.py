"""Offline checks for protocol v1 framing and payload decoding.

Run from policy/: python test_serial_protocol.py
No serial device is opened.
"""
from __future__ import annotations

import struct
import unittest
from collections import deque

import lowlevel_client as protocol


class FakeSerial:
    """In-memory peer used to exercise transactions without a device."""
    def __init__(self, *, drop_responses=False):
        self.rx = deque()
        self.timeout = 0.01
        self.write_timeout = 0.01
        self.drop_responses = drop_responses
        self.timestamp = 1000

    def write(self, wire):
        if self.drop_responses:
            return len(wire)
        frame_type, seq, payload = protocol.decode_frame(wire[:-1])
        if frame_type == protocol.CMD_HELLO:
            response_type = protocol.RESP_HELLO_ACK
            response = struct.pack("<BBHH", protocol.PROTOCOL_VERSION,
                                    protocol.MAX_PAYLOAD, 0x001F, 0)
        elif frame_type == protocol.CMD_GET_STATE:
            response_type = protocol.RESP_STATE
            self.timestamp += 2000
            response = struct.pack("<IffffHH", self.timestamp, 0.1, -0.2,
                                    0.3, -0.4, 0x001F, 0)
        else:
            response_type = protocol.RESP_ACK
            response = struct.pack("<BBH", frame_type, protocol.STATUS_OK, 0)
        self.rx.extend(protocol.encode_frame(response_type, seq, response))
        return len(wire)

    def flush(self):
        pass

    def read(self, count=1):
        return bytes([self.rx.popleft()]) if self.rx else b""

    def reset_input_buffer(self):
        self.rx.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        pass


class ProtocolCodecTests(unittest.TestCase):
    def test_crc_ccitt_false_reference(self):
        self.assertEqual(protocol.crc16_ccitt(b"123456789"), 0x29B1)

    def test_cobs_round_trip_binary_payloads(self):
        cases = [b"\x00", bytes(range(1, 32)), bytes(range(32)), b"\x00" * 31]
        cases.extend(bytes((i * 37 + n) & 0xFF for i in range(n)) for n in range(1, 65))
        for data in cases:
            with self.subTest(length=len(data), data=data[:8]):
                encoded = protocol.cobs_encode(data)
                self.assertNotIn(0, encoded)
                self.assertEqual(protocol.cobs_decode(encoded), data)

    def test_frame_round_trip_empty_float_and_max_state(self):
        frames = [
            (protocol.CMD_HELLO, 0xFFFF, b""),
            (protocol.CMD_SET_ACCEL, 0x1234, struct.pack("<f", -149.25)),
            (protocol.RESP_STATE, 0, struct.pack("<IffffHH", 0xFEDCBA98,
                                                  1.25, -2.5, 3.75, -4.0,
                                                  0x001F, 0x0021)),
        ]
        for frame_type, sequence, payload in frames:
            with self.subTest(type=frame_type, sequence=sequence):
                wire = protocol.encode_frame(frame_type, sequence, payload)
                self.assertEqual(wire[-1], 0)
                decoded_type, decoded_seq, decoded_payload = protocol.decode_frame(wire[:-1])
                self.assertEqual((decoded_type, decoded_seq, decoded_payload),
                                 (frame_type, sequence, payload))

    def test_crc_version_and_length_errors_rejected(self):
        wire = protocol.encode_frame(protocol.CMD_SET_ACCEL, 7, struct.pack("<f", 12.0))
        raw = bytearray(protocol.cobs_decode(wire[:-1]))
        raw[5] ^= 0x01
        with self.assertRaisesRegex(protocol.ProtocolError, "CRC"):
            protocol.decode_frame(protocol.cobs_encode(raw))

        raw = bytearray(protocol.cobs_decode(wire[:-1]))
        raw[0] = 2
        with self.assertRaisesRegex(protocol.ProtocolError, "version"):
            protocol.decode_frame(protocol.cobs_encode(raw))

        raw = bytearray(protocol.cobs_decode(wire[:-1]))
        raw[4] += 1
        with self.assertRaisesRegex(protocol.ProtocolError, "length"):
            protocol.decode_frame(protocol.cobs_encode(raw))

    def test_reject_oversized_payload(self):
        with self.assertRaises(ValueError):
            protocol.encode_frame(protocol.CMD_GET_STATE, 1, bytes(25))

    def test_client_transactions_use_matching_responses(self):
        client = protocol.LowLevelClient("fake", timeout=0.01)
        client._ser = FakeSerial()
        self.assertTrue(client.wait_until_ready(retries=1))
        state = client.get_state()
        self.assertEqual(state.time_us, 3000)
        self.assertAlmostEqual(state.pendulum_vel_rad_s, -0.4)
        self.assertEqual(state.health_flags, 0x001F)
        self.assertGreater(state.rx_monotonic_ns, 0)
        self.assertGreater(state.rtt_ns, 0)
        ack = client.set_acceleration(12.5)
        self.assertTrue(ack.ok)
        self.assertEqual(ack.request_type, protocol.CMD_SET_ACCEL)
        self.assertGreater(ack.rtt_ns, 0)
        self.assertTrue(client.arm().ok)
        self.assertTrue(client.disarm().ok)

    def test_client_timeout_is_bounded_and_typed(self):
        client = protocol.LowLevelClient("fake", timeout=0.002)
        client._ser = FakeSerial(drop_responses=True)
        with self.assertRaises(protocol.TransactionTimeout):
            client.get_state()

    def test_motion_command_rejects_nonfinite_input(self):
        client = protocol.LowLevelClient("fake")
        client._ser = FakeSerial()
        for value in (float("nan"), float("inf"), -float("inf"), 150.01, -150.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client.set_acceleration(value)

    def test_truncated_extra_and_crc_corrupt_frames_rejected(self):
        wire = protocol.encode_frame(protocol.CMD_SET_ACCEL, 7, struct.pack("<f", 12.0))
        encoded = wire[:-1]
        for damaged in (encoded[:-1], encoded + b"\x01"):
            with self.subTest(damaged=damaged), self.assertRaises(protocol.ProtocolError):
                protocol.decode_frame(damaged)
        raw = bytearray(protocol.cobs_decode(encoded))
        raw[-1] ^= 0x80
        with self.assertRaisesRegex(protocol.ProtocolError, "CRC"):
            protocol.decode_frame(protocol.cobs_encode(raw))

    def test_extra_payload_and_short_payload_are_rejected(self):
        # A frame can be structurally valid yet semantically invalid for a
        # command. The Arduino dispatcher must reject this five-byte command.
        _, _, extra_command_payload = protocol.decode_frame(
            protocol.encode_frame(protocol.CMD_SET_ACCEL, 1, b"\0" * 5)[:-1]
        )
        self.assertEqual(len(extra_command_payload), 5)
        raw = bytearray(protocol.cobs_decode(
            protocol.encode_frame(protocol.CMD_GET_STATE, 1)[:-1]
        ))
        raw[4] = 1  # Header claims a payload byte that is absent.
        raw[-2:] = struct.pack("<H", protocol.crc16_ccitt(raw[:-2]))
        with self.assertRaisesRegex(protocol.ProtocolError, "payload length"):
            protocol.decode_frame(protocol.cobs_encode(raw))

    def test_stale_sequence_response_cannot_satisfy_transaction(self):
        class StaleThenCurrent(FakeSerial):
            def write(self, wire):
                frame_type, seq, _ = protocol.decode_frame(wire[:-1])
                response = struct.pack("<BBH", frame_type, protocol.STATUS_OK, 0)
                self.rx.extend(protocol.encode_frame(protocol.RESP_ACK, (seq - 1) & 0xFFFF, response))
                self.rx.extend(protocol.encode_frame(protocol.RESP_ACK, seq, response))
                return len(wire)

        client = protocol.LowLevelClient("fake", timeout=0.01)
        client._ser = StaleThenCurrent()
        ack = client.set_acceleration(0.0)
        self.assertEqual(ack.sequence, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
