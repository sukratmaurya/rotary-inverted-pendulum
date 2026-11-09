#pragma once

#include <Arduino.h>
#include <math.h>
#include <string.h>

// Safety/state hooks are defined by the firmware safety implementation.
bool safety_is_disarmed();
bool safety_is_armed();
uint16_t safety_health_flags();
uint16_t safety_fault_flags();
uint8_t safety_arm();
void safety_disarm(uint8_t reason);
uint8_t safety_set_acceleration(float accel_rad_s2);
uint8_t safety_tare();
uint8_t safety_zero();
void safety_note_valid_accel(uint32_t now_ms);
void safety_check_command_lease(uint32_t now_ms);
bool safety_get_state(uint32_t* time_us, float* motor_pos, float* pend_pos,
                      float* motor_vel, float* pend_vel,
                      uint16_t* health_flags, uint16_t* fault_flags);

// Protocol v1 implementation. Frame contract: protocol/v1.md.
namespace ProtocolV1 {
static const uint8_t VERSION = 1;
static const uint8_t MAX_PAYLOAD = 24;
static const uint8_t TYPE_HELLO = 0x01;
static const uint8_t TYPE_GET_STATE = 0x02;
static const uint8_t TYPE_SET_ACCEL = 0x03;
static const uint8_t TYPE_ARM = 0x04;
static const uint8_t TYPE_DISARM = 0x05;
static const uint8_t TYPE_TARE = 0x06;
static const uint8_t TYPE_ZERO = 0x08;
static const uint8_t TYPE_ACK = 0x80;
static const uint8_t TYPE_HELLO_ACK = 0x81;
static const uint8_t TYPE_STATE = 0x82;

static const uint8_t STATUS_OK = 0;
static const uint8_t STATUS_BAD_VERSION = 1;
static const uint8_t STATUS_BAD_LENGTH = 2;
static const uint8_t STATUS_BAD_CRC = 3;
static const uint8_t STATUS_BAD_SEQUENCE = 4;
static const uint8_t STATUS_NOT_DISARMED = 5;
static const uint8_t STATUS_INVALID_VALUE = 6;
static const uint8_t STATUS_FAULT_LATCHED = 7;
static const uint8_t STATUS_NOT_ARMED = 8;
static const uint8_t STATUS_UNKNOWN_COMMAND = 9;
static const uint8_t STATUS_STATE_UNAVAILABLE = 10;

static const uint8_t MAX_RAW = 31;      // 5-byte header + 24 payload + 2 CRC
static const uint8_t MAX_ENCODED = 32;  // COBS overhead, excluding delimiter
static const uint8_t PARTIAL_TIMEOUT_MS = 25;

static uint8_t rx_encoded[MAX_ENCODED];
static uint8_t rx_len = 0;
static uint32_t rx_started_ms = 0;
static uint32_t malformed_frames = 0;
static bool discard_until_delimiter = false;
static bool have_sequence = false;
static uint16_t last_sequence = 0;

static uint16_t crc16(const uint8_t* data, uint8_t len)
{
    uint16_t crc = 0xFFFF;
    for (uint8_t i = 0; i < len; ++i)
    {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0; bit < 8; ++bit)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}

static uint8_t cobs_decode(const uint8_t* input, uint8_t input_len, uint8_t* output)
{
    uint8_t in = 0, out = 0;
    while (in < input_len)
    {
        uint8_t code = input[in++];
        if (code == 0) return 0;
        uint8_t count = (uint8_t)(code - 1);
        if ((uint16_t)in + count > input_len) return 0;
        if ((uint16_t)out + count > MAX_RAW) return 0;
        while (count--) output[out++] = input[in++];
        if (code != 0xFF && in < input_len)
        {
            if (out >= MAX_RAW) return 0;
            output[out++] = 0;
        }
    }
    return out;
}

static void send_frame(uint8_t type, uint16_t sequence,
                       const uint8_t* payload, uint8_t payload_len)
{
    if (payload_len > MAX_PAYLOAD) return;
    uint8_t raw[MAX_RAW];
    uint8_t raw_len = (uint8_t)(5 + payload_len);
    raw[0] = VERSION;
    raw[1] = type;
    raw[2] = (uint8_t)(sequence & 0xFF);
    raw[3] = (uint8_t)(sequence >> 8);
    raw[4] = payload_len;
    if (payload_len) memcpy(raw + 5, payload, payload_len);
    uint16_t crc = crc16(raw, raw_len);
    raw[raw_len++] = (uint8_t)(crc & 0xFF);
    raw[raw_len++] = (uint8_t)(crc >> 8);

    uint8_t encoded[MAX_ENCODED + 1];
    uint8_t out = 1, code_index = 0, code = 1;
    for (uint8_t i = 0; i < raw_len; ++i)
    {
        if (raw[i] == 0)
        {
            encoded[code_index] = code;
            code_index = out++;
            code = 1;
        }
        else
        {
            encoded[out++] = raw[i];
            if (++code == 0xFF)
            {
                encoded[code_index] = code;
                code_index = out++;
                code = 1;
            }
        }
    }
    encoded[code_index] = code;
    encoded[out++] = 0;
    Serial.write(encoded, out);
}

static void send_ack(uint16_t sequence, uint8_t request_type, uint8_t status)
{
    uint8_t payload[4] = {request_type, status,
        (uint8_t)(safety_fault_flags() & 0xFF),
        (uint8_t)(safety_fault_flags() >> 8)};
    send_frame(TYPE_ACK, sequence, payload, sizeof(payload));
}

static bool sequence_forward(uint16_t sequence)
{
    if (!have_sequence) return false;
    uint16_t delta = (uint16_t)(sequence - last_sequence);
    return delta != 0 && delta < 0x8000;
}

static void handle_decoded(const uint8_t* raw, uint8_t raw_len)
{
    if (raw_len < 7) { ++malformed_frames; return; }
    uint8_t type = raw[1];
    uint16_t seq = (uint16_t)raw[2] | ((uint16_t)raw[3] << 8);
    uint8_t payload_len = raw[4];
    if (raw[0] != VERSION)
    {
        send_ack(seq, type, STATUS_BAD_VERSION);
        ++malformed_frames;
        return;
    }
    if (payload_len > MAX_PAYLOAD || raw_len != (uint8_t)(7 + payload_len))
    {
        send_ack(seq, type, STATUS_BAD_LENGTH);
        ++malformed_frames;
        return;
    }
    uint16_t received_crc = (uint16_t)raw[raw_len - 2] |
                            ((uint16_t)raw[raw_len - 1] << 8);
    if (crc16(raw, (uint8_t)(raw_len - 2)) != received_crc)
    {
        send_ack(seq, type, STATUS_BAD_CRC);
        ++malformed_frames;
        return;
    }
    const uint8_t* payload = raw + 5;

    if (type == TYPE_HELLO)
    {
        if (payload_len != 0) { send_ack(seq, type, STATUS_BAD_LENGTH); return; }
        if (!safety_is_disarmed()) { send_ack(seq, type, STATUS_NOT_DISARMED); return; }
        have_sequence = true;
        last_sequence = seq;
        uint8_t hello[6] = {VERSION, MAX_PAYLOAD,
            (uint8_t)(safety_health_flags() & 0xFF), (uint8_t)(safety_health_flags() >> 8),
            (uint8_t)(safety_fault_flags() & 0xFF), (uint8_t)(safety_fault_flags() >> 8)};
        send_frame(TYPE_HELLO_ACK, seq, hello, sizeof(hello));
        return;
    }

    // DISARM is safety-priority: always invoke the idempotent stop for a valid
    // complete frame, even if its sequence is duplicate/old. It never refreshes
    // the command lease and does not roll the accepted sequence backwards.
    if (type == TYPE_DISARM)
    {
        if (payload_len != 0) { send_ack(seq, type, STATUS_BAD_LENGTH); return; }
        if (sequence_forward(seq)) { last_sequence = seq; have_sequence = true; }
        safety_disarm(0);
        send_ack(seq, type, STATUS_OK);
        return;
    }

    if (!sequence_forward(seq))
    {
        send_ack(seq, type, STATUS_BAD_SEQUENCE);
        return;
    }
    last_sequence = seq;

    uint8_t status = STATUS_OK;
    switch (type)
    {
    case TYPE_GET_STATE:
        if (payload_len != 0) { send_ack(seq, type, STATUS_BAD_LENGTH); return; }
        {
            uint32_t t_us;
            float mp, pp, mv, pv;
            uint16_t health, faults;
            if (!safety_get_state(&t_us, &mp, &pp, &mv, &pv, &health, &faults))
            {
                send_ack(seq, type, STATUS_STATE_UNAVAILABLE);
                return;
            }
            if (!isfinite(mp) || !isfinite(pp) || !isfinite(mv) || !isfinite(pv))
            {
                send_ack(seq, type, STATUS_INVALID_VALUE);
                return;
            }
            uint8_t state[24];
            memcpy(state, &t_us, 4);
            memcpy(state + 4, &mp, 4);
            memcpy(state + 8, &pp, 4);
            memcpy(state + 12, &mv, 4);
            memcpy(state + 16, &pv, 4);
            state[20] = (uint8_t)(health & 0xFF); state[21] = (uint8_t)(health >> 8);
            state[22] = (uint8_t)(faults & 0xFF); state[23] = (uint8_t)(faults >> 8);
            send_frame(TYPE_STATE, seq, state, sizeof(state));
        }
        return;
    case TYPE_SET_ACCEL:
        if (payload_len != 4) { status = STATUS_BAD_LENGTH; break; }
        {
            float accel;
            memcpy(&accel, payload, 4);
            if (!isfinite(accel)) { status = STATUS_INVALID_VALUE; break; }
            status = safety_set_acceleration(accel);
            if (status == STATUS_OK) safety_note_valid_accel(millis());
        }
        break;
    case TYPE_ARM:
        if (payload_len != 0) { status = STATUS_BAD_LENGTH; break; }
        status = safety_arm();
        break;
    case TYPE_TARE:
        if (payload_len != 0) { status = STATUS_BAD_LENGTH; break; }
        status = safety_tare();
        break;
    case TYPE_ZERO:
        if (payload_len != 0) { status = STATUS_BAD_LENGTH; break; }
        status = safety_zero();
        break;
    default:
        status = STATUS_UNKNOWN_COMMAND;
        break;
    }
    send_ack(seq, type, status);
}

static void finish_frame()
{
    uint8_t raw[MAX_RAW];
    uint8_t raw_len = cobs_decode(rx_encoded, rx_len, raw);
    if (raw_len == 0) { ++malformed_frames; return; }
    handle_decoded(raw, raw_len);
}

// Call on every main-loop pass. Drains currently available bytes only; never
// waits for a payload. RX timeout/overflow drops through the next delimiter.
static void protocol_poll()
{
    uint32_t now = millis();
    if (rx_len && (uint32_t)(now - rx_started_ms) > PARTIAL_TIMEOUT_MS)
    {
        rx_len = 0;
        discard_until_delimiter = true;
        ++malformed_frames;
    }
    while (Serial.available() > 0)
    {
        uint8_t byte = (uint8_t)Serial.read();
        now = millis();
        if (byte == 0)
        {
            if (!discard_until_delimiter && rx_len) finish_frame();
            rx_len = 0;
            discard_until_delimiter = false;
            continue;
        }
        if (discard_until_delimiter) continue;
        if (rx_len == 0) rx_started_ms = now;
        if (rx_len >= MAX_ENCODED)
        {
            rx_len = 0;
            discard_until_delimiter = true;
            ++malformed_frames;
            continue; // discard the rest of this frame through delimiter
        }
        rx_encoded[rx_len++] = byte;
    }
}
} // namespace ProtocolV1

using ProtocolV1::protocol_poll;
