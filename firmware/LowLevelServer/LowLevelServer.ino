#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>
#include "ProtocolV1.h"

// Communication speed
const long BAUD_RATE = 2000000;

// Command bytes

// Pin assignments. STEP must be on pin 9 (Timer1 OC1A on ATmega328) for
// FastAccelStepper. DIR and ENABLE can be any digital pin.
#define DIR_PIN 2
#define STEP_PIN 9
#define ENABLE_PIN 5

// Microstepping: the ONLY place to change it. Everything below derives from
// STEPS_PER_REVOLUTION. 32 is the recommended ratio on either driver, but the
// pin levels differ — DRV8825 M0=M1=M2=HIGH, TMC2209 MS1=HIGH/MS2=LOW.
// Legacy: 16 = TMC2209 MS1=MS2=HIGH, 8 = DRV8825 M0=M1=HIGH. Must match
// MOTOR_MICROSTEPS in pendulum_env.py and MICROSTEPS in RLControl.ino.
const int MICROSTEPS = 32;
const long STEPS_PER_REVOLUTION = 200L * MICROSTEPS;
const float STEPS_PER_RAD = (float)STEPS_PER_REVOLUTION / (2.0f * (float)PI);
const float RAD_PER_STEP = (2.0f * (float)PI) / (float)STEPS_PER_REVOLUTION;

// Accel-mode envelope. See pendulum_env.py for the corresponding sim
// constants. The velocity cap below corresponds to MAX_VELOCITY_RAD_S
// = 5 rad/s: 5 × (STEPS_PER_REVOLUTION / 2π) steps/s (≈2546 at 3200) ⇒ MOTOR_MIN_STEP_US below.
// NB: this boot-time speed cap applies to BOTH command paths — the accel
// mode it was sized for AND position-mode moveTo() tracking. Position-mode
// commanded slew is max_action_delta × control rate (3.5 rad/s at defaults),
// comfortably inside; raise those knobs past ~5 rad/s and this cap becomes
// the binding limit here while sim and RLControl.ino (~196 rad/s) model none.
const uint32_t MOTOR_MIN_STEP_US =  // ≈ 5 rad/s, derived so it tracks MICROSTEPS
    (uint32_t)(1.0e6f * 2.0f * (float)PI / (5.0f * (float)STEPS_PER_REVOLUTION));

// Ramp acceleration for position-mode moveTo() (CMD_SET_TARGET); matches
// RLControl.ino. Must be re-asserted on every CMD_SET_TARGET: CMD_ENGAGE_MOTOR's
// moveByAcceleration(0) resets the library acceleration to 1 step/s², which
// would make the motor merely creep toward the target. Accel mode ignores this.
const int32_t MOTOR_ACCEL_STEPS_S2 = 50000;

// Position safety limit (matches MOTOR_SAFE_LIMIT_RAD on the Python side,
// ±125°). Past the rail the firmware actively brakes (commands a fixed
// opposing accel) so the motor decelerates even if the host stops
// sending commands (USB hiccup, host hang). Clamping to zero instead
// would just let moveByAcceleration(0, true) coast the motor past the
// rail at peak velocity.
const int32_t MOTOR_SAFE_LIMIT_STEPS = (int32_t)((125.0f * PI / 180.0f) *
                                                  ((float)STEPS_PER_REVOLUTION / (2.0f * PI)));
// Brake authority when past the rail. 150 rad/s² matches the
// pendulum_env.py MAX_ACCEL_RAD_S2 — strong enough to bleed off the
// 5 rad/s vel cap within ~33 ms.
const int32_t MOTOR_BRAKE_ACCEL_STEPS_S2 =
    (int32_t)(150.0f * ((float)STEPS_PER_REVOLUTION / (2.0f * PI)));

// Encoder samples are kept in a ring buffer updated at 500 Hz; GET_STATE
// returns velocity computed as (newest - oldest)/Δt over a window of 5
// samples = 4 inter-sample gaps = 8 ms. Window halved from 10 → 5 on
// 2026-05-20 to cut ~9 ms of observation lag — the largest tunable
// component of the rig's policy-loop lag. Tradeoff: ~√2× noisier
// per-sample velocity estimate. See website/src/content/docs/reference/transport-delay.md.
const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  VEL_WINDOW       = 5;
// Only the latest VEL_WINDOW samples are used. Keeping extra history wastes
// 132 bytes of scarce Nano SRAM without affecting state or velocity results.
const uint8_t  BUFFER_SIZE      = VEL_WINDOW;
// Discard impossibly-large per-sample wraps as I²C glitches. Real
// pendulum tops out at ~50 rad/s, so in one 2 ms sample period the
// raw AS5600 reading can change by at most 50·0.002·4096/(2π) ≈ 65 LSB.
// Anything more is almost certainly a corrupted I²C transaction —
// keep the previous reading rather than letting it pollute the
// accumulator with a spurious ±2π wrap.
const long PEN_RAW_MAX_DELTA_LSB = 500;

static int32_t motor_step_buf[BUFFER_SIZE];   // raw stepper position (steps)
static float   pen_rad_buf[BUFFER_SIZE];      // accumulated pendulum angle (rad)
static uint32_t time_us_buf[BUFFER_SIZE];     // sample timestamps (µs)
static uint8_t  buf_head = 0;                 // next write index
static bool     buf_filled = false;           // becomes true after first full lap
static uint32_t last_sample_us = 0;

// Continuously-tracked pendulum angle (independent of GET_STATE cadence so
// rapid wraparounds aren't missed).
static long    pen_raw_prev   = -1;           // -1 = first read sentinel
static float   pen_position_rad = 0.0f;

// State variables
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5600 as5600;

// Motor state is owned by the safety hooks; no ISR accesses this flag.
bool motor_engaged = false;

// Safety and sensor-health flags shared with the framed protocol dispatcher.
// Keep these assignments in sync with protocol/v1.md.
static const uint16_t HEALTH_ENCODER_VALID  = 0x0001;
static const uint16_t HEALTH_SAMPLER_FRESH  = 0x0002;
static const uint16_t HEALTH_STEPPER_READY  = 0x0004;
static const uint16_t HEALTH_ARMED          = 0x0008;
static const uint16_t HEALTH_MAGNET_DETECTED = 0x0010;
static const uint16_t REQUIRED_ARM_HEALTH = 0x0017;

static const uint16_t FAULT_LEASE_EXPIRED = 0x0001;
static const uint16_t FAULT_ENCODER_STALE = 0x0002;
static const uint16_t FAULT_ENCODER_FROZEN = 0x0004;
static const uint16_t FAULT_ENCODER_INVALID = 0x0008;
static const uint16_t FAULT_HARD_LIMIT = 0x0010;
static const uint16_t FAULT_INTERNAL = 0x0020;

static const uint32_t ENCODER_STALE_US = 10000UL;
static const uint32_t MAGNET_CHECK_MS = 100UL;
static const uint32_t COMMAND_LEASE_MS = 100UL;
static const int32_t FROZEN_ENCODER_MOTOR_TRAVEL_STEPS = 6400;
static const uint32_t FROZEN_ENCODER_CONFIRM_MS = 1000UL;

static uint16_t latched_fault_flags = 0;
static bool encoder_sample_valid = false;
static bool encoder_magnet_detected = false;
static uint8_t consecutive_bad_encoder_samples = 0;
static uint32_t last_valid_encoder_us = 0;
static uint32_t last_magnet_check_ms = 0;
static uint32_t last_valid_accel_ms = 0;
static long frozen_reference_raw = -1;
static int32_t frozen_reference_motor_steps = 0;
static uint32_t frozen_reference_ms = 0;

// Function prototypes
void sampleState();
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s);
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

void setup()
{
    Serial.begin(BAUD_RATE);
    Wire.begin();
    Wire.setClock(400000);   // I²C fast mode for short transaction times.
    // Bound Wire's synchronous waits so a stuck bus cannot stall the lease
    // and sensor-fault checks indefinitely. Reset the TWI peripheral on expiry.
    Wire.setWireTimeout(3000, true);
    as5600.begin();

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        while (true) { /* halt: STEP_PIN is not Timer1 OC1A/OC1B */ }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);

    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        while (true) {}
    }
    // Acceleration is set per tick in CMD_SET_TARGET, not here (see that
    // constant's note).
    // Shrink the forward-planning window from the library's 20 ms default
    // to its documented minimum (8 ms = two cyclic-task periods). Default
    // adds ~20 ms of lag between a new accel command and any change in
    // emitted step intervals, because step pulses already queued can't be
    // retroactively edited. Measured (accel_step_probe.py) total firmware
    // lag at default: ~37 ms → expected ~15 ms after this change. At our
    // top step rate (≈1270 steps/s) 8 ms still queues ~10 pulses, well
    // above the "stepper starves and stalls at full speed" failure mode
    // the library warns about.
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    // Stay responsive and DISARMED even if USB is absent or the encoder
    // magnet is missing. ARM is rejected until the sampled health checks pass.
    encoder_magnet_detected = as5600.detectMagnet();
    last_magnet_check_ms = millis();
    last_sample_us = micros();
}

void loop()
{
    uint32_t now_us = micros();
    if ((uint32_t)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleState();
    }
    protocol_poll();
    safety_check_command_lease(millis());
}

/*
 * Append a (time, motor_position, pendulum_position) sample to the ring
 * buffer and update the pendulum wraparound accumulator.
 */
void sampleState()
{
    int32_t motor_step = stepper->getCurrentPosition();
    long raw = as5600.rawAngle();
    // AS5600.rawAngle() returns uint16_t and masks to 12 bits even after an
    // I2C error. Check lastError() or a disconnected sensor can look like a
    // valid zero-angle sample and evade the stale/fault path.
    if (as5600.lastError() != AS5600_OK) raw = -1;

    uint32_t now_ms = millis();
    if ((uint32_t)(now_ms - last_magnet_check_ms) >= MAGNET_CHECK_MS)
    {
        encoder_magnet_detected = as5600.detectMagnet();
        last_magnet_check_ms = now_ms;
    }
    if (motor_engaged && !encoder_magnet_detected)
        safety_disarm((uint8_t)FAULT_ENCODER_INVALID);

    // AS5600 raw angle is a 12-bit value. An out-of-range return generally
    // indicates a failed I2C transaction in the sensor library.
    if (raw < 0 || raw >= 4096)
    {
        encoder_sample_valid = false;
        if (consecutive_bad_encoder_samples < 255) consecutive_bad_encoder_samples++;
        if (consecutive_bad_encoder_samples >= 3 && motor_engaged)
            safety_disarm((uint8_t)FAULT_ENCODER_INVALID);
        motor_step_buf[buf_head] = motor_step;
        pen_rad_buf[buf_head] = pen_position_rad;
        time_us_buf[buf_head] = last_sample_us;
        buf_head = (buf_head + 1) % BUFFER_SIZE;
        if (buf_head == 0) buf_filled = true;
        return;
    }

    // Pendulum wraparound tracking (AS5600 is 12-bit; wrap threshold ±2048).
    if (pen_raw_prev < 0)
    {
        pen_raw_prev = raw;
    }
    else
    {
        long delta = raw - pen_raw_prev;
        if (delta >  2048) delta -= 4096;
        if (delta < -2048) delta += 4096;
        // Reject implausibly-large single-step deltas. One bad I²C read
        // sample that we incorrectly classify as a wrap would add ±2π to
        // pen_position_rad and contaminate every subsequent observation
        // (the accumulator never resets). Skip the update on glitches —
        // velocity for this tick will be slightly stale but stays sane.
        if (delta > PEN_RAW_MAX_DELTA_LSB || delta < -PEN_RAW_MAX_DELTA_LSB)
        {
            encoder_sample_valid = false;
            if (consecutive_bad_encoder_samples < 255) consecutive_bad_encoder_samples++;
            if (consecutive_bad_encoder_samples >= 3 && motor_engaged)
                safety_disarm((uint8_t)FAULT_ENCODER_INVALID);
            // Don't update pen_raw_prev either: next good read will
            // compute the delta against the last trustworthy reading.
        }
        else
        {
            pen_position_rad += (float)delta * ((2.0f * PI) / 4096.0f);
            pen_raw_prev = raw;
            encoder_sample_valid = true;
            consecutive_bad_encoder_samples = 0;
            last_valid_encoder_us = last_sample_us;
        }
    }

    // A stationary pendulum can legitimately repeat the same encoder code.
    // Only flag a frozen sensor after the motor has traversed two full turns
    // with no encoder-code change, then remained unchanged for another second.
    if (frozen_reference_raw < 0 || raw != frozen_reference_raw)
    {
        frozen_reference_raw = raw;
        frozen_reference_motor_steps = motor_step;
        frozen_reference_ms = now_ms;
    }
    else if (motor_engaged &&
             labs((long)motor_step - (long)frozen_reference_motor_steps) >=
                 FROZEN_ENCODER_MOTOR_TRAVEL_STEPS &&
             (uint32_t)(now_ms - frozen_reference_ms) >= FROZEN_ENCODER_CONFIRM_MS)
    {
        safety_disarm((uint8_t)FAULT_ENCODER_FROZEN);
    }

    motor_step_buf[buf_head] = motor_step;
    pen_rad_buf[buf_head]    = pen_position_rad;
    time_us_buf[buf_head]    = last_sample_us;
    buf_head = (buf_head + 1) % BUFFER_SIZE;
    if (buf_head == 0) buf_filled = true;
}

/*
 * Compute velocity over the most recent VEL_WINDOW samples as
 * (newest - oldest)/Δt. Returns 0 until the buffer holds enough samples.
 */
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s)
{
    uint8_t n_samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (n_samples < VEL_WINDOW)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1)            % BUFFER_SIZE);
    uint8_t oldest = (uint8_t)((buf_head + BUFFER_SIZE - VEL_WINDOW)   % BUFFER_SIZE);

    uint32_t t_new = time_us_buf[newest];
    uint32_t t_old = time_us_buf[oldest];
    float dt_s = (float)((uint32_t)(t_new - t_old)) * 1e-6f;
    if (dt_s <= 0.0f)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    int32_t motor_step_delta = motor_step_buf[newest] - motor_step_buf[oldest];
    *motor_vel_rad_s = ((float)motor_step_delta * ((2.0f * PI) / (float)STEPS_PER_REVOLUTION)) / dt_s;

    float pen_delta = pen_rad_buf[newest] - pen_rad_buf[oldest];
    *pen_vel_rad_s = pen_delta / dt_s;
}

// Safety-owned protocol hooks. The parser calls these only after frame, CRC,
// sequence and payload validation. No hook blocks waiting for serial input.
bool safety_is_disarmed() { return !motor_engaged; }
bool safety_is_armed() { return motor_engaged; }
uint16_t safety_fault_flags() { return latched_fault_flags; }

uint16_t safety_health_flags()
{
    uint16_t flags = 0;
    uint32_t now_us = micros();
    uint8_t samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (stepper != NULL) flags |= HEALTH_STEPPER_READY;
    if (motor_engaged) flags |= HEALTH_ARMED;
    if (encoder_magnet_detected &&
        (uint32_t)(millis() - last_magnet_check_ms) <= 3UL * MAGNET_CHECK_MS)
        flags |= HEALTH_MAGNET_DETECTED;
    if (samples >= VEL_WINDOW && encoder_sample_valid &&
        last_valid_encoder_us != 0 &&
        (uint32_t)(now_us - last_valid_encoder_us) <= ENCODER_STALE_US)
        flags |= HEALTH_ENCODER_VALID;
    if (samples >= VEL_WINDOW && last_sample_us != 0 &&
        (uint32_t)(now_us - last_sample_us) <= ENCODER_STALE_US)
        flags |= HEALTH_SAMPLER_FRESH;
    return flags;
}

void safety_disarm(uint8_t reason)
{
    if (reason != 0) latched_fault_flags |= (uint16_t)reason;
    motor_engaged = false;
    last_valid_accel_ms = 0;
    if (stepper != NULL)
    {
        stepper->forceStop();
        stepper->disableOutputs();
    }
}

uint8_t safety_arm()
{
    if (latched_fault_flags != 0) return ProtocolV1::STATUS_FAULT_LATCHED;
    if (motor_engaged) return ProtocolV1::STATUS_OK;
    if (stepper == NULL ||
        (safety_health_flags() & REQUIRED_ARM_HEALTH) != REQUIRED_ARM_HEALTH)
        return ProtocolV1::STATUS_INVALID_VALUE;

    int32_t current_steps = stepper->getCurrentPosition();
    int32_t hard_limit_steps = (int32_t)(132.0f * STEPS_PER_REVOLUTION / 360.0f);
    if (labs((long)current_steps) >= hard_limit_steps)
    {
        latched_fault_flags |= FAULT_HARD_LIMIT;
        return ProtocolV1::STATUS_FAULT_LATCHED;
    }

    motor_engaged = true;
    last_valid_accel_ms = millis();
    stepper->enableOutputs();
    stepper->moveByAcceleration(0, true);
    return ProtocolV1::STATUS_OK;
}

uint8_t safety_set_acceleration(float accel_rad_s2)
{
    if (!motor_engaged) return ProtocolV1::STATUS_NOT_ARMED;
    if (!isfinite(accel_rad_s2) || fabsf(accel_rad_s2) > 150.0f)
        return ProtocolV1::STATUS_INVALID_VALUE;

    int32_t current_steps = stepper->getCurrentPosition();
    int32_t hard_limit_steps = (int32_t)(132.0f * STEPS_PER_REVOLUTION / 360.0f);
    if (labs((long)current_steps) >= hard_limit_steps)
    {
        safety_disarm((uint8_t)FAULT_HARD_LIMIT);
        return ProtocolV1::STATUS_FAULT_LATCHED;
    }

    int32_t accel_steps_s2 = (int32_t)(accel_rad_s2 * STEPS_PER_RAD);
    if (current_steps >= MOTOR_SAFE_LIMIT_STEPS)
        accel_steps_s2 = -MOTOR_BRAKE_ACCEL_STEPS_S2;
    else if (current_steps <= -MOTOR_SAFE_LIMIT_STEPS)
        accel_steps_s2 = MOTOR_BRAKE_ACCEL_STEPS_S2;
    stepper->moveByAcceleration(accel_steps_s2, true);
    return ProtocolV1::STATUS_OK;
}

uint8_t safety_tare()
{
    if (motor_engaged) return ProtocolV1::STATUS_NOT_DISARMED;
    if (!encoder_sample_valid || !encoder_magnet_detected)
        return ProtocolV1::STATUS_INVALID_VALUE;
    noInterrupts();
    float offset = pen_position_rad;
    for (uint8_t i = 0; i < BUFFER_SIZE; ++i) pen_rad_buf[i] -= offset;
    pen_position_rad = 0.0f;
    interrupts();
    return ProtocolV1::STATUS_OK;
}

uint8_t safety_zero()
{
    if (motor_engaged) return ProtocolV1::STATUS_NOT_DISARMED;
    if (stepper == NULL) return ProtocolV1::STATUS_INVALID_VALUE;
    noInterrupts();
    int32_t offset = stepper->getCurrentPosition();
    for (uint8_t i = 0; i < BUFFER_SIZE; ++i) motor_step_buf[i] -= offset;
    stepper->setCurrentPosition(0);
    interrupts();
    return ProtocolV1::STATUS_OK;
}

void safety_note_valid_accel(uint32_t now_ms)
{
    last_valid_accel_ms = now_ms;
}

void safety_check_command_lease(uint32_t now_ms)
{
    if (motor_engaged &&
        (uint32_t)(now_ms - last_valid_accel_ms) >= COMMAND_LEASE_MS)
        safety_disarm((uint8_t)FAULT_LEASE_EXPIRED);

    if (motor_engaged && last_sample_us != 0 &&
        (uint32_t)(micros() - last_sample_us) > ENCODER_STALE_US)
        safety_disarm((uint8_t)FAULT_ENCODER_STALE);

    if (motor_engaged && stepper != NULL)
    {
        int32_t current_steps = stepper->getCurrentPosition();
        int32_t hard_limit_steps = (int32_t)(132.0f * STEPS_PER_REVOLUTION / 360.0f);
        if (labs((long)current_steps) >= hard_limit_steps)
        {
            safety_disarm((uint8_t)FAULT_HARD_LIMIT);
            return;
        }

        // Keep the ±125° rail brake active even if the host stops sending
        // commands between packets. Use the sampled velocity to avoid braking
        // motion that is already moving back into the allowed range.
        float motor_vel = 0.0f, pen_vel = 0.0f;
        computeVelocities(&motor_vel, &pen_vel);
        if (current_steps >= MOTOR_SAFE_LIMIT_STEPS && motor_vel > 0.0f)
            stepper->moveByAcceleration(-MOTOR_BRAKE_ACCEL_STEPS_S2, true);
        else if (current_steps <= -MOTOR_SAFE_LIMIT_STEPS && motor_vel < 0.0f)
            stepper->moveByAcceleration(MOTOR_BRAKE_ACCEL_STEPS_S2, true);
    }
}

bool safety_get_state(uint32_t* time_us, float* motor_pos, float* pend_pos,
                      float* motor_vel, float* pend_vel,
                      uint16_t* health_flags, uint16_t* fault_flags)
{
    uint8_t samples = buf_filled ? BUFFER_SIZE : buf_head;
    *health_flags = safety_health_flags();
    *fault_flags = latched_fault_flags;
    if (samples < VEL_WINDOW || stepper == NULL) return false;
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    *time_us = time_us_buf[newest];
    *motor_pos = (float)motor_step_buf[newest] * RAD_PER_STEP;
    *pend_pos = pen_rad_buf[newest];
    computeVelocities(motor_vel, pend_vel);
    return isfinite(*motor_pos) && isfinite(*pend_pos) &&
           isfinite(*motor_vel) && isfinite(*pend_vel);
}

// Binary command parsing/dispatch and state serialization are in ProtocolV1.h.
