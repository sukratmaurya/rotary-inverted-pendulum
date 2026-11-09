# System interface specification

This document is the shared deployment profile. The byte-level tethered protocol is maintained in [`protocol/v1.md`](../protocol/v1.md); changes to it must update the Python client, firmware parser, tests, and that protocol document together.

## Hardware and control profile

| Interface | Current source value | Source of truth |
|---|---:|---|
| MCU | ATmega328P, 16 MHz Nano | Arduino Nano build target |
| Driver / motor | TMC2209 / NEMA 17 | Hardware commissioning configuration; firmware does not identify the driver electrically |
| Encoder | AS5600, 12 bit / 4096 codes per revolution | `LowLevelServer.ino`, `RLControl.ino` |
| Motor microsteps | 32; 6400 steps/revolution | Both firmware sketches and `policy/pendulum_env.py` |
| Tethered baud | 2,000,000 | `LowLevelServer.ino`, `LowLevelClient`, `real_env.py`, `run_policy.py` |
| Standalone diagnostic baud | 500,000 | `RLControl.ino`; separate text telemetry interface, not protocol v1 |
| Sensor sampling | 500 Hz nominal (2,000 µs) | Both firmware sketches |
| Velocity window | 5 samples; 8 ms nominal span | Both firmware sketches and simulation sensor model |
| Policy loop | 50 Hz | Training 4 config, `run_policy.py`, RLControl |
| Observation | 24 float32 values: 4 oldest-to-newest frames × `[motor position, sin(theta), cos(theta), motor velocity, pendulum velocity, previous raw action]` | Training 4 config and RLControl |
| Upright convention | `theta = wrap_pi(phi - pi)`; theta 0 / cosine +1 is upright | Simulation, deploy controller, RLControl |
| Student action | tanh output in [-1, 1]; velocity setpoint × 3.5 rad/s | Training 4 config and RLControl |
| Acceleration cap | ±150 rad/s² | Training 4 config, environment, firmware |
| Action smoothing | 4-sample boxcar at actuator input; raw action remains in observation | Training 4 config, simulation, RLControl |
| Position soft / hard limits | ±125° / ±132° | Environment and both firmware safety paths |
| Tethered command lease | 100 ms; only accepted in-range SET_ACCEL refreshes it | Protocol v1 and LowLevelServer |

Training 4's checked-in run config records velocity mode, 4 observation frames, velocity observations, 50 Hz, 3.5 rad/s, 150 rad/s², 32 microsteps, and 4-sample smoothing. The standalone header is 24→16→16→1, ReLU/ReLU/tanh. The header's source comment names a distillation run, but that exact `student.pt` artifact is not in the repository, so checkpoint provenance has not been independently verified.

## Protocol v1 invariants

Frames are COBS records terminated by `0x00`, with version/type/u16 little-endian sequence/u8 payload length, payload, then little-endian CRC-16/CCITT-FALSE (poly `0x1021`, init `0xffff`, xorout 0). Payload maximum is 24 bytes. Response sequence must equal the request sequence. Sequence numbers advance modulo 16 bits; duplicate and backward commands are rejected except a valid DISARM is always honored. HELLO is disarmed-only and never arms. ARM requires a distinct valid ARM frame and healthy sensor/stepper state. Malformed, truncated, unknown, non-finite, wrong-length, or out-of-range commands do not call the valid-acceleration lease-refresh hook.

## State estimation and frame

Firmware unwraps AS5600 raw angle deltas across 0/4095, integrates radians using `2*pi/4096`, and rejects a single sample delta beyond 500 codes. The estimator returns newest position and finite-difference velocity over the oldest/newest samples in its 5-entry ring. Invalid samples retain the prior position, clear encoder-valid health, and repeated bad reads fault while armed. Freeze detection is intentionally conservative: repeated raw readings alone are allowed at rest; an armed motor must travel at least 6400 steps while the value remains unchanged for at least 1000 ms.

Motor coordinates are the raw stepper frame, with no Python-side sign reversal. Pendulum zero is set at engage in standalone mode; tethered tare requires disarmed operator action. Python rejects stale host-received state (80 ms default), missing required health bits, non-finite values, and any reported fault before a live policy action.

## Safety boundary

Boot must leave outputs disabled and control disarmed. The tethered arm command and standalone `E` command are explicit operator actions. Lease, sensor, timing, and hard-limit faults must stop software motion and latch as implemented. Fault clear never arms. The independent physical E-stop that removes driver enable or motor power remains mandatory; its wiring and effectiveness are unverified in software.
