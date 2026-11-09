# Final pre-hardware code review

Date: 2026-11-10. Review scope: current `LowLevelServer.ino`, `RLControl.ino`, `ProtocolV1.h`, installed AS5600 0.6.7 / FastAccelStepper 2.0.1 / Arduino AVR core 1.8.8 sources, current policy inference, fault transitions, and the offline verifier. No serial port was opened; no firmware was uploaded; no hardware or motor was accessed.

## Verdict summary

| Component | Verdict | Review result |
|---|---|---|
| Wire timeout handling | **PASS WITH LIMITATIONS** | 3 ms bounds core TWI waits and resets TWI on timeout. A timeout can exceed the nominal 2 ms sample period and no timing behavior was measured. |
| AS5600 `lastError()` handling | **PASS WITH LIMITATIONS** | Matches the installed library semantics; source inspection only, not a physical I2C fault test. |
| FastAccelStepper allocation | **PASS** | AVR uses a statically allocated pool; no runtime heap allocation/fragmentation in this build. |
| FastAccelStepper timing/ISR behavior | **PASS WITH LIMITATIONS** | Correct Timer1/cyclic ISR integration found; ISR nesting and timing are not bounded/measured here. |
| Protocol v1 command/lease path | **PASS WITH LIMITATIONS** | Source and offline client tests agree on framing/lease intent; the Arduino dispatcher has not been run in a host firmware harness. |
| RLControl policy mathematics | **PASS WITH LIMITATIONS** | Latest changes do not alter policy math; AVR compile and host parity pass, but AVR execution parity is not verified. |
| RLControl SRAM/stack safety | **NOT VERIFIED** | 759 B global-free SRAM; no release-LTO maximum stack bound or stack watermark. |
| Command watchdog / main-loop failure recovery | **PASS WITH LIMITATIONS** | Tethered 100 ms command lease exists, but it only runs when the loop runs. There is no MCU watchdog enabled. |
| Fault clearing and re-arming | **PASS WITH LIMITATIONS** | Standalone requires C then a separate E; tethered latched faults persist until reset, which boots disarmed. This reset path and physical outputs were not tested. |
| Full offline verification | **PASS** | `./scripts/verify_offline.sh`: 21 Python tests, host policy comparison, both Nano compile-only builds. |
| Hardware commissioning readiness | **NOT VERIFIED** | Code review and offline evidence do not establish stack margin, real-time deadlines, E-stop function, or safe physical torque behavior. |

## 1. Wire timeout and AS5600 errors

The installed Arduino AVR core's `TwoWire::setWireTimeout(timeout, reset)` configures the TWI driver's timeout checks in its synchronous wait loops. On expiry it sets an internal timeout flag and, with reset enabled, resets TWI and restores the prior bitrate/address registers. The firmware sets 3,000 µs after `Wire.begin()` and `Wire.setClock(400000)` in both sketches.

The AS5600 0.6.7 implementation was inspected directly:

- `rawAngle()` calls `readReg2()`, applies offset/direction, then masks the result with `0x0FFF`. A failed read can therefore return a plausible 12-bit value (typically zero) rather than an out-of-range sentinel.
- `readReg2()` resets `_error` at the start, sets a nonzero error if `endTransmission()` fails or `requestFrom()` returns fewer than two bytes, and otherwise returns the two-byte reading.
- `lastError()` returns `_error` and then resets it to `AS5600_OK`.
- Both firmware paths call `lastError()` immediately after `rawAngle()` and convert any non-OK result into the existing invalid-sample path. This is correctly ordered and consumes the read's error before another AS5600 method overwrites it.
- `detectMagnet()` calls `readStatus()`. A failed status register read yields zero through `readReg()` and thus reports no detected magnet. That blocks arm/clear health checks; while running, magnet loss faults/disarms on the next magnet check.

The timeout is not a 2 ms sample deadline. It permits a blocking call to last roughly 3 ms, longer than the nominal sampler period, and it does not ensure all recovery work finishes within the 20 ms policy tick. The timeout result is surfaced through AS5600's error state for the angle read; code does not explicitly query `Wire.getWireTimeoutFlag()`, but the library's `readReg2()` return-count/error path catches failed angle transactions. The separate timeout flag is therefore diagnostic-only at present. Real bus recovery, clock-stretch behavior, and the actual worst-case wait require hardware measurement.

**Verdict: PASS WITH LIMITATIONS.** The timeout does not by itself mark the sensor healthy: the library error is checked and invalidates the angle sample. Timing remains unmeasured, and the 3 ms bound can cause a missed nominal sample.

## 2. Fault transitions

| Event/state | LowLevelServer (tethered) | RLControl (standalone) | Verdict |
|---|---|---|---|
| Boot/reset | `motor_engaged=false`; after successful stepper initialization, outputs are disabled. Sensor absence no longer blocks setup. No automatic arm. | `state=WAITING`; after successful stepper setup, outputs are disabled. No automatic arm. | **PASS WITH LIMITATIONS** — software paths are disarmed after setup; electrical EN state during MCU reset/early setup depends on hardware pin bias and was not verified. |
| Disarmed | GET_STATE/HELLO allowed per protocol constraints; ARM requires fresh sampler, valid encoder, ready stepper, magnet, no latched fault, and position inside hard limit. | E requires WAITING, healthy encoder/magnet/fresh samples, no fault, and position inside soft limit. | **PASS WITH LIMITATIONS** — source review only. |
| Armed/running | Accepted SET_ACCEL commands refresh the 100 ms lease. Acceleration is finite and bounded to ±150 rad/s². | Policy runs autonomously at 50 Hz. USB serial is diagnostics/operator input, not a heartbeat; loss of USB does not stop autonomous control. | **PASS WITH LIMITATIONS** — tethered lease is main-loop serviced; standalone has no communication-loss stop because communication is not part of the control loop. |
| Sensor fault | I2C read errors are invalidated. Three consecutive bad/jump samples while engaged cause a latched encoder-invalid fault; stale sensor and missing magnet also disarm/fault. | A bad read clears encoder validity; control tick faults on invalid/stale state, with additional invalid/frozen/magnet handling. | **PASS WITH LIMITATIONS** — offline fixture tests model thresholds; actual sensor/bus path not fault-injected. |
| Communication timeout | A missing valid SET_ACCEL reaches the 100 ms lease and disarms with a latched lease fault. Parser partial frames time out after 25 ms and are discarded. | Not applicable as a control heartbeat. | **PASS WITH LIMITATIONS** — a blocked main loop delays lease servicing; no MCU watchdog is configured. |
| Hard limit | Hard-limit checks in command path and periodic safety loop disarm and latch. Soft rail applies opposing acceleration. | Crossing the hard limit latches FAULT and disables outputs; soft rail inhibits/brakes outward motion. | **PASS WITH LIMITATIONS** — no physical rail test. |
| Watchdog reset | No application `wdt_enable()`/`wdt_reset()` use found. | Same; no application watchdog. | **NOT VERIFIED** as a watchdog recovery feature because none is enabled. A normal MCU reset restarts firmware disarmed, but electrical reset/driver behavior is untested. |
| Fault clear | There is no protocol CLEAR_FAULT command. Latched faults remain until an operator resets the MCU; startup is disarmed. | C clears only an eligible latched fault while sensor and position checks pass. C does not arm; a separate E is required. Hard-limit fault also requires the explicit Z re-zero process first. | **PASS WITH LIMITATIONS** — deliberate action sequence is present; physical re-zero/operator procedure not validated. |
| Re-arm | ARM rejected while any volatile fault bit remains or required health is missing. Reset clears RAM faults but does not auto-arm. | E is accepted only from healthy WAITING with no latched fault; FAULT must first clear to WAITING. | **PASS WITH LIMITATIONS** — reset is an implicit fault clear for tethered firmware; a reset is deliberate but not a dedicated clear handshake. No physical re-arm test. |

No firmware watchdog means loop stalls are not independently recovered. Tethered lease, encoder staleness, and RLControl overrun checks all depend on main-loop progress. The hardware E-stop remains the independent stop path and must remove driver enable/power without firmware.

## 3. FastAccelStepper allocation and ISR integration

The earlier embedded-resource report incorrectly described the 93-byte `FastAccelStepper` object as heap-allocated on AVR. Source review of the installed FastAccelStepper 2.0.1 corrects this:

- `SUPPORT_DYNAMIC_ALLOCATION` is limited to its ESP32 IDF 5.3+ branch.
- AVR uses `FastAccelStepper fas_stepper[MAX_STEPPER]` static storage; `stepperConnectToPin()` returns a pointer to an available pool element.
- Nano's selected AVR configuration defines `MAX_STEPPER=2`. Each pool element is 93 bytes by an AVR-targeted `sizeof` compile probe. Those 186 bytes are already included in the firmware compiler's global SRAM total.
- No `malloc`, `new`, `String`, or recurring allocation was found in the active AVR application/library paths. There is no runtime fragmentation risk from stepper allocation on this target. The app owns no other heap use found in this review.

STEP output is driven through the FastAccelStepper AVR Timer1 compare ISR and queue replenishment through a cyclic timer ISR. The cyclic ISR disables its own overflow interrupt and then executes `sei()` while calling `manageSteppers()`, permitting other interrupts (including step compare, Timer0, UART, and TWI) to nest. This is normal library architecture, but it increases interrupt stack depth and timing interference. No ISR WCET or stack high-water was measured.

**Allocation verdict: PASS. ISR/timing verdict: PASS WITH LIMITATIONS.**

## 4. SRAM/stack and policy review

Latest verifier totals: LowLevelServer 20,586 B flash / 1,024 B globals; RLControl 27,574 B flash / 1,289 B globals. RLControl has 759 B between global SRAM and the 2,048 B SRAM limit. The 93-byte object is static, already included, and must not be subtracted again.

Non-LTO AVR GCC `-fstack-usage` instrumentation reported RLControl `policy_forward()` at 152 B and `loop()` at 39 B; LowLevelServer `loop()` at 104 B and `ProtocolV1::send_frame()` at 72 B. FastAccelStepper ISR frames were 16/18 B for compare and 19 B for cyclic overflow. These values do not form a complete worst-case bound: the instrumented build differs from release LTO, libc/libm and TWI C frames/nested callees are incomplete, and cyclic ISR nesting complicates the maximum. RLControl's 759 B free after globals is not sufficient evidence by itself to claim safety.

Policy shape remains 24→16→16→1, ReLU/ReLU/tanh, weights in PROGMEM, two 16-float activation buffers. The sampler ring reduction from 16 to 5 retains the exact five samples used by the estimator; no observation ordering, normalization, action scaling, smoothing, velocity law, or policy weight math changed in the reviewed changes. Wire timeout and lastError checks only alter failed-I2C handling. A compile-time `sizeof(float)==4` assertion protects the activation format.

The host validator still reports 93 observations, PyTorch-vs-NumPy max error `5.36e-7`, and PyTorch-vs-host-C++ max error `4.02e-6`. This is not an AVR runtime parity result. AVR float width and code generation are compile-verified; AVR `tanhf` numerical output has not been executed or compared. Exact student-checkpoint provenance remains unavailable.

**SRAM/stack verdict: NOT VERIFIED. Policy verdict: PASS WITH LIMITATIONS.**

## 5. Protocol/lease behavior

Protocol v1 validates frame size, version, CRC, sequence progression, command payload length, finite/range-checked acceleration, and explicit ARM. Only `safety_set_acceleration()` returning OK reaches `safety_note_valid_accel()`, so malformed, unknown, invalid, unarmed, or out-of-range commands do not refresh the lease. DISARM remains a safety-priority idempotent stop. Lease expiration is checked from the main loop and latches a fault.

Offline Python tests cover framing/CRC, malformed records, stale sequence responses, invalid numerical input, and missing responses. These tests exercise the host codec/client; they do not execute the Arduino `ProtocolV1.h` dispatcher. Source review supports the lease invariant, but compiled firmware fault injection remains unverified.

**Verdict: PASS WITH LIMITATIONS.**

## 6. Verification run

`./scripts/verify_offline.sh` completed successfully on 2026-11-10. It passed 21 Python tests, the host policy comparison on 93 observations, and compile-only builds of both Nano sketches. Output: LowLevelServer 20,586 B flash / 1,024 B globals; RLControl 27,574 B flash / 1,289 B globals. The builds used temporary directories and did not upload. No serial access or external hardware access occurred.

## Final safety disposition

The source changes reviewed here are internally consistent for the AS5600 error path, protocol lease policy, and unchanged policy mathematics. Static FastAccelStepper allocation removes the previously misstated heap-fragmentation concern. **The firmware is not established safe for hardware operation by this review.** Stack margin, deadlines, real I2C fault recovery, electrical boot-disable state, E-stop behavior, and actual fault-stop mechanics remain unverified and must be addressed under a supervised commissioning plan before enabling motor power.
