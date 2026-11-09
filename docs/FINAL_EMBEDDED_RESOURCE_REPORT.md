# Final embedded resource report

Date: 2026-11-10. Scope: source inspection, AVR Nano compile-only builds, compiler stack-frame reports, and host numerical tests. No serial port, upload, or physical hardware was accessed.

## Decision

**Software compile and host verification pass, but stack headroom and real-time timing are not proven. Treat the system as not yet cleared for motor-enabled commissioning.** Before supervised motion, obtain a release-build stack high-water result (or a complete AVR call-graph stack bound including libraries and nested ISRs) and measure control/sampler timing on a guarded, current-limited setup. The independent physical E-stop remains mandatory.

## AVR build resource totals

The normal Arduino CLI builds used Arduino AVR Boards core 1.8.8, AVR GCC 7.3.0-atmel3.6.1-arduino7, FastAccelStepper 2.0.1, AS5600 0.6.7, and `arduino:avr:nano:cpu=atmega328` (ATmega328P, 16 MHz). These are fresh compile-only results after the current sensor timeout/error-handling changes.

| Sketch | Flash used / limit | Flash remaining | Global SRAM / 2,048 B | SRAM remaining before heap/stack |
|---|---:|---:|---:|---:|
| LowLevelServer | 20,586 / 30,720 B (67%) | 10,134 B | 1,024 B (50%) | 1,024 B |
| RLControl | 27,574 / 30,720 B (89%) | 3,146 B | 1,289 B (62%) | 759 B |

Flash limit is the Nano application region reported by the board toolchain, not the full 32 KiB including bootloader reservation. RLControl has limited flash growth room; changing optimization flags changes code size and is not equivalent to the deploy build.

## SRAM, heap, and stack

### Static SRAM and heap

RLControl’s main SRAM residents include the four observation frames (96 B), five-entry synchronized motor/encoder/time sample ring (60 B), four-action smoothing ring (16 B), telemetry line (128 B), control state, Arduino UART rings, Wire state, FastAccelStepper engine/queue state, and library globals. The policy weights are `PROGMEM` and do not occupy SRAM. LowLevelServer has the same five-entry sample ring (60 B), protocol RX frame storage, state and fault tracking, plus core and library state.

The firmware contains no `String` use and no recurring allocation path. Correction from the final code review: although one FastAccelStepper source branch uses `new` for ESP32 IDF 5.3+, the installed AVR branch uses static allocation. `fas_stepper[MAX_STEPPER]` is global storage, and `stepperConnectToPin()` selects an element from that array on AVR. `MAX_STEPPER` is 2 for the Nano timer configuration. The AVR-targeted `sizeof(FastAccelStepper)` probe measured each pool element at 93 B; both elements are already included in the compiler's global SRAM figure. There is no runtime heap allocation or fragmentation from FastAccelStepper in this AVR build. Do not subtract another 93 B from the compiler-reported free SRAM. The remaining concern is stack use and its collision with statically allocated SRAM, not heap growth.

### Compiler stack data and estimate

AVR GCC `-fstack-usage` produced useful per-function figures only in separate `-Os -fno-lto` instrumentation builds; LTO is enabled in the deployment build and GCC emitted empty `.su` reports for it. The non-LTO binaries are not deployment artifacts (RLControl grew to 30,320 B flash, close to the 30,720 B app limit; LowLevelServer was 23,284 B). These frame sizes are therefore estimates, not a proof for the release image.

- RLControl: `policy_forward()` reported 152 B; `loop()` 39 B; `update_sample_buffer()` 12 B; `read_pendulum_rad()` 8 B; AS5600 `rawAngle()` and `readReg2()` reported 4 B and 6 B. The release ELF prologue for `policy_forward` reserves 132 B of local activation storage and saves 18 registers, consistent with the two 16-float hidden layers plus compiler temporaries. Inference therefore has a large contiguous stack frame.
- LowLevelServer: `loop()` reported 104 B; protocol `send_frame()` 72 B (64 B raw/encoded packet work area plus registers); `sampleState()` 16 B; `computeVelocities()` 18 B. The release ELF’s `send_frame` reserves 64 B local storage and saves six registers.
- FastAccelStepper AVR interrupt frames in the instrumented build: compare ISRs reported 16/18 B, cyclic overflow ISR 19 B, and UART RX ISR 17 B. The cyclic ISR explicitly enables nested interrupts while it calls `manageSteppers()`. Timer0 and TWI interrupts may also run during this interval. `manageSteppers`, `fill_queue`, ramp planning, and deeper helper frames add to the ISR call path.

For RLControl, an inference-time main path is already about 191 B from the `loop` and `policy_forward` reports before nested sample/I2C helpers; the AVR TWI C frames, libm internals (`sinf`, `cosf`, `tanhf`), return-address details, and interrupts are not all represented in those C++ `.su` totals. Adding the nested stepper cyclic/compare ISR path yields a rough **350–450 B** concurrent-stack estimate. This is not a hard upper bound. For LowLevelServer, loop plus `send_frame` is at least about 176 B on the instrumented paths; parser dispatch, sample/I2C work, and nested interrupt service suggest roughly **300–400 B**. This estimate also is not a hard bound.

Subtracting the rough stack estimates from compiler-reported free SRAM leaves approximately 309–409 B in RLControl and 624–724 B in LowLevelServer before unaccounted library/libm frames and instrumentation uncertainty. Because RLControl’s margin may be only a few hundred bytes and nested-stack maxima have not been proven, SRAM sufficiency cannot be certified from current evidence. No stack watermark has been collected.

## Neural-network resource requirements

RLControl’s deployed MLP is 24→16→16→1 with ReLU/ReLU/tanh. Its six float32 weight/bias arrays contain 689 floats (2,756 bytes) and are read from `PROGMEM`. Inference uses two 16-float activation arrays (128 B), approximately 656 multiply-accumulate operations, a tanh, and repeated weight reads from program memory. AVR `float` is 32-bit; a compile-time assertion now rejects a target/compiler with another size. The AVR compiles the exact firmware inference path successfully.

Host validation on 93 deterministic boundary observations reported PyTorch-vs-NumPy maximum absolute error `5.36e-7` and PyTorch-vs-host-C++ maximum absolute error `4.02e-6`. This validates parsed-header arithmetic on the host. No AVR simulator was installed, so AVR execution, AVR `tanhf` agreement, and policy output parity on an ATmega328P were **not** measured. The exact source student checkpoint remains unavailable; header provenance is declared in its comment but not byte-verified against a checkpoint.

## Real-time execution review

- RLControl schedules a 50 Hz policy tick (20 ms period) and services the 500 Hz sampler from the main loop and between hidden-layer rows. The sampler is cooperative, not interrupt-driven. Long control work can delay sampling; sensor freshness and control-overrun checks can fault, but actual worst-case timing has not been measured here.
- Both sketches use synchronous AS5600 reads over Wire at 400 kHz. Wire's default timeout was disabled in the installed Arduino core. Both setups now call `Wire.setWireTimeout(3000, true)`, bounding a stuck TWI wait to about 3 ms and resetting TWI on timeout. The sample loop can miss its 2 ms nominal period during that event; this is fault containment, not proof of real-time deadline compliance.
- Both sampling paths now check `AS5600.lastError()` immediately after `rawAngle()`. The installed library returns a masked 12-bit value on I2C read failure, so range checking alone could accept a disconnected sensor as angle zero. The read error is now converted into the existing invalid-sample path.
- RLControl telemetry uses a fixed line buffer and checks `Serial.availableForWrite()` before each chunk, avoiding a full-line wait in the main loop. Command diagnostics and setup prints use synchronous `Serial.print/println`; these can block if the UART transmit ring fills. LowLevelServer's framed response uses `Serial.write`; normal protocol frames are bounded, but a full UART TX buffer can also block.
- FastAccelStepper generates steps with Timer1 compare ISR and replenishes its queue in a cyclic timer ISR. The cyclic ISR enables nested interrupts while managing steppers. The motor pulse path is therefore independent of the policy loop, but interrupt occupancy and interference with I2C/serial are not timing-measured.
- Neither application enables a watchdog. The communication lease and loop overrun detection only run when the main loop runs; they cannot recover from an indefinitely blocked call. A watchdog was not added because reset/torque loss behavior must be considered mechanically, and the physical E-stop is an independent requirement.
- No worst-case execution time, jitter distribution, 50 Hz deadline compliance, 500 Hz sample interval distribution, UART blocking duration, or motor-step ISR latency has been measured in this report.

## Verification performed

`./scripts/verify_offline.sh` completed successfully. The verifier uses temporary build directories; it compiles only and contains no upload or serial-opening path. Results: 21 Python tests passed; 93-observation policy/header comparison passed; LowLevelServer built at 20,586 B flash / 1,024 B globals; RLControl built at 27,574 B flash / 1,289 B globals. The verifier now supports `ARDUINO_CLI_BIN` and `ARDUINO_LIBRARIES_DIR` overrides and locates the bundled Arduino IDE CLI when the Snap launcher is unusable.

The AVR ELF and stack-inspection builds were written under `/tmp`; no build products were placed in the project or original repository. Stack-instrumentation builds use different flags and must not be treated as production images.

## Remaining hardware requirements

- Measure stack high-water under simultaneous policy inference, TWI sampling, serial traffic, and timer interrupts; inspect the deployed LTO binary/toolchain and record a safe margin.
- Measure actual 50 Hz control period, 500 Hz sample spacing, inference duration, interrupt latency, and serial blocking on the intended Nano/driver with a guarded, current-limited commissioning setup.
- Verify AS5600 direction, magnet health, I2C timeout recovery, and sensor-disconnect fault response on the actual unit.
- Verify driver enable polarity, motor current, step timing, mechanical limits, and all fault stop behavior with an operator at the independent E-stop.
- Verify that the independent physical E-stop removes motor power/driver enable independently of firmware and USB.
- Do not interpret compile success, host numerical parity, or simulation as hardware validation.
