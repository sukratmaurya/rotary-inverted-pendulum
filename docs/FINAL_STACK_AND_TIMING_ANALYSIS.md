# Final ATmega328P stack and timing analysis

Date: 2026-11-10. Scope: fresh LTO Nano ELFs and maps, AVR disassembly, non-LTO `-fstack-usage` builds, firmware and library source, and the offline verification run. Target is ATmega328P at 16 MHz (`arduino:avr:nano:cpu=atmega328`). No serial port or hardware was accessed; nothing was uploaded and no motor operation occurred.

## Verdicts

| Area | Verdict | Basis |
|---|---|---|
| SRAM safety | **PASS WITH LIMITATIONS** | Static globals are 1,289 B (RLControl) and 1,024 B (LowLevelServer); free SRAM before stack is 759 B and 1,024 B. Heap growth was not found, but peak stack/high-water is not established. |
| Stack safety | **NOT VERIFIED** | LTO disassembly exposes substantial frames and important ISR nesting. Available compiler reports are non-LTO; there is no complete bounded call-graph proof or runtime stack watermark. |
| ISR safety | **PASS WITH LIMITATIONS** | Timer1 queue ISR behavior and entry frames were inspected. The cyclic ISR enables nesting and calls substantial queue/ramp code; stack and execution-time bounds are not proven. |
| Control-loop timing | **PASS WITH LIMITATIONS** | The late-tick path now skips/faults beyond a 1 ms lateness allowance and never replays missed slots. Actual deadline performance remains unmeasured. |
| I2C fault recovery | **PASS WITH LIMITATIONS** | 3 ms Wire timeout and immediate AS5600 error handling contain failed reads in source, but a timeout exceeds the 2 ms sample period and recovery timing was not measured. |
| Overall hardware readiness | **FAIL** | Late-tick behavior is corrected and offline-tested, but release-build stack and target timing margins remain unmeasured. Hardware behavior and the independent physical E-stop remain unverified. |

## Artifacts and method

Release-configuration LTO ELFs and linker maps were built in `/tmp/rip-stack-lto-rl` and `/tmp/rip-stack-lto-ll` with the Arduino AVR toolchain and current installed libraries. `avr-nm` and `avr-objdump` from AVR GCC 7.3.0-atmel3.6.1-arduino7 were used to inspect release symbols and actual prologues. Separate `-fno-lto -fstack-usage` builds were placed in `/tmp/rip-stack-su-rl` and `/tmp/rip-stack-su-ll`. Those `.su` reports are useful for function frame clues but do not describe the LTO release call graph and are not used as a release worst-case bound.

The Nano linker map and Arduino size report agree with the offline verifier:

| Sketch | Flash | Global/static SRAM | SRAM left for heap + stack |
|---|---:|---:|---:|
| RLControl | 27,808 B / 30,720 B | 1,289 B / 2,048 B | 759 B |
| LowLevelServer | 20,586 B / 30,720 B | 1,024 B / 2,048 B | 1,024 B |

The SRAM remainder is not a stack allowance: it is shared by stack and any heap. The inspected AVR FastAccelStepper path allocates its stepper objects in the static `fas_stepper[MAX_STEPPER]` pool (`MAX_STEPPER=2`), not with runtime `new`. The 93 B per-object size is already included in global SRAM. No recurring `String` or dynamic-allocation path was found. Thus no heap fragmentation risk was identified in the reviewed application/library path, but generic libc allocation was not proven unreachable by a formal whole-program analysis. Both firmware paths should be treated as static/no-heap by design; adding dynamic allocation later would invalidate this conclusion.

## LTO stack and call paths

AVR call instructions push a two-byte return address on this ATmega328P. Figures below count observed saved registers and explicit local stack reservation in LTO disassembly; where a call is present its return address is additional. They are frame evidence, not complete maximum stack depths.

### RLControl

The refreshed release LTO ELF after the scheduler change contains `main` as the Arduino-generated whole-sketch loop body. Its entry prologue saves `r28/r29` and reserves 22 bytes: **24 B** of its own frame. The hot policy function `policy_forward` still saves 18 registers and reserves 132 local bytes: **150 B** of its own frame. Its 132-byte local area is consistent with two 16-element float activation buffers plus compiler temporaries. This confirms neural-network inference has a large stack frame. The weights reside in program memory and are not copied wholesale to SRAM.

Observed policy-path calls include:

```text
main/loop
  -> control_tick / observation preparation (substantial code in main)
  -> policy_forward (150 B frame; float arithmetic and tanh)
       -> update_sample_buffer (called between inference rows)
            -> read_pendulum_rad
                 -> AS5600 rawAngle/readReg2
                      -> Wire transaction / TWI wait and interrupt service
```

LTO symbols show `update_sample_buffer` has a 0x23e-byte code body; its non-LTO stack report is 12 B, while `read_pendulum_rad` reports 8 B. The release machine code additionally calls AVR floating-point runtime helpers and `tanh`; an observation-building path also uses trigonometric library functions. Those callees and all Wire/core helper frames must be included to calculate a true peak. Policy invocation may also be interrupted while executing; interrupt frames stack on top of the active main/policy/sensor frames.

Minimum explicitly observed main + policy nesting is 24 B main frame + 2 B call return + 150 B policy frame = **176 B**, before any policy callees or interrupts. This is a lower bound for that call state, not a “worst-case stack” figure. The separate non-LTO `.su` result for `loop()` (39 B) must not be substituted for the release LTO main frame.

### LowLevelServer

Release `main` saves two registers and reserves 74 local bytes: **76 B**. Protocol dispatch is largely inlined into that large function. `ProtocolV1::send_frame` saves six registers and reserves 64 bytes: **70 B**. A common response path includes a small `send_ack` frame and then `send_frame`; serial/core callees add more. Sensor sampling enters the AS5600/Wire/TWI path while the large `main` frame remains live.

Observed call-path shapes:

```text
main/loop
  -> protocol receive/dispatch (inlined)
       -> safety/command validation
       -> send_ack/send_state
            -> send_frame (70 B frame plus return addresses)
                 -> Serial.write / UART ring handling

main/loop
  -> sampleState (inlined or specialized in LTO)
       -> AS5600 rawAngle/readReg2
            -> Wire/TWI transaction
```

At minimum, main + call + `send_frame` occupies 76 + 2 + 70 = **148 B**, before a wrapper frame/return, UART callees, or an interrupt. The non-LTO reports (`loop` 104 B, `send_frame` 72 B, `sampleState` 16 B) differ from the release build and are supporting clues only.

### What cannot be claimed statically from these data

No complete numeric maximum is reported. The LTO `main` function contains several inlined paths, library calls have independent frames, and the Timer1 overflow ISR can nest another ISR. A maximum requires summing the deepest feasible path through each selected branch and helper, plus hardware return addresses, saved registers, locals, and the allowed nested interrupt path. The non-LTO `.su` data does not provide this for the release image. An interrupt can occur at any instruction where enabled; therefore the active main call chain may coexist with the ISR chain.

To close the stack question, use a release-identical binary with an SRAM stack watermark/canary initialized over the unused stack region and read it after stress covering: policy inference at worst input, sensor success and timeout/error paths, command/frame transmission with UART buffers saturated, queue refill, and Timer1 compare/TWI/Timer0/UART interrupt overlap. Record minimum SP or maximum overwritten watermark and preserve a guard margin. Instrumentation itself changes timing and should be checked; ideally measure an otherwise identical image and then verify the production image’s compiler/linker identity. A host simulator can aid instruction-path inspection but is not a substitute for target stack high-water evidence.

## Interrupt and FastAccelStepper behavior

The installed FastAccelStepper AVR backend selects Timer1 (`FAS_TIMER_MODULE 1`) with STEP on OC1A. The Timer1 overflow cyclic service runs on the 16-bit timer overflow, once per 65,536 timer counts at prescaler 1: **4.096 ms / approximately 244 Hz** at 16 MHz. This replenishes the step queue; Timer1 compare ISRs generate the step pulse events.

The LTO release ELF contains the same ISR bodies in both sketches. `__vector_13` (Timer1 overflow) has a prologue saving 33 register/SREG bytes and reserves 65 local bytes, plus the two-byte hardware return address: **100 B at ISR entry before helper-call frames**. In the source, this ISR disables its own overflow interrupt, executes `sei()`, calls `manageSteppers()`, then disables interrupts before restoring the overflow interrupt. Therefore another enabled interrupt can nest during queue management. The release compare-B ISR entry is 16 saved bytes plus the two-byte return address (**18 B**); compare-A is about 16 B including return. Thus a known direct nested entry combination of overflow + compare-B is **118 B**, excluding `manageSteppers()` and any additional helper frames. It is not the ISR maximum. The child ISR runs with interrupts disabled, so ordinary ISR-on-ISR recursion is not supported by that path; after the overflow ISR reenables interrupts, one child can preempt it.

The queue-management path invokes ramp planning, queue insertion, stepper state queries, and AVR arithmetic helpers. These add stack and consume CPU while interrupts are enabled. A TWI, Timer0, UART, or compare interrupt can be the nested event; exact maximum nesting depends on interrupt flags and timing. Some ISR entry paths are shorter than compare-B, but without a fully enumerated set of simultaneous pending sources and all helper frames, 118 B is only the observed direct-entry subtotal. No ISR execution-time or pulse-jitter measurement exists.

The stepper engine is not evidence of a bounded control task: step pulses are interrupt-driven, while the RL policy and sampler are cooperative main-loop work. Queue refill delay/underrun behavior and interaction with serial/TWI load need target measurement under guarded commissioning conditions.

## I2C timeout, sample loss, and late control recovery

Both sketches configure Wire for 400 kHz and call `Wire.setWireTimeout(3000, true)`. The timeout is a maximum wait bound for applicable synchronous TWI wait loops, not a deadline guarantee for the whole sensor call or loop. It exceeds the configured 2,000 µs sensor sample period. A sensor transaction that reaches the timeout can therefore cause at least one missed/delayed 500 Hz sample and may delay the next policy work. It does not follow that every timeout will miss a 20 ms policy deadline, nor that the loop always recovers within that period; neither has been measured.

The AS5600 path checks `lastError()` immediately after `rawAngle()` and marks failed I2C reads invalid. RLControl faults on invalid/stale encoder state in its control safety path. LowLevelServer records invalid samples and disarms on the third consecutive bad sample while engaged; its command lease is separately serviced by the main loop. These source paths do not equate a 3 ms timeout with a valid angle. Physical bus recovery and error-path duration remain untested.

RLControl now uses a fixed-rate scheduler gate. It preserves `CONTROL_FREQUENCY_HZ=50`, `CONTROL_DT_S=0.02`, the P-law gain, velocity integrator, smoothing, and trained observation/action semantics. It does not substitute measured elapsed time into those equations. One due slot is consumed and rebased at a time, so no missed slots are replayed as multiple catch-up updates. The allowed lateness is 1,000 µs (5% of the period), matching the simulator's configured ±5% control-period randomization envelope. A running tick later than 21 ms from the previous consumed slot latches `FAULT_CONTROL_OVERRUN` and skips policy/controller execution. Serial commands are also deferred on a rejected slot, preventing an ARM command from enabling outputs without a same-slot policy update. This threshold is a software fault policy; it is not evidence the target meets timing.

The sample timestamp selected for the observation is checked against `ENCODER_STALE_US` (10 ms) before inference, after inference, and immediately before `moveByAcceleration()`. The last successful sensor-read timestamp and encoder/magnet validity are checked at the latter points too. Thus sampler work performed during inference does not make the older observation itself appear fresh. On a late-tick path the control-start watchdog timestamp is not refreshed. The host tethered runner also treats a cycle longer than its nominal period as a deadline miss and aborts rather than using variable-dt control math.

This change contains scheduler-delay behavior in source and is exercised by deterministic host tests, including one bounded late tick, a missed slot, repeated polling/no catch-up, stale samples, and `micros()` rollover. It does not prove that real inference/I2C/serial workloads stay within 21 ms or that hardware disable behavior is mechanically safe. A target timing measurement remains required.

### Timing evidence required

No WCET, observed interval, jitter, or deadline measurement was produced. Source statements and the 3 ms Wire timeout cannot establish 50 Hz or 500 Hz timing. On a guarded test fixture, measure with GPIO timing markers or equivalent timestamp capture:

- control tick start-to-start intervals and execution duration, including policy `tanh` and observation work;
- sensor transaction duration and sample-to-sample intervals for successful, NACK/disconnected, and timeout paths;
- Timer1 overflow queue-refill ISR and compare pulse ISR duration, nested-interrupt latency, and queue underruns;
- UART service/response duration when RX/TX buffers are stressed;
- simultaneous worst-case load and the number of late ticks/faults.

Capture enough repetitions to report maxima and distributions, not just averages. Use the independent physical E-stop and a mechanically restrained/current-limited setup during any later physical measurement. These measurements are not part of this offline task.

## Verification run

`./scripts/verify_offline.sh` was run for this investigation and passed:

- Python syntax checks passed.
- 21 Python offline unit tests passed.
- Deterministic fixed-rate scheduler tests passed (normal cadence, bounded lateness, missed slots, no catch-up, freshness boundary, and timestamp rollover).
- Header-policy numerical comparison passed on 93 deterministic observations: PyTorch vs NumPy max absolute error `5.3644e-7`; PyTorch vs host C++ scalar max absolute error `4.0233e-6`.
- Compile-only LowLevelServer: 20,586 B flash, 1,024 B global SRAM.
- Compile-only RLControl after scheduler fix: 27,808 B flash, 1,289 B global SRAM.

The policy comparison is host-side and is not AVR execution validation. Compile success is not timing or physical safety validation. All investigation builds were temporary; checkpoints, experiment outputs, CSV files, existing project changes, and the original repository were not modified by this task.

## Final disposition

The ELF proves that RLControl's 150 B inference frame and the 100 B Timer1 overflow ISR frame coexist with a limited 759 B SRAM remainder; static analysis performed here does not establish their full peak overlap or a safe stack margin. LowLevelServer has more static headroom but similarly lacks a proven ISR-inclusive bound. FastAccelStepper uses static allocation on AVR, so the reviewed allocation does not create heap fragmentation.

The nominal loop deadlines have not been measured. The late-tick source behavior is now fail-closed and regression-tested, but its 21 ms limit has not been measured on target. **Overall verdict remains FAIL for proceeding to motor-enabled commissioning** until release-build stack high-water and target timing margins are measured. Maintain an independent physical E-stop; firmware torque disable alone is not assumed mechanically safe.
