# Final control timing fix

Date: 2026-11-10. Scope: RLControl fixed-rate scheduling and observation freshness. No serial device or physical hardware was accessed; no firmware was uploaded and no motor was operated.

## Findings before the fix

RLControl already used a fixed 50 Hz policy period (`20,000 µs`) and fixed `CONTROL_DT_S=0.02`. The acceleration law uses the fixed 50 Hz gain, and the commanded-velocity integrator advances by `accel_cmd * 0.02`. This is deliberate discrete-time behavior; replacing 0.02 with measured elapsed time would change the controller and trained deployment semantics.

The old loop ran a policy tick when elapsed time was at least 20 ms. It counted an overrun after 30 ms but still ran the tick with fixed-step mathematics; its separate fault only triggered after the gap between control starts exceeded 60 ms. Although the loop reset its timer on each due poll and did not intentionally replay a backlog, a moderately late update could still apply an action calculated from a stale observation and would not fault promptly.

The training/simulation source confirms that policy cadence is fixed by `control_freq_hz`, and the simulator optionally randomizes control-period duration by ±5%. The tethered `run_policy.py` treats a cycle longer than its nominal period as a deadline miss and exits; it does not stretch controller `dt` to recover.

## Correction

`firmware/RLControl/ControlTiming.h` now provides a small, host-testable fixed-rate scheduler and wrap-safe timestamp-age check. RLControl uses that same scheduler:

- Nominal policy rate remains 50 Hz; action mapping, observation construction, smoothing, P-law gain, acceleration limit, integrator step, and policy weights are unchanged.
- A due tick up to 1,000 µs late is run once. The tolerance is 5% of the 20 ms period, matching the configured simulation timing-randomization envelope. It is not variable-dt integration.
- A running tick more than 1,000 µs late is consumed, skipped, and latches `FAULT_CONTROL_OVERRUN`. It does not run policy inference or controller math and does not refresh `last_control_start_us`.
- Serial commands are not dispatched on a rejected slot, so an `E` arm request cannot enable outputs during a late slot and wait for a later policy tick. It is serviced at a subsequent nominal slot.
- Each due slot is rebased exactly once. Repeated loop polling cannot generate a catch-up burst. A rejected slot is consumed too; the faulted controller does not replay it later.
- The timestamp of the specific sample snapshot used to build the policy observation is checked against the 10 ms stale limit before inference, after inference, and immediately before `moveByAcceleration()`. Sensor validity, magnet status, and latest successful sensor timestamp are also checked before command issue. Sampling during inference cannot refresh the timestamp of the already-built observation.
- All time differences use unsigned 32-bit subtraction and are covered at `micros()` rollover.

The fix intentionally keeps the trained fixed-step mathematics. A delay outside the allowed timing envelope transitions to the existing latched fault path rather than trying to compensate with a new, unvalidated `dt`.

## Deterministic tests

`policy/test_control_timing.cpp` tests the production scheduler helper with deterministic timestamps:

- before-deadline wait, exact 20 ms tick, and bounded ±5% lateness;
- late tick just outside tolerance, and delays of one or multiple full periods;
- repeated polling after a consumed tick and after a rejected slot, proving no catch-up burst;
- fresh/stale observation sample boundary and 32-bit timestamp rollover.

The test is part of `./scripts/verify_offline.sh`, compiled with host C++11 into the verifier's temporary directory. It validates scheduler decisions, not the physical actuator response or all firmware state transitions in an on-target emulator.

## Verification results

The complete `./scripts/verify_offline.sh` run passed after the change:

- Python syntax checks passed.
- 21 Python offline unit tests passed.
- Fixed-rate scheduler/freshness tests passed.
- Host policy/header numerical comparison passed on 93 observations (PyTorch versus NumPy maximum absolute error `5.3644e-7`; PyTorch versus host C++ maximum absolute error `4.0233e-6`).
- Compile-only LowLevelServer: 20,586 B flash, 1,024 B global SRAM.
- Compile-only RLControl: 27,808 B flash, 1,289 B global SRAM; 759 B remain before stack.

The firmware compile verifies the AVR-targeted code builds; the timing unit test is host-side. No target timing, stack high-water, sensor fault recovery, motor output, or hardware E-stop test was performed.

## Remaining limits and commissioning gate

The 21 ms policy-start limit is a selected fail-safe threshold anchored to the simulation's ±5% timing variation, not a measured Nano worst-case deadline. Measure tick intervals, execution duration, I2C success/failure latency, interrupt load, and observation-to-command age on the actual guarded setup before motor-enabled commissioning. Also establish stack high-water with the release-identical build. Keep the independent physical E-stop mandatory.

**Verdict:** source-level late-tick handling is corrected and deterministically unit-tested. Overall hardware readiness remains **NOT VERIFIED** until on-target timing and stack margins are measured.
