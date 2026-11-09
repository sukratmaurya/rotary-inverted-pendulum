# Final software hardening record

## Implemented in this project copy

- Preserved existing changes, checkpoints, and evaluation artifacts. The original repository was not edited.
- Retained protocol v1 framing and explicit-disarm/arm architecture; added offline tests for truncated/extra/corrupt frames, non-finite and out-of-range acceleration values, payload mismatch, delayed sequence responses, and bounded missing-response timeouts.
- Reduced each firmware sensor history ring from 16 entries to the 5 entries actually consumed by the estimator. This saves 132 bytes of static SRAM per sketch and does not change the five-sample newest/oldest estimator.
- Added deterministic AS5600 fault-injection fixtures for wraparound, invalid/disconnected samples, implausible jumps, recovery, stationary repeated readings, and the firmware's frozen-sensor threshold.
- Kept the existing policy/header numerical validator and fail-closed state/arming guards.

## Verification status

The Python test suite and policy validation completed in this session. `./scripts/verify_offline.sh` reached both firmware compile steps, but the installed `arduino-cli` is a Snap and refused to run because `snap-confine` reports that elevated permissions are not confined. Therefore this invocation did **not** compile the current ring-buffer edits. A prior build record in `FINAL_SYSTEM.md` reports successful compile-only builds of the earlier source; those size figures are historical and are not claimed for this edited version.

No hardware, serial device, upload, or motor operation was used. Sensor tests exercise a Python offline model of the firmware's stated thresholds; they do not execute AS5600 hardware or prove the C++ sensor path.

## Memory review

The project copy's previous compile record reports LowLevelServer 1,156 bytes static SRAM and RLControl 1,421 bytes static SRAM. Removing eleven samples × twelve bytes per sketch gives estimated current static totals of 1,024 and 1,289 bytes, leaving approximately 1,024 and 759 bytes respectively for stack, library runtime state, serial buffers, and interrupt context. These are arithmetic estimates from the prior build record, **not freshly measured compiler reports**.

Policy weights are in `PROGMEM`; RLControl's observation history, action smoothing, sample arrays, and telemetry buffer reside in SRAM. Protocol code allocates bounded local COBS/raw/state arrays (up to roughly 100 bytes in nested send/dispatch paths). FastAccelStepper, Wire, Arduino core, serial RX/TX rings, and interrupt entry frames consume additional SRAM/stack. No compiler stack-usage report or reliable worst-case interrupt/main stack watermark is available, so stack headroom cannot be proven from static global usage. The 759-byte estimated RLControl margin is a material residual risk until current compile output and a stack watermark/analysis are recorded.

## Control / policy facts

The Training 4 run config specifies 24 observations, four six-value frames, 50 Hz, velocity mode, 3.5 rad/s action scale, 150 rad/s² acceleration limit, 32 microsteps, and smoothing window 4. Offline header numerical comparison used 93 deterministic edge observations: PyTorch vs NumPy max absolute error `5.36e-7`; PyTorch vs host C++ max `4.02e-6`; NumPy vs host C++ max `4.02e-6`. The checked-in header's exact source student checkpoint is absent, so this establishes arithmetic equivalence for parsed header weights, not source-checkpoint provenance or AVR floating-point parity.

See [`FINAL_TEST_REPORT.md`](FINAL_TEST_REPORT.md), [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md), and [`HARDWARE_COMMISSIONING.md`](HARDWARE_COMMISSIONING.md) for evidence boundaries and remaining work.
