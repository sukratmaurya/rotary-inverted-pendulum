# Final project integration guide

This directory is the second, independent project copy. The source repository was not modified. The copy preserves the source Git metadata, uncommitted source changes, Training 3/4 checkpoints, evaluation CSVs, and other experiment artifacts. Only the 4.9 GB `policy/.venv` dependency cache was excluded; recreate dependencies for this copy with `uv sync --project policy` when network/package access is available.

## System boundaries

- `policy/` contains training, evaluation, Python policy inference, state estimation, and the tethered control loop.
- `firmware/LowLevelServer/` implements the tethered Arduino server and sensor/stepper safety hooks.
- `firmware/RLControl/` implements standalone student-policy inference and firmware safety behavior.
- `protocol/v1.md` defines the shared tethered wire/API contract. The host and Arduino use COBS-delimited frames, protocol version 1, u16 request sequence, CRC-16/CCITT-FALSE, bounded payloads, typed responses, and explicit health/fault flags.
- `docs/HARDWARE_COMMISSIONING.md` is a future supervised checklist. It is not a test report.

For protocol v1, only a valid, in-range `SET_ACCEL` refreshes the 100 ms lease. `HELLO` does not arm the motor. The tethered client does not automatically retry `ARM` or motion commands. Sensor freshness/health and latched faults must be checked before arming. See the protocol document for full packet layouts and exact flag definitions.

## Safe software startup

Both firmware paths must boot with outputs disabled and disarmed. Tethered Python environments remain disarmed on reset; control entrypoints require explicit operator opt-in and confirmation before calling ARM. In standalone firmware, boot leaves the controller DISARMED; `E` explicitly arms, `D` disarms, `C` clears eligible latched faults, and `Z` performs operator-confirmed center re-zero only with outputs disabled. Clearing a fault still requires a separate `E`. Sensor, hard-limit, and timing faults latch. There is no absolute arm-position sensor, so `Z` relies on operator attestation that the arm is physically centered. Do not bypass these guards for unattended launch.

The hardware E-stop must remain an independent physical circuit that removes driver enable or motor power. Firmware and Python commands are not substitutes for it. The repository does not establish that a physical E-stop is wired or verified; commissioning must verify the actual circuit.

## Offline verification

Run the project-owned verifier from the project root:

```bash
./scripts/verify_offline.sh
```

It runs Python unit tests, the read-only header/Python/NumPy/host-C++ numerical check, and compile-only Arduino builds for `LowLevelServer` and `RLControl`. Arduino core `arduino:avr` plus `FastAccelStepper` and `AS5600` libraries must already be installed. Build output goes to a temporary directory. The script contains no upload command and does not connect to a serial port.

For a Python dependency environment:

```bash
uv sync --project policy
```

For a serial-free protocol test only:

```bash
cd policy
python test_serial_protocol.py
```

For numerical validation of the checked-in rig1 TMC2209 header only:

```bash
python policy/validate_deployment.py \
  --header firmware/RLControl/policy_weights_rig1_tmc2209_32.h \
  --random-edge-rows 32
```

The named distilled student checkpoint is not present in the copied run artifacts, so the header's source comment cannot currently be proven against checkpoint bytes. Numerical agreement between parsed header weights and host Python/NumPy/C++ does not prove parity on the AVR binary.

## Evidence limits

Offline unit tests and firmware compilation establish software behavior only for the tested inputs and host toolchain. The policy numerical validator does not run on an AVR binary. No physical motor operation, firmware upload, E-stop test, timing measurement, sensor fault injection, or hardware validation has been performed in this project build. Track those checks in the commissioning record before making deployment claims.

## Current offline verification record

The integrated `./scripts/verify_offline.sh` run completed successfully (exit code 0). It reported:

- 8 protocol framing/client tests passed using an in-memory fake serial peer.
- 14 combined protocol and control-safety tests passed; no serial device was opened.
- Host numerical check on 93 deterministic edge observations: PyTorch-vs-NumPy max absolute action error `5.36e-7`; PyTorch-vs-host-compiled-C++ and NumPy-vs-host-compiled-C++ max error `4.02e-6`.
- Both Arduino Nano sketches compiled with the AVR toolchain; this is compile evidence only, not an upload or hardware result.

The verifier compiled its scalar C++ harness and both Arduino sketches in temporary directories, then removed those temporary build products. These results do not establish AVR floating-point parity, timing under load, physical E-stop behavior, or successful motor control.

Compile record from the firmware owner: Arduino AVR Boards core 1.8.8, AS5600 0.6.7, FastAccelStepper 2.0.1, `arduino-cli` at `/snap/bin/arduino-cli`, FQBN `arduino:avr:nano:cpu=atmega328`. `LowLevelServer` compiled to 20,342 bytes flash / 1,156 bytes globals; `RLControl` compiled to 27,390 bytes flash / 1,421 bytes globals in the integrated verifier (89% flash, 69% SRAM). An earlier owner build reported 27,258 bytes; the integrated build is the current record. Builds used `/tmp` output paths and did not upload.
