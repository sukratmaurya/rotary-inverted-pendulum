# Hardware commissioning checklist

This checklist is a procedure for a future, supervised commissioning session. No hardware validation is claimed by this project until each physical check is completed and recorded.

## Before connecting motor power

- [ ] Confirm the correct firmware target, driver, microstep setting, motor wiring, supply voltage, and current limit for this rig.
- [ ] Confirm the independent, physical E-stop removes driver enable or motor power without relying on USB, Arduino code, or Python. Record the circuit and test it with motor power isolated.
- [ ] Verify the E-stop remains electrically independent after any wiring or enclosure changes. Do not substitute a software command for it.
- [ ] Run the offline Python test suite and firmware compile. Record toolchain versions, commit, and build output.
- [ ] Verify firmware reset and power-up leave outputs disabled and the controller DISARMED. Verify no startup delay, serial handshake, or policy inference can arm the motor.
- [ ] Verify the operator ARM action is explicit, and DISARM is available independently of policy execution.
- [ ] Confirm the encoder reports valid samples, correct direction, wrap handling, and plausible velocity while the motor driver remains disabled.

## Powered checks with motion still inhibited

- [ ] Keep the physical E-stop engaged and motor mechanically restrained. Confirm serial HELLO/GET_STATE, CRC rejection, sequence rejection, packet timeout/resynchronization, and fault reporting.
- [ ] Confirm malformed, stale, duplicate, out-of-order, NaN/Inf, and out-of-range commands are rejected without enabling the driver.
- [ ] Confirm stale/frozen/invalid encoder conditions prevent ARM and latch a visible fault.
- [ ] Confirm Arduino reset, USB disconnect, and host-process termination leave the firmware disarmed or cause the configured command lease to expire and disable outputs.
- [ ] Confirm the physical E-stop still works if the Arduino is held in reset or the USB cable is removed.

## First supervised motion session

- [ ] Remove restraints only after the previous checks pass; use a clear travel area, mechanical guards, and an operator at the independent E-stop.
- [ ] Start with the lowest practical current, velocity, acceleration, and travel envelope. Use short, supervised moves before enabling a learned policy.
- [ ] Confirm commanded direction, encoder sign, step counter direction, and soft/hard-limit behavior at low speed.
- [ ] Measure state age, loop period, inference time, command ACK time, motor command-to-motion delay, missed deadlines, and fault response under the intended deployment computer and USB link.
- [ ] Test lease expiry and sensor-fault response with the motor disabled first. Any later fault-injection test involving motion requires a separate risk review and a guarded test setup.
- [ ] Validate the exact policy header/checkpoint pair and record policy hash, firmware build, rig identity, configuration, environment, duration, and outcomes.
- [ ] Stop and investigate if the control rate misses its deadline, sensor values become stale, the motor position disagrees with physical motion, or any fault is not clearly reported.

## Sign-off record

Record operator, date, rig and driver identifiers, firmware commit/build, policy/header hash, E-stop verification method, test conditions, observed timing, faults, and explicit pass/fail for every item. A successful simulation, host-compiled numerical comparison, or firmware compile is not a substitute for physical validation.
