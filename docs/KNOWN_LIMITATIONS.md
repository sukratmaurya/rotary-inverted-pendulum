# Known limitations

- The Arduino compiler was not runnable in the latest verification attempt: Snap confinement rejected `arduino-cli`. Prior compile figures are historical and predate the 5-entry ring optimization. Current flash/global SRAM values need a fresh compile-only build.
- Static SRAM estimates do not prove stack safety. Maximum nested protocol stack, FastAccelStepper/Wire/Serial runtime allocations, and interrupt stack usage have not been bounded or measured. RLControl's estimated remaining SRAM after the ring reduction is about 759 bytes before those costs.
- No physical validation has occurred: encoder direction and magnet integrity, TMC2209 enable polarity/current, FastAccelStepper timing, USB 2 Mbaud reliability, watchdog behavior, mechanical hard limits, and E-stop effectiveness all require supervised checks.
- The physical E-stop is an independent commissioning prerequisite. Firmware torque disable is not assumed mechanically safe for every pendulum pose.
- Sensor fault fixtures are host-side models of documented thresholds, not executions of the actual firmware parser/sampler or AS5600 library. Disconnected-I2C return behavior depends on the installed library and bus condition.
- Tethered response/lease semantics have Python tests and source review; there is no host-compiled automated execution of the Arduino `ProtocolV1.h` dispatcher. Invalid-packet lease non-refresh is established by control flow inspection, not exhaustive firmware fault injection.
- Header/PyTorch/NumPy/host-C++ equivalence is based on parsed header weights. The exact header-producing student checkpoint is unavailable. AVR numerical parity is unmeasured.
- No Training 4 simulation result is claimed unless listed in the final test report with a completed run. Simulation is not hardware evidence.
- USB serial connection/open, firmware upload, motor movement, and hardware tests were intentionally not performed under this task's safety constraints.
