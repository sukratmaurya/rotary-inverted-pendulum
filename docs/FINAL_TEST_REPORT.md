# Final test report

Date: 2026-11-09. Scope: offline software checks only; no serial port or physical hardware access.

## Results

| Check | Result | Evidence / limits |
|---|---|---|
| Python unit tests | Passed, 21 cases | `./scripts/verify_offline.sh`; tests use fake serial, local state, and modeled sensor fixtures |
| Policy numerical comparison | Passed, 93 observations | PyTorch vs NumPy max `5.3644e-7`; PyTorch vs host C++ max `4.0233e-6`; P99 values and sample count are printed by `validate_deployment.py` |
| LowLevelServer compile | Not verified for current source | Current verify attempt blocked by Snap confinement before the compile invocation could execute |
| RLControl compile | Not verified for current source | Same blocker; prior successful figures predate the ring-size edit |
| Sensor failure model | Offline tests added | Does not exercise Arduino code or physical AS5600 |
| Training 4 MuJoCo evaluation | Passed, 5/5 episodes | Deterministic nominal run; see results below. Simulation evidence only. |
| Hardware operation / E-stop | Not performed | Explicitly prohibited by task constraints; independent physical E-stop remains required |

The latest verifier attempt passed all 21 Python tests and numerical equivalence, then failed at the first firmware compile because `snap-confine` reported an environment confinement error. The Arduino compile steps were not executed successfully.

## Training 4 nominal MuJoCo run

Checkpoint: `policy/runs/cpu_training_4_lag_robust/best_model.zip`; run config loaded from the adjacent `config.json`. Command used five episodes, seed `20261109`, 8 seconds each, 50 Hz, `--no-randomize`.

- Solved: 5/5 (100%); each episode had 100% upright occupancy in its final one-second hold window.
- Mean reward: -465.84.
- Hard-stop terminations: 0.
- Peak motor positions: 50.6°, 33.0°, 69.2°, 56.3°, 56.8°.

The identical-seed run produced the same episode rewards, upright fractions, peak motor positions, and aggregate metrics, confirming deterministic replay for this configuration. This finite nominal simulation sample does not establish randomized robustness or hardware performance.

## Policy numerical detail from the completed run

- Test corpus: 93 deterministic edge/boundary observations; `--random-edge-rows 32`.
- PyTorch vs NumPy: max `5.364418029785156e-7`, P99 `5.090236641080992e-7`.
- PyTorch vs host C++: max `4.023313522338867e-6`, P99 `3.9410592762578744e-6`.
- NumPy vs host C++: max `4.023313522338867e-6`, P99 `3.4475326629035408e-6`.

No fresh test result is asserted for changes made after that run. The single safe command is `./scripts/verify_offline.sh`; it is compile-only and uses temporary build paths when the Arduino toolchain is available.
