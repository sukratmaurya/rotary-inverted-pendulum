#!/usr/bin/env bash
# Serial-free/offline checks and compile-only Arduino builds. Never uploads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="$ROOT/policy"
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT
export UV_CACHE_DIR="$BUILD_TMP/uv-cache"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY=("$PYTHON_BIN")
elif [[ -x "$POLICY/.venv/bin/python" ]]; then
  PY=("$POLICY/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run --offline --project "$POLICY" python)
else
  PY=(python3)
fi

printf '%s\n' '== Python syntax checks =='
"${PY[@]}" -m py_compile \
  "$POLICY/lowlevel_client.py" \
  "$POLICY/control_safety.py" \
  "$POLICY/real_env.py" \
  "$POLICY/async_control.py" \
  "$POLICY/run_policy.py" \
  "$POLICY/finetune_async.py" \
  "$POLICY/finetune_real.py" \
  "$POLICY/sysid_accel.py" \
  "$POLICY/accel_step_probe.py" \
  "$POLICY/accel_lag_moving_probe.py" \
  "$POLICY/sysid_wizard.py" \
  "$POLICY/tick_budget_probe.py"

printf '%s\n' '== Python offline unit tests =='
(cd "$POLICY" && "${PY[@]}" -m unittest discover -p 'test*.py' -v)

printf '%s\n' '== Fixed-rate control timing tests =='
HOST_CXX="${HOST_CXX:-$(command -v c++ || true)}"
if [[ -z "$HOST_CXX" ]]; then
  printf '%s\n' 'C++ compiler not found; control timing tests cannot run.' >&2
  exit 2
fi
"$HOST_CXX" -std=c++11 -Wall -Wextra -Werror \
  "$POLICY/test_control_timing.cpp" -o "$BUILD_TMP/test_control_timing"
"$BUILD_TMP/test_control_timing"

printf '%s\n' '== Policy/header numerical check =='
"${PY[@]}" "$POLICY/validate_deployment.py" \
  --header "$ROOT/firmware/RLControl/policy_weights_rig1_tmc2209_32.h" \
  --random-edge-rows 32

ARDUINO_CLI_BIN="${ARDUINO_CLI_BIN:-$(command -v arduino-cli || true)}"
if [[ -n "$ARDUINO_CLI_BIN" && "$(readlink -f "$ARDUINO_CLI_BIN" 2>/dev/null || true)" == */snap ]]; then
  # Snap's launcher may reject confinement in restricted build containers.
  # Prefer an installed Arduino IDE bundled CLI when available.
  for candidate in "$HOME"/Downloads/*/resources/app/lib/backend/resources/arduino-cli; do
    if [[ -x "$candidate" ]]; then ARDUINO_CLI_BIN="$candidate"; break; fi
  done
fi
if [[ -z "$ARDUINO_CLI_BIN" ]]; then
  printf '%s\n' 'arduino-cli not found; firmware compile checks skipped.' >&2
  exit 2
fi
ARDUINO_LIBRARIES_DIR="${ARDUINO_LIBRARIES_DIR:-$HOME/Arduino/libraries}"
if [[ ! -d "$ARDUINO_LIBRARIES_DIR/FastAccelStepper" ]]; then
  for candidate in "$HOME"/snap/arduino-cli/*/Arduino/libraries; do
    if [[ -d "$candidate/FastAccelStepper" && -d "$candidate/AS5600" ]]; then
      ARDUINO_LIBRARIES_DIR="$candidate"
      break
    fi
  done
fi
ARDUINO_LIB_ARGS=()
for lib in FastAccelStepper AS5600; do
  if [[ -d "$ARDUINO_LIBRARIES_DIR/$lib" ]]; then
    ARDUINO_LIB_ARGS+=(--library "$ARDUINO_LIBRARIES_DIR/$lib")
  fi
done
mkdir -p "$BUILD_TMP/lowlevel" "$BUILD_TMP/rlcontrol"
FQBN="${ARDUINO_FQBN:-arduino:avr:nano:cpu=atmega328}"
printf '%s\n' '== Compile LowLevelServer (no upload) =='
"$ARDUINO_CLI_BIN" compile "${ARDUINO_LIB_ARGS[@]}" --fqbn "$FQBN" --build-path "$BUILD_TMP/lowlevel" \
  "$ROOT/firmware/LowLevelServer"
printf '%s\n' '== Compile RLControl (no upload) =='
"$ARDUINO_CLI_BIN" compile "${ARDUINO_LIB_ARGS[@]}" --fqbn "$FQBN" --build-path "$BUILD_TMP/rlcontrol" \
  "$ROOT/firmware/RLControl"
printf '%s\n' 'All requested offline checks passed.'
