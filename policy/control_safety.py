"""Shared fail-closed checks for live control state and timing."""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass


# protocol/v1.md: require a valid encoder, a fresh sampler, a ready stepper,
# and detected AS5600 magnet. ARMED is informational because state reads are
# also used while disarmed.
HEALTH_ENCODER_VALID = 0x0001
HEALTH_SAMPLER_FRESH = 0x0002
HEALTH_STEPPER_READY = 0x0004
HEALTH_ARMED = 0x0008
HEALTH_MAGNET_DETECTED = 0x0010
REQUIRED_STATE_HEALTH = (
    HEALTH_ENCODER_VALID | HEALTH_SAMPLER_FRESH | HEALTH_STEPPER_READY
    | HEALTH_MAGNET_DETECTED
)


class UnsafeState(RuntimeError):
    """State is stale, invalid, or reports an unsafe firmware condition."""


@dataclass(frozen=True)
class StateAge:
    age_ns: int
    rtt_ns: int


def validate_state(state, *, max_age_s: float = 0.080,
                   required_health_mask: int = REQUIRED_STATE_HEALTH,
                   require_no_faults: bool = True,
                   now_ns: int | None = None) -> StateAge:
    """Reject state that must not be passed into a live policy.

    `rx_monotonic_ns` and `now_ns` share the host monotonic clock. Arduino
    `time_us` is deliberately not compared here because there is no clock
    synchronization protocol.
    """
    values = (
        state.motor_pos_rad,
        state.pendulum_pos_rad,
        state.motor_vel_rad_s,
        state.pendulum_vel_rad_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise UnsafeState("state contains NaN or infinity")
    missing = required_health_mask & ~int(state.health_flags)
    if missing:
        raise UnsafeState(f"required sensor/firmware health missing: 0x{missing:04x}")
    if require_no_faults and int(state.fault_flags) != 0:
        raise UnsafeState(f"firmware fault flags set: 0x{int(state.fault_flags):04x}")
    received_ns = int(state.rx_monotonic_ns)
    now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    age_ns = now_ns - received_ns
    if age_ns < 0:
        raise UnsafeState("state receive timestamp is in the future")
    max_age_ns = int(max_age_s * 1_000_000_000)
    if age_ns > max_age_ns:
        raise UnsafeState(
            f"state is stale: {age_ns / 1e6:.1f} ms > {max_age_s * 1e3:.1f} ms"
        )
    rtt_ns = int(state.rtt_ns)
    if rtt_ns < 0:
        raise UnsafeState("state has invalid negative request RTT")
    return StateAge(age_ns=age_ns, rtt_ns=rtt_ns)


def require_ack(ack, *, operation: str):
    """Require a successful command acknowledgement."""
    if ack is None or not bool(getattr(ack, "ok", False)):
        raise UnsafeState(f"{operation} was not acknowledged")
    return ack


def confirm_operator_arm(*, requested: bool, prompt_fn=input,
                         is_interactive: bool | None = None,
                         phrase: str = "ARM") -> bool:
    """Require an explicit flag and typed terminal confirmation to arm.

    Tests can inject `prompt_fn` and `is_interactive`; production entry points
    use stdin and never prompt when arming was not explicitly requested.
    """
    if not requested:
        return False
    if is_interactive is None:
        is_interactive = sys.stdin.isatty()
    if not is_interactive:
        raise UnsafeState("arming requires an interactive terminal confirmation")
    answer = prompt_fn(f"Type {phrase!r} to confirm motor arming: ")
    if answer != phrase:
        raise UnsafeState("motor arming confirmation did not match")
    return True
