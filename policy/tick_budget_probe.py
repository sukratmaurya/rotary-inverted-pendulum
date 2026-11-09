"""Profile per-tick wall-clock cost of the deploy loop.

Answers: at the current setup, how fast can we run the control loop
before per-tick work runs over the budget? Output is a percentile
breakdown of every component:

    get_state:  serial round-trip (Python write → Arduino reply)
    inference:  policy.predict()  (or a dummy callable)
    total:      end-to-end tick wall time
    sleep_jit:  difference between intended next-tick time and actual wake

The driver remains disarmed, so this measures state-read and compute costs.

Usage:
    # With a real policy
    python tick_budget_probe.py --port /dev/cu.usbserial-1130 \\
        --policy runs/<run>/best_model.zip

    # Without a policy (dummy callable, e.g. to isolate serial cost)
    python tick_budget_probe.py --port /dev/cu.usbserial-1130

    # Probe a specific control rate (default: run flat-out, no sleep)
    python tick_budget_probe.py --port ... --control-hz 50 --duration 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from lowlevel_client import LowLevelClient


def _percentiles(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {k: float("nan") for k in ("p50", "p95", "p99", "min", "max", "mean")}
    return dict(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
    )


def _fmt_us(s: float) -> str:
    """Format seconds as a human-readable µs/ms string."""
    if not np.isfinite(s):
        return "    nan"
    if s < 1e-3:
        return f"{s * 1e6:6.0f} µs"
    return f"{s * 1e3:6.2f} ms"


def _print_row(label: str, stats: dict) -> None:
    print(f"  {label:>12}  "
          f"p50={_fmt_us(stats['p50'])}  "
          f"p95={_fmt_us(stats['p95'])}  "
          f"p99={_fmt_us(stats['p99'])}  "
          f"max={_fmt_us(stats['max'])}")


def load_policy(policy_path: str | None):
    """Load an SB3 SAC policy, or return None for the dummy case."""
    if policy_path is None:
        return None
    from stable_baselines3 import SAC
    print(f"Loading policy from {policy_path} ...")
    model = SAC.load(policy_path, device="cpu")

    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return predict


def build_dummy_obs(prev_action: float = 0.0) -> np.ndarray:
    """Build a synthetic observation matching pendulum_env's layout.

    The env uses (motor_pos, motor_vel, pen_pos_cos, pen_pos_sin, pen_vel,
    prev_action) — but we don't import the env here to keep the profiler
    standalone. For the dummy policy the exact shape doesn't matter; for
    the real-policy case the user must pass a model trained on this
    observation layout (current env default).
    """
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0, prev_action], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--policy", type=str, default=None,
                   help="Path to SAC .zip (omitted → use dummy zero-action policy)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="How long to profile (seconds). Default 5.")
    p.add_argument("--control-hz", type=float, default=None,
                   help="If set, sleep to maintain this control rate "
                        "(reports sleep jitter). If omitted, run flat-out.")
    args = p.parse_args(argv)

    predict = load_policy(args.policy)
    if predict is None:
        print("No policy specified — using dummy zero-action callable.")
        def predict(obs):
            return np.array([0.0], dtype=np.float32)

    n_max = int(args.duration * 5000)  # plenty of room (even at 1 kHz)
    t_get   = np.zeros(n_max)
    t_inf   = np.zeros(n_max)
    t_set   = np.zeros(n_max)
    t_total = np.zeros(n_max)
    t_jit   = np.zeros(n_max)

    with LowLevelClient(args.port) as client:
        if not client.wait_until_ready():
            print("Arduino not responding", file=sys.stderr)
            return 1
        # Motor stays DISENGAGED — we only profile the I/O + compute
        # cost; we don't want the rig moving while we measure.
        client.disengage_motor()
        time.sleep(0.2)

        if args.control_hz is None:
            print(f"Profiling flat-out for {args.duration:g} s ...")
        else:
            print(f"Profiling at {args.control_hz:g} Hz for {args.duration:g} s ...")

        t_loop_start = time.monotonic()
        prev_action = 0.0
        next_tick = t_loop_start
        dt_target = (1.0 / args.control_hz) if args.control_hz else 0.0

        i = 0
        while True:
            tick_start = time.monotonic()
            if (tick_start - t_loop_start) > args.duration or i >= n_max:
                break

            t0 = time.monotonic()
            s = client.get_state()
            t1 = time.monotonic()

            obs = build_dummy_obs(prev_action=prev_action)
            action = predict(obs)
            prev_action = float(action[0])
            t2 = time.monotonic()

            # Do not arm merely to benchmark an acceleration-command write.
            # This probe intentionally measures state-read and inference only.
            t3 = t2

            t_get[i]   = t1 - t0
            t_inf[i]   = t2 - t1
            t_set[i]   = t3 - t2
            t_total[i] = t3 - tick_start

            if dt_target > 0.0:
                next_tick += dt_target
                wake_target = next_tick
                sleep_for = wake_target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                t_jit[i] = time.monotonic() - wake_target  # ≥ 0 if late
            else:
                t_jit[i] = 0.0

            i += 1

        client.disengage_motor()

    t_get   = t_get[:i]
    t_inf   = t_inf[:i]
    t_set   = t_set[:i]
    t_total = t_total[:i]
    t_jit   = t_jit[:i]

    print(f"\nCollected {i} ticks "
          f"({i / max(args.duration, 1e-9):.0f} Hz effective).\n")

    print("Per-tick component breakdown:")
    _print_row("get_state",  _percentiles(t_get))
    _print_row("inference",  _percentiles(t_inf))
    print("  set_accel: not measured (driver intentionally disarmed)")
    _print_row("total tick", _percentiles(t_total))
    if args.control_hz is not None:
        print()
        print(f"Sleep jitter (target {args.control_hz:g} Hz = "
              f"{1000/args.control_hz:.2f} ms period):")
        _print_row("late by",  _percentiles(t_jit))

    print()
    p99_total = float(np.percentile(t_total, 99))
    safe_period_s = p99_total * 2.0  # 2× headroom against p99 jitter
    safe_hz = 1.0 / safe_period_s if safe_period_s > 0 else float("inf")
    print(f"p99 per-tick cost: {_fmt_us(p99_total)}")
    print(f"→ recommended safe max control rate: ~{safe_hz:.0f} Hz "
          f"(2× headroom over p99)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
