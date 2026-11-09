"""Run a trained policy on the real rotary inverted pendulum.

Accepts either:
  - a Stable-Baselines3 SAC checkpoint (`.zip`) — the standard teacher path, or
  - a distilled student MLP (`.pt`) produced by `distill.py` — used to
    validate the student on the rig before flashing
    `firmware/RLControl/RLControl.ino`.

Drives the device through the LowLevelServer binary protocol at a fixed
control rate. The observation matches what `pendulum_env.py` produces:

    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel]

with theta=0 at upright. Velocities are computed by finite difference and
low-pass filtered to attenuate quantisation noise.

Safety:
- The commanded motor target is clamped to ±125° (inside the ±135°
  mechanical hard stops).
- A staleness check kills the loop if a get_state call returns garbage
  or stalls.
- Ctrl-C disengages the motor cleanly via LowLevelClient.__exit__.

Usage (teacher):
    python run_policy.py --policy runs/desktop_run_2026-05-01/best_model.zip \\
        --port /dev/cu.usbserial-1130

Usage (distilled student, before flashing the Nano):
    python run_policy.py \\
        --policy runs/async_35hz_v2_extend/distill_h32_aug/student.pt \\
        --port /dev/cu.usbserial-1130 \\
        --duration-s 30
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from lowlevel_client import LowLevelClient
from control_safety import (
    REQUIRED_STATE_HEALTH,
    UnsafeState,
    confirm_operator_arm,
    require_ack,
    validate_state,
)
from run_config import check_config, find_run_config


MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)
MOTOR_LIMIT_RAD = math.radians(135.0)  # obs-space bound; matches pendulum_env


def _wrap_pi(x: float) -> float:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def make_obs(motor_pos: float, phi: float, motor_vel: float, pen_vel: float,
             prev_action: float, include_velocities: bool = True) -> np.ndarray:
    """Build one observation frame matching pendulum_env.py.

    phi is the pendulum joint angle (0 = hanging down, +/- pi = upright).
    theta = phi - pi   (so theta=0 at upright, theta=+/-pi at hanging down).
    prev_action is the action issued last tick, in [-1, 1] — restores Markov
    property under action delay by giving the policy a read on its own queue.
    With include_velocities=False the frame is positions-only and the policy
    derives velocities from the stacked history.
    """
    theta = _wrap_pi(phi - math.pi)
    if include_velocities:
        return np.array([
            motor_pos,
            math.sin(theta),
            math.cos(theta),
            motor_vel,
            pen_vel,
            prev_action,
        ], dtype=np.float32)
    return np.array([
        motor_pos,
        math.sin(theta),
        math.cos(theta),
        prev_action,
    ], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a trained SAC policy on the device")
    p.add_argument("--policy", required=True, help="path to a .zip checkpoint")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbserial-1130")
    p.add_argument("--baud", type=int, default=2_000_000)
    p.add_argument("--control-freq", type=float, default=50.0,
                   help="control loop frequency in Hz. MUST match the rate "
                        "the policy was trained at — see "
                        "website/src/content/docs/reference/control-rate.md. Default 50 Hz "
                        "matches this rig's canonical design rate.")
    p.add_argument("--action-mode", choices=("accel", "velocity"),
                   default="velocity",
                   help="how the action drives the motor. 'accel' (default): via "
                        "CMD_SET_ACCEL. 'velocity': velocity setpoint converted "
                        "host-side to an accel command each tick. Must match training.")
    p.add_argument("--max-accel-rad-s2", type=float, default=150.0,
                   help="accel/velocity mode: accel command clamp [-max, +max] "
                        "rad/s². Must match training-time max_accel_rad_s2 (150).")
    p.add_argument("--max-velocity-rad-s", type=float, default=3.5,
                   help="velocity mode: action maps to velocity setpoint "
                        "[-max, +max] rad/s. Must match training-time "
                        "max_velocity_rad_s (3.5).")
    p.add_argument("--max-action-delta-rad", type=float, default=0.10,
                   help="position_delta mode: per-tick motor-target delta of "
                        "action × this (rad). Must match training (default 0.10).")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--arm", action="store_true",
                   help="explicitly request ARM; motor remains disarmed without this flag")
    p.add_argument("--max-state-age-ms", type=float, default=80.0,
                   help="maximum host-receipt age accepted for a policy input (default 80 ms)")
    p.add_argument("--dry-run", action="store_true",
                   help="observation-only run; never arm or send acceleration")
    p.add_argument("--log", default=None,
                   help="save the trajectory (state + action) to this .npz path "
                        "for refined sysid / sim-to-real analysis")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy's Gaussian (matches SAC "
                        "training-time behaviour). Default is deterministic = mean. "
                        "Useful while ent_coef is still high and the deterministic "
                        "mean lands in degenerate compromises.")
    p.add_argument("--obs-history-len", type=int, default=None,
                   help="frames stacked into the observation. Default None → "
                        "read from the checkpoint's config.json (1 for legacy "
                        "checkpoints). Must match training.")
    p.add_argument("--ignore-config-mismatch", action="store_true",
                   help="downgrade the config.json validation abort to a warning")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.arm and args.dry_run:
        p.error("--arm and --dry-run are incompatible")
    if args.max_state_age_ms <= 0:
        p.error("--max-state-age-ms must be positive")
    if args.action_mode not in ("accel", "velocity"):
        p.error("protocol v1 supports acceleration commands, not position targets")

    saved_cfg = find_run_config(args.policy) or {}
    obs_history_len = (
        int(args.obs_history_len) if args.obs_history_len is not None
        else int(saved_cfg.get("obs_history_len") or 1)
    )
    obs_include_velocities = bool(saved_cfg.get("obs_include_velocities", True))
    # Actuator-side action smoothing — inherited from the checkpoint's
    # training config, never a deploy-time choice (a policy trained with
    # the boxcar expects its 1.5-tick delay; one trained without would
    # get an unmodelled lag). Mirrors ACTION_SMOOTH_WINDOW in RLControl.ino.
    action_smooth_window = int(saved_cfg.get("action_smooth_window") or 1)

    # The log is written with np.savez — require a .npz path so a slip like
    # `--log runs/<run>/last.zip` can't shadow (or, with a .npz-suffixed
    # typo, overwrite) a model checkpoint.
    if args.log and not str(args.log).endswith(".npz"):
        p.error(f"--log path must end in .npz, got {args.log!r}")

    # Validate flags against the checkpoint's recorded training config
    # (both action modes share obs/action spaces, so a mismatch would
    # otherwise fail silently on the rig). Checks the mode-relevant
    # action scale only.
    expected = {"action_mode": args.action_mode,
                "control_freq_hz": float(args.control_freq),
                "obs_history_len": obs_history_len,
                "obs_include_velocities": obs_include_velocities}
    if args.action_mode in ("accel", "velocity"):
        expected["max_accel_rad_s2"] = float(args.max_accel_rad_s2)
        if args.action_mode == "velocity":
            expected["max_velocity_rad_s"] = float(args.max_velocity_rad_s)

    check_config(args.policy, expected, ignore=args.ignore_config_mismatch)

    arm_requested = confirm_operator_arm(
        requested=args.arm,
        phrase="ARM",
    )

    print(f"Loading policy from {args.policy}")
    if str(args.policy).endswith(".pt"):
        # Distilled student MLP (distill.py / dagger_distill.py output).
        from distill import StudentMLP, _student_predict_factory
        import torch
        from gymnasium import spaces
        ckpt = torch.load(args.policy, map_location=args.device, weights_only=True)
        student = StudentMLP(
            hidden=int(ckpt["hidden"]),
            obs_dim=int(ckpt["obs_dim"]),
            act_dim=int(ckpt["act_dim"]),
        )
        student.load_state_dict(ckpt["state_dict"])
        predict_fn = _student_predict_factory(student, device=args.device)
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])

        class _StudentShim:
            # Real Box objects so the obs/action-space prints match what
            # `SAC.load(...).observation_space` produces. Bounds match the
            # sim env (`pendulum_env.py`): obs are loose, actions clamped
            # to [-1, 1] by SAC's tanh squash.
            observation_space = spaces.Box(
                low=np.array(
                    [-MOTOR_LIMIT_RAD, -1.0, -1.0, -200.0, -200.0, -1.0],
                    dtype=np.float32),
                high=np.array(
                    [MOTOR_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0],
                    dtype=np.float32),
                dtype=np.float32,
            )
            action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
            )

            def predict(self, obs, deterministic: bool = True):
                return predict_fn(obs, deterministic=deterministic)

        model = _StudentShim()
        print(f"  loaded float distilled student "
              f"({ckpt['obs_dim']}->{ckpt['hidden']}->{ckpt['hidden']}->{ckpt['act_dim']}, "
              f"val_mse={ckpt.get('val_mse', float('nan')):.6f})")
    else:
        model = SAC.load(args.policy, device=args.device)
    print(f"Policy obs space: {model.observation_space}")
    print(f"Policy action space: {model.action_space}")

    dt = 1.0 / args.control_freq
    print(f"Control dt = {dt*1000:.2f} ms")

    interrupted = False

    def _interrupt(*_):
        nonlocal interrupted
        interrupted = True

    # SIGTERM handler matters because `timeout` and many shell-level kills
    # send SIGTERM, which would otherwise skip Python's cleanup path and
    # leave the motor engaged.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)

    with LowLevelClient(args.port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print("ERROR: LowLevelServer did not respond. Is it flashed?", file=sys.stderr)
            return 1
        print("Arduino ready.")

        if arm_requested:
            state = client.get_state()
            validate_state(
                state,
                max_age_s=args.max_state_age_ms / 1000.0,
                required_health_mask=REQUIRED_STATE_HEALTH,
            )
            require_ack(client.arm(), operation="ARM")
            try:
                require_ack(client.set_acceleration(0.0), operation="initial SET_ACCEL")
            except Exception:
                client.disarm()
                raise
            print("Motor armed after explicit --arm and typed confirmation.")
        else:
            print("Motor remains DISARMED (observation-only).")

        prev_action = 0.0

        # Observation frame stack (oldest → newest), mirroring the envs:
        # seeded with K copies of the first frame, then one append per tick.
        from collections import deque
        obs_history: deque = deque(maxlen=obs_history_len)

        loop_count = 0
        next_tick = time.monotonic()
        max_steps = int(args.duration_s * args.control_freq)
        ep_reward_proxy = 0.0
        deadline_misses = 0
        max_cycle_ms = 0.0

        # Honest balance counters (mirrors analyze_deploy.honest_balance_metrics).
        # "Balanced" requires near-upright AND slow — the legacy cos-proxy above
        # scores ~0.8 for a continuously spinning pendulum and 0.95+ for a
        # vibrational (Kapitza-style) policy, so it can't certify true balance.
        BAL_THETA_RAD = math.radians(15.0)
        # Gate raised 2.0 -> 4.0 on 2026-07-22: the old value punished the tight ~5 Hz micro-oscillation of fast balancing policies (theta_dot peaks past 2 rad/s at 1.7 deg amplitude) while spin-through and vibrational stabilisation run >= 7-20 rad/s, so 4.0 keeps the gate's anti-spoof teeth without underrating genuine tight balance.
        BAL_PEN_VEL_RAD_S = 4.0
        bal_steps = 0          # total balanced steps
        bal_streak = 0         # current balanced streak (steps)
        bal_streak_max = 0     # longest balanced streak (steps)
        phi_travel = 0.0       # gross pendulum travel (rad) — spins rack this up
        phi_last = None
        v_cmd = 0.0            # velocity-mode commanded-velocity integrator
        from collections import deque as _deque
        a_smooth_buf = _deque([0.0] * action_smooth_window,
                              maxlen=action_smooth_window)
        if action_smooth_window > 1:
            print(f"Action smoothing: boxcar over last {action_smooth_window} actions")

        # Trajectory log buffers (sim convention throughout for ease of analysis)
        log_t_us = np.zeros(max_steps, dtype=np.int64)
        log_motor_pos = np.zeros(max_steps, dtype=np.float32)
        log_pen_pos = np.zeros(max_steps, dtype=np.float32)
        log_motor_vel = np.zeros(max_steps, dtype=np.float32)
        log_pen_vel = np.zeros(max_steps, dtype=np.float32)
        log_accel_cmd = np.zeros(max_steps, dtype=np.float32)
        log_action = np.zeros(max_steps, dtype=np.float32)

        try:
            while loop_count < max_steps and not interrupted:
                tick_start = time.monotonic()
                sleep_for = next_tick - tick_start
                if sleep_for > 0:
                    time.sleep(sleep_for)
                tick_start = time.monotonic()

                s = client.get_state()
                validate_state(
                    s,
                    max_age_s=args.max_state_age_ms / 1000.0,
                    required_health_mask=REQUIRED_STATE_HEALTH,
                )

                # Firmware frame end-to-end; values feed the observation
                # as-is.
                motor_pos = s.motor_pos_rad
                phi = s.pendulum_pos_rad
                motor_vel = s.motor_vel_rad_s
                pen_vel = s.pendulum_vel_rad_s

                frame = make_obs(motor_pos, phi, motor_vel, pen_vel,
                                 prev_action, obs_include_velocities)
                if not obs_history:
                    obs_history.extend([frame] * obs_history_len)
                else:
                    obs_history.append(frame)
                obs = (frame if obs_history_len == 1
                       else np.concatenate(obs_history))
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action_values = np.asarray(action, dtype=np.float32).reshape(-1)
                if action_values.size != 1 or not np.isfinite(action_values[0]):
                    raise UnsafeState("policy returned a non-finite or non-scalar action")
                a = float(np.clip(action_values[0], -1.0, 1.0))
                # Actuator sees the boxcar average; obs/logs keep the raw action.
                if action_smooth_window > 1:
                    a_smooth_buf.append(a)
                    a_cmd = sum(a_smooth_buf) / action_smooth_window
                else:
                    a_cmd = a

                if args.action_mode in ("accel", "velocity"):
                    # Accel-mode: action maps directly to commanded angular accel.
                    # Velocity-mode: action is a velocity setpoint, converted to
                    # an accel command via a saturating P-law on the firmware-
                    # reported motor velocity (same CMD_SET_ACCEL transport).
                    # Safety clamp: if we're at the position limit and the policy
                    # would push us further into it, zero the accel command.
                    if args.action_mode == "velocity":
                        # P-law feedback is the host's own commanded-velocity
                        # integrator (v_cmd), not the firmware-measured
                        # velocity — the measurement's ±0.5 rad/s quantisation
                        # times the control-frequency gain injected a
                        # ±17 rad/s² accel dither. The complementary
                        # correction below heals integrator drift while
                        # attenuating that noise ~10×. Mirrors real_env.py
                        # and the sim's velocity mode.
                        v_des = a_cmd * args.max_velocity_rad_s
                        cmd_value = float(np.clip(
                            (v_des - v_cmd) * args.control_freq,
                            -args.max_accel_rad_s2, args.max_accel_rad_s2,
                        ))
                    else:
                        cmd_value = a_cmd * args.max_accel_rad_s2
                    if motor_pos >= MOTOR_SAFE_LIMIT_RAD and cmd_value > 0.0:
                        cmd_value = 0.0
                    elif motor_pos <= -MOTOR_SAFE_LIMIT_RAD and cmd_value < 0.0:
                        cmd_value = 0.0
                    if args.action_mode == "velocity":
                        v_cmd = float(np.clip(
                            v_cmd + cmd_value / args.control_freq,
                            -args.max_velocity_rad_s, args.max_velocity_rad_s,
                        ))
                        v_cmd += 0.1 * (motor_vel - v_cmd)
                if args.arm and not args.dry_run:
                    validate_state(
                        s,
                        max_age_s=args.max_state_age_ms / 1000.0,
                        required_health_mask=REQUIRED_STATE_HEALTH,
                    )
                    require_ack(client.set_acceleration(cmd_value), operation="SET_ACCEL")

                cycle_s = time.monotonic() - tick_start
                max_cycle_ms = max(max_cycle_ms, cycle_s * 1000.0)
                if cycle_s > dt:
                    deadline_misses += 1
                    raise UnsafeState(
                        f"control deadline missed: {cycle_s * 1000:.2f} ms > {dt * 1000:.2f} ms"
                    )
                next_tick += dt
                if next_tick < time.monotonic():
                    next_tick = time.monotonic() + dt

                # Reward for live monitoring
                theta = _wrap_pi(s.pendulum_pos_rad - math.pi)
                ep_reward_proxy += 0.5 * (1.0 + math.cos(theta))

                # Honest balance counters.
                if abs(theta) <= BAL_THETA_RAD and abs(pen_vel) <= BAL_PEN_VEL_RAD_S:
                    bal_steps += 1
                    bal_streak += 1
                    bal_streak_max = max(bal_streak_max, bal_streak)
                else:
                    bal_streak = 0
                if phi_last is not None:
                    phi_travel += abs(phi - phi_last)
                phi_last = phi

                # Log this step (firmware frame, same as the observation).
                # The "commanded" quantity is the angular accel (accel-mode) or
                # the position target (position-delta mode) — logged under the
                # same array name for downstream tooling compatibility. The
                # saved npz records `action_mode` so consumers can disambiguate.
                if args.log:
                    log_t_us[loop_count] = s.time_us
                    log_motor_pos[loop_count] = motor_pos
                    log_pen_pos[loop_count] = phi
                    log_motor_vel[loop_count] = motor_vel
                    log_pen_vel[loop_count] = pen_vel
                    log_accel_cmd[loop_count] = cmd_value
                    log_action[loop_count] = a

                if loop_count % args.control_freq == 0:
                    cmd_label = "accel_cmd"
                    print(
                        f"t={loop_count * dt:.1f}s  motor={motor_pos:+.3f} "
                        f"{cmd_label}={cmd_value:+6.2f}  theta={theta:+.3f}  "
                        f"upright={0.5 * (1.0 + math.cos(theta)):.2f}"
                    )

                prev_action = a
                loop_count += 1
        finally:
            # Disarm is the fail-closed host cleanup. The firmware command
            # lease independently stops/disables the driver if this ACK fails.
            try:
                client.disarm()
            except Exception:
                pass
            print(f"Loop finished. Steps: {loop_count}, "
                  f"max cycle: {max_cycle_ms:.2f} ms, deadline misses: {deadline_misses}, "
                  f"avg upright proxy: {ep_reward_proxy / max(1, loop_count):.3f}, "
                  f"motor disarmed.")
            if loop_count > 0:
                print(f"Honest balance (|θ|≤15° AND |θ̇|≤{BAL_PEN_VEL_RAD_S:.0f} rad/s): "
                      f"fraction {bal_steps / loop_count:.3f}, "
                      f"longest streak {bal_streak_max * dt:.2f} s, "
                      f"pendulum travel {phi_travel / (2 * math.pi):.1f} rev "
                      f"(spinning if ≫1 after swing-up)")

            if args.log and loop_count > 0:
                Path(args.log).parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    args.log,
                    time_us=log_t_us[:loop_count],
                    motor_pos_rad=log_motor_pos[:loop_count],
                    pendulum_pos_rad=log_pen_pos[:loop_count],
                    motor_vel_rad_s=log_motor_vel[:loop_count],
                    pendulum_vel_rad_s=log_pen_vel[:loop_count],
                    accel_cmd_rad_s2=log_accel_cmd[:loop_count],
                    action=log_action[:loop_count],
                    control_freq_hz=np.float32(args.control_freq),
                    max_accel_rad_s2=np.float32(args.max_accel_rad_s2),
                    max_velocity_rad_s=np.float32(args.max_velocity_rad_s),
                    max_action_delta_rad=np.float32(args.max_action_delta_rad),
                    action_mode=str(args.action_mode),
                    policy_path=str(args.policy),
                )
                print(f"Saved trajectory log to {args.log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
