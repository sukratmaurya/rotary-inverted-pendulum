
import math
import csv

import mujoco
import numpy as np
from stable_baselines3 import SAC

from pendulum_env import RotaryInvertedPendulumEnv

MODEL_PATH = "runs/cpu_training_3/best_model.zip"
OUTPUT_CSV = "action_lag_ab_results.csv"

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0
UPRIGHT_TOLERANCE_DEG = 15.0
HOLD_SECONDS = 1.0

MAX_STEPS = round(CONTROL_HZ * EPISODE_SECONDS)
HOLD_STEPS = round(CONTROL_HZ * HOLD_SECONDS)

# Compare no action lag against 30 ms action lag.
CONDITIONS = {
    "zero_lag": 0.0,
    "lag_30ms": 0.030,
}


def upright_error(phi):
    return abs(math.atan2(
        math.sin(phi - math.pi),
        math.cos(phi - math.pi),
    ))


def make_env(lag):
    return RotaryInvertedPendulumEnv(
        render_mode=None,
        control_freq_hz=CONTROL_HZ,
        action_mode="velocity",
        max_accel_rad_s2=150.0,
        max_velocity_rad_s=3.5,
        max_action_delta_rad=0.1,
        reward_alive_offset=15.0,
        reward_upright_alive_weight=5.0,
        reward_stillness_bonus_weight=5.0,
        reward_motor_pos_weight=0.5,
        reward_motor_vel_weight=0.005,
        upright_reset_frac=1.0,
        obs_history_len=4,
        obs_include_velocities=True,
        firmware_obs_model=True,
        action_smooth_window=4,

        # Disable other episode-level DR so only lag differs.
        domain_randomization=False,
        motor_max_accel_rad_s2=150.0,
        action_delay_steps=0,
        action_lag_tau_s=lag,
        motor_tau_s=0.0,
        dr_theta_bias_max_rad=0.0,
        dr_base_tilt_max_rad=0.0,
        dr_control_dt_jitter_frac=0.0,
        obs_staleness_s=0.005,
    )


def set_initial_state(env, state):
    motor_pos, pen_phi, motor_vel, pen_vel = state

    env.data.qpos[env._motor_qpos_addr] = motor_pos
    env.data.qpos[env._pen_qpos_addr] = pen_phi
    env.data.qvel[env._motor_qvel_addr] = motor_vel
    env.data.qvel[env._pen_qvel_addr] = pen_vel

    env._motor_target = motor_pos
    env._lagged_target = motor_pos
    env._prev_cmd_pos = motor_pos
    env._motor_vel = motor_vel
    env._prev_action = 0.0
    env._prev_motor_vel = motor_vel

    env._action_queue.clear()
    env._lagged_action = 0.0
    env._lagged_accel_cmd = 0.0
    env._accel_cmd_queue.clear()
    env._kick_ticks_left = 0
    env._action_smooth_buf.clear()
    env._action_smooth_buf.extend([0.0] * env.action_smooth_window)

    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    env._obs_history.clear()

    return env._obs()


def run_trial(model, env, state):
    obs = set_initial_state(env, state)

    consecutive = 0
    longest = 0
    max_abs_action = 0.0
    success_time = None
    terminated_flag = False

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(np.asarray(action).flatten()[0])
        max_abs_action = max(max_abs_action, abs(action_value))

        obs, _, terminated, truncated, info = env.step(action)

        error = upright_error(float(info["phi"]))

        if error <= math.radians(UPRIGHT_TOLERANCE_DEG):
            consecutive += 1
            longest = max(longest, consecutive)

            if consecutive >= HOLD_STEPS:
                success_time = (step - HOLD_STEPS + 2) / CONTROL_HZ
                break
        else:
            consecutive = 0

        if terminated or truncated:
            terminated_flag = bool(terminated)
            break

    return {
        "success": success_time is not None,
        "stable_entry_s": success_time,
        "longest_upright_s": longest / CONTROL_HZ,
        "max_abs_action": max_abs_action,
        "terminated": terminated_flag,
    }


def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    rows = []

    print("\nCONTROLLED ACTION-LAG A/B TEST")
    print("=" * 68)
    print(f"Trials per condition: {TRIALS}")
    print(f"Control frequency: {CONTROL_HZ} Hz")
    print("All non-lag environment randomization disabled")
    print("=" * 68)

    # Same initial states for both conditions.
    rng = np.random.default_rng(24001)
    states = []

    while len(states) < TRIALS:
        motor_pos = rng.uniform(-1.0, 1.0)
        pen_phi = rng.uniform(0.0, 2.0 * math.pi)

        if upright_error(pen_phi) < math.radians(30.0):
            continue

        states.append((
            motor_pos,
            pen_phi,
            rng.uniform(-0.5, 0.5),
            rng.uniform(-1.0, 1.0),
        ))

    envs = {name: make_env(lag) for name, lag in CONDITIONS.items()}

    try:
        for trial, state in enumerate(states, start=1):
            initial_error = math.degrees(upright_error(state[1]))

            # Use the same physical initial state in both conditions.
            for condition, lag in CONDITIONS.items():
                env = envs[condition]
                env.reset(seed=8000 + trial)

                result = run_trial(model, env, state)

                row = {
                    "trial": trial,
                    "condition": condition,
                    "action_lag_tau_s": lag,
                    "initial_error_deg": initial_error,
                    "initial_motor_pos_rad": state[0],
                    "initial_motor_vel_rad_s": state[2],
                    "initial_pendulum_vel_rad_s": state[3],
                    **result,
                }
                rows.append(row)

                label = "SUCCESS" if result["success"] else "FAIL"
                time_text = (
                    f"{result['stable_entry_s']:.2f}s"
                    if result["success"]
                    else f"upright={result['longest_upright_s']:.2f}s"
                )

                print(
                    f"Trial {trial:02d} | {condition:8s} | "
                    f"{label:7s} | error={initial_error:6.1f} deg | "
                    f"{time_text} | "
                    f"max_action={result['max_abs_action']:.3f}"
                )

    finally:
        for env in envs.values():
            env.close()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY")
    print("=" * 68)

    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        successes = [r for r in group if r["success"]]
        failures = len(group) - len(successes)

        times = [r["stable_entry_s"] for r in successes]

        print(
            f"{condition}: {len(successes)}/{len(group)} successes "
            f"({100 * len(successes) / len(group):.1f}%), "
            f"{failures} failures"
        )

        if times:
            print(
                f"  Mean stable-entry time: {np.mean(times):.2f}s; "
                f"slowest: {max(times):.2f}s"
            )

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
