
import csv
import math

import mujoco
import numpy as np
from stable_baselines3 import SAC

from pendulum_env import RotaryInvertedPendulumEnv

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/cpu_training_3/best_model.zip"
OUTPUT_CSV = "action_lag_20ms_results.csv"

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0

UPRIGHT_TOLERANCE_DEG = 15.0
HOLD_SECONDS = 1.0
MIN_INITIAL_ERROR_DEG = 30.0

ACTION_LAG_TAU_S = 0.020

MAX_STEPS = round(CONTROL_HZ * EPISODE_SECONDS)
HOLD_STEPS = round(CONTROL_HZ * HOLD_SECONDS)


# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
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

        # Disable randomized conditions to isolate action lag.
        domain_randomization=False,
        motor_max_accel_rad_s2=150.0,
        action_delay_steps=0,
        action_lag_tau_s=ACTION_LAG_TAU_S,
        motor_tau_s=0.0,

        dr_theta_bias_max_rad=0.0,
        dr_base_tilt_max_rad=0.0,
        dr_control_dt_jitter_frac=0.0,
        obs_staleness_s=0.005,
    )


# ============================================================
# HELPERS
# ============================================================

def upright_error(phi):
    """Absolute angular distance from the upright position."""
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def make_initial_states():
    """Generate reproducible initial states for paired comparisons."""
    rng = np.random.default_rng(24001)
    states = []

    while len(states) < TRIALS:
        motor_pos = rng.uniform(-1.0, 1.0)
        pen_phi = rng.uniform(0.0, 2.0 * math.pi)

        error = upright_error(pen_phi)

        if error < math.radians(MIN_INITIAL_ERROR_DEG):
            continue

        motor_vel = rng.uniform(-0.5, 0.5)
        pen_vel = rng.uniform(-1.0, 1.0)

        states.append({
            "motor_pos": motor_pos,
            "pen_phi": pen_phi,
            "motor_vel": motor_vel,
            "pen_vel": pen_vel,
        })

    return states


def set_initial_state(env, state):
    """Set the same physical state for every comparison."""
    motor_pos = state["motor_pos"]
    pen_phi = state["pen_phi"]
    motor_vel = state["motor_vel"]
    pen_vel = state["pen_vel"]

    env.data.qpos[env._motor_qpos_addr] = motor_pos
    env.data.qpos[env._pen_qpos_addr] = pen_phi

    env.data.qvel[env._motor_qvel_addr] = motor_vel
    env.data.qvel[env._pen_qvel_addr] = pen_vel

    # Synchronize controller state.
    env._motor_target = motor_pos
    env._lagged_target = motor_pos
    env._prev_cmd_pos = motor_pos
    env._motor_vel = motor_vel
    env._prev_action = 0.0
    env._prev_motor_vel = motor_vel

    # Reset transport and smoothing states.
    env._action_queue.clear()
    env._lagged_action = 0.0
    env._lagged_accel_cmd = 0.0
    env._accel_cmd_queue.clear()
    env._kick_ticks_left = 0

    env._action_smooth_buf.clear()
    env._action_smooth_buf.extend(
        [0.0] * env.action_smooth_window
    )

    # Refresh physics and measurement history.
    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    env._obs_history.clear()

    return env._obs()


# ============================================================
# TRIAL
# ============================================================

def run_trial(model, env, state):
    obs = set_initial_state(env, state)

    consecutive_upright = 0
    longest_upright = 0
    max_abs_action = 0.0

    stable_entry_time = None
    terminated_flag = False

    for step in range(MAX_STEPS):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action_value = float(
            np.asarray(action).flatten()[0]
        )

        max_abs_action = max(
            max_abs_action,
            abs(action_value),
        )

        obs, _, terminated, truncated, info = env.step(action)

        error = upright_error(float(info["phi"]))

        if error <= math.radians(UPRIGHT_TOLERANCE_DEG):
            consecutive_upright += 1

            longest_upright = max(
                longest_upright,
                consecutive_upright,
            )

            if consecutive_upright >= HOLD_STEPS:
                stable_entry_time = (
                    step - HOLD_STEPS + 2
                ) / CONTROL_HZ
                break
        else:
            consecutive_upright = 0

        if terminated or truncated:
            terminated_flag = bool(terminated)
            break

    return {
        "success": stable_entry_time is not None,
        "stable_entry_s": (
            stable_entry_time
            if stable_entry_time is not None
            else ""
        ),
        "longest_upright_s": (
            longest_upright / CONTROL_HZ
        ),
        "max_abs_action": max_abs_action,
        "terminated": terminated_flag,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    model = SAC.load(MODEL_PATH, device="cpu")
    env = make_env()

    states = make_initial_states()
    results = []

    print("\nACTION-LAG TEST: 15 ms")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Trials: {TRIALS}")
    print(f"Control frequency: {CONTROL_HZ} Hz")
    print(f"Action-lag time constant: {ACTION_LAG_TAU_S * 1000:.1f} ms")
    print(f"Episode duration: {EPISODE_SECONDS:.1f} s")
    print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEG:.1f} degrees")
    print(f"Required upright hold: {HOLD_SECONDS:.1f} s")
    print("Other domain randomization: disabled")
    print("=" * 70)

    try:
        for index, state in enumerate(states, start=1):
            env.reset(seed=8000 + index)

            result = run_trial(model, env, state)

            initial_error = math.degrees(
                upright_error(state["pen_phi"])
            )

            row = {
                "trial": index,
                "action_lag_tau_s": ACTION_LAG_TAU_S,
                "initial_error_deg": initial_error,
                "initial_motor_pos_rad": state["motor_pos"],
                "initial_motor_vel_rad_s": state["motor_vel"],
                "initial_pendulum_vel_rad_s": state["pen_vel"],
                **result,
            }

            results.append(row)

            label = (
                "SUCCESS"
                if result["success"]
                else "FAIL"
            )

            timing = (
                f"stable-entry={result['stable_entry_s']:.2f}s"
                if result["success"]
                else (
                    "longest-upright="
                    f"{result['longest_upright_s']:.2f}s"
                )
            )

            print(
                f"Trial {index:02d} | {label:7s} | "
                f"initial error={initial_error:6.1f} deg | "
                f"{timing} | "
                f"max action={result['max_abs_action']:.3f}"
            )

    finally:
        env.close()

    fields = list(results[0].keys())

    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    successes = [
        result for result in results
        if result["success"]
    ]

    failures = TRIALS - len(successes)

    print("\nSUMMARY")
    print("=" * 70)
    print(f"Successful trials: {len(successes)}/{TRIALS}")
    print(f"Failed trials: {failures}")
    print(f"Success rate: {100 * len(successes) / TRIALS:.1f}%")

    if successes:
        times = [
            result["stable_entry_s"]
            for result in successes
        ]

        print(f"Mean stable-entry time: {np.mean(times):.2f} s")
        print(f"Slowest stable-entry time: {max(times):.2f} s")

    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
