import csv
import math

import mujoco
import numpy as np
from stable_baselines3 import SAC

from pendulum_env import RotaryInvertedPendulumEnv

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODELS = {
    "training3": "runs/cpu_training_3/best_model.zip",
    "training4": "runs/cpu_training_4_lag_robust/best_model.zip",
}

LAG_VALUES_S = [0.0, 0.015, 0.020, 0.025, 0.030]

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0

UPRIGHT_TOLERANCE_DEG = 15.0
HOLD_SECONDS = 1.0
MIN_INITIAL_ERROR_DEG = 30.0

OUTPUT_CSV = "training3_vs_training4_lag_comparison.csv"

MAX_STEPS = round(CONTROL_HZ * EPISODE_SECONDS)
HOLD_STEPS = round(CONTROL_HZ * HOLD_SECONDS)


def upright_error(phi):
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def make_initial_states():
    """Use identical reproducible states for both models."""
    rng = np.random.default_rng(24001)
    states = []

    while len(states) < TRIALS:
        motor_pos = rng.uniform(-1.0, 1.0)
        pen_phi = rng.uniform(0.0, 2.0 * math.pi)

        if upright_error(pen_phi) < math.radians(
            MIN_INITIAL_ERROR_DEG
        ):
            continue

        states.append({
            "motor_pos": motor_pos,
            "pen_phi": pen_phi,
            "motor_vel": rng.uniform(-0.5, 0.5),
            "pen_vel": rng.uniform(-1.0, 1.0),
        })

    return states


def make_env(lag_tau_s):
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
        domain_randomization=False,
        motor_max_accel_rad_s2=150.0,
        action_delay_steps=0,
        action_lag_tau_s=lag_tau_s,
        motor_tau_s=0.0,
        dr_theta_bias_max_rad=0.0,
        dr_base_tilt_max_rad=0.0,
        dr_control_dt_jitter_frac=0.0,
        obs_staleness_s=0.005,
    )


def set_initial_state(env, state):
    env.data.qpos[env._motor_qpos_addr] = state["motor_pos"]
    env.data.qpos[env._pen_qpos_addr] = state["pen_phi"]

    env.data.qvel[env._motor_qvel_addr] = state["motor_vel"]
    env.data.qvel[env._pen_qvel_addr] = state["pen_vel"]

    env._motor_target = state["motor_pos"]
    env._lagged_target = state["motor_pos"]
    env._prev_cmd_pos = state["motor_pos"]
    env._motor_vel = state["motor_vel"]
    env._prev_action = 0.0
    env._prev_motor_vel = state["motor_vel"]

    env._action_queue.clear()
    env._lagged_action = 0.0
    env._lagged_accel_cmd = 0.0
    env._accel_cmd_queue.clear()
    env._kick_ticks_left = 0

    env._action_smooth_buf.clear()
    env._action_smooth_buf.extend(
        [0.0] * env.action_smooth_window
    )

    mujoco.mj_forward(env.model, env.data)
    env.seed_measurement_ring()
    env._obs_history.clear()

    return env._obs()


def run_trial(model, env, state):
    obs = set_initial_state(env, state)

    consecutive_upright = 0
    longest_upright = 0
    stable_entry_time = None
    max_abs_action = 0.0
    terminated_flag = False

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(np.asarray(action).flatten()[0])
        max_abs_action = max(max_abs_action, abs(action_value))

        obs, _, terminated, truncated, info = env.step(action)

        error = upright_error(float(info["phi"]))

        if error <= math.radians(UPRIGHT_TOLERANCE_DEG):
            consecutive_upright += 1
            longest_upright = max(longest_upright, consecutive_upright)

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
            stable_entry_time if stable_entry_time is not None else ""
        ),
        "longest_upright_s": longest_upright / CONTROL_HZ,
        "max_abs_action": max_abs_action,
        "terminated": terminated_flag,
    }


def main():
    states = make_initial_states()
    rows = []

    print("TRAINING 3 vs TRAINING 4 — ACTION-LAG COMPARISON")
    print("=" * 76)
    print(f"Trials per model/lag: {TRIALS}")
    print(f"Lags: {[int(x * 1000) for x in LAG_VALUES_S]} ms")
    print(f"Control frequency: {CONTROL_HZ} Hz")
    print(f"Success: upright within +/-{UPRIGHT_TOLERANCE_DEG} degrees "
          f"for {HOLD_SECONDS:.1f} continuous second")
    print("Other domain randomization: disabled")
    print("=" * 76)

    try:
        for lag in LAG_VALUES_S:
            env = make_env(lag)
            try:
                for model_name, model_path in MODELS.items():
                    print(
                        f"\nModel={model_name} | "
                        f"lag_tau={lag * 1000:.0f} ms"
                    )
                    model = SAC.load(model_path, device="cpu")
                    outcomes = []

                    for index, state in enumerate(states, start=1):
                        env.reset(seed=8000 + index)
                        result = run_trial(model, env, state)

                        initial_error = math.degrees(
                            upright_error(state["pen_phi"])
                        )

                        row = {
                            "model": model_name,
                            "model_path": model_path,
                            "action_lag_tau_ms": lag * 1000,
                            "trial": index,
                            "initial_error_deg": initial_error,
                            "initial_motor_pos_rad": state["motor_pos"],
                            "initial_motor_vel_rad_s": state["motor_vel"],
                            "initial_pendulum_vel_rad_s": state["pen_vel"],
                            **result,
                        }
                        rows.append(row)
                        outcomes.append(row)

                        print(
                            f"  Trial {index:02d}: "
                            f"{'SUCCESS' if result['success'] else 'FAIL':7s} | "
                            f"initial_error={initial_error:6.1f} deg | "
                            f"stable={result['stable_entry_s']} | "
                            f"longest_upright={result['longest_upright_s']:.2f}s"
                        )

                    successes = [r for r in outcomes if r["success"]]
                    mean_time = (
                        np.mean([r["stable_entry_s"] for r in successes])
                        if successes else float("nan")
                    )
                    print(
                        f"SUMMARY {model_name} at {lag * 1000:.0f} ms: "
                        f"{len(successes)}/{TRIALS} successes "
                        f"({100 * len(successes) / TRIALS:.1f}%), "
                        f"mean stable-entry={mean_time:.2f}s"
                    )
            finally:
                env.close()

    finally:
        if rows:
            with open(OUTPUT_CSV, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print("\nOVERALL SUMMARY")
    print("=" * 76)
    for lag in LAG_VALUES_S:
        for model_name in MODELS:
            subset = [
                r for r in rows
                if r["model"] == model_name
                and r["action_lag_tau_ms"] == lag * 1000
            ]
            successes = [r for r in subset if r["success"]]
            times = [r["stable_entry_s"] for r in successes]
            mean_time = np.mean(times) if times else float("nan")
            print(
                f"{model_name:10s} | {lag * 1000:2.0f} ms | "
                f"{len(successes):2d}/{TRIALS} "
                f"({100 * len(successes) / TRIALS:5.1f}%) | "
                f"mean stable-entry={mean_time:.2f}s"
            )

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
