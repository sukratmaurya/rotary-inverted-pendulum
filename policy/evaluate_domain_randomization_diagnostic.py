
import csv
import math
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from pendulum_env import RotaryInvertedPendulumEnv


MODEL_PATH = "runs/cpu_training_3/best_model.zip"
OUTPUT_CSV = "domain_randomization_results.csv"

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0

MIN_INITIAL_ERROR_DEG = 30.0
UPRIGHT_TOLERANCE_DEG = 15.0
REQUIRED_HOLD_SECONDS = 1.0

MAX_STEPS = round(EPISODE_SECONDS * CONTROL_HZ)
HOLD_STEPS = round(REQUIRED_HOLD_SECONDS * CONTROL_HZ)


def upright_error(phi):
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def randomized_initial_angle(rng):
    minimum = math.radians(MIN_INITIAL_ERROR_DEG)

    while True:
        phi = rng.uniform(0.0, 2.0 * math.pi)
        if upright_error(phi) >= minimum:
            return phi


env = RotaryInvertedPendulumEnv(
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
    domain_randomization=True,
)

model = SAC.load(MODEL_PATH, device="cpu")

fields = [
    "trial",
    "success",
    "initial_error_deg",
    "initial_motor_pos_rad",
    "initial_motor_vel_rad_s",
    "initial_pendulum_vel_rad_s",
    "motor_max_accel_rad_s2",
    "action_delay_steps",
    "action_lag_tau_s",
    "motor_tau_s",
    "obs_staleness_s",
    "theta_bias_rad",
    "base_tilt_rad",
    "motor_frictionloss_Nm",
    "pendulum_mass_kg",
    "pendulum_friction",
    "pendulum_coulomb_friction",
    "stable_entry_s",
    "longest_upright_s",
    "max_abs_action",
    "terminated",
]

rows = []

print("\nDOMAIN-RANDOMIZATION DIAGNOSTIC")
print("=" * 68)

try:
    for trial in range(TRIALS):
        rng = np.random.default_rng(12000 + trial)

        obs, _ = env.reset(seed=5000 + trial)

        motor_pos = rng.uniform(-1.4, 1.4)
        pen_phi = randomized_initial_angle(rng)
        motor_vel = rng.uniform(-0.5, 0.5)
        pen_vel = rng.uniform(-1.0, 1.0)

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

        mujoco.mj_forward(env.model, env.data)

        if env.firmware_obs_model:
            env.seed_measurement_ring()

        env._obs_history.clear()
        obs = env._obs()

        initial_error = math.degrees(
            upright_error(float(env.data.qpos[env._pen_qpos_addr]))
        )

        # Capture per-episode randomized values after reset.
        row = {
            "trial": trial + 1,
            "success": False,
            "initial_error_deg": initial_error,
            "initial_motor_pos_rad": motor_pos,
            "initial_motor_vel_rad_s": motor_vel,
            "initial_pendulum_vel_rad_s": pen_vel,
            "motor_max_accel_rad_s2": env._motor_max_accel_rad_s2,
            "action_delay_steps": env._action_delay_steps,
            "action_lag_tau_s": env._action_lag_tau_s,
            "motor_tau_s": env._motor_tau_s,
            "obs_staleness_s": env._obs_staleness_s,
            "theta_bias_rad": env._theta_bias_rad,
            "base_tilt_rad": math.acos(
                np.clip(
                    -env.model.opt.gravity[2] / 9.81,
                    -1.0,
                    1.0,
                )
            ),
            "motor_frictionloss_Nm": env.model.dof_frictionloss[
                env._motor_dof_addr
            ],
            "pendulum_mass_kg": env.model.body_mass[env._pen_body_id],
            "pendulum_friction": env.model.dof_damping[
                env._pen_dof_addr
            ],
            "pendulum_coulomb_friction": env.model.dof_frictionloss[
                env._pen_dof_addr
            ],
            "stable_entry_s": "",
            "longest_upright_s": 0.0,
            "max_abs_action": 0.0,
            "terminated": False,
        }

        consecutive = 0
        longest = 0
        success = False
        max_abs_action = 0.0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            max_abs_action = max(
                max_abs_action,
                abs(float(np.asarray(action).flatten()[0])),
            )

            obs, _, terminated, truncated, info = env.step(action)

            error = upright_error(float(info["phi"]))

            if error <= math.radians(UPRIGHT_TOLERANCE_DEG):
                consecutive += 1
                longest = max(longest, consecutive)

                if consecutive >= HOLD_STEPS:
                    success = True
                    row["stable_entry_s"] = (
                        step - HOLD_STEPS + 2
                    ) / CONTROL_HZ
                    break
            else:
                consecutive = 0

            if terminated or truncated:
                row["terminated"] = bool(terminated)
                break

        row["success"] = success
        row["longest_upright_s"] = longest / CONTROL_HZ
        row["max_abs_action"] = max_abs_action

        rows.append(row)

        result = "SUCCESS" if success else "FAIL"
        stable_time = (
            f"{row['stable_entry_s']:.2f}s"
            if success
            else f"longest upright {row['longest_upright_s']:.2f}s"
        )

        print(
            f"Trial {trial + 1:02d}: {result} | "
            f"Error {initial_error:6.1f} deg | "
            f"Accel {row['motor_max_accel_rad_s2']:6.1f} | "
            f"Lag {row['action_lag_tau_s'] * 1000:5.1f} ms | "
            f"Bias {math.degrees(row['theta_bias_rad']):5.2f} deg | "
            f"{stable_time}"
        )

finally:
    env.close()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

successes = [r for r in rows if r["success"]]
failures = [r for r in rows if not r["success"]]

print("\nSUMMARY")
print("=" * 68)
print(f"Trials: {len(rows)}")
print(f"Successful: {len(successes)}")
print(f"Failed: {len(failures)}")
print(f"Success rate: {100 * len(successes) / len(rows):.1f}%")

if successes:
    times = [float(r["stable_entry_s"]) for r in successes]
    print(f"Mean stable-entry time: {np.mean(times):.2f} s")
    print(f"Slowest stable-entry time: {np.max(times):.2f} s")

print(f"CSV saved to: {Path(OUTPUT_CSV).resolve()}")

