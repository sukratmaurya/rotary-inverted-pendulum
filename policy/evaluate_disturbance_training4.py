
import math
import numpy as np
import mujoco
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

MODEL_PATH = "runs/cpu_training_4_lag_robust/best_model.zip"

TRIALS = 20
CONTROL_HZ = 50
SETTLE_SECONDS = 1.0
RECOVERY_TIMEOUT = 3.0
DISTURBANCE_DEGREES = 60.0
UPRIGHT_TOLERANCE_DEGREES = 15.0
HOLD_SECONDS = 0.2

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
)

model = SAC.load(MODEL_PATH, device="cpu")

settle_steps = round(SETTLE_SECONDS * CONTROL_HZ)
recovery_steps = round(RECOVERY_TIMEOUT * CONTROL_HZ)
hold_steps = round(HOLD_SECONDS * CONTROL_HZ)
tolerance = math.radians(UPRIGHT_TOLERANCE_DEGREES)
disturbance = math.radians(DISTURBANCE_DEGREES)

results = []

def upright_error(phi):
    return abs(math.atan2(math.sin(phi - math.pi),
                          math.cos(phi - math.pi)))

print("\nDISTURBANCE RECOVERY TEST V2")
print("=" * 60)
print(f"Control frequency: {CONTROL_HZ} Hz")
print(f"Disturbance magnitude: +/-{DISTURBANCE_DEGREES:.1f} degrees")
print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEGREES:.1f} degrees")
print(f"Trials: {TRIALS}")
print("Disturbance type: instantaneous joint-position displacement")
print("")

try:
    for trial in range(TRIALS):
        obs, _ = env.reset(seed=3000 + trial)
        terminated = False
        truncated = False

        # Settle the policy at upright.
        for _ in range(settle_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if terminated or truncated:
            print(f"Trial {trial + 1}: INVALID - ended during settling")
            results.append(None)
            continue

        # Alternate the perturbation direction.
        direction = 1.0 if trial % 2 == 0 else -1.0
        env.data.qpos[env._pen_qpos_addr] += direction * disturbance
        env.data.qpos[env._pen_qpos_addr] %= 2.0 * math.pi
        mujoco.mj_forward(env.model, env.data)

        # Reset firmware-style measurements from the disturbed state.
        if env.firmware_obs_model:
            env.seed_measurement_ring()

        # Clear stale pre-disturbance observation frames.
        env._obs_history.clear()
        obs = env._obs()

        phi_after = float(env.data.qpos[env._pen_qpos_addr])
        initial_error = math.degrees(upright_error(phi_after))

        # Verify the perturbation is actually outside the target region.
        if initial_error <= UPRIGHT_TOLERANCE_DEGREES:
            print(
                f"Trial {trial + 1}: INVALID - actual initial error "
                f"{initial_error:.1f} degrees is inside tolerance"
            )
            results.append(None)
            continue

        continuous_upright = 0
        recovery_time = None
        longest_upright = 0

        for step in range(recovery_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            phi = float(info["phi"])
            error = upright_error(phi)

            if error <= tolerance:
                continuous_upright += 1
                longest_upright = max(longest_upright, continuous_upright)

                if continuous_upright >= hold_steps:
                    first_in_region_step = step - hold_steps + 1
                    recovery_time = (first_in_region_step + 1) / CONTROL_HZ
                    break
            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        if recovery_time is None:
            print(
                f"Trial {trial + 1}: NO RECOVERY | "
                f"Initial error: {initial_error:.1f} degrees | "
                f"Longest upright interval: "
                f"{longest_upright / CONTROL_HZ:.2f}s"
            )
            results.append(None)
        else:
            print(
                f"Trial {trial + 1}: RECOVERED | "
                f"Initial error: {initial_error:.1f} degrees | "
                f"Recovery time: {recovery_time:.2f}s"
            )
            results.append(recovery_time)

finally:
    env.close()

valid = [x for x in results if x is not None]
failed = sum(x is None for x in results)

print("\nSUMMARY")
print("=" * 60)
print(f"Valid trials: {len(results) - sum('INVALID' in str(x) for x in [])}")
print(f"Successful recoveries: {len(valid)}")
print(f"Failures or invalid trials: {failed}")
print(f"Success rate among all trials: {100 * len(valid) / TRIALS:.1f}%")

if valid:
    print(f"Mean recovery time: {np.mean(valid):.2f}s")
    print(f"Slowest successful recovery: {max(valid):.2f}s")
