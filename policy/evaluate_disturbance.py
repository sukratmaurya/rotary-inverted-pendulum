import math
import numpy as np
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

MODEL_PATH = "runs/cpu_training_3/best_model.zip"

TRIALS = 50
CONTROL_HZ = 20
SETTLE_SECONDS = 1.0
RECOVERY_TIMEOUT = 3.0
DISTURBANCE_DEGREES = 45.0
UPRIGHT_TOLERANCE_DEGREES = 15.0

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

settle_steps = int(SETTLE_SECONDS * CONTROL_HZ)
recovery_steps = int(RECOVERY_TIMEOUT * CONTROL_HZ)
tolerance = math.radians(UPRIGHT_TOLERANCE_DEGREES)
disturbance = math.radians(DISTURBANCE_DEGREES)

results = []

print("\nDISTURBANCE RECOVERY TEST")
print("=" * 60)
print(f"Disturbance: {DISTURBANCE_DEGREES:.1f} degrees")
print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEGREES:.1f} degrees")
print(f"Trials: {TRIALS}\n")

try:
    for trial in range(TRIALS):
        obs, _ = env.reset(seed=2000 + trial)

        # Allow the policy to settle before applying the disturbance.
        for _ in range(settle_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if terminated or truncated:
            print(f"Trial {trial + 1}: could not stabilize before disturbance")
            results.append(None)
            continue

        # Apply an instantaneous angular displacement to the pendulum.
        # This is a simulated position perturbation, not a physical impulse.
        env.data.qpos[env._pen_qpos_addr] += disturbance
        env.data.qpos[env._pen_qpos_addr] = (
            env.data.qpos[env._pen_qpos_addr] % (2 * math.pi)
        )
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        # Refresh the firmware-style measurement ring after perturbation
        # so the controller receives the disturbed pendulum measurement.
        if env.firmware_obs_model:
            env.seed_measurement_ring()

        obs = env._obs()

        recovered_steps = None
        continuous_upright = 0
        longest_upright = 0

        for step in range(recovery_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            # phi = 0 is hanging; phi = pi is upright.
            phi = float(info["phi"])
            angle_error = abs(math.atan2(math.sin(phi - math.pi),
                                         math.cos(phi - math.pi)))

            if angle_error <= tolerance:
                continuous_upright += 1
                longest_upright = max(longest_upright, continuous_upright)

                # Count recovery after 0.2 seconds continuously upright.
                if continuous_upright >= int(0.2 * CONTROL_HZ):
                    recovered_steps = step - continuous_upright + 1
                    break
            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        if recovered_steps is None:
            print(
                f"Trial {trial + 1}: NO RECOVERY | "
                f"Longest upright interval: {longest_upright / CONTROL_HZ:.2f}s"
            )
            results.append(None)
        else:
            recovery_time = recovered_steps / CONTROL_HZ
            print(
                f"Trial {trial + 1}: RECOVERED | "
                f"Recovery time: {recovery_time:.2f}s"
            )
            results.append(recovery_time)

finally:
    env.close()

successful = [x for x in results if x is not None]

print("\nSUMMARY")
print("=" * 60)
print(f"Successful recoveries: {len(successful)}/{TRIALS}")
print(f"Recovery success rate: {100 * len(successful) / TRIALS:.1f}%")

if successful:
    print(f"Mean recovery time: {np.mean(successful):.2f}s")
    print(f"Slowest successful recovery: {max(successful):.2f}s")
