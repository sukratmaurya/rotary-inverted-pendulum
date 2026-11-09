import math
import numpy as np
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

MODEL_PATH = "runs/cpu_training_3/best_model.zip"
TRIALS = 10
MAX_STEPS = 400
UPRIGHT_TOLERANCE = math.radians(15)
BALANCE_TOLERANCE = math.radians(15)

env = RotaryInvertedPendulumEnv(
    render_mode=None,
    control_freq_hz=50.0,
    action_mode="velocity",
    max_accel_rad_s2=150.0,
    max_velocity_rad_s=3.5,
    max_action_delta_rad=0.1,
    reward_alive_offset=15.0,
    reward_upright_alive_weight=5.0,
    reward_stillness_bonus_weight=5.0,
    reward_motor_pos_weight=0.5,
    reward_motor_vel_weight=0.005,
    upright_reset_frac=0.0,
    obs_history_len=4,
    obs_include_velocities=True,
    firmware_obs_model=True,
    action_smooth_window=4,
)

model = SAC.load(MODEL_PATH, device="cpu")

successes = 0
swingup_times = []
balance_durations = []
mean_actions = []

print("\nHANGING-START SWING-UP EVALUATION")
print("=" * 58)

try:
    for trial in range(TRIALS):
        obs, _ = env.reset(seed=1000 + trial)

        first_upright_step = None
        continuous_upright = 0
        longest_upright = 0
        action_values = []
        success = False

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            action_values.append(float(np.mean(np.abs(action))))

            # The environment defines theta = 0 as upright.
            # Recover theta from the latest observation frame.
            latest = obs[-6:]
            theta = math.atan2(float(latest[1]), float(latest[2]))
            angle_error = abs(theta)

            if angle_error <= UPRIGHT_TOLERANCE:
                if first_upright_step is None:
                    first_upright_step = step
                    success = True
                continuous_upright += 1
                longest_upright = max(longest_upright, continuous_upright)
            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        if success:
            successes += 1
            swingup_times.append(first_upright_step / 50.0)

        balance_durations.append(longest_upright / 50.0)
        mean_actions.append(float(np.mean(action_values)))

        print(
            f"Trial {trial + 1:02d}: "
            f"{'UPRIGHT REACHED' if success else 'NO SWING-UP'} | "
            f"Time: {first_upright_step / 50.0:.2f}s"
            if success else
            f"Trial {trial + 1:02d}: NO SWING-UP | "
            f"Longest upright interval: {longest_upright / 50.0:.2f}s"
        )

    print("\nSUMMARY")
    print("=" * 58)
    print(f"Trials: {TRIALS}")
    print(f"Swing-up success rate: {100 * successes / TRIALS:.1f}%")

    if swingup_times:
        print(f"Mean time to first upright: {np.mean(swingup_times):.2f}s")

    print(
        f"Mean longest upright interval: "
        f"{np.mean(balance_durations):.2f}s"
    )
    print(f"Mean absolute action: {np.mean(mean_actions):.3f}")

finally:
    env.close()

