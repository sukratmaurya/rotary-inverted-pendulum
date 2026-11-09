import time
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

MODEL_PATH = "runs/cpu_training_4_lag_robust/best_model.zip"
SIMULATION_SECONDS = 30
CONTROL_FREQ_HZ = 50

env = RotaryInvertedPendulumEnv(
    render_mode="human",
    control_freq_hz=CONTROL_FREQ_HZ,
    action_mode="velocity",
    obs_history_len=4,
    obs_include_velocities=True,
    firmware_obs_model=True,
    action_smooth_window=4,
    max_accel_rad_s2=150.0,
    max_velocity_rad_s=3.5,
    max_action_delta_rad=0.1,
    reward_alive_offset=15.0,
    reward_upright_alive_weight=5.0,
    reward_stillness_bonus_weight=5.0,
    reward_motor_pos_weight=0.5,
    reward_motor_vel_weight=0.005,
    upright_reset_frac=0.0,
)

model = SAC.load(MODEL_PATH, device="cpu")

obs, info = env.reset(seed=123)

print("Starting hanging-position swing-up visualization.")
print("Close the MuJoCo window or press Ctrl+C to stop.")

dt = 1.0 / CONTROL_FREQ_HZ
next_tick = time.monotonic()
steps = int(SIMULATION_SECONDS * CONTROL_FREQ_HZ)

try:
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {step}; resetting.")
            obs, info = env.reset()

        next_tick += dt
        sleep_for = next_tick - time.monotonic()

        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

    print("Visualization finished.")

finally:
    env.close()
