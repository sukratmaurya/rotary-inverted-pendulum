import math
import numpy as np
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/cpu_training_3/best_model.zip"

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0

MIN_INITIAL_ERROR_DEG = 30.0
UPRIGHT_TOLERANCE_DEG = 15.0
REQUIRED_HOLD_SECONDS = 1.0

# ============================================================
# ENVIRONMENT
# ============================================================

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

    # Enable the environment's configured randomization.
    domain_randomization=True,
)

model = SAC.load(MODEL_PATH, device="cpu")

MAX_STEPS = round(EPISODE_SECONDS * CONTROL_HZ)
HOLD_STEPS = round(REQUIRED_HOLD_SECONDS * CONTROL_HZ)
MIN_INITIAL_ERROR = math.radians(MIN_INITIAL_ERROR_DEG)
UPRIGHT_TOLERANCE = math.radians(UPRIGHT_TOLERANCE_DEG)


def upright_error(phi):
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def randomized_initial_angle(rng):
    while True:
        phi = rng.uniform(0.0, 2.0 * math.pi)
        if upright_error(phi) >= MIN_INITIAL_ERROR:
            return phi


results = []

print("\nDOMAIN-RANDOMIZED ROBUSTNESS TEST")
print("=" * 68)
print(f"Model: {MODEL_PATH}")
print(f"Control frequency: {CONTROL_HZ} Hz")
print(f"Trials: {TRIALS}")
print(f"Episode limit: {EPISODE_SECONDS:.1f} s")
print(f"Minimum initial error: {MIN_INITIAL_ERROR_DEG:.1f} degrees")
print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEG:.1f} degrees")
print(f"Required upright hold: {REQUIRED_HOLD_SECONDS:.1f} s")
print("Domain randomization: enabled")
print("=" * 68)

try:
    for trial in range(TRIALS):
        rng = np.random.default_rng(12000 + trial)

        obs, _ = env.reset(seed=5000 + trial)

        # Randomize the initial physical state.
        motor_pos = rng.uniform(-1.4, 1.4)
        pen_phi = randomized_initial_angle(rng)
        motor_vel = rng.uniform(-0.5, 0.5)
        pen_vel = rng.uniform(-1.0, 1.0)

        env.data.qpos[env._motor_qpos_addr] = motor_pos
        env.data.qpos[env._pen_qpos_addr] = pen_phi
        env.data.qvel[env._motor_qvel_addr] = motor_vel
        env.data.qvel[env._pen_qvel_addr] = pen_vel

        # Synchronize controller internals with the new motor state.
        env._motor_target = motor_pos
        env._lagged_target = motor_pos
        env._prev_cmd_pos = motor_pos
        env._motor_vel = motor_vel
        env._prev_action = 0.0
        env._prev_motor_vel = motor_vel

        # Reinitialize the physics and simulated measurement pipeline.
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        if env.firmware_obs_model:
            env.seed_measurement_ring()

        env._obs_history.clear()
        obs = env._obs()

        initial_error = math.degrees(
            upright_error(
                float(env.data.qpos[env._pen_qpos_addr])
            )
        )

        continuous_upright = 0
        longest_upright = 0
        success = False
        stable_entry_time = None

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            error = upright_error(float(info["phi"]))

            if error <= UPRIGHT_TOLERANCE:
                continuous_upright += 1
                longest_upright = max(
                    longest_upright,
                    continuous_upright,
                )

                if continuous_upright >= HOLD_STEPS:
                    success = True
                    first_stable_step = step - HOLD_STEPS + 1
                    stable_entry_time = (
                        first_stable_step + 1
                    ) / CONTROL_HZ
                    break
            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        longest_seconds = longest_upright / CONTROL_HZ

        if success:
            print(
                f"Trial {trial + 1:02d}: SUCCESS | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Stable-entry time: {stable_entry_time:.2f} s"
            )
            results.append({
                "success": True,
                "time": stable_entry_time,
                "longest": longest_seconds,
            })
        else:
            print(
                f"Trial {trial + 1:02d}: FAIL | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Longest upright interval: {longest_seconds:.2f} s"
            )
            results.append({
                "success": False,
                "time": None,
                "longest": longest_seconds,
            })

finally:
    env.close()

successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]
import math
import numpy as np
from stable_baselines3 import SAC
from pendulum_env import RotaryInvertedPendulumEnv

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/cpu_training_3/best_model.zip"

TRIALS = 30
CONTROL_HZ = 50
EPISODE_SECONDS = 8.0

MIN_INITIAL_ERROR_DEG = 30.0
UPRIGHT_TOLERANCE_DEG = 15.0
REQUIRED_HOLD_SECONDS = 1.0

# ============================================================
# ENVIRONMENT
# ============================================================

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

    # Enable the environment's configured randomization.
    domain_randomization=True,
)

model = SAC.load(MODEL_PATH, device="cpu")

MAX_STEPS = round(EPISODE_SECONDS * CONTROL_HZ)
HOLD_STEPS = round(REQUIRED_HOLD_SECONDS * CONTROL_HZ)
MIN_INITIAL_ERROR = math.radians(MIN_INITIAL_ERROR_DEG)
UPRIGHT_TOLERANCE = math.radians(UPRIGHT_TOLERANCE_DEG)


def upright_error(phi):
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def randomized_initial_angle(rng):
    while True:
        phi = rng.uniform(0.0, 2.0 * math.pi)
        if upright_error(phi) >= MIN_INITIAL_ERROR:
            return phi


results = []

print("\nDOMAIN-RANDOMIZED ROBUSTNESS TEST")
print("=" * 68)
print(f"Model: {MODEL_PATH}")
print(f"Control frequency: {CONTROL_HZ} Hz")
print(f"Trials: {TRIALS}")
print(f"Episode limit: {EPISODE_SECONDS:.1f} s")
print(f"Minimum initial error: {MIN_INITIAL_ERROR_DEG:.1f} degrees")
print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEG:.1f} degrees")
print(f"Required upright hold: {REQUIRED_HOLD_SECONDS:.1f} s")
print("Domain randomization: enabled")
print("=" * 68)

try:
    for trial in range(TRIALS):
        rng = np.random.default_rng(12000 + trial)

        obs, _ = env.reset(seed=5000 + trial)

        # Randomize the initial physical state.
        motor_pos = rng.uniform(-1.4, 1.4)
        pen_phi = randomized_initial_angle(rng)
        motor_vel = rng.uniform(-0.5, 0.5)
        pen_vel = rng.uniform(-1.0, 1.0)

        env.data.qpos[env._motor_qpos_addr] = motor_pos
        env.data.qpos[env._pen_qpos_addr] = pen_phi
        env.data.qvel[env._motor_qvel_addr] = motor_vel
        env.data.qvel[env._pen_qvel_addr] = pen_vel

        # Synchronize controller internals with the new motor state.
        env._motor_target = motor_pos
        env._lagged_target = motor_pos
        env._prev_cmd_pos = motor_pos
        env._motor_vel = motor_vel
        env._prev_action = 0.0
        env._prev_motor_vel = motor_vel

        # Reinitialize the physics and simulated measurement pipeline.
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        if env.firmware_obs_model:
            env.seed_measurement_ring()

        env._obs_history.clear()
        obs = env._obs()

        initial_error = math.degrees(
            upright_error(
                float(env.data.qpos[env._pen_qpos_addr])
            )
        )

        continuous_upright = 0
        longest_upright = 0
        success = False
        stable_entry_time = None

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            error = upright_error(float(info["phi"]))

            if error <= UPRIGHT_TOLERANCE:
                continuous_upright += 1
                longest_upright = max(
                    longest_upright,
                    continuous_upright,
                )

                if continuous_upright >= HOLD_STEPS:
                    success = True
                    first_stable_step = step - HOLD_STEPS + 1
                    stable_entry_time = (
                        first_stable_step + 1
                    ) / CONTROL_HZ
                    break
            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        longest_seconds = longest_upright / CONTROL_HZ

        if success:
            print(
                f"Trial {trial + 1:02d}: SUCCESS | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Stable-entry time: {stable_entry_time:.2f} s"
            )
            results.append({
                "success": True,
                "time": stable_entry_time,
                "longest": longest_seconds,
            })
        else:
            print(
                f"Trial {trial + 1:02d}: FAIL | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Longest upright interval: {longest_seconds:.2f} s"
            )
            results.append({
                "success": False,
                "time": None,
                "longest": longest_seconds,
            })

finally:
    env.close()

successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print("\nSUMMARY")
print("=" * 68)
print(f"Total trials: {len(results)}")
print(f"Successful trials: {len(successful)}")
print(f"Failed trials: {len(failed)}")
print(f"Success rate: {100 * len(successful) / len(results):.1f}%")

if successful:
    times = [r["time"] for r in successful]
    print(f"Mean stable-entry time: {np.mean(times):.2f} s")
    print(f"Slowest stable-entry time: {max(times):.2f} s")

if failed:
    intervals = [r["longest"] for r in failed]
    print(
        "Mean longest upright interval among failures: "
        f"{np.mean(intervals):.2f} s"
    )

print("=" * 68)
