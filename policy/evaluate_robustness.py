
import math
import numpy as np
import mujoco
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

# Randomized starting-state limits
MOTOR_POS_LIMIT_RAD = 1.4
MOTOR_VEL_LIMIT_RAD_S = 0.5
PENDULUM_VEL_LIMIT_RAD_S = 1.0

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
)

model = SAC.load(MODEL_PATH, device="cpu")

MAX_STEPS = round(EPISODE_SECONDS * CONTROL_HZ)
HOLD_STEPS = round(REQUIRED_HOLD_SECONDS * CONTROL_HZ)

MIN_INITIAL_ERROR = math.radians(MIN_INITIAL_ERROR_DEG)
UPRIGHT_TOLERANCE = math.radians(UPRIGHT_TOLERANCE_DEG)


# ============================================================
# ANGLE UTILITIES
# ============================================================

def upright_error(phi):
    """Absolute angular error from upright, where phi = pi."""
    return abs(
        math.atan2(
            math.sin(phi - math.pi),
            math.cos(phi - math.pi),
        )
    )


def randomized_initial_angle(rng):
    """
    Select a pendulum angle at least MIN_INITIAL_ERROR_DEG
    away from upright.

    Upright is phi = pi. The returned angle is in [0, 2*pi).
    """
    while True:
        phi = rng.uniform(0.0, 2.0 * math.pi)

        if upright_error(phi) >= MIN_INITIAL_ERROR:
            return phi


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\nRANDOMIZED ROBUSTNESS TEST")
print("=" * 68)
print(f"Model: {MODEL_PATH}")
print(f"Control frequency: {CONTROL_HZ} Hz")
print(f"Trials: {TRIALS}")
print(f"Episode limit: {EPISODE_SECONDS:.1f} seconds")
print(f"Minimum initial error: {MIN_INITIAL_ERROR_DEG:.1f} degrees")
print(f"Upright tolerance: +/-{UPRIGHT_TOLERANCE_DEG:.1f} degrees")
print(f"Required continuous upright time: {REQUIRED_HOLD_SECONDS:.1f} seconds")
print("Randomized states: pendulum angle/velocity and motor position/velocity")
print("=" * 68)

try:
    for trial in range(TRIALS):

        # Use independent, repeatable random seeds.
        rng = np.random.default_rng(9000 + trial)

        # Reset the environment to initialize internal buffers.
        obs, _ = env.reset(seed=4000 + trial)

        # ----------------------------------------------------
        # RANDOMIZE THE INITIAL STATE
        # ----------------------------------------------------

        motor_pos = rng.uniform(
            -MOTOR_POS_LIMIT_RAD,
            MOTOR_POS_LIMIT_RAD,
        )

        pen_phi = randomized_initial_angle(rng)

        motor_vel = rng.uniform(
            -MOTOR_VEL_LIMIT_RAD_S,
            MOTOR_VEL_LIMIT_RAD_S,
        )

        pen_vel = rng.uniform(
            -PENDULUM_VEL_LIMIT_RAD_S,
            PENDULUM_VEL_LIMIT_RAD_S,
        )

        env.data.qpos[env._motor_qpos_addr] = motor_pos
        env.data.qpos[env._pen_qpos_addr] = pen_phi

        env.data.qvel[env._motor_qvel_addr] = motor_vel
        env.data.qvel[env._pen_qvel_addr] = pen_vel

        # Synchronize internal controller state with motor position.
        env._motor_target = motor_pos
        env._lagged_target = motor_pos
        env._prev_cmd_pos = motor_pos
        env._motor_vel = motor_vel
        env._prev_action = 0.0
        env._prev_motor_vel = motor_vel

        # Update MuJoCo physics after directly changing joint state.
        mujoco.mj_forward(env.model, env.data)

        # Reinitialize simulated encoder measurements.
        if env.firmware_obs_model:
            env.seed_measurement_ring()

        # Remove observations from the original reset state.
        env._obs_history.clear()
        obs = env._obs()

        initial_error = math.degrees(
            upright_error(
                float(env.data.qpos[env._pen_qpos_addr])
            )
        )

        # ----------------------------------------------------
        # RUN THE POLICY
        # ----------------------------------------------------

        continuous_upright = 0
        longest_upright = 0
        success = False
        stable_entry_time = None

        for step in range(MAX_STEPS):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            phi = float(info["phi"])
            error = upright_error(phi)

            if error <= UPRIGHT_TOLERANCE:

                continuous_upright += 1

                longest_upright = max(
                    longest_upright,
                    continuous_upright,
                )

                if continuous_upright >= HOLD_STEPS:
                    success = True

                    # Time from the start of this trial to the
                    # beginning of the sustained upright interval.
                    first_stable_step = step - HOLD_STEPS + 1
                    stable_entry_time = (
                        first_stable_step + 1
                    ) / CONTROL_HZ

                    break

            else:
                continuous_upright = 0

            if terminated or truncated:
                break

        # ----------------------------------------------------
        # RECORD RESULTS
        # ----------------------------------------------------

        longest_upright_seconds = longest_upright / CONTROL_HZ

        if success:

            print(
                f"Trial {trial + 1:02d}: SUCCESS | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Motor position: {math.degrees(motor_pos):6.1f} deg | "
                f"Stable-entry time: {stable_entry_time:.2f}s"
            )

            results.append({
                "success": True,
                "initial_error": initial_error,
                "stable_entry_time": stable_entry_time,
                "longest_upright": longest_upright_seconds,
            })

        else:

            print(
                f"Trial {trial + 1:02d}: FAIL | "
                f"Initial error: {initial_error:6.1f} deg | "
                f"Motor position: {math.degrees(motor_pos):6.1f} deg | "
                f"Longest upright interval: "
                f"{longest_upright_seconds:.2f}s"
            )

            results.append({
                "success": False,
                "initial_error": initial_error,
                "stable_entry_time": None,
                "longest_upright": longest_upright_seconds,
            })

finally:
    env.close()


# ============================================================
# SUMMARY
# ============================================================

successful = [
    result for result in results
    if result["success"]
]

failed = [
    result for result in results
    if not result["success"]
]

print("\nSUMMARY")
print("=" * 68)
print(f"Total trials: {len(results)}")
print(f"Successful trials: {len(successful)}")
print(f"Failed trials: {len(failed)}")
print(
    f"Success rate: "
    f"{100.0 * len(successful) / len(results):.1f}%"
)

if successful:
    times = [
        result["stable_entry_time"]
        for result in successful
    ]

    print(
        f"Mean stable-entry time: "
        f"{np.mean(times):.2f}s"
    )

    print(
        f"Slowest stable-entry time: "
        f"{max(times):.2f}s"
    )

if failed:
    longest_intervals = [
        result["longest_upright"]
        for result in failed
    ]

    print(
        f"Mean longest upright interval among failures: "
        f"{np.mean(longest_intervals):.2f}s"
    )

print("=" * 68)
print("Evaluation complete.")

