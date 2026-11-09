"""Gymnasium environment that uses the *real* rotary inverted pendulum
as its dynamics model.

This is the Phase 4 "sim-to-real fine-tuning" environment: SAC's
`learn()` collects (s, a, r, s') transitions by physically driving the
device, instead of stepping MuJoCo. The policy thus adapts to the
actual hardware dynamics, closing the residual sim-to-real gap that
pure sim training can't.

Observation, action, and reward exactly mirror `pendulum_env.py` so
that a sim-trained checkpoint loads without modification.

Reset behaviour:
    - Disengage the motor.
    - Wait `reset_settle_s` seconds for the pendulum to stop swinging
      under bearing friction.
    - Re-engage the motor at its current position. The motor stays where
      the previous episode ended (no homing) — this matches the typical
      Furuta sim-to-real fine-tuning recipe and avoids unnecessary motor
      wear.

Termination/Truncation:
    - Truncated when `episode_length_s` elapses.
    - Terminated (with `hard_stop_penalty`) if the motor
      position |motor_pos| exceeds MOTOR_LIMIT_RAD (the firmware also
      enforces this in hardware, so the policy should never command
      past it; the check here is the policy's "feedback" if it does).

Safety:
    - SIGINT/SIGTERM handlers disengage the motor and close the serial
      port cleanly.
    - Failures during `step()` (lost serial, etc.) raise immediately;
      the SAC training loop will surface them.
"""

from __future__ import annotations

import math
import signal
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lowlevel_client import LowLevelClient
from reward import RewardWeights, compute_reward
from control_safety import REQUIRED_STATE_HEALTH, UnsafeState, require_ack, validate_state
from pendulum_env import (
    MAX_ACTION_DELTA_RAD,
    MOTOR_LIMIT_RAD,
    MOTOR_SAFE_LIMIT_RAD,
    PendulumParams,
    _wrap_pi,
)


HERE = Path(__file__).resolve().parent

# Rest-detection internals (used by _wait_for_pendulum_rest):
# - 0.5 rad/s is above the firmware vel-regression noise floor. AS5600
#   is 12-bit (2π/4096 ≈ 1.53 mrad/count), the regression window is 8 ms
#   (5 samples × 2 ms), so ±1 count of quantisation/I²C jitter alone
#   produces velocity spikes up to ~0.4 rad/s on a stationary pendulum.
#   Earlier threshold of 0.1 was below this noise floor — a "still"
#   pendulum could never accumulate enough consecutive sub-threshold
#   samples to register as rested.
# - 1.0 s of sustained sub-threshold velocity rejects long-tail noise
#   transients (a single I²C glitch or a tiny breath of air won't
#   trigger a false-positive tare).
REST_THRESHOLD_RAD_S = 0.5
REST_DURATION_S = 1.0


class RealRotaryInvertedPendulumEnv(gym.Env):
    """Drives the actual hardware rig as the RL environment.

    Operates over the LowLevelServer binary protocol. Sign conventions
    match `run_policy.py`: server flips motor and pendulum positions on
    output in the firmware frame — the training frame, used everywhere with
    no flips. The action is interpreted per
    `action_mode`: "accel" (angular acceleration) or "position_delta"
    (per-tick motor-target increment) — must match the mode the sim policy
    was trained with.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        port: str = "/dev/cu.usbserial-110",
        baud: int = 2_000_000,
        control_freq_hz: float = 50.0,  # canonical for this rig — see website/src/content/docs/reference/control-rate.md
        action_mode: str = "velocity",  # must match sim training
        max_accel_rad_s2: float = 100.0,
        max_velocity_rad_s: float = 3.5,  # velocity-mode action scale; matches sim MAX_VELOCITY_RAD_S
        max_action_delta_rad: float = MAX_ACTION_DELTA_RAD,  # position-mode per-step target delta
        episode_length_s: float = 6.0,
        # Max seconds to wait for the pendulum to come to rest between
        # episodes before giving up. While waiting, polls pen_vel; once
        # |pen_vel| stays below REST_THRESHOLD_RAD_S for REST_DURATION_S
        # consecutive seconds, we tare and start the next episode. On
        # timeout (operator didn't stop the pendulum, bearing issue, etc.)
        # we SKIP the tare and proceed using the previous tare's zero —
        # safer than capturing a moving pendulum as the new zero. 15 s
        # gives the operator generous time to settle the pendulum.
        reset_settle_s: float = 15.0,
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # Reward weights (default = Quanser quadratic-cost form, same as
        # the sim env so a sim-trained checkpoint produces equivalent
        # advantage estimates on the first real rollout).
        reward_motor_pos_weight: float = 0.5,
        reward_motor_vel_weight: float = 0.005,
        reward_action_weight: float = 0.20,
        reward_pen_vel_weight: float = 0.001,
        # Smoothness penalty on (a_t - a_{t-1})². Default None → 0
        # (canonical disabled). Mirror of pendulum_env's flag.
        reward_action_rate_weight: float | None = None,
        # Stillness bonus near upright — mirrors pendulum_env.py
        # exactly so sim-trained policies have continuity in the
        # gradient signal when fine-tuned on rig. Default None → 0
        # (disabled). Required if the sim training used this bonus —
        # otherwise fine-tune will gradient-descend out of the bonus
        # basin back toward whatever the classic reward prefers.
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,
        # Alive terms — must match sim training (see reward.py, audit F1).
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        obs_history_len: int = 4,  # frames per observation — must match sim training
        obs_include_velocities: bool = True,  # False: positions-only frames — must match sim training
        # Actuator receives the moving average of the last N actions —
        # mirror of pendulum_env's action_smooth_window and of
        # ACTION_SMOOTH_WINDOW in RLControl.ino. Must match sim training.
        action_smooth_window: int = 4,
        max_state_age_s: float = 0.080,
        required_health_mask: int = REQUIRED_STATE_HEALTH,
        params_path: str | Path | None = None,
    ):
        super().__init__()
        if action_mode not in ("accel", "velocity"):
            raise ValueError(
                f"action_mode must be 'accel' or 'velocity' for protocol v1, "
                f"got {action_mode!r}"
            )
        self.port = port
        self.baud = baud
        self.control_freq_hz = control_freq_hz
        self.action_mode = action_mode
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_action_delta_rad = max_action_delta_rad
        if int(action_smooth_window) < 1:
            raise ValueError(
                f"action_smooth_window must be >= 1, got {action_smooth_window}"
            )
        self.action_smooth_window = int(action_smooth_window)
        if max_state_age_s <= 0:
            raise ValueError("max_state_age_s must be positive")
        self.max_state_age_s = float(max_state_age_s)
        self.required_health_mask = int(required_health_mask)
        self._last_state_age_ms = math.inf
        self._last_state_rtt_ms = math.inf
        self._last_command_rtt_ms = math.inf
        self._action_smooth_buf: deque = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )
        self.episode_length_s = episode_length_s
        self.reset_settle_s = reset_settle_s  # rest-detection timeout
        if int(obs_history_len) < 1:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
        self.obs_history_len = int(obs_history_len)
        self.obs_include_velocities = bool(obs_include_velocities)
        if not self.obs_include_velocities and self.obs_history_len < 2:
            raise ValueError(
                "obs_include_velocities=False needs obs_history_len >= 2"
            )
        self._obs_history: deque = deque(maxlen=self.obs_history_len)
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        # All reward terms live in a single RewardWeights dataclass shared
        # with pendulum_env.py via reward.py — single source of truth so
        # the sim training and rig fine-tune optimise the SAME objective.
        # Add new reward terms to reward.py, not here.
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_action_rate=(
                float(reward_action_rate_weight)
                if reward_action_rate_weight is not None else 0.0
            ),
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
            k_alive_offset=(
                float(reward_alive_offset)
                if reward_alive_offset is not None else 0.0
            ),
            k_upright_alive=(
                float(reward_upright_alive_weight)
                if reward_upright_alive_weight is not None else 0.0
            ),
        )
        # Motor-jerk term needs prev_motor_vel tracking. real_env doesn't
        # currently expose a CLI flag for it (sim-only feature), but the
        # compute_reward signature takes it — initialise to 0 and never update,
        # which makes k_motor_jerk=0 the only sensible weight for real env.
        self._prev_motor_vel = 0.0

        self._dt = 1.0 / control_freq_hz
        self._max_steps = int(episode_length_s * control_freq_hz)

        # Lazy-init the serial connection on the first reset() so that
        # SB3's vectorised env wrappers don't open the port at construction
        # time (which would race during process spawning).
        self._client: LowLevelClient | None = None
        self._motor_engaged = False

        # Per-episode state
        self._step_count = 0
        self._last_accel_cmd = 0.0  # most recent accel command sent (rad/s²)
        self._motor_target = 0.0    # position-mode: integrated target sent to moveTo (rad)
        self._prev_action = 0.0     # last applied action (∈ [-1, 1]); fed into obs
        self._motor_pos_prev = 0.0
        self._phi_prev = 0.0
        self._motor_vel = 0.0
        self._pen_vel = 0.0
        self._next_tick = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Observation bounds match the sim env exactly. Last dim is
        # prev_action ∈ [-1, 1], present from the accel-mode rework.
        frame_high = (
            np.array([MOTOR_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0], dtype=np.float32)
            if self.obs_include_velocities
            else np.array([MOTOR_LIMIT_RAD, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        obs_high = np.tile(frame_high, self.obs_history_len)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Cooperative shutdown flag. NOT auto-wired to signal handlers
        # anymore — `finetune_async.main()` installs handlers on the main
        # thread and owns shutdown. Direct callers who want signal-driven
        # shutdown can wire `signal.signal(SIGINT, self._on_signal)`
        # themselves.
        self._sigterm_received = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_signal(self, *_):
        self._sigterm_received = True
        try:
            if self._client is not None:
                self._client.disarm()
        except Exception:
            pass

    def _ensure_open(self) -> LowLevelClient:
        if self._client is None:
            self._client = LowLevelClient(self.port, baud=self.baud)
            self._client.__enter__()
            if not self._client.wait_until_ready():
                raise RuntimeError(
                    "LowLevelServer did not respond. Is the sketch flashed and "
                    "the port correct?"
                )
        return self._client

    def _read_raw_state(self):
        try:
            s = self._client.get_state()  # type: ignore[union-attr]
            age = validate_state(
                s,
                max_age_s=self.max_state_age_s,
                required_health_mask=self.required_health_mask,
            )
        except Exception:
            self.disengage_safely()
            raise
        self._last_state_age_ms = age.age_ns / 1e6
        self._last_state_rtt_ms = age.rtt_ns / 1e6
        return (s.time_us, s.motor_pos_rad, s.pendulum_pos_rad,
                s.motor_vel_rad_s, s.pendulum_vel_rad_s)

    def _wait_for_pendulum_rest(self, client: LowLevelClient) -> bool:
        """Block until the pendulum has been at rest for REST_DURATION_S.

        At rest = |pen_vel| < REST_THRESHOLD_RAD_S. Polls at 20 Hz; resets
        the streak any time a single sample exceeds the threshold so noise
        transients don't false-positive.

        Returns True if rest was achieved within `reset_settle_s`. Returns
        False on timeout — the caller (real_env.reset) skips the tare in
        that case rather than capturing a moving pendulum's reading as the
        new zero.
        """
        poll_dt_s = 0.05  # 20 Hz
        required_consecutive = int(REST_DURATION_S / poll_dt_s)
        t_start = time.monotonic()
        consecutive_rest = 0
        last_pen_vel = 0.0
        while time.monotonic() - t_start < self.reset_settle_s:
            _, _, _, _, pen_vel = self._read_raw_state()
            last_pen_vel = pen_vel
            if abs(pen_vel) < REST_THRESHOLD_RAD_S:
                consecutive_rest += 1
                if consecutive_rest >= required_consecutive:
                    return True
            else:
                consecutive_rest = 0
            time.sleep(poll_dt_s)
        print(
            f"  [warn] pendulum did not settle within "
            f"{self.reset_settle_s:.1f}s (last |pen_vel|={abs(last_pen_vel):.3f} "
            f"rad/s, threshold={REST_THRESHOLD_RAD_S:.3f}); "
            f"skipping tare for this episode"
        )
        return False

    def _build_obs(self, motor_pos: float, phi: float, *,
                   push: bool = True) -> np.ndarray:
        """Build the stacked observation from the current state.

        `push=True` (normal reset/step path) appends the new frame to the
        history first. Error paths pass `push=False` so a failed serial read
        doesn't advance the frame clock with a stale duplicate.
        """
        theta = _wrap_pi(phi - math.pi)
        if self.obs_include_velocities:
            frame = np.array(
                [
                    motor_pos,
                    math.sin(theta),
                    math.cos(theta),
                    self._motor_vel,
                    self._pen_vel,
                    self._prev_action,
                ],
                dtype=np.float32,
            )
        else:
            # Positions-only frame: the firmware velocities are still read
            # and used for the reward and rest detection — they just aren't
            # shown to the policy, which derives its own from the history.
            frame = np.array(
                [motor_pos, math.sin(theta), math.cos(theta),
                 self._prev_action],
                dtype=np.float32,
            )
        if not self._obs_history:
            self._obs_history.extend([frame] * self.obs_history_len)
        elif push:
            self._obs_history.append(frame)
        if self.obs_history_len == 1:
            return frame
        return np.concatenate(self._obs_history)

    def _reward(self, action: float, motor_pos: float, phi: float) -> float:
        # Delegates to reward.compute_reward — same code path the sim
        # env uses, so SAC's gradient signal is identical in sim and on
        # the rig. Add new terms via reward.py.
        return compute_reward(
            theta=_wrap_pi(phi - math.pi),
            pen_vel=self._pen_vel,
            motor_pos=motor_pos,
            motor_vel=self._motor_vel,
            action=action,
            prev_action=self._prev_action,
            prev_motor_vel=self._prev_motor_vel,
            weights=self._reward_weights,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        client = self._ensure_open()

        # Disengage the motor and wait for the pendulum to come to rest.
        # Adaptive — polls pen_vel and only proceeds once the pendulum has
        # been at rest for `rest_duration_s` continuous seconds. Fixed-sleep
        # was unreliable: hard-spun pendulums can still oscillate after 3 s.
        # Taring a moving pendulum captures a non-rest sample as the new
        # encoder zero, polluting the bias signal we rely on.
        client.disarm()
        self._motor_engaged = False
        print("  [reset] re-centre the arm and let the pendulum settle — the "
              "next episode starts once it is at rest")
        rested = self._wait_for_pendulum_rest(client)

        # Re-tare the pendulum encoder — but ONLY if the pendulum is
        # actually at rest. If the rest check timed out (operator wasn't
        # paying attention, pendulum still oscillating from previous
        # episode), skipping the tare is safer than capturing a moving
        # sample as the new encoder zero. The previous tare's zero
        # stays in effect for this episode; the next reset gets another
        # chance.
        #
        # When it does fire, the tare records the rig's current
        # static-friction-bounded rest position (±~1.9° from true
        # vertical-down) as the new encoder zero. Across many successful
        # resets the tare value varies within that band — effectively
        # per-episode bias sampling from the rig's actual physical
        # distribution (the hardware analogue of sim's bias DR).
        if rested:
            ok = client.tare_pendulum()
            if not ok:
                raise RuntimeError(
                    "tare_pendulum did not ack — re-flash LowLevelServer.ino "
                    "to pick up the CMD_TARE_PENDULUM=0x06 handler."
                )
            # Verify the tare captured HANGING, not a perfectly-balanced
            # upright. A well-balancing policy can leave the pendulum at
            # true upright with ~zero velocity at disengage; it then sits
            # still long enough to pass the rest check, the tare records
            # upright as phi=0, and the whole episode runs in an inverted
            # frame (observed ep36, vel_v9_async). Discrimination is free:
            # a hanging pendulum stays near phi=0, while an unactuated
            # upright one MUST fall (unstable, ~0.1 s time constant) —
            # so watch for 2 s and re-settle + re-tare if it moves.
            for attempt in range(3):
                fell = False
                t_end = time.monotonic() + 2.0
                while time.monotonic() < t_end:
                    _, _, phi_chk, _, _ = self._read_raw_state()
                    if abs(phi_chk) > 0.8:
                        fell = True
                        break
                    time.sleep(0.05)
                if not fell:
                    break
                print("  [reset] tare captured a balanced-upright pendulum "
                      "(it fell during verification) — re-settling + re-taring")
                self._wait_for_pendulum_rest(client)
                if not client.tare_pendulum():
                    raise RuntimeError("tare_pendulum did not ack on re-tare")

        # Reset deliberately leaves the driver disarmed. Arming is a separate
        # explicit operation and must only be called after operator approval.
        _, motor_pos, phi, _, _ = self._read_raw_state()
        self._motor_target = float(motor_pos)

        self._motor_pos_prev = motor_pos
        self._phi_prev = phi
        self._motor_vel = 0.0
        self._pen_vel = 0.0
        self._v_cmd = 0.0  # velocity-mode commanded-velocity integrator
        self._action_smooth_buf = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )
        self._prev_action = 0.0
        self._step_count = 0
        self._next_tick = time.monotonic()

        self._obs_history.clear()  # reseeded by _build_obs with the initial frame
        return self._build_obs(motor_pos, phi), {}

    @property
    def is_armed(self) -> bool:
        return self._motor_engaged

    def arm(self) -> None:
        """Arm only on an explicit operator-authorized call; reset never arms."""
        client = self._ensure_open()
        if self._motor_engaged:
            return
        self._read_raw_state()
        try:
            require_ack(client.arm(), operation="ARM")
            self._motor_engaged = True
            ack = require_ack(client.set_acceleration(0.0), operation="initial SET_ACCEL")
            self._last_command_rtt_ms = int(ack.rtt_ns) / 1e6
        except Exception:
            self.disengage_safely()
            raise

    def apply_action(self, action) -> float:
        """Send `action` to the motor, interpreted per `action_mode`.

        Returns the clipped action actually applied (fed into the reward /
        replay-buffer transition).

        Accel mode: action → angular accel ∈ [-max, +max] rad/s² via
        moveByAcceleration; zeroed if at the safety rail pushing outward.
        Position-delta mode: action → per-tick motor-target delta of
        action × max_action_delta_rad, integrated, clamped to the rail, and
        sent via moveTo. The firmware clamps the target independently too.
        """
        if self._sigterm_received:
            raise KeyboardInterrupt("SIGTERM received during step")
        client = self._client
        if client is None or not self._motor_engaged:
            raise RuntimeError("env.apply_action() called before reset() or motor disengaged")

        action_values = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_values.size != 1 or not np.isfinite(action_values[0]):
            self.disengage_safely()
            raise UnsafeState("policy action must be one finite scalar")
        a = float(np.clip(action_values[0], -1.0, 1.0))
        # Actuator-side smoothing: the motor chases the boxcar average of
        # the last N actions; the RAW action still goes to the reward and
        # replay buffer (return value below), mirroring pendulum_env.
        if self.action_smooth_window > 1:
            self._action_smooth_buf.append(a)
            a_cmd = sum(self._action_smooth_buf) / self.action_smooth_window
        else:
            a_cmd = a
        if self.action_mode in ("accel", "velocity"):
            if self.action_mode == "velocity":
                # Saturating P-law tracking the velocity setpoint. Feedback
                # is the host's OWN commanded-velocity integrator, not the
                # firmware-measured velocity: the measurement is quantised
                # to ±(LSB/window) ≈ ±0.5 rad/s, and multiplying that error
                # by the control-frequency gain injected a persistent
                # ±17 rad/s² accel dither (the wobble every raw deploy
                # showed). The stepper executes commands essentially
                # perfectly, so the integrator is accurate by construction;
                # a slow complementary correction from the measurement
                # (after send, below) heals drift from rail clamps or
                # skipped steps while attenuating quantisation noise ~10×.
                v_des = a_cmd * self.max_velocity_rad_s
                accel_cmd = float(np.clip(
                    (v_des - self._v_cmd) * self.control_freq_hz,
                    -self.max_accel_rad_s2, self.max_accel_rad_s2,
                ))
            else:
                accel_cmd = a_cmd * self.max_accel_rad_s2
            if self._motor_pos_prev >= MOTOR_SAFE_LIMIT_RAD and accel_cmd > 0.0:
                accel_cmd = 0.0
            elif self._motor_pos_prev <= -MOTOR_SAFE_LIMIT_RAD and accel_cmd < 0.0:
                accel_cmd = 0.0
            self._last_accel_cmd = accel_cmd
            try:
                ack = require_ack(client.set_acceleration(accel_cmd), operation="SET_ACCEL")
                self._last_command_rtt_ms = int(ack.rtt_ns) / 1e6
            except Exception:
                self.disengage_safely()
                raise
            if self.action_mode == "velocity":
                self._v_cmd = float(np.clip(
                    self._v_cmd + accel_cmd / self.control_freq_hz,
                    -self.max_velocity_rad_s, self.max_velocity_rad_s,
                ))
                self._v_cmd += 0.1 * (self._motor_vel - self._v_cmd)
        else:  # guarded in __init__; retained as a defensive invariant
            raise RuntimeError("protocol v1 supports acceleration commands only")
        self._prev_action = a
        return a

    def observe_and_step(self, applied_action: float):
        """Read state from the rig, update filtered velocities, advance step
        counter, compute reward + termination + truncation, and return the
        gym-style 5-tuple `(obs, reward, terminated, truncated, info)`.

        Call this AFTER the control-loop sleep, so the motor has had time to
        react to whatever `apply_action` last sent.
        """
        try:
            t_us, motor_pos, phi, motor_vel, pen_vel = self._read_raw_state()
        except Exception:
            self.disengage_safely()
            return (self._build_obs(self._motor_pos_prev, self._phi_prev,
                                    push=False), 0.0, True, False, {})

        self._motor_vel = motor_vel
        self._pen_vel = pen_vel

        self._step_count += 1
        reward = self._reward(applied_action, motor_pos, phi)

        terminated = False
        if self.terminate_on_hard_stop and abs(motor_pos) >= MOTOR_LIMIT_RAD:
            terminated = True
            reward -= self.hard_stop_penalty

        truncated = self._step_count >= self._max_steps
        info = {
            "motor_pos": motor_pos,
            "phi": phi,
            "accel_cmd_rad_s2": getattr(self, "_last_accel_cmd", 0.0),
            "motor_target_rad": self._motor_target,
            "time_us": int(t_us),
            "state_age_ms": self._last_state_age_ms,
            "state_rtt_ms": self._last_state_rtt_ms,
            "command_rtt_ms": self._last_command_rtt_ms,
        }

        self._motor_pos_prev = motor_pos
        self._phi_prev = phi
        return self._build_obs(motor_pos, phi), reward, terminated, truncated, info

    def step(self, action):
        """Synchronous step: apply action, sleep to maintain the configured
        control rate, then observe. Used by single-threaded callers / tests
        and by the legacy synchronous `finetune_real.py` flow.

        The async control loop in `async_control.py` does NOT call this; it
        composes `apply_action` + externally-paced sleep + `observe_and_step`
        directly so the control rate is held even when the learner is busy.
        """
        try:
            a = self.apply_action(action)
        except Exception:
            self.disengage_safely()
            return (self._build_obs(self._motor_pos_prev, self._phi_prev,
                                    push=False), 0.0, True, False, {})

        # Pace to the requested control rate. Note: this is the bug-prone
        # part — if the SAC training loop runs gradient updates between
        # step() calls and they exceed `_dt`, we silently drop the rate.
        # The async architecture in `async_control.py` paces externally so
        # this `time.sleep` is bypassed.
        self._next_tick += self._dt
        sleep_for = self._next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_tick = time.monotonic()

        return self.observe_and_step(a)

    def disengage_safely(self) -> None:
        """Disengage the motor from any thread.

        Safe to call from a signal handler or from the orchestrator's main
        thread while a control thread is mid-`get_state` — the lock inside
        `LowLevelClient` serialises serial transactions cleanly.
        """
        try:
            if self._client is not None:
                self._client.disarm()
                self._motor_engaged = False
        except Exception:
            pass

    def close(self):
        if self._client is not None:
            try:
                self._client.disarm()
            except Exception:
                pass
            self._client.__exit__(None, None, None)
            self._client = None
            self._motor_engaged = False
