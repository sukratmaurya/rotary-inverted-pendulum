"""Offline safety checks; no serial device or motor access."""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

from control_safety import (
    REQUIRED_STATE_HEALTH,
    UnsafeState,
    confirm_operator_arm,
    validate_state,
)
from lowlevel_client import Ack, State
from real_env import RealRotaryInvertedPendulumEnv


def make_state(*, age_ms: float = 1.0, health: int = REQUIRED_STATE_HEALTH,
               faults: int = 0, value: float = 0.0) -> State:
    return State(
        time_us=1,
        motor_pos_rad=value,
        pendulum_pos_rad=0.1,
        motor_vel_rad_s=0.0,
        pendulum_vel_rad_s=0.0,
        health_flags=health,
        fault_flags=faults,
        rx_monotonic_ns=time.monotonic_ns() - int(age_ms * 1e6),
        rtt_ns=100_000,
    )


class FakeClient:
    instances = []
    state_factory = staticmethod(make_state)

    def __init__(self, *args, **kwargs):
        self.arm_calls = 0
        self.disarm_calls = 0
        self.acceleration_calls = []
        self.state_reads = 0
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def wait_until_ready(self):
        return True

    def get_state(self):
        self.state_reads += 1
        return type(self).state_factory()

    def arm(self):
        self.arm_calls += 1
        return Ack(0x04, 0, 0, self.arm_calls, 100_000)

    def disarm(self):
        self.disarm_calls += 1
        return Ack(0x05, 0, 0, self.disarm_calls, 100_000)

    def set_acceleration(self, value):
        self.acceleration_calls.append(value)
        return Ack(0x03, 0, 0, len(self.acceleration_calls), 100_000)

    def tare_pendulum(self):
        return True


class ControlSafetyTests(unittest.TestCase):
    def test_valid_state_passes_and_reports_age(self):
        state = make_state(age_ms=2.5)
        age = validate_state(state, now_ns=state.rx_monotonic_ns + 2_500_000)
        self.assertEqual(age.age_ns, 2_500_000)
        self.assertEqual(age.rtt_ns, 100_000)

    def test_stale_health_fault_and_nonfinite_state_rejected(self):
        cases = (
            make_state(age_ms=81.0),
            make_state(health=REQUIRED_STATE_HEALTH & ~0x2),
            make_state(faults=0x1),
            make_state(value=float("nan")),
        )
        for state in cases:
            with self.subTest(state=state), self.assertRaises(UnsafeState):
                validate_state(state)

    def test_no_arm_request_never_prompts(self):
        def should_not_prompt(_):
            raise AssertionError("must not prompt when arming was not requested")

        self.assertFalse(confirm_operator_arm(
            requested=False, prompt_fn=should_not_prompt, is_interactive=False
        ))

    def test_explicit_arm_requires_typed_interactive_confirmation(self):
        self.assertTrue(confirm_operator_arm(
            requested=True,
            prompt_fn=lambda _: "ARM EPISODE 1",
            is_interactive=True,
            phrase="ARM EPISODE 1",
        ))
        with self.assertRaises(UnsafeState):
            confirm_operator_arm(
                requested=True, prompt_fn=lambda _: "no", is_interactive=True
            )
        with self.assertRaises(UnsafeState):
            confirm_operator_arm(
                requested=True, prompt_fn=lambda _: "ARM", is_interactive=False
            )

    def test_reset_stays_disarmed_until_explicit_arm(self):
        FakeClient.instances.clear()
        with patch("real_env.LowLevelClient", FakeClient):
            env = RealRotaryInvertedPendulumEnv(reset_settle_s=0.0)
            try:
                obs, _ = env.reset()
                client = FakeClient.instances[-1]
                self.assertEqual(obs.shape, env.observation_space.shape)
                self.assertFalse(env.is_armed)
                self.assertEqual(client.arm_calls, 0)
                self.assertEqual(client.acceleration_calls, [])
                with self.assertRaises(RuntimeError):
                    env.apply_action(np.array([0.0], dtype=np.float32))
                env.arm()
                self.assertTrue(env.is_armed)
                self.assertEqual(client.arm_calls, 1)
                self.assertEqual(client.acceleration_calls, [0.0])
            finally:
                env.close()

    def test_bad_reset_state_never_arms(self):
        FakeClient.instances.clear()
        FakeClient.state_factory = staticmethod(lambda: make_state(faults=1))
        try:
            with patch("real_env.LowLevelClient", FakeClient):
                env = RealRotaryInvertedPendulumEnv(reset_settle_s=0.0)
                with self.assertRaises(UnsafeState):
                    env.reset()
                client = FakeClient.instances[-1]
                self.assertEqual(client.arm_calls, 0)
                env.close()
        finally:
            FakeClient.state_factory = staticmethod(make_state)


if __name__ == "__main__":
    unittest.main()
