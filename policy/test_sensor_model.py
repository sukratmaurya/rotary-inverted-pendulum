"""Offline sensor fault-injection fixtures; no I2C or hardware access."""
from __future__ import annotations

import math
import unittest

from sensor_model import AngleTracker, frozen_fault


class SensorModelTests(unittest.TestCase):
    def test_both_encoder_wrap_directions_are_small_continuous_steps(self):
        forward = AngleTracker(previous_raw=4095)
        self.assertTrue(forward.sample(0))
        self.assertAlmostEqual(forward.position_rad, 2 * math.pi / 4096)
        reverse = AngleTracker(previous_raw=0)
        self.assertTrue(reverse.sample(4095))
        self.assertAlmostEqual(reverse.position_rad, -2 * math.pi / 4096)

    def test_disconnected_and_invalid_readings_do_not_corrupt_position(self):
        tracker = AngleTracker()
        tracker.sample(2000)
        before = tracker.position_rad
        for raw in (None, -1, 4096):
            self.assertFalse(tracker.sample(raw))
            self.assertFalse(tracker.valid)
        self.assertEqual(tracker.bad_samples, 3)
        self.assertEqual(tracker.position_rad, before)
        self.assertTrue(tracker.sample(2001))
        self.assertTrue(tracker.valid)
        self.assertEqual(tracker.bad_samples, 0)

    def test_sudden_jump_is_rejected_and_next_good_read_recovers(self):
        tracker = AngleTracker()
        tracker.sample(100)
        self.assertFalse(tracker.sample(1000))
        self.assertEqual(tracker.previous_raw, 100)
        self.assertTrue(tracker.sample(101))
        self.assertEqual(tracker.bad_samples, 0)

    def test_repeated_samples_allowed_at_rest_but_fault_after_armed_travel(self):
        tracker = AngleTracker()
        tracker.sample(1234)
        for _ in range(20):
            self.assertTrue(tracker.sample(1234))
        self.assertFalse(frozen_fault(6399, 2000))
        self.assertFalse(frozen_fault(6400, 999))
        self.assertTrue(frozen_fault(6400, 1000))


if __name__ == "__main__":
    unittest.main()
