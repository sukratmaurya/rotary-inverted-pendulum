"""Offline AS5600 sample tracker matching the firmware acceptance rules.

This is a deterministic fault-injection fixture, not a replacement estimator
and not evidence of physical sensor behavior. Firmware integration remains a
separate commissioning check.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AngleTracker:
    previous_raw: int | None = None
    position_rad: float = 0.0
    bad_samples: int = 0
    valid: bool = False

    def sample(self, raw: int | None) -> bool:
        """Return whether this sample was accepted; invalid/jump reads retain state."""
        if raw is None or not 0 <= raw < 4096:
            self.valid = False
            self.bad_samples = min(255, self.bad_samples + 1)
            return False
        if self.previous_raw is None:
            self.previous_raw = raw
            self.valid = True
            self.bad_samples = 0
            return True
        delta = raw - self.previous_raw
        if delta > 2048:
            delta -= 4096
        elif delta < -2048:
            delta += 4096
        if abs(delta) > 500:
            self.valid = False
            self.bad_samples = min(255, self.bad_samples + 1)
            return False
        self.position_rad += delta * (2.0 * 3.141592653589793 / 4096.0)
        self.previous_raw = raw
        self.valid = True
        self.bad_samples = 0
        return True


def frozen_fault(motor_travel_steps: int, unchanged_ms: int) -> bool:
    """Mirror the firmware's conservative armed-motion frozen-sensor gate."""
    return abs(motor_travel_steps) >= 6400 and unchanged_ms >= 1000
