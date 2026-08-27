"""Stationary detection, which drives both ZUPT and ZARU.

OWNER: Sikruti  |  MILESTONE: M3  |  Spec: docs/BUILD_PLAN.md section 6.6

Standing still is free information and dead reckoning normally throws it away. Detect
it and you get two corrections at once: velocity is exactly zero (ZUPT) and angular
rate is exactly zero, which pins the gyro bias (ZARU).

It is also the most legible moment in the live demo. The presenter stops walking, the
lamp lights, the ellipse tightens, drift stops climbing. Judges see the mechanism work
rather than being told it does.

Done when: standing still for 10 s produces no position creep, the bias converges, and
the lamp fires on cue during the scripted arc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_core.types import ImuSample


@dataclass(frozen=True, slots=True)
class StationaryConfig:
    """Thresholds for the detector.

    Tune on a recording that contains both a genuine stop and slow walking. The failure
    mode to avoid is firing during a slow shuffle, which pins the state to a position
    the person is actually leaving.
    """

    window_s: float = 0.5
    accel_var_threshold: float = 0.05  # (m/s^2)^2
    gyro_var_threshold: float = 0.01  # (rad/s)^2
    min_duration_s: float = 0.3  # must hold this long before firing


class StationaryDetector:
    """Sliding-window variance test over accelerometer and gyroscope."""

    def __init__(self, config: StationaryConfig | None = None, rate_hz: float = 200.0) -> None:
        raise NotImplementedError("M3 -- owner: Sikruti")

    def update(self, sample: ImuSample) -> bool:
        """Feed one sample. Returns whether the device is currently stationary."""
        raise NotImplementedError("M3 -- owner: Sikruti")

    @property
    def is_stationary(self) -> bool:
        """Current verdict, without advancing the window."""
        raise NotImplementedError("M3 -- owner: Sikruti")

    @property
    def stationary_duration_s(self) -> float:
        """How long the current stationary episode has lasted. Zero if moving."""
        raise NotImplementedError("M3 -- owner: Sikruti")
