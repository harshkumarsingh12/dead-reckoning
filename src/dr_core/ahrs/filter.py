"""Madgwick AHRS wrapper: gyro + accel + gated magnetometer -> quaternion.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 6.3

Done when: orientation is stable over minutes of walking with turns, and a deliberately
introduced magnet visibly triggers rejection rather than corrupting heading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_core.ahrs.mag_gate import MagGate
    from dr_core.preprocess.calibrate import CalibrationResult
    from dr_core.types import ImuSample, OrientationEstimate


class AhrsFilter:
    """Stateful orientation estimator, one instance per session.

    Wraps ``imufusion.Ahrs``. The only logic added on top is the magnetometer triple
    gate: a rejected reading is dropped before it reaches the AHRS, so a magnetic
    disturbance degrades heading to gyro-only drift rather than yanking it.
    """

    def __init__(
        self,
        calibration: CalibrationResult,
        mag_gate: MagGate,
        rate_hz: float = 200.0,
    ) -> None:
        raise NotImplementedError("M1 -- owner: Sristee")

    def update(self, sample: ImuSample) -> OrientationEstimate:
        """Advance the filter by one sample and return the current orientation.

        The returned estimate carries the magnetometer gate verdict, which the
        telemetry strip displays live.
        """
        raise NotImplementedError("M1 -- owner: Sristee")

    @property
    def heading_rad(self) -> float:
        """Current yaw in the world ENU frame, radians, 0 = East, CCW positive."""
        raise NotImplementedError("M1 -- owner: Sristee")

    def reset(self) -> None:
        """Discard accumulated state. Used between replay runs."""
        raise NotImplementedError("M1 -- owner: Sristee")
