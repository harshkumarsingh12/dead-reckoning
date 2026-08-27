"""Madgwick AHRS wrapper: gyro + accel + gated magnetometer -> quaternion.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 6.3

Done when: orientation is stable over minutes of walking with turns, and a deliberately
introduced magnet visibly triggers rejection rather than corrupting heading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imufusion
import numpy as np

from dr_core.types import MagGateVerdict, OrientationEstimate

if TYPE_CHECKING:
    from dr_core.ahrs.mag_gate import MagGate
    from dr_core.preprocess.calibrate import CalibrationResult
    from dr_core.types import ImuSample

_GRAVITY = 9.80665  # m/s^2; imufusion wants acceleration in units of g


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
        self._calibration = calibration
        self._mag_gate = mag_gate
        self._rate_hz = rate_hz
        self._dt = 1.0 / rate_hz
        self._ahrs = imufusion.Ahrs()
        self._ahrs.set_sample_period(self._dt)
        self._apply_enu_settings()

    def _apply_enu_settings(self) -> None:
        """Configure imufusion: ENU world frame plus explicit rejection settings.

        Set by attribute assignment, not the positional constructor: in imufusion 1.3.2
        the positional form takes ``sample_rate`` first and silently drops ``convention``.
        There is deliberately no try/except -- if a future imufusion renames a field this
        must fail loudly rather than fall back to library defaults. That silent fallback
        was the prior bug: a wrong-order positional call raised and was swallowed, so the
        filter ran on NWU with acceleration_rejection disabled (90 deg).
        """
        settings = imufusion.AhrsSettings()
        settings.sample_rate = self._rate_hz
        settings.convention = imufusion.CONVENTION_ENU
        settings.gain = 0.5
        settings.gyroscope_range = 2000.0  # deg/s, typical MEMS full scale
        settings.acceleration_rejection = 10.0  # deg: distrust accel as a gravity
        settings.magnetic_rejection = 10.0  # deg  reference when it deviates beyond this
        settings.rejection_timeout = 5.0  # s before a rejected sensor is trusted again
        self._ahrs.set_settings(settings)
        self._settings = settings

    def update(self, sample: ImuSample) -> OrientationEstimate:
        """Advance the filter by one sample and return the current orientation.

        The returned estimate carries the magnetometer gate verdict, which the
        telemetry strip displays live.
        """
        w_body = np.asarray(sample.w_body, dtype=np.float64) - self._calibration.gyro_bias_body
        a_body = np.asarray(sample.a_body, dtype=np.float64) - self._calibration.accel_bias_body
        gyro_deg = np.degrees(w_body)
        accel_g = a_body / _GRAVITY

        verdict = MagGateVerdict.REJECTED_INNOVATION  # default when no reading is fused
        if sample.m_body is not None:
            m_corrected = np.asarray(sample.m_body, dtype=np.float64) - (
                self._calibration.mag_hard_iron_body
            )
            # Gravity points opposite the measured specific force of a level device.
            verdict = self._mag_gate.check(m_corrected, -a_body)
            if verdict is MagGateVerdict.ACCEPTED:
                self._ahrs.update(gyro_deg, accel_g, m_corrected)
            else:
                self._ahrs.update_no_magnetometer(gyro_deg, accel_g)
        else:
            self._ahrs.update_no_magnetometer(gyro_deg, accel_g)

        q = np.asarray(self._ahrs.get_quaternion(), dtype=np.float64)  # (w, x, y, z)
        return OrientationEstimate(t_ns=sample.t_ns, q_world_body=q, mag_verdict=verdict)

    @property
    def heading_rad(self) -> float:
        """Current yaw in the world ENU frame, radians, 0 = East, CCW positive."""
        w, x, y, z = (float(c) for c in self._ahrs.get_quaternion())
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def reset(self) -> None:
        """Discard accumulated state. Used between replay runs."""
        self._ahrs.restart()
        self._ahrs.set_sample_period(self._dt)
        self._apply_enu_settings()
