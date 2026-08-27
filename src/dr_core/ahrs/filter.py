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
        """Configure the world frame as ENU (x East, y North, z Up).

        The ``AhrsSettings`` constructor arity varies across ``imufusion`` builds, so we
        try the known signatures and fall back to the library default. ENU and the
        default NWU are both z-up, which is all the gravity-removal path depends on; the
        convention only changes the heading zero, verified by the turn tests.
        """
        conv = imufusion.CONVENTION_ENU
        recovery = int(5 * self._rate_hz)
        for args in (
            (conv, 0.5, 2000, 10, 10, recovery),
            (conv, 0.5, 2000.0, 10.0, 10.0, recovery),
            (conv, 0.5, 10.0, 10.0, recovery),
            (conv, 0.5),
        ):
            try:
                self._ahrs.set_settings(imufusion.AhrsSettings(*args))
                return
            except (TypeError, ValueError):
                continue

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
