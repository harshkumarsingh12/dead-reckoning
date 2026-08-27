"""Per-session sensor calibration, captured once at session start.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 6.2

Done when: a stationary phone reports near-zero world-frame velocity over 60 s, and
the training and live paths import these exact functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import ImuSample

    Vec3 = npt.NDArray[np.float64]
    Mat3 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Everything measured during the session-start calibration ritual."""

    gyro_bias_body: Vec3  # rad/s
    accel_bias_body: Vec3  # m/s^2
    mag_hard_iron_body: Vec3  # tesla, the offset to subtract
    mag_soft_iron: Mat3 | None = None  # optional 3x3 correction
    expected_field_strength_t: float = 50e-6  # local geomagnetic magnitude
    expected_dip_rad: float = 0.0  # local inclination
    stationary_samples: int = 0


def estimate_gyro_bias(samples: list[ImuSample]) -> Vec3:
    """Mean angular rate over a stationary window at session start.

    Args:
        samples: IMU samples captured while the phone is provably still.

    Returns:
        Per-axis bias, rad/s, to be subtracted from every subsequent reading.

    Raises:
        ValueError: if the window shows too much motion to be trusted as stationary.
    """
    raise NotImplementedError("M1 -- owner: Sristee")


def fit_hard_iron(mag_samples: list[Vec3]) -> tuple[Vec3, float]:
    """Fit a hard-iron offset from a 10-second figure-8 sweep.

    Fits a sphere to the sampled field vectors; the centre is the hard-iron offset and
    the radius is the local field magnitude the triple gate will compare against.

    Args:
        mag_samples: raw magnetometer vectors, tesla, from the figure-8.

    Returns:
        (offset_to_subtract, fitted_field_magnitude_t).

    Raises:
        ValueError: if the sweep did not cover enough of the sphere for a stable fit.
    """
    raise NotImplementedError("M1 -- owner: Sristee")


def calibrate_session(
    stationary: list[ImuSample],
    figure_eight: list[ImuSample],
    latitude_deg: float,
) -> CalibrationResult:
    """Run the full session-start calibration and bundle the result.

    Args:
        stationary: a few seconds of the phone sitting still.
        figure_eight: the 10 s figure-8 sweep for the magnetometer.
        latitude_deg: used to look up the expected geomagnetic dip angle.
    """
    raise NotImplementedError("M1 -- owner: Sristee")
