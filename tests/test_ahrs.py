"""AHRS configuration unit tests.

Spec: docs/BUILD_PLAN.md section 6.3  |  OWNER: Sristee  |  MILESTONE: M1

These check the wiring into imufusion, not orientation accuracy -- heading-tracking
quality is exercised by the frame tests and (later) the fusion consistency checks.
"""

from __future__ import annotations

import imufusion
import numpy as np
import pytest

from dr_core.ahrs import AhrsFilter, MagGate
from dr_core.preprocess import CalibrationResult


def _filter() -> AhrsFilter:
    calib = CalibrationResult(
        gyro_bias_body=np.zeros(3),
        accel_bias_body=np.zeros(3),
        mag_hard_iron_body=np.zeros(3),
    )
    return AhrsFilter(calib, MagGate(calib.expected_field_strength_t, calib.expected_dip_rad))


def test_ahrs_settings_reach_imufusion() -> None:
    """The ENU convention and acceleration_rejection must actually be applied.

    Regression guard: the settings were once built with a wrong-order positional
    constructor that raised and was silently swallowed, so the filter ran on imufusion
    defaults -- NWU, with acceleration_rejection disabled at 90 deg. Assert the values we
    set are the values the applied settings object carries.
    """
    ahrs = _filter()
    assert ahrs._settings.convention == imufusion.CONVENTION_ENU
    assert ahrs._settings.acceleration_rejection == pytest.approx(10.0)
