"""Unit tests for the geomagnetic dip model and session-calibration wiring.

Spec: docs/BUILD_PLAN.md section 6.2  |  OWNER: Sristee  |  MILESTONE: M1

Two things in isolation: the dipole dip-from-latitude approximation, and that
CalibrationResult.for_session populates a real expected dip (with an explicit override
path) instead of the 0.0 default that made the mag gate reject clean fields (#59).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dr_core.preprocess import CalibrationResult
from dr_core.preprocess.calibrate import expected_dip_from_latitude


def test_dip_at_the_equator_is_zero() -> None:
    assert expected_dip_from_latitude(0.0) == pytest.approx(0.0)


def test_dip_at_the_pole_is_ninety_degrees() -> None:
    assert expected_dip_from_latitude(90.0) == pytest.approx(math.pi / 2)


def test_dip_at_forty_five_matches_atan_two() -> None:
    # dipole: tan(dip) = 2*tan(45) = 2, so dip = atan(2); assert against atan(2) directly
    assert expected_dip_from_latitude(45.0) == pytest.approx(math.atan(2.0))


def test_explicit_dip_overrides_latitude() -> None:
    calib = CalibrationResult.for_session(latitude_deg=20.0, expected_dip_rad=1.2)
    assert calib.expected_dip_rad == 1.2


def test_latitude_populates_dip_and_biases_are_placeholder_zeros() -> None:
    calib = CalibrationResult.for_session(latitude_deg=45.0)
    assert calib.expected_dip_rad == pytest.approx(math.atan(2.0))  # not the 0.0 default
    # partial result: biases are placeholder zeros pending fit_hard_iron / estimate_gyro_bias
    assert np.array_equal(calib.gyro_bias_body, np.zeros(3))
    assert np.array_equal(calib.accel_bias_body, np.zeros(3))
    assert np.array_equal(calib.mag_hard_iron_body, np.zeros(3))
