"""Unit tests for the magnetometer triple gate (magnitude AND dip).

Spec: docs/BUILD_PLAN.md section 6.3  |  OWNER: Sristee  |  MILESTONE: M1

The innovation check lives in the filter; here we pin the two checks MagGate owns --
field magnitude and dip angle -- plus the accept-rate bookkeeping the telemetry reads.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dr_core.ahrs import MagGate
from dr_core.types import MagGateVerdict

if TYPE_CHECKING:
    import numpy.typing as npt

    Vec3 = npt.NDArray[np.float64]

EXPECTED_MAG_T = 50e-6
EXPECTED_DIP_RAD = math.radians(60.0)
DOWN = np.array([0.0, 0.0, -1.0])  # gravity_body: points down


def _field(magnitude_t: float, dip_rad: float) -> Vec3:
    """A field vector of the given magnitude and dip, in the DOWN=[0,0,-1] frame."""
    return np.array(
        [magnitude_t * math.cos(dip_rad), 0.0, -magnitude_t * math.sin(dip_rad)],
        dtype=np.float64,
    )


def _gate() -> MagGate:
    return MagGate(EXPECTED_MAG_T, EXPECTED_DIP_RAD)


def test_clean_on_target_field_is_accepted() -> None:
    gate = _gate()
    verdict = gate.check(_field(EXPECTED_MAG_T, EXPECTED_DIP_RAD), DOWN)
    assert verdict == MagGateVerdict.ACCEPTED
    assert gate.accept_rate == 1.0


@pytest.mark.parametrize("scale", [1.5, 0.5])
def test_magnitude_out_of_tolerance_is_rejected(scale: float) -> None:
    # dip stays on target; only the magnitude is off, so magnitude is the cause
    gate = _gate()
    verdict = gate.check(_field(EXPECTED_MAG_T * scale, EXPECTED_DIP_RAD), DOWN)
    assert verdict == MagGateVerdict.REJECTED_MAGNITUDE


def test_rotated_field_at_normal_strength_is_rejected_on_dip() -> None:
    # correct magnitude, horizontal (dip 0): the indoor-disturbance case the gate exists for
    gate = _gate()
    verdict = gate.check(_field(EXPECTED_MAG_T, 0.0), DOWN)
    assert verdict == MagGateVerdict.REJECTED_DIP


def test_magnitude_is_checked_before_dip() -> None:
    # fails BOTH magnitude and dip; the reported reason must be magnitude (checked first)
    gate = _gate()
    verdict = gate.check(_field(EXPECTED_MAG_T * 1.5, 0.0), DOWN)
    assert verdict == MagGateVerdict.REJECTED_MAGNITUDE


@pytest.mark.parametrize(
    ("gate", "m_body", "gravity_body"),
    [
        # zero field, magnitude check disabled -> hits the |m| ~ 0 guard
        (MagGate(0.0, EXPECTED_DIP_RAD), np.zeros(3), DOWN),
        # zero gravity, valid field -> hits the |gravity| ~ 0 guard
        (
            MagGate(EXPECTED_MAG_T, EXPECTED_DIP_RAD),
            _field(EXPECTED_MAG_T, EXPECTED_DIP_RAD),
            np.zeros(3),
        ),
    ],
)
def test_degenerate_inputs_are_rejected_not_waved_through(
    gate: MagGate, m_body: Vec3, gravity_body: Vec3
) -> None:
    assert gate.check(m_body, gravity_body) == MagGateVerdict.REJECTED_MAGNITUDE


def test_accept_rate_is_zero_on_a_fresh_gate() -> None:
    # no calls -> 0.0, and no ZeroDivisionError from _accepted / _total
    assert _gate().accept_rate == 0.0
