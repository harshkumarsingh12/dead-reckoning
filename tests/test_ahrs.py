"""AHRS configuration and heading-tracking tests.

Spec: docs/BUILD_PLAN.md section 6.3  |  OWNER: Sristee  |  MILESTONE: M1
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import imufusion
import numpy as np
import pytest

from dr_core.ahrs import AhrsFilter, MagGate
from dr_core.preprocess import CalibrationResult
from dr_core.types import ImuSample, MagGateVerdict

if TYPE_CHECKING:
    import numpy.typing as npt

GRAVITY = 9.80665
NS_PER_S = 1_000_000_000


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


def _rotz(psi: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _north_field_turn() -> tuple[list[ImuSample], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A phone circling at constant yaw rate, with a physically correct magnetic field.

    The field's horizontal component points NORTH (+y ENU) because that is where a real
    geomagnetic field's horizontal points -- toward magnetic north. It is generated
    locally rather than reusing conftest's pure_turn, whose field points East and would
    put any mag-fused heading test 90 deg off (issue #66).
    """
    rate_hz = 200.0
    speed, radius = 1.4, 20.0
    omega = speed / radius
    duration = 2.0 * math.pi / omega
    n = int(duration * rate_hz)
    dt = 1.0 / rate_hz
    g = np.array([0.0, 0.0, GRAVITY])
    field_world = np.array([0.0, 20e-6, -45e-6])  # North horizontal, downward -> dip ~66 deg
    samples: list[ImuSample] = []
    true_psi = np.empty(n)
    for i in range(n):
        t = i * dt
        true_psi[i] = omega * t
        r = _rotz(omega * t)
        a_world = np.array(
            [-speed * omega * math.cos(omega * t), -speed * omega * math.sin(omega * t), 0.0]
        )
        samples.append(
            ImuSample(
                t_ns=round(t * NS_PER_S),
                a_body=r.T @ (a_world + g),
                w_body=np.array([0.0, 0.0, omega]),
                m_body=r.T @ field_world,
            )
        )
    return samples, true_psi, field_world


def test_populated_dip_lets_the_gate_accept_and_heading_track() -> None:
    """#59 end to end: a populated expected dip makes the gate accept a clean field, and
    heading then tracks a sustained turn to about a degree.

    F' passes expected_dip_rad explicitly, so it uses the OVERRIDE path deliberately --
    the latitude_deg argument is NOT what sets the dip here. The field points North
    because that is physically correct, not because an East-pointing field failed.
    """
    samples, true_psi, field_world = _north_field_turn()
    horizontal = float(math.hypot(field_world[0], field_world[1]))
    vertical = float(-field_world[2])
    field_dip = math.atan2(vertical, horizontal)  # the field's own dip, not a tuned target

    calib = CalibrationResult.for_session(latitude_deg=20.35, expected_dip_rad=field_dip)
    gate = MagGate(calib.expected_field_strength_t, calib.expected_dip_rad)
    ahrs = AhrsFilter(calib, gate)

    errors: list[float] = []
    for sample, psi_true in zip(samples, true_psi, strict=True):
        ahrs.update(sample)
        err = (ahrs.heading_rad - float(psi_true) + math.pi) % (2 * math.pi) - math.pi
        errors.append(abs(err))

    warmup = int(5.0 * 200.0)  # let the filter settle
    assert gate.accept_rate > 0.95  # the clean field is fused, not rejected
    assert max(errors[warmup:]) < math.radians(3.0)  # headroom over the measured ~1.3 deg


def test_default_zero_dip_wrongly_rejects_the_same_clean_field() -> None:
    """Before/after for #59: expected_dip = 0 (the old default) rejects the clean field
    on dip; the populated dip accepts it. Same field, same gravity."""
    field = np.array([0.0, 20e-6, -45e-6])  # North horizontal, physically correct
    down = np.array([0.0, 0.0, -GRAVITY])
    field_strength = float(np.linalg.norm(field))
    field_dip = math.atan2(-float(field[2]), math.hypot(float(field[0]), float(field[1])))

    rejected = MagGate(field_strength, 0.0).check(field, down)
    accepted = MagGate(field_strength, field_dip).check(field, down)
    assert rejected == MagGateVerdict.REJECTED_DIP
    assert accepted == MagGateVerdict.ACCEPTED
