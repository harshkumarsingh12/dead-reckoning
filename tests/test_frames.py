"""Coordinate-frame invariants on synthetic motion. The cheapest insurance we buy.

Spec: docs/BUILD_PLAN.md section 9

Most trajectory bugs live in the frames. These three cases cost nothing to run and
catch the whole class. Read the rotation-in-place docstring before deciding any of them
is redundant -- straight line and pure turn both pass under a wrong-frame
implementation, and that one does not.

This module runs as its own named CI check (``pytest -m frames``) so it is impossible
to lose in the noise of a long test log.
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.baselines import RawIntegrator
from dr_core.types import ImuSample, Trajectory

pytestmark = pytest.mark.frames

OWNER = "Sristee"


@pytest.mark.xfail(reason="M1 -- AHRS + integration unimplemented (owner: Sristee)", strict=True)
def test_straight_line_travels_the_expected_distance(
    straight_line: tuple[list[ImuSample], Trajectory],
) -> None:
    """Walking due East for 20 s at 1.4 m/s ends up ~26 m East and ~0 m North."""
    samples, truth = straight_line
    integrator = RawIntegrator()
    p = np.zeros(2)
    for s in samples:
        p = integrator.update(s, _flat_orientation(s.t_ns))
    expected = truth.p_world[-1]
    assert p[0] == pytest.approx(expected[0], rel=0.05)
    assert abs(p[1]) < 1.0


@pytest.mark.xfail(reason="M1 -- AHRS + integration unimplemented (owner: Sristee)", strict=True)
def test_pure_turn_closes_the_circle(pure_turn: tuple[list[ImuSample], Trajectory]) -> None:
    """One full lap of a 20 m circle returns to the start.

    Catches a sign error in the yaw rate, which a straight line cannot see.
    """
    samples, truth = pure_turn
    integrator = RawIntegrator()
    p = np.zeros(2)
    for s in samples:
        p = integrator.update(s, _flat_orientation(s.t_ns))
    assert np.linalg.norm(p - truth.p_world[0]) < 5.0


@pytest.mark.xfail(reason="M1 -- AHRS + integration unimplemented (owner: Sristee)", strict=True)
def test_rotation_in_place_produces_zero_displacement(
    rotation_in_place: tuple[list[ImuSample], Trajectory],
) -> None:
    """THE important one.

    The phone is spun about yaw while the person stands still. Any implementation that
    confuses the body and world frames, or that carries a lever-arm term it should not,
    will walk the position off the origin here while passing both tests above. If this
    goes red, stop and fix the frames before touching anything downstream.
    """
    samples, _truth = rotation_in_place
    integrator = RawIntegrator()
    p = np.zeros(2)
    for s in samples:
        p = integrator.update(s, _flat_orientation(s.t_ns))
    assert np.linalg.norm(p) < 0.5, f"spun in place but moved {np.linalg.norm(p):.2f} m"


@pytest.mark.xfail(reason="M1 -- AHRS unimplemented (owner: Sristee)", strict=True)
def test_stationary_phone_reports_near_zero_world_velocity(
    stationary: list[ImuSample],
) -> None:
    """60 s still, world-frame velocity stays near zero (build plan 6.2, done-when)."""
    integrator = RawIntegrator()
    for s in stationary:
        integrator.update(s, _flat_orientation(s.t_ns))
    drift = float(np.linalg.norm(integrator.trajectory.p_world[-1]))
    assert drift < 5.0, f"stationary for 60 s but drifted {drift:.1f} m"


def _flat_orientation(t_ns: int):
    """Placeholder orientation for a device held flat.

    Replaced by the real AhrsFilter output when M1 lands; these tests then exercise the
    genuine AHRS-to-integrator path rather than an idealised one.
    """
    from dr_core.types import OrientationEstimate

    return OrientationEstimate(t_ns=t_ns, q_world_body=np.array([1.0, 0.0, 0.0, 0.0]))
