"""Unit tests for ModelOnlyIntegrator: the model-only trajectory baseline.

Spec: docs/BUILD_PLAN.md section 8  |  OWNER: Sumedha  |  MILESTONE: M2

No torch needed here -- ModelOnlyIntegrator only integrates VelocityEstimates the
runtime already produced, so this runs in the default (non-ml) test job.
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.models import ModelOnlyIntegrator
from dr_core.types import VelocityEstimate

NS_PER_S = 1_000_000_000


def _estimate(t_ns: int, v_dev: tuple[float, float]) -> VelocityEstimate:
    return VelocityEstimate(t_ns=t_ns, v_dev=np.array(v_dev), cov=np.eye(2) * 0.01)


def test_constant_forward_velocity_facing_east_moves_east() -> None:
    integrator = ModelOnlyIntegrator()
    dt_ns = NS_PER_S // 10  # 0.1 s steps
    p = np.zeros(2)
    for i in range(101):  # 10 s at 1.4 m/s -> 14 m
        p = integrator.update(_estimate(i * dt_ns, (1.4, 0.0)), psi_rad=0.0)
    assert p[0] == pytest.approx(14.0, rel=0.01)
    assert abs(p[1]) < 1e-6


def test_heading_rotates_device_velocity_into_world() -> None:
    """Facing North (psi=pi/2) and reporting "forward" (v_dev=[v, 0]) must move North
    in world ENU, not East -- this is the whole reason rotate_dev_to_world exists."""
    integrator = ModelOnlyIntegrator()
    dt_ns = NS_PER_S // 10
    p = np.zeros(2)
    for i in range(101):
        p = integrator.update(_estimate(i * dt_ns, (1.4, 0.0)), psi_rad=np.pi / 2.0)
    assert abs(p[0]) < 1e-6
    assert p[1] == pytest.approx(14.0, rel=0.01)


def test_zero_velocity_produces_no_movement() -> None:
    integrator = ModelOnlyIntegrator()
    p = np.zeros(2)
    for i in range(50):
        p = integrator.update(_estimate(i * (NS_PER_S // 10), (0.0, 0.0)), psi_rad=1.23)
    np.testing.assert_allclose(p, np.zeros(2))


def test_trajectory_is_a_valid_interchange_trajectory() -> None:
    integrator = ModelOnlyIntegrator()
    for i in range(5):
        integrator.update(_estimate(i * (NS_PER_S // 10), (1.0, 0.0)), psi_rad=0.0)
    traj = integrator.trajectory
    assert traj.label == "model_only"
    assert traj.t_ns.dtype == np.int64
    assert traj.p_world.shape == (len(traj.t_ns), 2)
    assert len(traj.t_ns) == 5


def test_reset_clears_state() -> None:
    integrator = ModelOnlyIntegrator()
    for i in range(10):
        integrator.update(_estimate(i * (NS_PER_S // 10), (1.0, 0.0)), psi_rad=0.0)
    assert len(integrator.trajectory.t_ns) == 10

    integrator.reset()
    assert len(integrator.trajectory.t_ns) == 0
    assert integrator.trajectory.p_world.shape == (0, 2)

    p = integrator.update(_estimate(0, (0.0, 0.0)), psi_rad=0.0)
    np.testing.assert_allclose(p, np.zeros(2))  # first update after reset: dt=0
