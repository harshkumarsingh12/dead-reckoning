"""Evaluation harness: the numbers the whole project is judged on.

Spec: docs/BUILD_PLAN.md section 8  |  OWNER: Sikruti / Sumedha  |  MILESTONE: M0
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.eval.metrics import ate, calibration_coverage, drift_pct, final_error, rte
from dr_core.types import Trajectory

NS_PER_S = 1_000_000_000


def _line(n: int = 100, step: float = 1.0, label: str = "x") -> Trajectory:
    t = np.arange(n, dtype=np.int64) * (NS_PER_S // 10)
    p = np.column_stack([np.arange(n) * step, np.zeros(n)])
    return Trajectory(t_ns=t, p_world=p, label=label)


@pytest.mark.xfail(reason="M0 -- ATE unimplemented (owner: Sikruti)", strict=True)
def test_ate_of_a_trajectory_against_itself_is_zero() -> None:
    traj = _line()
    assert ate(traj, traj) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.xfail(reason="M0 -- RTE unimplemented (owner: Sikruti)", strict=True)
def test_rte_is_zero_for_a_perfect_estimate() -> None:
    traj = _line()
    assert rte(traj, traj, window_s=1.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.xfail(reason="M0 -- final_error unimplemented (owner: Sikruti)", strict=True)
def test_final_error_measures_the_endpoint_gap() -> None:
    truth = _line(label="truth")
    est = Trajectory(t_ns=truth.t_ns, p_world=truth.p_world + np.array([3.0, 4.0]), label="est")
    assert final_error(est, truth) == pytest.approx(5.0)


@pytest.mark.xfail(reason="M0 -- drift_pct unimplemented (owner: Sikruti)", strict=True)
def test_drift_pct_is_final_error_over_distance() -> None:
    """The headline number. 5 m off after a 100 m walk is 5%."""
    truth = _line(n=101, step=1.0, label="truth")  # 100 m travelled
    est = Trajectory(t_ns=truth.t_ns, p_world=truth.p_world + np.array([5.0, 0.0]), label="est")
    assert drift_pct(est, truth) == pytest.approx(5.0, rel=0.02)


@pytest.mark.xfail(reason="M2 -- coverage test unimplemented (owner: Sumedha)", strict=True)
def test_calibration_coverage_of_a_well_calibrated_model_is_about_68_percent() -> None:
    """MANDATORY gate before the model's covariance is allowed to become the filter's R.

    Draw errors from exactly the distribution the model claims and roughly 68% must
    land inside 1 sigma. A model reporting materially less is over-confident, which
    silently poisons fusion and makes the on-screen ellipse indefensible.
    """
    rng = np.random.default_rng(26168)
    sigmas = np.full(20_000, 0.2)
    errors = rng.normal(0.0, 0.2, 20_000)
    assert calibration_coverage(errors, sigmas, k=1.0) == pytest.approx(0.68, abs=0.02)


@pytest.mark.xfail(reason="M2 -- coverage test unimplemented (owner: Sumedha)", strict=True)
def test_calibration_coverage_catches_an_overconfident_model() -> None:
    """Claimed sigma is half the true error scale, so coverage should fall well short."""
    rng = np.random.default_rng(26168)
    errors = rng.normal(0.0, 0.4, 20_000)
    sigmas = np.full(20_000, 0.2)
    assert calibration_coverage(errors, sigmas, k=1.0) < 0.45
