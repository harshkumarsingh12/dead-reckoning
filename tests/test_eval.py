"""Evaluation harness: the numbers the whole project is judged on.

Spec: docs/BUILD_PLAN.md section 8  |  OWNER: Sikruti / Sumedha  |  MILESTONE: M0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dr_core.eval.metrics import ate, calibration_coverage, drift_pct, final_error, rte
from dr_core.types import Trajectory

NS_PER_S = 1_000_000_000


def _line(n: int = 100, step: float = 1.0, label: str = "x") -> Trajectory:
    t = np.arange(n, dtype=np.int64) * (NS_PER_S // 10)
    p = np.column_stack([np.arange(n) * step, np.zeros(n)])
    return Trajectory(t_ns=t, p_world=p, label=label)


def test_ate_of_a_trajectory_against_itself_is_zero() -> None:
    traj = _line()
    assert ate(traj, traj) == pytest.approx(0.0, abs=1e-9)


def test_rte_is_zero_for_a_perfect_estimate() -> None:
    traj = _line()
    assert rte(traj, traj, window_s=1.0) == pytest.approx(0.0, abs=1e-9)


def test_final_error_measures_the_endpoint_gap() -> None:
    truth = _line(label="truth")
    est = Trajectory(t_ns=truth.t_ns, p_world=truth.p_world + np.array([3.0, 4.0]), label="est")
    assert final_error(est, truth) == pytest.approx(5.0)


def test_drift_pct_is_final_error_over_distance() -> None:
    """The headline number. 5 m off after a 100 m walk is 5%."""
    truth = _line(n=101, step=1.0, label="truth")  # 100 m travelled
    est = Trajectory(t_ns=truth.t_ns, p_world=truth.p_world + np.array([5.0, 0.0]), label="est")
    assert drift_pct(est, truth) == pytest.approx(5.0, rel=0.02)


def test_resample_to_handles_angle_wrapping() -> None:
    """Interpolating an angle crossing the +/-pi boundary must not swing through zero."""
    from dr_core.eval.metrics import resample_to

    # Trajectory going from 3.0 rad to -3.0 rad (turning across pi)
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    p = np.zeros((2, 2), dtype=np.float64)
    psi = np.array([3.0, -3.0], dtype=np.float64)
    traj = Trajectory(t_ns=t, p_world=p, psi_rad=psi)

    # Resample at midpoint t = 0.5s
    t_mid = np.array([500_000_000], dtype=np.int64)
    res = resample_to(traj, t_mid)
    assert res.psi_rad is not None
    # Midpoint of 3.0 and -3.0 wrapping across pi should be near pi or -pi (~3.14159), not 0.0
    assert abs(res.psi_rad[0]) > 3.0


def test_generate_report_and_cli(tmp_path: Path) -> None:
    from dr_core.eval.cli import EXIT_USAGE, main
    from dr_core.eval.report import generate_report
    from dr_core.fusion.gating import NisLogger

    truth = _line(n=10, step=1.0, label="truth")
    est = Trajectory(t_ns=truth.t_ns, p_world=truth.p_world + 0.1, label="est")
    logger = NisLogger({"velocity": 2})
    logger.record("velocity", 2.0, accepted=True)

    report = generate_report(
        estimate=est,
        truth=truth,
        baselines={},
        output_dir=tmp_path,
        run_id="test_run",
        nis_logger=logger,
    )
    assert (tmp_path / "trajectory.png").exists()
    assert (tmp_path / "error_time.png").exists()
    assert (tmp_path / "error_cdf.png").exists()
    assert (tmp_path / "nis.png").exists()
    assert (tmp_path / "report.json").exists()
    assert report.drift_pct >= 0.0

    # Test CLI missing file handling
    code = main(["non_existent_file.jsonl.gz"])
    assert code == EXIT_USAGE


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


def test_calibration_coverage_catches_an_overconfident_model() -> None:
    """Claimed sigma is half the true error scale, so coverage should fall well short."""
    rng = np.random.default_rng(26168)
    errors = rng.normal(0.0, 0.4, 20_000)
    sigmas = np.full(20_000, 0.2)
    assert calibration_coverage(errors, sigmas, k=1.0) < 0.45
