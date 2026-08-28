"""ESKF behaviour, gating, and the stationary corrections.

Spec: docs/BUILD_PLAN.md sections 6.6 and 7  |  OWNER: Sikruti  |  MILESTONE: M3

Every assertion here is copied from a "done when" clause in the build plan. Deleting
one is deleting an acceptance criterion, which is a spec change and needs saying so in
the PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.fusion import ChiSquareGate, Eskf, NisLogger, StationaryDetector
from dr_core.types import ImuSample

pytestmark = pytest.mark.gating


def test_ten_second_stop_produces_zero_position_creep(stationary: list[ImuSample]) -> None:
    """Standing still for 10 s must not move the estimate.

    The most visible moment in the live demo: the presenter stops, the ZUPT lamp fires,
    the ellipse tightens, the drift counter freezes. If this test is red, that moment
    does not land.
    """
    eskf = Eskf()
    detector = StationaryDetector()
    start = eskf.state.p_world.copy()
    for s in stationary[: 10 * 200]:
        eskf.predict(s.t_ns, float(s.w_body[2]))
        if detector.update(s):
            eskf.update_zupt(s.t_ns)
            eskf.update_zaru(s.t_ns, float(s.w_body[2]))
    creep = float(np.linalg.norm(eskf.state.p_world - start))
    assert creep < 0.1, f"crept {creep:.3f} m while standing still"


def test_zaru_converges_the_gyro_bias(stationary: list[ImuSample]) -> None:
    """A stationary phone pins the yaw bias, so heading stops drifting for free."""
    eskf = Eskf()
    detector = StationaryDetector()
    for s in stationary:
        eskf.predict(s.t_ns, float(s.w_body[2]))
        if detector.update(s):
            eskf.update_zaru(s.t_ns, float(s.w_body[2]))
    assert abs(eskf.state.gyro_bias_z) < 0.01


def test_device_frame_velocity_update_corrects_heading() -> None:
    """The single most important wiring decision in the filter (build plan 7.1).

    Fusing the learned velocity in the DEVICE frame puts a dh/dpsi term in the
    Jacobian, so a velocity update also corrects heading. Start the filter with a
    deliberate heading error, feed consistent velocity, and the heading must converge.
    If it does not, the update was wired in the world frame and the path will bend
    through every turn.
    """
    from dr_core.types import VelocityEstimate

    eskf = Eskf()
    truth_psi = 0.0
    eskf.predict(0, 0.5)  # inject a heading error via a bogus gyro rate
    for i in range(1, 400):
        est = VelocityEstimate(
            t_ns=i * 5_000_000,
            v_dev=np.array([1.4, 0.0]),
            cov=np.diag([0.01, 0.01]),
        )
        eskf.predict(est.t_ns, 0.0)
        eskf.update_velocity(est)
    assert abs(eskf.state.psi_rad - truth_psi) < 0.1


def test_chi_square_gate_rejects_an_injected_outlier() -> None:
    """Outlier injection (build plan section 9). A wild innovation must be rejected and
    must still show up in the NIS log -- a silently dropped measurement teaches nobody
    anything."""
    gate = ChiSquareGate(dof=2, level=0.95)
    good, nis_good = gate.accept(np.array([0.1, 0.1]), np.eye(2))
    bad, nis_bad = gate.accept(np.array([50.0, 50.0]), np.eye(2))
    assert good is True
    assert bad is False
    assert nis_bad > nis_good


def test_nis_logger_reports_per_channel_consistency() -> None:
    """A consistent filter has mean NIS near the channel's degrees of freedom."""
    logger = NisLogger({"velocity": 2, "heading": 1})
    rng = np.random.default_rng(0)
    for _ in range(500):
        logger.record("velocity", float(rng.chisquare(2)), accepted=True)
        logger.record("heading", float(rng.chisquare(1)), accepted=True)
    assert logger.is_consistent() == {"velocity": True, "heading": True}


def test_velocity_scale_freezes_when_gps_drops() -> None:
    """Scale and speed are not separable without GPS (build plan 7.2).

    Letting `s` keep adapting in the tunnel is how a filter talks itself into a
    confidently wrong speed.
    """
    eskf = Eskf()
    eskf.set_gps_enabled(False)
    before = eskf.state.scale
    from dr_core.types import VelocityEstimate

    for i in range(1, 200):
        eskf.predict(i * 5_000_000, 0.0)
        eskf.update_velocity(
            VelocityEstimate(
                t_ns=i * 5_000_000,
                v_dev=np.array([2.5, 0.0]),
                cov=np.diag([0.01, 0.01]),
            )
        )
    assert eskf.state.scale == pytest.approx(before)


# -------------- edge cases (hardening, not acceptance criteria) ----------------


def test_covariance_remains_positive_definite_over_long_run() -> None:
    """Joseph-form update must keep P symmetric positive-definite forever.

    A naive (I - KH)P form loses symmetry through floating-point accumulation.
    The Joseph form was chosen specifically to avoid this, so we prove it holds
    over a realistically long cycle count.
    """
    from dr_core.types import VelocityEstimate

    eskf = Eskf()
    for i in range(1, 1001):
        eskf.predict(i * 5_000_000, 0.1)
        if i % 5 == 0:
            eskf.update_velocity(
                VelocityEstimate(
                    t_ns=i * 5_000_000,
                    v_dev=np.array([1.0, 0.0]),
                    cov=np.diag([0.05, 0.05]),
                )
            )
    cov = eskf.state.cov
    # Symmetric
    assert np.allclose(cov, cov.T, atol=1e-12), "covariance lost symmetry"
    # Positive definite (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), f"non-positive eigenvalue: {eigenvalues.min():.2e}"


def test_magnetometer_gate_rejects_large_heading_innovation() -> None:
    """A heading innovation far outside the chi-square gate must be rejected.

    Indoor environments produce rotated fields at near-normal strength. The gate
    must catch these; if it does not, the filter trusts a corrupted reading and
    the path bends.
    """
    eskf = Eskf()
    eskf.predict(5_000_000, 0.0)  # initialise time

    # Current heading is near 0.0. Feed a heading 3 rad away — clearly wrong.
    accepted = eskf.update_magnetometer(10_000_000, 3.0, sigma_rad=0.1)
    assert accepted is False, "gate accepted a 3-rad heading innovation"


def test_gps_cog_heading_only_updates_above_speed_threshold() -> None:
    """Below 1 m/s, GPS course-over-ground is noise. The filter must ignore it.

    BUILD_PLAN section 7.6: course heading correction only at 'sufficient speed'.
    The code uses speed > 1.0 m/s as the threshold.
    """
    from dr_core.types import GpsFix

    eskf = Eskf()
    eskf.predict(5_000_000, 0.0)
    heading_before = eskf.state.psi_rad

    # GPS fix at very low speed — should NOT update heading via COG
    slow_fix = GpsFix(
        t_ns=10_000_000,
        lat_deg=20.3535,
        lon_deg=85.8164,
        accuracy_m=3.0,
        speed_mps=0.5,  # below threshold
        course_rad=1.5,  # very different from heading_before
    )
    eskf.update_gps(slow_fix)
    heading_after = eskf.state.psi_rad

    # Position may have changed, but heading should NOT have been corrected by COG.
    # (It may shift slightly from the position update's effect on the error state,
    # but not by the 1.5 rad the COG heading would impose.)
    assert abs(heading_after - heading_before) < 0.3, (
        f"heading jumped {abs(heading_after - heading_before):.2f} rad from a low-speed COG"
    )


def test_double_zupt_from_same_window_is_safe() -> None:
    """Calling ZUPT twice while stationary must not corrupt the state.

    This can happen if the detector fires on consecutive IMU samples within the
    same stationary window, and the caller applies ZUPT for each one.
    """
    eskf = Eskf()
    eskf.predict(5_000_000, 0.0)
    eskf.predict(10_000_000, 0.0)

    eskf.update_zupt(10_000_000)
    eskf.update_zupt(10_000_000)
    st = eskf.state

    # Velocity should still be near zero (not NaN or diverged)
    assert np.all(np.isfinite(st.v_world))
    assert np.all(np.isfinite(st.p_world))
    assert np.all(np.isfinite(st.cov))
    # Covariance should still be positive definite
    eigenvalues = np.linalg.eigvalsh(st.cov)
    assert np.all(eigenvalues > 0), (
        f"non-positive eigenvalue after double ZUPT: {eigenvalues.min():.2e}"
    )


def test_large_imu_dt_does_not_cause_numerical_overflow() -> None:
    """A 2-second gap in IMU (e.g. Android app backgrounded) must not blow P to infinity.

    The predict step scales process noise by dt, so a large dt produces a large Q addition.
    P should grow but remain finite and positive-definite.
    """
    eskf = Eskf()
    eskf.predict(1_000_000_000, 0.0)  # t = 1 s
    eskf.predict(3_000_000_000, 0.0)  # t = 3 s — a 2-second gap

    cov = eskf.state.cov
    assert np.all(np.isfinite(cov)), "covariance contains inf/nan after a 2 s gap"
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), (
        f"non-positive eigenvalue after large dt: {eigenvalues.min():.2e}"
    )
    # Covariance should have grown but not to absurd values
    assert np.max(cov) < 1e6, f"covariance exploded to {np.max(cov):.2e}"


def test_heading_wraps_correctly_across_pi_boundary() -> None:
    """Driving heading past +pi must wrap to -pi, not accumulate unbounded.

    The wrap_angle function is the defence, and it sits on the hot path of every
    predict and inject_and_reset call.
    """
    from dr_core.fusion.eskf import wrap_angle

    # Exact boundary cases
    assert wrap_angle(np.pi) == pytest.approx(-np.pi, abs=1e-10)
    assert wrap_angle(-np.pi) == pytest.approx(-np.pi, abs=1e-10)
    assert wrap_angle(0.0) == pytest.approx(0.0, abs=1e-10)

    # Just past +pi wraps to just past -pi
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1, abs=1e-10)
    # Just past -pi wraps correctly
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1, abs=1e-10)

    # Large accumulated angle wraps properly
    assert -np.pi <= wrap_angle(10.0) < np.pi
    assert -np.pi <= wrap_angle(-10.0) < np.pi

    # Now verify the filter itself: accumulate heading past ±pi
    eskf = Eskf()
    # 100 predict steps with a large positive gyro rate to push heading past pi
    for i in range(1, 101):
        eskf.predict(i * 10_000_000, 5.0)  # 5 rad/s * 0.01 s = 0.05 rad per step
    psi = eskf.state.psi_rad
    assert -np.pi <= psi < np.pi, f"heading {psi:.4f} outside [-pi, pi)"
