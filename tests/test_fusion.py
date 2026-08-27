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
