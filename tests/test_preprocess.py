"""Unit tests for the shared preprocessing path: resampling and window construction.

Spec: docs/BUILD_PLAN.md sections 4 and 6.2  |  MILESTONE: M1/M2

``resample_uniform`` and ``prepare_window`` are the two functions the roadmap calls out
as "still open for M2" (docs/ROADMAP.md, M1 table) -- the rest of ``dr_core.preprocess``
is Sristee's M1 work and already covered elsewhere. These tests exist so training and
live cannot silently diverge on rate, shape, or frame (AGENTS.md rule 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.preprocess import align_gravity, prepare_window, resample_uniform
from dr_core.types import ImuSample, OrientationEstimate

NS_PER_S = 1_000_000_000


def test_resample_uniform_produces_the_requested_rate() -> None:
    """A ragged input timeline still comes out on an exact 1/rate_hz grid."""
    rng = np.random.default_rng(26168)
    t_ns = np.sort(rng.integers(0, 5 * NS_PER_S, size=50)).astype(np.int64)
    t_ns[0] = 0
    values = rng.normal(size=(50, 3))

    t_uniform, values_uniform = resample_uniform(t_ns, values, rate_hz=100.0)

    diffs_ns = np.diff(t_uniform)
    np.testing.assert_array_equal(diffs_ns, np.full(diffs_ns.shape, 10_000_000))
    assert values_uniform.shape == (t_uniform.shape[0], 3)


def test_resample_uniform_matches_linear_interpolation() -> None:
    """A known linear ramp resampled onto a finer grid reproduces the same line."""
    t_ns = np.array([0, 1 * NS_PER_S, 2 * NS_PER_S], dtype=np.int64)
    values = np.array([[0.0], [10.0], [20.0]])  # 10 units/s

    t_uniform, values_uniform = resample_uniform(t_ns, values, rate_hz=10.0)

    expected = (t_uniform.astype(np.float64) / NS_PER_S) * 10.0
    np.testing.assert_allclose(values_uniform[:, 0], expected, atol=1e-9)


def test_resample_uniform_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        resample_uniform(np.array([0], dtype=np.int64), np.zeros((1, 3)))


def _flat_stream(
    n: int, rate_hz: float, a_body_z: float = 9.80665
) -> tuple[list[ImuSample], list[OrientationEstimate]]:
    """A stationary, level phone: gravity only, identity orientation."""
    dt_ns = round(NS_PER_S / rate_hz)
    samples = [
        ImuSample(
            t_ns=i * dt_ns,
            a_body=np.array([0.0, 0.0, a_body_z]),
            w_body=np.zeros(3),
        )
        for i in range(n)
    ]
    orientations = [
        OrientationEstimate(t_ns=s.t_ns, q_world_body=np.array([1.0, 0.0, 0.0, 0.0]))
        for s in samples
    ]
    return samples, orientations


def test_prepare_window_shape_is_channels_by_window_samples() -> None:
    samples, orientations = _flat_stream(n=250, rate_hz=200.0)
    window = prepare_window(samples, orientations, rate_hz=200.0, window_s=1.0)
    assert window.shape == (6, 200)


def test_prepare_window_matches_align_gravity_per_sample() -> None:
    """The window is built from the SAME align_gravity every other consumer uses --
    a stationary window must show near-zero accel (gravity removed) and near-zero
    gyro, exactly like the raw integrator's stationary invariant."""
    samples, orientations = _flat_stream(n=250, rate_hz=200.0)
    window = prepare_window(samples, orientations, rate_hz=200.0, window_s=1.0)

    a_dev, w_dev = window[0:3], window[3:6]
    assert np.max(np.abs(a_dev)) < 1e-6
    assert np.max(np.abs(w_dev)) < 1e-6

    # Cross-check against a direct call, so a future change to either path is caught.
    a_expected, w_expected = align_gravity(samples[-1].a_body, samples[-1].w_body, orientations[-1])
    np.testing.assert_allclose(a_dev[:, -1], a_expected, atol=1e-9)
    np.testing.assert_allclose(w_dev[:, -1], w_expected, atol=1e-9)


def _forward_push_stream(
    n: int, rate_hz: float, psi_rad: float, push_mps2: float = 0.5
) -> tuple[list[ImuSample], list[OrientationEstimate]]:
    """A phone held flat, facing a fixed heading, accelerating along its own forward axis."""
    dt_ns = round(NS_PER_S / rate_hz)
    q = np.array([np.cos(psi_rad / 2.0), 0.0, 0.0, np.sin(psi_rad / 2.0)])
    a_body = np.array([push_mps2, 0.0, 9.80665])
    samples = [
        ImuSample(t_ns=i * dt_ns, a_body=a_body.copy(), w_body=np.zeros(3)) for i in range(n)
    ]
    orientations = [OrientationEstimate(t_ns=s.t_ns, q_world_body=q) for s in samples]
    return samples, orientations


def test_prepare_window_is_heading_agnostic() -> None:
    """Two sessions facing different absolute directions but identical RELATIVE motion
    must produce identical device-frame windows -- the RoNIN trick, and the reason
    align_gravity's docstring says the yaw-removal step belongs here. Facing North
    instead of East and pushing "forward" must not change the window at all."""
    east_samples, east_orient = _forward_push_stream(n=250, rate_hz=200.0, psi_rad=0.0)
    north_samples, north_orient = _forward_push_stream(n=250, rate_hz=200.0, psi_rad=np.pi / 2.0)

    window_east = prepare_window(east_samples, east_orient, rate_hz=200.0, window_s=1.0)
    window_north = prepare_window(north_samples, north_orient, rate_hz=200.0, window_s=1.0)

    np.testing.assert_allclose(window_east, window_north, atol=1e-9)
    # And it is not a trivially-zero window -- the push is actually present.
    assert np.max(np.abs(window_east[0:2])) > 0.1


def test_prepare_window_rejects_mismatched_lengths() -> None:
    samples, orientations = _flat_stream(n=250, rate_hz=200.0)
    with pytest.raises(ValueError, match="same length"):
        prepare_window(samples, orientations[:-1], rate_hz=200.0, window_s=1.0)


def test_prepare_window_rejects_a_span_shorter_than_the_window() -> None:
    samples, orientations = _flat_stream(n=50, rate_hz=200.0)  # 0.25 s, window wants 1 s
    with pytest.raises(ValueError, match="window"):
        prepare_window(samples, orientations, rate_hz=200.0, window_s=1.0)
