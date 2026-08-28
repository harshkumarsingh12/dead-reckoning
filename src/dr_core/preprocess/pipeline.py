"""Resample, unit-normalise, gravity-align. The one path both training and live take.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md sections 4 and 6.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import ImuSample, OrientationEstimate

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

_GRAVITY = np.array([0.0, 0.0, 9.80665])  # world ENU: gravity reaction is +z

DEFAULT_RATE_HZ = 200.0
DEFAULT_WINDOW_S = 1.0
DEFAULT_HOP_S = 0.2  # <= 200 ms keeps effective model delay near 0.2 s (build plan 5)


def resample_uniform(
    t_ns: IntArray,
    values: Array,
    rate_hz: float = DEFAULT_RATE_HZ,
) -> tuple[IntArray, Array]:
    """Resample an irregularly sampled channel onto a uniform grid.

    Phone IMUs deliver at a nominally fixed rate with real jitter, and the public
    datasets are recorded at different rates entirely. Everything is normalised here so
    the model never sees a rate it was not trained on.

    Linear interpolation, per channel, onto a grid starting at ``t_ns[0]`` spaced by
    ``1 / rate_hz`` seconds and not extrapolating past ``t_ns[-1]``.

    Args:
        t_ns: (n,) int64 capture timestamps, strictly increasing.
        values: (n, k) samples.
        rate_hz: target rate.

    Returns:
        (t_ns_uniform, values_uniform).

    Raises:
        ValueError: fewer than two input samples, so there is no interval to resample.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if t_ns.shape[0] < 2:
        raise ValueError(f"need at least two samples to resample, got {t_ns.shape[0]}")

    period_ns = round(1.0e9 / rate_hz)
    t_start_ns, t_end_ns = int(t_ns[0]), int(t_ns[-1])
    n_out = int((t_end_ns - t_start_ns) // period_ns) + 1
    t_uniform_ns = t_start_ns + np.arange(n_out, dtype=np.int64) * period_ns

    t_src = t_ns.astype(np.float64)
    t_dst = t_uniform_ns.astype(np.float64)
    values_uniform = np.column_stack(
        [np.interp(t_dst, t_src, values[:, k]) for k in range(values.shape[1])]
    ).astype(np.float64)

    return t_uniform_ns, values_uniform


def _rotation_body_to_world(q_world_body: Array) -> Array:
    """Rotation matrix that takes a body-frame vector to world, from a (w, x, y, z) quat."""
    w, x, y, z = (float(c) for c in q_world_body)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def align_gravity(
    a_body: Array,
    w_body: Array,
    orientation: OrientationEstimate,
) -> tuple[Array, Array]:
    """Rotate raw body-frame IMU into the world ENU frame and remove gravity.

    The accelerometer measures specific force -- linear acceleration plus the gravity
    reaction. Rotated into world by the AHRS orientation and with the constant gravity
    reaction subtracted, what remains is the true linear acceleration in world ENU, which
    is what the raw integrator (and any world-frame consumer) integrates.

    Gravity removal is centralised here on purpose: doing it anywhere else is the exact
    train/live drift the shared module exists to prevent (AGENTS.md rule on preprocessing).

    Note: the velocity model wants this in a *heading-agnostic* device frame rather than
    world; that yaw-removal is applied downstream in ``prepare_window``, not here.

    Returns:
        (a_world, w_world): linear acceleration and angular rate in world ENU, gravity
        removed from the linear channel.
    """
    r_wb = _rotation_body_to_world(np.asarray(orientation.q_world_body, dtype=np.float64))
    a_world = r_wb @ np.asarray(a_body, dtype=np.float64) - _GRAVITY
    w_world = r_wb @ np.asarray(w_body, dtype=np.float64)
    return a_world, w_world


def heading_rad(q_world_body: Array) -> float:
    """Yaw of a world<-body quaternion: 0 = East, CCW positive (docs/CONVENTIONS.md).

    Shared by ``prepare_window`` (to find "now"'s heading, for the device-aligned
    rotation below) and by anything else that needs the same number
    ``dr_core.ahrs.AhrsFilter.heading_rad`` reports live -- one formula, not two.
    """
    w, x, y, z = (float(c) for c in q_world_body)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rotate_world_to_dev(vector_world: Array, psi_rad: float) -> Array:
    """Rotate a planar world-frame vector into the device-aligned frame at heading psi.

    ``v_dev = R(-psi) @ v_world`` -- the exact convention the ESKF's device-frame
    velocity measurement model assumes (build plan section 7.1: ``h(x) = (1/s) *
    R(-psi) * v_world``) and the one ``prepare_window`` uses internally for the IMU
    channels. Shared so a training-time velocity LABEL and a live filter update can
    never disagree on which way "device-aligned" rotates.
    """
    c, s = np.cos(-psi_rad), np.sin(-psi_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    result: Array = rot @ np.asarray(vector_world, dtype=np.float64)
    return result


def rotate_dev_to_world(vector_dev: Array, psi_rad: float) -> Array:
    """Rotate a planar device-aligned vector back into world ENU: the inverse of
    ``rotate_world_to_dev`` (docs/CONVENTIONS.md section 1: ``v_world = R(psi) @
    v_dev``). What a model-only integrator needs to turn the network's raw output
    back into a plottable world-frame trajectory.
    """
    c, s = np.cos(psi_rad), np.sin(psi_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    result: Array = rot @ np.asarray(vector_dev, dtype=np.float64)
    return result


def _rotate_horizontal_by(vectors: Array, psi_rad: float) -> Array:
    """Rotate the horizontal (x, y) columns of an (n, 3) world-frame array by -psi.

    The vertical (Up, column 2) component is invariant under a rotation about the
    vertical axis and is passed through unchanged. Same convention as
    ``rotate_world_to_dev``, batched over many samples instead of one vector.
    """
    c, s = np.cos(-psi_rad), np.sin(-psi_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    out = np.array(vectors, dtype=np.float64, copy=True)
    out[:, 0:2] = (rot @ out[:, 0:2].T).T
    return out


def prepare_window(
    samples: list[ImuSample],
    orientations: list[OrientationEstimate],
    rate_hz: float = DEFAULT_RATE_HZ,
    window_s: float = DEFAULT_WINDOW_S,
) -> Array:
    """Turn a causal window of raw samples into a model input tensor.

    The window ENDS at the current instant -- no future context. That is what keeps the
    effective model delay near 0.2 s instead of the 0.5-1.0 s a centred window costs.

    Two steps, deliberately kept separate (see ``align_gravity``'s docstring):

    1. Every sample is rotated by ``align_gravity`` (the SAME function the
       raw-integration baseline uses -- one shared preprocessing path, not a second
       implementation to drift from) into world ENU, gravity removed.
    2. The whole window is then rotated by ``-psi_now`` about Up, where ``psi_now`` is
       the heading at the window's LAST (current) sample -- a single, constant rotation
       applied uniformly across the window, not a per-sample one, so a window keeps one
       internally consistent frame even while the person is turning mid-window. This is
       the model's device-aligned ``_dev`` frame (docs/CONVENTIONS.md section 1):
       heading-agnostic not because heading is zeroed out, but because ``psi_now``
       varies window to window (and training additionally scrambles it via
       ``augment_random_yaw``), so the network cannot learn a fixed absolute heading.

    Args:
        samples: raw IMU covering at least ``window_s`` and ending at now.
        orientations: AHRS output covering the same span, one per sample.
        rate_hz: target rate.
        window_s: window length in seconds.

    Returns:
        (channels, window_samples) float64 -- accel (3) then gyro (3), ready for the
        model.

    Raises:
        ValueError: if the samples do not cover the requested window, or if
            ``samples`` and ``orientations`` are not the same length.
    """
    if len(samples) != len(orientations):
        raise ValueError(
            f"samples ({len(samples)}) and orientations ({len(orientations)}) "
            "must be the same length"
        )
    if len(samples) < 2:
        raise ValueError(f"need at least two samples to prepare a window, got {len(samples)}")

    t_ns = np.array([s.t_ns for s in samples], dtype=np.int64)
    span_s = float(t_ns[-1] - t_ns[0]) / 1.0e9
    if span_s < window_s:
        raise ValueError(f"samples span {span_s:.3f} s, less than the {window_s:.3f} s window")

    a_world = np.empty((len(samples), 3), dtype=np.float64)
    w_world = np.empty((len(samples), 3), dtype=np.float64)
    for i, (sample, orientation) in enumerate(zip(samples, orientations, strict=True)):
        a_world[i], w_world[i] = align_gravity(sample.a_body, sample.w_body, orientation)

    t_uniform_ns, a_uniform = resample_uniform(t_ns, a_world, rate_hz=rate_hz)
    _, w_uniform = resample_uniform(t_ns, w_world, rate_hz=rate_hz)

    window_samples = round(window_s * rate_hz)
    if t_uniform_ns.shape[0] < window_samples:
        raise ValueError(
            f"only {t_uniform_ns.shape[0]} resampled samples available, need "
            f"{window_samples} for a {window_s:.3f} s window at {rate_hz:.1f} Hz"
        )

    psi_now = heading_rad(np.asarray(orientations[-1].q_world_body, dtype=np.float64))
    a_window = _rotate_horizontal_by(a_uniform[-window_samples:], psi_now)
    w_window = _rotate_horizontal_by(w_uniform[-window_samples:], psi_now)
    return np.concatenate([a_window.T, w_window.T], axis=0)
