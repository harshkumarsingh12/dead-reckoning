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


def prepare_window(
    samples: list[ImuSample],
    orientations: list[OrientationEstimate],
    rate_hz: float = DEFAULT_RATE_HZ,
    window_s: float = DEFAULT_WINDOW_S,
) -> Array:
    """Turn a causal window of raw samples into a model input tensor.

    The window ENDS at the current instant -- no future context. That is what keeps the
    effective model delay near 0.2 s instead of the 0.5-1.0 s a centred window costs.

    Each sample is rotated by ``align_gravity`` (the SAME function the raw-integration
    baseline uses -- one shared preprocessing path, not a second implementation to drift
    from). That yields z-true-Up, gravity-removed accel and gyro; the horizontal axes
    carry whatever heading the AHRS orientation happens to have at that instant, which is
    exactly the model's device-aligned ``_dev`` frame (docs/CONVENTIONS.md section 1):
    heading-agnostic not because heading is zeroed out, but because the model is trained
    (via ``augment_random_yaw``) to be indifferent to whatever it is.

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

    a_dev = np.empty((len(samples), 3), dtype=np.float64)
    w_dev = np.empty((len(samples), 3), dtype=np.float64)
    for i, (sample, orientation) in enumerate(zip(samples, orientations, strict=True)):
        a_dev[i], w_dev[i] = align_gravity(sample.a_body, sample.w_body, orientation)

    t_uniform_ns, a_uniform = resample_uniform(t_ns, a_dev, rate_hz=rate_hz)
    _, w_uniform = resample_uniform(t_ns, w_dev, rate_hz=rate_hz)

    window_samples = round(window_s * rate_hz)
    if t_uniform_ns.shape[0] < window_samples:
        raise ValueError(
            f"only {t_uniform_ns.shape[0]} resampled samples available, need "
            f"{window_samples} for a {window_s:.3f} s window at {rate_hz:.1f} Hz"
        )

    a_window = a_uniform[-window_samples:]
    w_window = w_uniform[-window_samples:]
    return np.concatenate([a_window.T, w_window.T], axis=0)
