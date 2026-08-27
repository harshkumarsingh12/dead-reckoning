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

_GRAVITY = np.array([0.0, 0.0, 9.80665])  # world ENU: gravity reaction is +z

DEFAULT_RATE_HZ = 200.0
DEFAULT_WINDOW_S = 1.0
DEFAULT_HOP_S = 0.2  # <= 200 ms keeps effective model delay near 0.2 s (build plan 5)


def resample_uniform(
    t_ns: Array,
    values: Array,
    rate_hz: float = DEFAULT_RATE_HZ,
) -> tuple[Array, Array]:
    """Resample an irregularly sampled channel onto a uniform grid.

    Phone IMUs deliver at a nominally fixed rate with real jitter, and the public
    datasets are recorded at different rates entirely. Everything is normalised here so
    the model never sees a rate it was not trained on.

    Args:
        t_ns: (n,) int64 capture timestamps, strictly increasing.
        values: (n, k) samples.
        rate_hz: target rate.

    Returns:
        (t_ns_uniform, values_uniform).
    """
    raise NotImplementedError("M1 -- owner: Sristee")


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

    Args:
        samples: raw IMU covering at least ``window_s`` and ending at now.
        orientations: AHRS output covering the same span.
        rate_hz: target rate.
        window_s: window length in seconds.

    Returns:
        (channels, window_samples) float64, ready for the model.

    Raises:
        ValueError: if the samples do not cover the requested window.
    """
    raise NotImplementedError("M1 -- owner: Sristee")
