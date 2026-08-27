"""Resample, unit-normalise, gravity-align. The one path both training and live take.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md sections 4 and 6.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import ImuSample, OrientationEstimate

    Array = npt.NDArray[np.float64]

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


def align_gravity(
    a_body: Array,
    w_body: Array,
    orientation: OrientationEstimate,
) -> tuple[Array, Array]:
    """Rotate raw body-frame IMU into the gravity-aligned device frame.

    The result has a known down direction but an arbitrary heading -- that is the
    heading-agnostic frame the velocity model regresses into, and it is what makes the
    model robust to how the phone happens to be held (build plan 6.4).

    Returns:
        (a_dev, w_dev), with gravity removed from the linear channel.
    """
    raise NotImplementedError("M1 -- owner: Sristee")


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
