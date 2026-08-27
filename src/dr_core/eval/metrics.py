"""ATE, RTE, drift, and the model calibration coverage test.

OWNER: Sikruti  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 8

Targets on a 100-300 m indoor loop:

    metric                      acceptable        strong
    drift (final / distance)    < 5%              < 2-3%
    RTE over 60 s               a few metres      1-2 m
    NIS / NEES                  within bounds     across carry positions
    model coverage at 1 sigma   ~68%              holds across carry positions
    inference per window        < 10 ms           on-device viable
    raw-integration baseline    > 100%            (contrast, not a target)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import Trajectory

    Array = npt.NDArray[np.float64]


def ate(estimate: Trajectory, truth: Trajectory, align: bool = True) -> float:
    """Absolute Trajectory Error: RMSE of position after alignment.

    Args:
        estimate: the estimated path.
        truth: ground truth. Resampled onto the estimate's timestamps internally.
        align: apply an SE(2) alignment first. Standard for ATE, but report the
            unaligned number too when the start point is genuinely known -- in this
            demo it is, since the walk starts on a marked spot.

    Returns:
        RMSE in metres.
    """
    raise NotImplementedError("M0 -- owner: Sikruti")


def rte(estimate: Trajectory, truth: Trajectory, window_s: float = 60.0) -> float:
    """Relative Trajectory Error over a fixed window.

    Reflects the drift RATE rather than accumulated error, so it stays comparable
    across runs of different lengths. This is the number to quote when comparing two
    models on differently sized loops.
    """
    raise NotImplementedError("M0 -- owner: Sikruti")


def final_error(estimate: Trajectory, truth: Trajectory) -> float:
    """Distance between the estimated and true endpoints, metres.

    On a closed loop this is the loop-closure error -- the closing shot of the demo.
    """
    raise NotImplementedError("M0 -- owner: Sikruti")


def drift_pct(estimate: Trajectory, truth: Trajectory) -> float:
    """final_error / distance travelled, as a percentage. The headline number."""
    raise NotImplementedError("M0 -- owner: Sikruti")


def calibration_coverage(errors: Array, sigmas: Array, k: float = 1.0) -> float:
    """Fraction of held-out errors falling inside k sigma, per axis.

    MANDATORY before the model's covariance is allowed anywhere near the filter's R.
    A well-calibrated 1-sigma lands near 0.68. Materially below that means the model
    is over-confident, which silently poisons fusion and makes the on-screen ellipse
    indefensible; materially above means it is under-confident and the filter is
    ignoring information it has.
    """
    raise NotImplementedError("M2 -- owner: Sumedha")


def resample_to(trajectory: Trajectory, t_ns: Array) -> Trajectory:
    """Interpolate a trajectory onto a given timebase. Used by every metric above."""
    raise NotImplementedError("M0 -- owner: Sikruti")
