"""Classical pedestrian dead reckoning: step count x step length, plus heading.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 8

The honest classical alternative, and the number the learned model must beat before
anyone can claim the ML is earning its place. Target for M1 is under 10% drift on the
test loop; if PDR alone gets there, that is a useful and slightly uncomfortable data
point worth reporting rather than burying.

Shares the AHRS heading with everything else, so a heading bug shows up here too --
which is a feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import ImuSample, Trajectory

    Vec2 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PdrConfig:
    """Step detection and stride model parameters."""

    min_step_interval_s: float = 0.25  # rejects double-counting one footfall
    accel_peak_threshold: float = 1.2  # m/s^2 above the gravity-removed baseline
    # Weinberg stride estimate: k * (a_max - a_min) ** 0.25. Beats a fixed length
    # because it tracks walking speed, and calibrating k on one GPS walk is cheap.
    weinberg_k: float = 0.5
    fixed_step_length_m: float | None = None  # set to bypass Weinberg entirely


class PdrTracker:
    """Step-and-heading dead reckoning."""

    def __init__(self, config: PdrConfig | None = None) -> None:
        raise NotImplementedError("M1 -- owner: Sristee")

    def update(self, sample: ImuSample, heading_rad: float) -> Vec2:
        """Feed one sample. Position advances only when a step is detected."""
        raise NotImplementedError("M1 -- owner: Sristee")

    def calibrate_stride(self, gps_distance_m: float, step_count: int) -> float:
        """Fit the stride constant from one GPS-tracked outdoor walk.

        Returns:
            The fitted ``weinberg_k``.
        """
        raise NotImplementedError("M1 -- owner: Sristee")

    @property
    def step_count(self) -> int:
        raise NotImplementedError("M1 -- owner: Sristee")

    @property
    def trajectory(self) -> Trajectory:
        raise NotImplementedError("M1 -- owner: Sristee")
