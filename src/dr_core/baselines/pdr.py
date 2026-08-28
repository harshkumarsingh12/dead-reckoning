"""Classical pedestrian dead reckoning: step count x fixed step length, plus heading.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 8

The honest classical alternative -- the number the learned model must beat. Kept as
simple as the raw-integration baseline: a fixed-stride step counter, no per-run GPS
tuning. A tuned baseline would make "does the model beat PDR" hinge on how well PDR was
tuned, which defeats the comparison. Shares the AHRS heading with everything else, so a
heading bug shows up here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dr_core.types import Trajectory

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import ImuSample

    Vec2 = npt.NDArray[np.float64]

_GRAVITY = 9.80665
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class PdrConfig:
    """Step detection and stride parameters."""

    min_step_interval_s: float = 0.25  # rejects double-counting one footfall
    accel_peak_threshold: float = 1.2  # m/s^2 above the gravity-removed baseline
    step_length_m: float = 0.7  # fixed stride; honest baselines do not tune per run


class PdrTracker:
    """Step-and-heading dead reckoning with a fixed stride."""

    def __init__(self, config: PdrConfig | None = None) -> None:
        self._config = config if config is not None else PdrConfig()
        self._position = np.zeros(2)
        self._step_count = 0
        self._prev_above = False
        self._last_step_t_ns: int | None = None
        self._t_ns: list[int] = []
        self._positions: list[npt.NDArray[np.float64]] = []
        self._headings: list[float] = []

    def update(self, sample: ImuSample, heading_rad: float) -> Vec2:
        """Feed one sample. Position advances only when a step is detected."""
        signal = float(np.linalg.norm(sample.a_body)) - _GRAVITY
        above = signal > self._config.accel_peak_threshold
        rising_edge = above and not self._prev_above
        self._prev_above = above

        if rising_edge:
            interval_ns = self._config.min_step_interval_s * _NS_PER_S
            far_enough = (
                self._last_step_t_ns is None or (sample.t_ns - self._last_step_t_ns) >= interval_ns
            )
            if far_enough:
                self._step_count += 1
                self._last_step_t_ns = sample.t_ns
                stride = self._config.step_length_m * np.array(
                    [np.cos(heading_rad), np.sin(heading_rad)]
                )
                self._position = self._position + stride

        self._t_ns.append(sample.t_ns)
        self._positions.append(self._position.copy())
        self._headings.append(heading_rad)
        return self._position.copy()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(
            t_ns=np.array(self._t_ns, dtype=np.int64),
            p_world=np.array(self._positions, dtype=np.float64).reshape(-1, 2),
            psi_rad=np.array(self._headings, dtype=np.float64),
            label="pdr",
        )

    def reset(self) -> None:
        self._position = np.zeros(2)
        self._step_count = 0
        self._prev_above = False
        self._last_step_t_ns = None
        self._t_ns = []
        self._positions = []
        self._headings = []
