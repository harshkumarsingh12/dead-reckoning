"""Unit tests for the classical PDR baseline (fixed-stride step counter).

Spec: docs/BUILD_PLAN.md section 8  |  OWNER: Sristee  |  MILESTONE: M1

No acceptance dataset exists in the repo, so these pin the deterministic behavior:
step detection (count, debounce, threshold), heading-directed advance, the Trajectory
interchange shape, and reset.
"""

from __future__ import annotations

import math

import numpy as np

from dr_core.baselines import PdrTracker
from dr_core.baselines.pdr import PdrConfig
from dr_core.types import ImuSample

GRAVITY = 9.80665
RATE_HZ = 100.0
NS_PER_S = 1_000_000_000


def _stepping_samples(
    step_times_s: list[float], peak: float = 3.0, duration_s: float = 3.0
) -> list[ImuSample]:
    """IMU samples where |a_body| - g spikes to `peak` at each listed time, else 0."""
    n = int(duration_s * RATE_HZ)
    step_idx = {round(t * RATE_HZ) for t in step_times_s}
    samples: list[ImuSample] = []
    for i in range(n):
        s = peak if i in step_idx else 0.0
        samples.append(
            ImuSample(
                t_ns=round(i / RATE_HZ * NS_PER_S),
                a_body=np.array([0.0, 0.0, GRAVITY + s]),
                w_body=np.zeros(3),
            )
        )
    return samples


def _run(
    samples: list[ImuSample], heading_rad: float, config: PdrConfig | None = None
) -> PdrTracker:
    pdr = PdrTracker(config)
    for s in samples:
        pdr.update(s, heading_rad)
    return pdr


def test_counts_clean_footfalls() -> None:
    pdr = _run(_stepping_samples([0.5, 1.0, 1.5, 2.0, 2.5]), heading_rad=0.0)
    assert pdr.step_count == 5


def test_debounces_peaks_within_min_interval() -> None:
    # two peaks 0.1 s apart, below the 0.25 s min interval -> a single step
    pdr = _run(_stepping_samples([0.5, 0.6]), heading_rad=0.0)
    assert pdr.step_count == 1


def test_sub_threshold_peaks_are_not_steps() -> None:
    # peak 1.0 m/s^2 is below the 1.2 threshold -> nothing counts
    pdr = _run(_stepping_samples([0.5, 1.0, 1.5], peak=1.0), heading_rad=0.0)
    assert pdr.step_count == 0


def test_advances_only_on_steps_along_heading() -> None:
    heading = math.radians(30.0)
    config = PdrConfig(step_length_m=0.7)
    pdr = _run(_stepping_samples([0.5, 1.0, 1.5, 2.0]), heading_rad=heading, config=config)
    assert pdr.step_count == 4
    expected = 4 * 0.7 * np.array([math.cos(heading), math.sin(heading)])
    np.testing.assert_allclose(pdr.trajectory.p_world[-1], expected)
    # no footfalls -> never leaves the origin
    still = _run(_stepping_samples([]), heading_rad=heading, config=config)
    assert still.step_count == 0
    np.testing.assert_allclose(still.trajectory.p_world[-1], np.zeros(2))


def test_trajectory_is_a_valid_interchange_trajectory() -> None:
    traj = _run(_stepping_samples([0.5, 1.0]), heading_rad=0.0).trajectory
    assert traj.label == "pdr"
    assert traj.t_ns.dtype == np.int64
    assert traj.p_world.shape == (len(traj.t_ns), 2)
    assert len(traj.t_ns) == len(traj.p_world)


def test_reset_clears_state() -> None:
    pdr = _run(_stepping_samples([0.5, 1.0, 1.5]), heading_rad=0.0)
    assert pdr.step_count == 3
    pdr.reset()
    assert pdr.step_count == 0
    assert len(pdr.trajectory.t_ns) == 0
    assert pdr.trajectory.p_world.shape == (0, 2)
