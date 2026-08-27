"""Double integration of acceleration. The thing this project exists to not do.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 8

Kept deliberately naive -- gravity removed via the AHRS, then integrate twice, no
corrections of any kind. Making it "a bit better" would be dishonest framing and would
weaken the contrast. Expected drift is well over 100%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import ImuSample, OrientationEstimate, Trajectory

    Vec2 = npt.NDArray[np.float64]


class RawIntegrator:
    """Stateful naive integrator, run live alongside the filter as the second dot."""

    def __init__(self) -> None:
        raise NotImplementedError("M1 -- owner: Sristee")

    def update(self, sample: ImuSample, orientation: OrientationEstimate) -> Vec2:
        """Integrate one sample. Returns the current world-frame position estimate."""
        raise NotImplementedError("M1 -- owner: Sristee")

    @property
    def trajectory(self) -> Trajectory:
        """Everything integrated so far, for the post-run plot."""
        raise NotImplementedError("M1 -- owner: Sristee")

    def reset(self) -> None:
        raise NotImplementedError("M1 -- owner: Sristee")
