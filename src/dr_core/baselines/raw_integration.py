"""Double integration of acceleration. The thing this project exists to not do.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 8

Kept deliberately naive -- gravity removed via the AHRS, then integrate twice, no
corrections of any kind. Making it "a bit better" would be dishonest framing and would
weaken the contrast. Expected drift is well over 100%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dr_core.preprocess import align_gravity
from dr_core.types import Trajectory

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import ImuSample, OrientationEstimate

    Vec2 = npt.NDArray[np.float64]

_NS_PER_S = 1_000_000_000


class RawIntegrator:
    """Stateful naive integrator, run live alongside the filter as the second dot.

    Deliberately naive: gravity is removed via the AHRS orientation (through the shared
    ``align_gravity``), then acceleration is integrated twice with no correction of any
    kind. It exists to show the drift the rest of the system defeats; making it "better"
    would be dishonest framing (AGENTS.md).
    """

    def __init__(self) -> None:
        self._t_prev: int | None = None
        self._v_world = np.zeros(2)
        self._p_world = np.zeros(2)
        self._t_ns: list[int] = []
        self._positions: list[npt.NDArray[np.float64]] = []
        self._headings: list[float] = []

    def update(self, sample: ImuSample, orientation: OrientationEstimate) -> Vec2:
        """Integrate one sample. Returns the current world-frame position estimate."""
        a_world, _w_world = align_gravity(sample.a_body, sample.w_body, orientation)

        dt = 0.0 if self._t_prev is None else (sample.t_ns - self._t_prev) / _NS_PER_S
        self._t_prev = sample.t_ns

        self._v_world = self._v_world + a_world[:2] * dt
        self._p_world = self._p_world + self._v_world * dt

        self._t_ns.append(sample.t_ns)
        self._positions.append(self._p_world.copy())
        q = np.asarray(orientation.q_world_body, dtype=np.float64)
        w, x, y, z = (float(c) for c in q)
        self._headings.append(float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        return self._p_world.copy()

    @property
    def trajectory(self) -> Trajectory:
        """Everything integrated so far, for the post-run plot."""
        return Trajectory(
            t_ns=np.array(self._t_ns, dtype=np.int64),
            p_world=np.array(self._positions, dtype=np.float64).reshape(-1, 2),
            psi_rad=np.array(self._headings, dtype=np.float64),
            label="raw_integration",
        )

    def reset(self) -> None:
        self._t_prev = None
        self._v_world = np.zeros(2)
        self._p_world = np.zeros(2)
        self._t_ns = []
        self._positions = []
        self._headings = []
