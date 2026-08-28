"""Model-only trajectory reconstruction: the learned velocity alone, no filter.

OWNER: Sumedha  |  MILESTONE: M2  |  Spec: docs/BUILD_PLAN.md section 8

The third baseline plotted alongside ``dr_core.baselines.RawIntegrator`` (double
integration, the naive failure mode) and ``dr_core.baselines.PdrTracker`` (the classical
alternative): M2 is not "done enough" until integrating THIS beats PDR on held-out data
(docs/EVALUATION.md). Same shape and pattern as those two -- a stateful ``update`` per
sample, a ``trajectory`` property, and a ``reset`` -- so it drops into the same plotting
and eval code with no special-casing.

Free of any torch import, like ``dr_core.models.runtime`` -- this only integrates
``VelocityEstimate``s the ONNX runtime already produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dr_core.preprocess import rotate_dev_to_world
from dr_core.types import Trajectory

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import VelocityEstimate

    Vec2 = npt.NDArray[np.float64]

_NS_PER_S = 1_000_000_000


class ModelOnlyIntegrator:
    """Stateful integrator over the model's own velocity estimates, no filter.

    Unlike ``RawIntegrator`` (which integrates acceleration -- position from a second
    integral) this integrates velocity directly -- one integral -- which is the whole
    point: the learned model already IS the bounded-error replacement for double
    integration, so proving it beats PDR requires nothing more than accumulating its
    output honestly.
    """

    def __init__(self) -> None:
        self._t_prev: int | None = None
        self._p_world = np.zeros(2)
        self._t_ns: list[int] = []
        self._positions: list[Vec2] = []
        self._headings: list[float] = []

    def update(self, estimate: VelocityEstimate, psi_rad: float) -> Vec2:
        """Integrate one velocity estimate.

        Args:
            estimate: the model's device-frame velocity for this instant (from
                ``VelocityModelRuntime.predict``).
            psi_rad: the heading at this same instant (from the AHRS), used to rotate
                ``v_dev`` back into world ENU for plotting -- the same convention
                ``prepare_window`` and ``dr_core.fusion.eskf``'s device-frame update
                share (docs/CONVENTIONS.md section 1).

        Returns:
            The current world-frame position estimate.
        """
        v_world = rotate_dev_to_world(estimate.v_dev, psi_rad)

        dt = 0.0 if self._t_prev is None else (estimate.t_ns - self._t_prev) / _NS_PER_S
        self._t_prev = estimate.t_ns

        self._p_world = self._p_world + v_world * dt

        self._t_ns.append(estimate.t_ns)
        self._positions.append(self._p_world.copy())
        self._headings.append(psi_rad)
        return self._p_world.copy()

    @property
    def trajectory(self) -> Trajectory:
        """Everything integrated so far, for the post-run plot and the drift-%
        comparison against PDR."""
        return Trajectory(
            t_ns=np.array(self._t_ns, dtype=np.int64),
            p_world=np.array(self._positions, dtype=np.float64).reshape(-1, 2),
            psi_rad=np.array(self._headings, dtype=np.float64),
            label="model_only",
        )

    def reset(self) -> None:
        self._t_prev = None
        self._p_world = np.zeros(2)
        self._t_ns = []
        self._positions = []
        self._headings = []
