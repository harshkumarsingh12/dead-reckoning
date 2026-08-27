"""The two baselines that make the result legible.

OWNER: Sristee (backup: Sikruti)  |  MILESTONE: M1
Spec: docs/BUILD_PLAN.md section 8

These are not throwaway comparisons. Raw double integration spiralling off the map as a
second dot, live, alongside the tracked one is the single most persuasive thing on
screen: it shows the problem being solved rather than asserting it was. PDR is the
honest classical alternative and the number the learned model has to beat.

Both are plotted alongside the fused result in every evaluation run, always.
"""

from dr_core.baselines.pdr import PdrTracker
from dr_core.baselines.raw_integration import RawIntegrator

__all__ = ["PdrTracker", "RawIntegrator"]
