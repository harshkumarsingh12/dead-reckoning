"""Dataset loaders. Everything lands as a Trajectory plus ImuSample sequences.

OWNER: Sumedha (backup: Sristee)  |  MILESTONE: M0
Spec: docs/BUILD_PLAN.md section 4

Data is the make-or-break, and dataset ACCESS is the long pole -- RoNIN and OxIOD both
require a request that can take days to approve. That request goes out on day one, in
parallel with everything else, because no amount of clever modelling recovers a week
lost waiting for an email.

Two hard rules, both of which exist because breaking them makes the metrics lie:

  1. SPLIT BY TRAJECTORY, never by window. Windows from one walk share so much context
     that a window-level split leaks the answer and every number afterwards is fiction.
  2. Every loader returns data that has been through ``dr_core.preprocess``. No loader
     does its own resampling.
"""

from dr_core.datasets.loaders import (
    Recording,
    load_own_recording,
    load_oxiod,
    load_ronin,
    split_by_trajectory,
)

__all__ = [
    "Recording",
    "load_own_recording",
    "load_oxiod",
    "load_ronin",
    "split_by_trajectory",
]
