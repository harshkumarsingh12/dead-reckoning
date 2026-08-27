"""Orientation, and the magnetometer gate that keeps it honest indoors.

OWNER: Sristee (backup: Sumedha)  |  MILESTONE: M1
Spec: docs/BUILD_PLAN.md section 6.3

Orientation comes first: nothing downstream can run without knowing which way is down
and which way is forward. Uses ``imufusion`` (Madgwick) rather than a hand-rolled
filter -- see docs/AGENTS.md, the library is battle-tested and this is not where the
project differentiates.
"""

from dr_core.ahrs.filter import AhrsFilter
from dr_core.ahrs.mag_gate import MagGate, MagGateConfig

__all__ = ["AhrsFilter", "MagGate", "MagGateConfig"]
