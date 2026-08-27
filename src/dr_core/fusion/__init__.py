"""2D error-state Kalman filter and everything that feeds it.

OWNER: Sikruti (backup: Sristee)  |  MILESTONE: M3
Spec: docs/BUILD_PLAN.md section 7 -- read it before touching anything here, the
wiring is specified explicitly and deviating from it silently is how this subsystem
becomes unreviewable.

Reference: Sola, "Quaternion kinematics for the error-state Kalman filter" (2017).
The 2D case here is a simplification of that machinery.
"""

from dr_core.fusion.eskf import Eskf, EskfConfig
from dr_core.fusion.gating import ChiSquareGate, NisLogger
from dr_core.fusion.zupt import StationaryDetector

__all__ = ["ChiSquareGate", "Eskf", "EskfConfig", "NisLogger", "StationaryDetector"]
