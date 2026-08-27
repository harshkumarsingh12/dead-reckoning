"""THE shared preprocessing module. Training and live BOTH import this.

OWNER: Sristee (backup: Sumedha)  |  MILESTONE: M1
Spec: docs/BUILD_PLAN.md sections 4 and 6.2

Any mismatch between how training data and live data are prepared silently degrades the
model, and the degradation is invisible until the live demo underperforms with no
explanation. Sharing one code path makes the mismatch impossible.

Rule: if you find yourself writing a resample, a unit conversion, or a gravity
alignment anywhere else in this repository, you are creating the bug this module
exists to prevent. Put it here instead.
"""

from dr_core.preprocess.calibrate import (
    CalibrationResult,
    estimate_gyro_bias,
    fit_hard_iron,
)
from dr_core.preprocess.pipeline import (
    align_gravity,
    prepare_window,
    resample_uniform,
)

__all__ = [
    "CalibrationResult",
    "align_gravity",
    "estimate_gyro_bias",
    "fit_hard_iron",
    "prepare_window",
    "resample_uniform",
]
