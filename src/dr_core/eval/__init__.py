"""The evaluation harness. Built at M0, before there is anything to evaluate.

OWNER: Sikruti (backup: Sumedha)  |  MILESTONE: M0
Spec: docs/BUILD_PLAN.md section 8

Measured, not vibed. "Strong prototype" is a number on a known loop, and the harness
that produces that number exists from milestone zero so nobody is tempted to build the
scoring after seeing the score.

Two families, and both are reported every run:
  * Accuracy -- ATE, RTE, final error, drift %.
  * Honesty  -- NIS, NEES, model calibration coverage.
"""

from dr_core.eval.metrics import ate, drift_pct, final_error, rte
from dr_core.eval.report import RunReport, generate_report

__all__ = ["RunReport", "ate", "drift_pct", "final_error", "generate_report", "rte"]
