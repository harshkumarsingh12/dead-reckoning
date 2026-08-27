"""The post-run report, generated the moment a walk ends.

OWNER: Sikruti (plots) + Akshit (presentation)  |  MILESTONE: M0 skeleton, M4 polish
Spec: docs/BUILD_PLAN.md sections 6.8 and 8

Auto-generated on the spot: an error-vs-time strip chart, an error CDF, the
loop-closure error in metres, and the drift percentage, with the PDR and
raw-integration baselines plotted alongside. Having this appear by itself the instant
the presenter stops walking is worth more than any slide, because it is visibly not
prepared in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from dr_core.types import Trajectory


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything measured about one run. Serialises to JSON for the results table."""

    run_id: str
    distance_m: float
    duration_s: float
    ate_m: float
    rte_60s_m: float
    final_error_m: float
    drift_pct: float
    baseline_drift_pct: dict[str, float] = field(default_factory=dict)
    nis_consistent: dict[str, bool] = field(default_factory=dict)
    coverage_1sigma: float | None = None
    inference_ms_median: float | None = None
    notes: str = ""

    def to_json(self) -> str:
        raise NotImplementedError("M0 -- owner: Sikruti")

    def summary_line(self) -> str:
        """One line for the terminal and the group chat. The number people repeat."""
        raise NotImplementedError("M0 -- owner: Sikruti")


def generate_report(
    estimate: Trajectory,
    truth: Trajectory,
    baselines: dict[str, Trajectory],
    output_dir: Path,
    run_id: str,
) -> RunReport:
    """Compute every metric and write the plots.

    Writes into ``output_dir``:
        trajectory.png    estimate, truth, and every baseline on one set of axes
        error_time.png    error vs time strip chart
        error_cdf.png     error CDF
        nis.png           per-channel NIS against its chi-square bounds
        report.json       the RunReport
    """
    raise NotImplementedError("M0 -- owner: Sikruti")
