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

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dr_core.eval.metrics import ate, drift_pct, final_error, resample_to, rte

if TYPE_CHECKING:
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
        return json.dumps(dataclasses.asdict(self), indent=2)

    def summary_line(self) -> str:
        """One line for the terminal and the group chat. The number people repeat."""
        return (
            f"[{self.run_id}] drift={self.drift_pct:.2f}% | ATE={self.ate_m:.2f}m | "
            f"RTE(60s)={self.rte_60s_m:.2f}m | final_err={self.final_error_m:.2f}m | "
            f"dist={self.distance_m:.1f}m"
        )


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
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_r = resample_to(truth, estimate.t_ns)
    diffs = np.diff(truth_r.p_world, axis=0)
    distance_m = float(np.sum(np.linalg.norm(diffs, axis=1)))
    duration_s = float((estimate.t_ns[-1] - estimate.t_ns[0]) * 1e-9) if len(estimate) > 1 else 0.0

    ate_m = ate(estimate, truth, align=True)
    rte_60s_m = rte(estimate, truth, window_s=60.0)
    final_error_m = final_error(estimate, truth)
    drift_percent = drift_pct(estimate, truth)
    baseline_drift = {name: drift_pct(base, truth) for name, base in baselines.items()}

    report = RunReport(
        run_id=run_id,
        distance_m=distance_m,
        duration_s=duration_s,
        ate_m=ate_m,
        rte_60s_m=rte_60s_m,
        final_error_m=final_error_m,
        drift_pct=drift_percent,
        baseline_drift_pct=baseline_drift,
    )

    # Plot generation
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Trajectory plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        truth_r.p_world[:, 0],
        truth_r.p_world[:, 1],
        "k--",
        label=f"Truth ({truth.label})",
        linewidth=2,
    )
    ax.plot(
        estimate.p_world[:, 0],
        estimate.p_world[:, 1],
        "b-",
        label=f"Estimate ({estimate.label})",
        linewidth=2,
    )
    for name, base in baselines.items():
        base_r = resample_to(base, estimate.t_ns)
        ax.plot(
            base_r.p_world[:, 0],
            base_r.p_world[:, 1],
            ":",
            label=f"Baseline: {name}",
            alpha=0.7,
        )
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Trajectory Comparison — {run_id}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend()
    fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Error vs Time plot
    errors = np.linalg.norm(estimate.p_world - truth_r.p_world, axis=1)
    t_s = (estimate.t_ns - estimate.t_ns[0]) * 1e-9
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_s, errors, "r-", label="Position Error (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Position Error vs Time — {run_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.legend()
    fig.savefig(output_dir / "error_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Error CDF plot
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sorted_errors, cdf, "g-", linewidth=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Position Error CDF — {run_id}")
    ax.set_xlabel("Position Error (m)")
    ax.set_ylabel("Cumulative Probability")
    fig.savefig(output_dir / "error_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. NIS plot placeholder
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "Per-channel NIS logged during live / replay session",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_title(f"NIS Consistency — {run_id}")
    fig.savefig(output_dir / "nis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. JSON report
    (output_dir / "report.json").write_text(report.to_json(), encoding="utf-8")

    return report
