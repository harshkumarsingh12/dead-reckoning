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
from scipy import stats

from dr_core.eval.metrics import _umeyama_se2, ate, drift_pct, final_error, resample_to, rte
from dr_core.fusion.gating import nees

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.fusion.gating import NisLogger
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
    nees_mean: float | None = None
    nees_consistent: bool | None = None
    coverage_1sigma: float | None = None
    inference_ms_median: float | None = None
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    def summary_line(self) -> str:
        """One line for the terminal and the group chat. The number people repeat."""
        nees_str = f" | NEES={self.nees_mean:.2f}" if self.nees_mean is not None else ""
        return (
            f"[{self.run_id}] drift={self.drift_pct:.2f}% | ATE={self.ate_m:.2f}m | "
            f"RTE(60s)={self.rte_60s_m:.2f}m | final_err={self.final_error_m:.2f}m | "
            f"dist={self.distance_m:.1f}m{nees_str}"
        )


def generate_report(
    estimate: Trajectory,
    truth: Trajectory,
    baselines: dict[str, Trajectory],
    output_dir: Path,
    run_id: str,
    nis_logger: NisLogger | None = None,
    cov_history: list[npt.NDArray[np.float64]] | None = None,
) -> RunReport:
    """Compute every metric and write the plots.

    Writes into ``output_dir``:
        trajectory.png    estimate, truth, and every baseline on one set of axes
        error_time.png    error vs time strip chart (aligned to match ATE)
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
    nis_consistent = nis_logger.is_consistent() if nis_logger is not None else {}

    # Compute offline NEES if covariance history is provided
    nees_mean: float | None = None
    nees_consistent: bool | None = None
    nees_history: list[float] = []
    if cov_history is not None and len(cov_history) == len(estimate) and len(estimate) > 0:
        for i in range(len(estimate)):
            err = estimate.p_world[i] - truth_r.p_world[i]
            cov = cov_history[i]
            cov_pos = cov[0:2, 0:2] if cov.shape == (7, 7) else cov
            nees_history.append(nees(err, cov_pos))

        if nees_history:
            n_samples = len(nees_history)
            nees_mean = float(np.mean(nees_history))
            low = float(stats.chi2.ppf(0.025, df=n_samples * 2) / n_samples)
            high = float(stats.chi2.ppf(0.975, df=n_samples * 2) / n_samples)
            nees_consistent = low <= nees_mean <= high

    report = RunReport(
        run_id=run_id,
        distance_m=distance_m,
        duration_s=duration_s,
        ate_m=ate_m,
        rte_60s_m=rte_60s_m,
        final_error_m=final_error_m,
        drift_pct=drift_percent,
        baseline_drift_pct=baseline_drift,
        nis_consistent=nis_consistent,
        nees_mean=nees_mean,
        nees_consistent=nees_consistent,
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

    # 2. Error vs Time plot (SE(2) aligned to match ATE)
    est_p_aligned = estimate.p_world.copy()
    if len(estimate) >= 2:
        rot, trans = _umeyama_se2(est_p_aligned, truth_r.p_world)
        est_p_aligned = (rot @ est_p_aligned.T).T + trans
    errors = np.linalg.norm(est_p_aligned - truth_r.p_world, axis=1)

    t_s = (estimate.t_ns - estimate.t_ns[0]) * 1e-9 if len(estimate) > 0 else np.array([])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_s, errors, "r-", label="Aligned Position Error (m)")
    ax.axhline(ate_m, color="k", linestyle="--", alpha=0.7, label=f"ATE = {ate_m:.2f} m")
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

    # 4. NIS plot
    fig, ax = plt.subplots(figsize=(10, 5))
    has_nis_data = False
    if nis_logger is not None:
        for ch, ch_stats in nis_logger._channels.items():
            if ch_stats.nis_history:
                has_nis_data = True
                samples = np.arange(len(ch_stats.nis_history))
                ax.plot(
                    samples, ch_stats.nis_history, label=f"{ch} (dof={ch_stats.dof})", alpha=0.7
                )
                _low, high = ch_stats.bounds
                ax.axhline(high, linestyle=":", alpha=0.5, label=f"{ch} 95% bound ({high:.2f})")

    if not has_nis_data:
        ax.text(
            0.5,
            0.5,
            "No per-channel NIS telemetry recorded for this run",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"NIS Consistency — {run_id}")
    ax.set_xlabel("Measurement Index")
    ax.set_ylabel("Normalized Innovation Squared")
    if has_nis_data:
        ax.legend()
    fig.savefig(output_dir / "nis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. NEES plot (if covariance history provided)
    if cov_history is not None and nees_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        samples = np.arange(len(nees_history))
        ax.plot(samples, nees_history, "m-", label="Position NEES (dof=2)", alpha=0.7)
        low_bound = float(stats.chi2.ppf(0.025, df=len(nees_history) * 2) / len(nees_history))
        high_bound = float(stats.chi2.ppf(0.975, df=len(nees_history) * 2) / len(nees_history))
        ax.axhline(
            high_bound, color="r", linestyle=":", alpha=0.7, label=f"97.5% Bound ({high_bound:.2f})"
        )
        ax.axhline(
            low_bound, color="b", linestyle=":", alpha=0.7, label=f"2.5% Bound ({low_bound:.2f})"
        )
        if nees_mean is not None:
            ax.axhline(
                nees_mean,
                color="k",
                linestyle="--",
                alpha=0.6,
                label=f"Mean NEES = {nees_mean:.2f}",
            )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(f"NEES Consistency (Offline) — {run_id}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Normalized Estimation Error Squared")
        ax.legend()
        fig.savefig(output_dir / "nees.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 6. JSON report
    (output_dir / "report.json").write_text(report.to_json(), encoding="utf-8")

    return report
