#!/usr/bin/env python
"""Evaluate a trained velocity model against PDR and raw integration. Needs [ml].

OWNER: Sumedha  |  MILESTONE: M2

    python scripts/evaluate_model.py --checkpoint models/tcn.pt \
        --data data/own/outdoor --oxiod-data data/oxiod \
        --oxiod-carry-positions handheld pocket --out reports/m2_eval.json

Reconstructs the SAME held-out split scripts/train.py produced (identical recordings,
identical seed, via dr_core.datasets.load_combined_recordings + split_by_trajectory --
never re-derive a split by hand, or "held-out" stops meaning anything) and, for every
held-out recording:

  * integrates the model's own predictions (dr_core.models.ModelOnlyIntegrator)
  * integrates the classical PDR baseline (dr_core.baselines.PdrTracker)
  * integrates raw double integration (dr_core.baselines.RawIntegrator) -- "always
    plotted alongside", per dr_core.baselines' own module docstring
  * scores all three against ground truth (dr_core.eval.metrics: ATE, RTE, drift %)
  * pools the model's per-window velocity residuals into a calibration-coverage number

This is what M2's "done when" actually means: model-only beats PDR on held-out data,
coverage lands near 68%. Nothing here is claimed until it is measured -- a missed
target is printed and written to the report exactly as measured, not hidden
(AGENTS.md: "do not invent numbers").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from dr_core.baselines import PdrTracker, RawIntegrator
from dr_core.datasets import (
    Recording,
    load_combined_recordings,
    orientations_for_recording,
    split_by_trajectory,
    windows_for_recording,
)
from dr_core.eval.metrics import ate, calibration_coverage, drift_pct, final_error, rte
from dr_core.models import ModelOnlyIntegrator
from dr_core.models.runtime import _cholesky_output_to_cov
from dr_core.models.tcn import build_model
from dr_core.preprocess import DEFAULT_HOP_S, DEFAULT_RATE_HZ, heading_rad
from dr_core.types import VelocityEstimate

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

EXIT_OK = 0
EXIT_USAGE = 2


def _evaluate_recording(
    recording: Recording, model: torch.nn.Module, window_s: float, hop_s: float, rate_hz: float
) -> dict[str, object] | None:
    if recording.truth is None or len(recording.truth) < 2 or len(recording.imu) < 2:
        return None

    orientations = orientations_for_recording(recording, rate_hz)
    windows, targets, window_t_ns = windows_for_recording(
        recording, orientations, window_s, hop_s, rate_hz
    )
    if windows.shape[0] == 0:
        return None

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows.astype(np.float32))
        pred = model(x)[:, :, -1].numpy()  # (n, 5): mean(2) + cholesky(3)

    means = pred[:, 0:2]
    residuals = means - targets  # device-frame velocity error, same units both sides

    imu_t_ns = np.array([s.t_ns for s in recording.imu], dtype=np.int64)
    sigmas = np.empty_like(residuals)
    model_integrator = ModelOnlyIntegrator()

    for i in range(windows.shape[0]):
        cov = _cholesky_output_to_cov(pred[i, 2:5])
        sigmas[i] = np.sqrt(np.diag(cov))

        idx = min(int(np.searchsorted(imu_t_ns, window_t_ns[i])), len(orientations) - 1)
        psi_now = heading_rad(np.asarray(orientations[idx].q_world_body, dtype=np.float64))
        model_integrator.update(
            VelocityEstimate(t_ns=int(window_t_ns[i]), v_dev=means[i], cov=cov), psi_now
        )

    pdr = PdrTracker()
    raw = RawIntegrator()
    for sample, orientation in zip(recording.imu, orientations, strict=True):
        psi = heading_rad(np.asarray(orientation.q_world_body, dtype=np.float64))
        pdr.update(sample, psi)
        raw.update(sample, orientation)

    model_traj = model_integrator.trajectory
    pdr_traj = pdr.trajectory
    raw_traj = raw.trajectory
    truth = recording.truth

    if len(model_traj) < 2 or len(pdr_traj) < 2:
        return None

    return {
        "session_id": recording.meta.session_id,
        "carry_position": recording.meta.carry_position.value,
        "distance_m": recording.distance_m,
        "n_windows": int(windows.shape[0]),
        "model_drift_pct": drift_pct(model_traj, truth),
        "model_ate_m": ate(model_traj, truth),
        "model_ate_unaligned_m": ate(model_traj, truth, align=False),
        "model_rte_60s_m": rte(model_traj, truth),
        "model_final_error_m": final_error(model_traj, truth),
        "pdr_drift_pct": drift_pct(pdr_traj, truth),
        "pdr_ate_m": ate(pdr_traj, truth),
        "pdr_rte_60s_m": rte(pdr_traj, truth),
        "raw_drift_pct": drift_pct(raw_traj, truth),
        "residuals_x": residuals[:, 0],
        "residuals_y": residuals[:, 1],
        "sigmas_x": sigmas[:, 0],
        "sigmas_y": sigmas[:, 1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="directory of our own recordings")
    parser.add_argument("--oxiod-data", default=None, help="OxIOD dataset root")
    parser.add_argument("--oxiod-carry-positions", nargs="*", default=None)
    parser.add_argument(
        "--seed", type=int, default=26168, help="must match the training run's seed"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="test = genuinely held out, never used for training or checkpoint selection",
    )
    parser.add_argument("--out", default=None, help="optional path to write a JSON report")
    args = parser.parse_args()

    if args.data is None and args.oxiod_data is None:
        print("evaluate_model: pass --data, --oxiod-data, or both", file=sys.stderr)
        return EXIT_USAGE

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"evaluate_model: no such checkpoint: {args.checkpoint}", file=sys.stderr)
        return EXIT_USAGE

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    window_s = checkpoint.get("window_s", 1.0)
    rate_hz = checkpoint.get("rate_hz", DEFAULT_RATE_HZ)
    seed = checkpoint.get("seed", args.seed)
    if "seed" not in checkpoint:
        print(
            f"evaluate_model: checkpoint has no recorded seed, using --seed {seed} -- "
            "if this doesn't match the training run's seed, this is NOT the same "
            "held-out split and the numbers below are not trustworthy.",
            file=sys.stderr,
        )

    try:
        recordings = load_combined_recordings(
            args.data, args.oxiod_data, args.oxiod_carry_positions
        )
    except FileNotFoundError as e:
        print(f"evaluate_model: {e}", file=sys.stderr)
        return EXIT_USAGE

    with_truth = [r for r in recordings if r.truth is not None]
    if not with_truth:
        print("evaluate_model: no recordings with ground truth", file=sys.stderr)
        return EXIT_USAGE

    train_recs, val_recs, test_recs = split_by_trajectory(with_truth, seed=seed)
    split_map = {"train": train_recs, "val": val_recs, "test": test_recs, "all": with_truth}
    target_recs = split_map[args.split]
    print(f"evaluate_model: evaluating {len(target_recs)} '{args.split}' recordings (seed={seed})")

    results: list[dict[str, object]] = []
    for recording in target_recs:
        r = _evaluate_recording(recording, model, window_s, DEFAULT_HOP_S, rate_hz)
        if r is None:
            print(
                f"evaluate_model: skipping {recording.meta.session_id} "
                "(too short / no usable windows)",
                file=sys.stderr,
            )
            continue
        results.append(r)
        print(
            f"  {r['session_id']!s:42s} carry={r['carry_position']!s:8s} "
            f"dist={r['distance_m']:6.1f}m  "
            f"model={r['model_drift_pct']:6.2f}%  pdr={r['pdr_drift_pct']:6.2f}%  "
            f"raw={r['raw_drift_pct']:7.1f}%"
        )

    if not results:
        print("evaluate_model: nothing evaluated", file=sys.stderr)
        return EXIT_USAGE

    model_drifts = np.array([r["model_drift_pct"] for r in results], dtype=np.float64)
    pdr_drifts = np.array([r["pdr_drift_pct"] for r in results], dtype=np.float64)
    raw_drifts = np.array([r["raw_drift_pct"] for r in results], dtype=np.float64)
    all_res_x = np.concatenate([r["residuals_x"] for r in results])  # type: ignore[misc]
    all_res_y = np.concatenate([r["residuals_y"] for r in results])  # type: ignore[misc]
    all_sig_x = np.concatenate([r["sigmas_x"] for r in results])  # type: ignore[misc]
    all_sig_y = np.concatenate([r["sigmas_y"] for r in results])  # type: ignore[misc]
    coverage_x = calibration_coverage(all_res_x, all_sig_x, k=1.0)
    coverage_y = calibration_coverage(all_res_y, all_sig_y, k=1.0)
    beats_pdr = int((model_drifts < pdr_drifts).sum())

    print("\n=== Aggregate, held-out ===")
    print(
        f"model mean drift:  {model_drifts.mean():6.2f}%  (median {np.median(model_drifts):.2f}%)"
    )
    print(f"pdr   mean drift:  {pdr_drifts.mean():6.2f}%  (median {np.median(pdr_drifts):.2f}%)")
    print(f"raw   mean drift:  {raw_drifts.mean():6.2f}%  (contrast, not a target)")
    print(f"model beats pdr on {beats_pdr}/{len(results)} recordings")
    print(f"calibration coverage @1sigma: x={coverage_x:.3f}  y={coverage_y:.3f}  (target ~0.68)")

    by_carry: dict[str, list[float]] = {}
    for r in results:
        by_carry.setdefault(str(r["carry_position"]), []).append(float(r["model_drift_pct"]))
    print("\n=== By carry position (model drift %) ===")
    for carry, drifts in sorted(by_carry.items()):
        arr = np.array(drifts)
        print(
            f"  {carry:10s} n={len(drifts):3d}  "
            f"mean={arr.mean():6.2f}%  median={np.median(arr):.2f}%"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "seed": seed,
            "n_recordings": len(results),
            "model_drift_pct_mean": float(model_drifts.mean()),
            "model_drift_pct_median": float(np.median(model_drifts)),
            "pdr_drift_pct_mean": float(pdr_drifts.mean()),
            "pdr_drift_pct_median": float(np.median(pdr_drifts)),
            "raw_drift_pct_mean": float(raw_drifts.mean()),
            "model_beats_pdr_count": beats_pdr,
            "model_beats_pdr_total": len(results),
            "coverage_1sigma_x": coverage_x,
            "coverage_1sigma_y": coverage_y,
            "by_carry_position_model_drift_pct_mean": {
                k: float(np.mean(v)) for k, v in by_carry.items()
            },
            "per_recording": [
                {k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results
            ],
        }
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nevaluate_model: wrote {out_path}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
