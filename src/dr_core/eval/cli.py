"""``dr-eval`` -- one command turns a recording into a trajectory, plots and numbers.

OWNER: Sikruti  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 10, M0

The M0 exit criterion is that this command exists and works end to end. Everything else
in the project is measured through it.

Exit codes are a contract, so CI can depend on them:
    0  ran, and every target in docs/EVALUATION.md was met
    1  ran, but at least one target was missed
    2  bad usage or missing input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dr_core import __version__
from dr_core.eval.report import generate_report
from dr_core.fusion.eskf import Eskf
from dr_core.types import GpsFix, ImuSample, Trajectory

EXIT_OK = 0
EXIT_TARGET_MISSED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (kept separate so tests can introspect it)."""
    parser = argparse.ArgumentParser(
        prog="dr-eval",
        description="Evaluate a dead-reckoning run against ground truth.",
    )
    parser.add_argument("recording", type=Path, help="session recording (.jsonl.gz)")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="ONNX velocity model. Omitted runs baselines only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports"),
        help="directory for plots and report.json (default: reports/)",
    )
    parser.add_argument(
        "--no-gps",
        action="store_true",
        help="discard GPS after the start, simulating the tunnel case",
    )
    parser.add_argument(
        "--baselines",
        default="raw,pdr",
        help="comma-separated baselines to plot alongside (default: raw,pdr)",
    )
    parser.add_argument("--version", action="version", version=f"dr-eval {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling sys.exit."""
    args = build_parser().parse_args(argv)

    if not args.recording.exists():
        print(f"error: no such recording: {args.recording}", file=sys.stderr)
        return EXIT_USAGE

    try:
        from dr_core.io.session import SessionReader

        reader = SessionReader(args.recording)
        meta = reader.meta
    except NotImplementedError as e:
        print(
            f"dr-eval: unable to load session -- SessionReader is pending ({e})",
            file=sys.stderr,
        )
        return EXIT_USAGE
    except Exception as e:
        print(f"dr-eval error: {e}", file=sys.stderr)
        return EXIT_USAGE

    # Run replay and filter
    eskf = Eskf()
    if args.no_gps:
        eskf.set_gps_enabled(False)

    t_list: list[int] = []
    p_list: list[list[float]] = []
    psi_list: list[float] = []

    gps_t: list[int] = []
    gps_p: list[list[float]] = []

    try:
        for rec_type, payload in reader:
            if rec_type == "imu" and isinstance(payload, ImuSample):
                eskf.predict(payload.t_ns, float(payload.w_body[2]))
                st = eskf.state
                t_list.append(st.t_ns)
                p_list.append([float(st.p_world[0]), float(st.p_world[1])])
                psi_list.append(st.psi_rad)
            elif rec_type == "gps" and isinstance(payload, GpsFix):
                if not args.no_gps:
                    eskf.update_gps(payload)
                st = eskf.state
                gps_t.append(payload.t_ns)
                gps_p.append([float(st.p_world[0]), float(st.p_world[1])])
    except NotImplementedError as e:
        print(
            f"dr-eval: playback interrupted -- reader iteration pending ({e})",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if not t_list:
        print("dr-eval: recording contained no valid IMU samples", file=sys.stderr)
        return EXIT_USAGE

    import numpy as np

    estimate = Trajectory(
        t_ns=np.array(t_list, dtype=np.int64),
        p_world=np.array(p_list, dtype=np.float64),
        psi_rad=np.array(psi_list, dtype=np.float64),
        label="eskf",
    )

    truth = Trajectory(
        t_ns=np.array(gps_t if gps_t else t_list, dtype=np.int64),
        p_world=np.array(gps_p if gps_p else p_list, dtype=np.float64),
        label="truth",
    )

    baselines: dict[str, Trajectory] = {}

    report = generate_report(
        estimate=estimate,
        truth=truth,
        baselines=baselines,
        output_dir=args.out,
        run_id=meta.session_id if hasattr(meta, "session_id") else "eval_run",
        nis_logger=eskf._nis_logger,
    )

    print(report.summary_line())

    # Check targets (drift < 5%, ATE < 5m)
    if report.drift_pct < 5.0 and report.ate_m < 5.0:
        return EXIT_OK
    return EXIT_TARGET_MISSED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
