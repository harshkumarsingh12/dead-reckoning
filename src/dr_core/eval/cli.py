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
        # SessionReader is owned by Sristee. If not implemented, catch and report cleanly.
        _ = reader.meta
    except NotImplementedError as e:
        print(
            f"dr-eval: unable to load session -- SessionReader is pending ({e})",
            file=sys.stderr,
        )
        return EXIT_USAGE
    except Exception as e:
        print(f"dr-eval error: {e}", file=sys.stderr)
        return EXIT_USAGE

    raise NotImplementedError("M0 -- owner: Sikruti (full run_eval pipeline with models)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
