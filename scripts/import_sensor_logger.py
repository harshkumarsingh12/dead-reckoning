#!/usr/bin/env python
"""Convert Sensor Logger export(s) into dr_core.io's session format.

OWNER: Sumedha  |  MILESTONE: M2

    python scripts/import_sensor_logger.py recording.zip \
        --out data/own/outdoor/walk1.jsonl.gz --carry-position hand

    # or batch-convert a folder of exports, one output file per input:
    python scripts/import_sensor_logger.py "data recordings"/*.zip \
        --out-dir data/own/outdoor --carry-position hand

Thin by design -- see dr_core.datasets.sensor_logger for the real conversion logic and
the column-mapping notes (verified against real exports, not assumed from Sensor
Logger's docs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dr_core.datasets.sensor_logger import convert
from dr_core.types import CarryPosition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Sensor Logger .zip export(s) or extracted folder(s)"
    )
    parser.add_argument("--out", type=Path, default=None, help="output path (single input only)")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="write one file per input, named after it"
    )
    parser.add_argument(
        "--carry-position",
        choices=[p.value for p in CarryPosition],
        default=CarryPosition.UNKNOWN.value,
    )
    args = parser.parse_args()

    if args.out is not None and len(args.inputs) != 1:
        print(
            "import_sensor_logger: --out only valid with exactly one input; "
            "use --out-dir for more than one",
            file=sys.stderr,
        )
        return 2
    if args.out is None and args.out_dir is None:
        print(
            "import_sensor_logger: pass --out (single input) or --out-dir (one or more inputs)",
            file=sys.stderr,
        )
        return 2

    carry_position = CarryPosition(args.carry_position)
    exit_code = 0

    for input_path in args.inputs:
        if not input_path.exists():
            print(f"import_sensor_logger: no such path: {input_path}", file=sys.stderr)
            exit_code = 1
            continue
        out_path = (
            args.out if args.out is not None else args.out_dir / f"{input_path.stem}.jsonl.gz"
        )
        try:
            result = convert(input_path, out_path, carry_position, session_id=input_path.stem)
        except (FileNotFoundError, ValueError) as e:
            print(f"import_sensor_logger: skipping {input_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue
        print(
            f"import_sensor_logger: wrote {result.out_path} "
            f"({result.imu_samples} IMU samples @ ~{result.imu_rate_hz:.0f} Hz, "
            f"{result.gps_fixes} GPS fixes)"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
