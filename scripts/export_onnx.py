#!/usr/bin/env python
"""Export the trained model to ONNX, int8-quantized.

OWNER: Sumedha  |  MILESTONE: M2

Run this EARLY, on the first checkpoint that trains at all, not at the end. It
de-risks on-device inference and produces a real per-window latency number to quote
instead of a hope. Target is under 10 ms per window on the laptop.

    python scripts/export_onnx.py models/tcn.pt --out models/tcn.onnx --benchmark
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="models/tcn.onnx")
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="report median inference ms")
    parser.parse_args()

    raise NotImplementedError("M2 -- owner: Sumedha")


if __name__ == "__main__":
    sys.exit(main())
