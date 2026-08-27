#!/usr/bin/env python
"""Train the causal TCN velocity model. Needs the [ml] extra.

OWNER: Sumedha  |  MILESTONE: M2

    pip install -e ".[ml]"
    python scripts/train.py --data data/ronin --epochs 60 --out models/tcn.pt

Imports dr_core.preprocess -- the SAME module the live pipeline imports. If you ever
find yourself preparing data any other way here, stop: that divergence is the exact
failure the shared module exists to prevent, and it will show up as an unexplained
live-demo underperformance rather than as an error.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset root")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--out", default="models/tcn.pt")
    parser.add_argument("--seed", type=int, default=26168)
    parser.add_argument(
        "--no-yaw-aug",
        action="store_true",
        help="disable random-yaw augmentation (ablation only -- it is what enforces "
        "heading-agnosticism, and without it the model fails from the pocket)",
    )
    parser.parse_args()

    raise NotImplementedError("M2 -- owner: Sumedha")


if __name__ == "__main__":
    sys.exit(main())
