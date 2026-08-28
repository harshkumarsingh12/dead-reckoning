#!/usr/bin/env python
"""Download and lay out the public datasets.

OWNER: Sumedha  |  MILESTONE: M0

Both RoNIN and OxIOD require an ACCESS REQUEST that can take days to approve. Send
both on day one. No amount of clever modelling recovers a week lost waiting on an
email, and this is the single longest lead time in the project.

    python scripts/fetch_datasets.py --dataset ronin --out data/ronin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_INFO = {
    "ronin": {
        "paper": "Herath, Yan, Furukawa -- RoNIN (ICRA 2020)",
        "access": "https://ronin.cs.sfu.ca/",
        "why": "large multi-subject smartphone IMU with good truth; also the method template",
    },
    "oxiod": {
        "paper": "Chen et al. -- OxIOD",
        "access": "http://deepio.cs.ox.ac.uk/",
        "why": "Vicon truth across hand / pocket / bag / trolley -- the carry-position story",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_INFO), required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--info", action="store_true", help="print access details and exit")
    args = parser.parse_args()

    info = DATASET_INFO[args.dataset]
    if args.info:
        for key, value in info.items():
            print(f"{key:8} {value}")
        return 0

    # Neither dataset is a plain public download: both gate access behind a request
    # that can take days to approve (data/README.md, docs/BUILD_PLAN.md section 4).
    # There is no URL this script could fetch from without that approval, so rather
    # than pretend otherwise, it prepares the destination and prints exactly what to
    # do once access lands -- consistent with load_ronin/load_oxiod's own guard,
    # which checks for this same directory.
    out_root = Path(args.out) if args.out else Path("data") / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{args.dataset}: no direct-download URL -- access requires manual approval.")
    print(f"  1. Request access: {info['access']}")
    print("  2. Once approved, download the archive and extract its contents under:")
    print(f"       {out_root}")
    print(
        f"  3. dr_core.datasets.load_{args.dataset} checks that directory; run this "
        "again with --info any time to re-check the access URL."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
