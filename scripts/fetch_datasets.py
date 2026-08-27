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

    raise NotImplementedError("M0 -- owner: Sumedha")


if __name__ == "__main__":
    sys.exit(main())
