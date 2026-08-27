#!/usr/bin/env python
"""Build the offline MBTiles for the demo area.

OWNER: Harsh  |  MILESTONE: M4

Do this DAYS before the event, on a connection you trust, and verify the result with
the laptop's Wi-Fi physically switched off. "We will grab the tiles at the venue" is
how a demo dies (build plan risk register).

Output is gitignored -- hundreds of MB has no business in a repository. Keep the file
in the team drive and note its checksum in docs/DEMO_RUNBOOK.md.

    python scripts/make_tiles.py --bbox 85.810,20.348,85.824,20.360 --zoom 14-19 \
        --out tiles/kiit.mbtiles
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--zoom", default="14-19")
    parser.add_argument("--out", default="tiles/demo.mbtiles")
    parser.add_argument("--source", default="osm", help="tile source; respect its usage policy")
    parser.parse_args()

    raise NotImplementedError("M4 -- owner: Harsh")


if __name__ == "__main__":
    sys.exit(main())
