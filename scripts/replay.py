#!/usr/bin/env python
"""Push a recorded golden run through the live pipeline. The demo's safety net.

OWNER: Harsh  |  MILESTONE: M4

The replay goes through the IDENTICAL server, filter and UI as a live walk -- same
code, same sockets, same rendering. That is what makes it a real fallback rather than
a second demo path nobody has tested. Rehearse switching to it until it is boring.

    python scripts/replay.py data/golden/kiit_loop_01.jsonl.gz --speed 1.0
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--speed", type=float, default=1.0, help="0 runs as fast as possible")
    parser.add_argument("--server", default="ws://127.0.0.1:8000/ingest")
    parser.add_argument("--loop", action="store_true", help="restart when it finishes")
    parser.parse_args()

    raise NotImplementedError("M4 -- owner: Harsh")


if __name__ == "__main__":
    sys.exit(main())
