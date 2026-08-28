#!/usr/bin/env python
"""Push a recorded golden run through the live pipeline. The demo's safety net.

OWNER: Harsh  |  MILESTONE: M4

The replay goes through the IDENTICAL server, filter and UI as a live walk -- same
code, same sockets, same rendering. That is what makes it a real fallback rather than
a second demo path nobody has tested. Rehearse switching to it until it is boring.

    python scripts/replay.py data/golden/kiit_loop_01.jsonl.gz --speed 1.0

Recorded ``gps_off``/``gps_on`` event markers are replayed as real ``POST
/control/gps`` calls against the gateway, not just as bytes on ``/ingest`` -- see
`services/gateway/replay.py`, which holds the actual logic so it is covered by tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dr_core.io.session import SessionReader
from services.gateway.replay import control_base_url, replay_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="0 runs as fast as possible")
    parser.add_argument("--server", default="ws://127.0.0.1:8000/ingest")
    parser.add_argument("--loop", action="store_true", help="restart when it finishes")
    args = parser.parse_args()

    reader = SessionReader(args.recording)
    control_base = control_base_url(args.server)
    print(
        f"replaying {args.recording} ({reader.meta.session_id}) "
        f"at speed={args.speed} -> {args.server}"
    )

    try:
        while True:
            imu_n, gps_n, event_n = replay_once(reader, args.server, control_base, args.speed)
            print(f"done: {imu_n} imu, {gps_n} gps, {event_n} event messages sent")
            if not args.loop:
                break
            print("--loop set, restarting from the beginning")
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
