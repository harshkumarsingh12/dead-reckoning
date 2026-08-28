"""Push a recorded session's messages through the gateway, indistinguishable from live.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.1

`scripts/replay.py` is the thin CLI; the logic lives here so it is covered by
`tests/test_replay.py` rather than only ever exercised by hand against a running
gateway.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.sync.client import connect

from dr_core.io.session import SessionEvent
from dr_core.types import GpsFix, ImuSample
from services.gateway.wire import (
    encode_event_for_ingest,
    encode_gps_for_ingest,
    encode_imu_for_ingest,
)

if TYPE_CHECKING:
    from dr_core.io.session import SessionReader

# The only event names that correspond to a real live control call today. Every other
# marker (tap, calib_*, corner_N, manual_marker, ...) still goes out on /ingest for
# faithfulness, but the gateway has nothing wired to act on them yet
# (services/gateway/app.py's "event" case is a no-op) -- see hub.py's module docstring
# for the same pattern on the IMU side.
GPS_TOGGLE_EVENTS: dict[str, bool] = {"gps_off": False, "gps_on": True}


def control_base_url(ingest_url: str) -> str:
    """Derive the gateway's HTTP base from its /ingest WebSocket URL.

    ws:// becomes http://, wss:// becomes https://, and the /ingest path is dropped --
    the toggle lives at POST /control/gps on the same host, not on the socket itself.
    """
    parts = urlsplit(ingest_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, "", "", ""))


def toggle_gps(control_base: str, enabled: bool) -> None:
    """Reproduce a recorded gps_off/gps_on marker as the real live control call.

    Best-effort: a dropped toggle during a replay is a bad demo moment, worth printing
    loudly, but not a reason to kill the whole fallback run over one flaky localhost
    request (build plan's "the demo survives a dead network" applies to the replay
    path too).
    """
    try:
        httpx.post(f"{control_base}/control/gps", json={"enabled": enabled}, timeout=5.0)
    except httpx.HTTPError as exc:
        print(f"warning: /control/gps toggle failed: {exc}", file=sys.stderr)


def replay_once(
    reader: SessionReader, server: str, control_base: str, speed: float
) -> tuple[int, int, int]:
    """One pass through the recording, paced by SessionReader.replay.

    Returns:
        (imu_count, gps_count, event_count) actually sent, for the CLI's summary line.
    """
    imu_count = gps_count = event_count = 0
    with connect(server) as ws:
        for record_type, payload in reader.replay(speed=speed):
            if record_type == "imu":
                assert isinstance(payload, ImuSample)
                ws.send(json.dumps(encode_imu_for_ingest(payload)))
                imu_count += 1
            elif record_type == "gps":
                assert isinstance(payload, GpsFix)
                ws.send(json.dumps(encode_gps_for_ingest(payload)))
                gps_count += 1
            elif record_type == "event":
                assert isinstance(payload, SessionEvent)
                ws.send(json.dumps(encode_event_for_ingest(payload.t_ns, payload.name)))
                if payload.name in GPS_TOGGLE_EVENTS:
                    toggle_gps(control_base, GPS_TOGGLE_EVENTS[payload.name])
                event_count += 1
    return imu_count, gps_count, event_count
