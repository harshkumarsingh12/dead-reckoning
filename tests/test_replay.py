"""Replay through a REAL, running gateway -- the actual bar issue #37 sets.

Spec: docs/BUILD_PLAN.md section 6.1  |  OWNER: Harsh  |  MILESTONE: M4

`test_gateway.py` uses FastAPI's `TestClient`, which never opens a real socket -- fine
for exercising the app's wiring, but `scripts/replay.py`/`services/gateway/replay.py`
is itself a real WebSocket + real HTTP client. Proving it works means starting an
actual uvicorn server on a real port and connecting to it for real, which is what this
module does. Marked ``slow`` because it is (server startup, real sockets).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from websockets.sync.client import connect

from dr_core.io.session import SessionReader, SessionWriter
from dr_core.types import GpsFix, ImuSample, SessionMeta
from services.gateway import create_app
from services.gateway.replay import control_base_url, replay_once

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port


@pytest.fixture
def live_gateway_port() -> int:
    """A real gateway, bound to an ephemeral port, for the duration of one test."""
    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "gateway did not start within 5s"

    yield port

    server.should_exit = True
    thread.join(timeout=5.0)


def _write_sample_session(path: Path) -> None:
    meta = SessionMeta(session_id="test-replay", device_model="pytest", imu_rate_hz=200.0)
    with SessionWriter(path, meta) as writer:
        writer.write_imu(
            ImuSample(t_ns=1_000_000_000, a_body=np.array([0.0, 0.0, 9.81]), w_body=np.zeros(3))
        )
        writer.write_event(t_ns=1_050_000_000, name="gps_off")
        writer.write_gps(
            GpsFix(t_ns=1_100_000_000, lat_deg=20.3535, lon_deg=85.8164, accuracy_m=5.0)
        )


def test_control_base_url_derives_http_from_the_ingest_websocket_url() -> None:
    assert control_base_url("ws://127.0.0.1:8000/ingest") == "http://127.0.0.1:8000"
    assert control_base_url("wss://demo.example/ingest") == "https://demo.example"


def test_replay_reproduces_live_mechanics(tmp_path: Path, live_gateway_port: int) -> None:
    """The actual acceptance bar: a recorded run flows through /ingest, /control/gps,
    and out the other side on /live -- same server, same code path as a live walk."""
    recording = tmp_path / "session.jsonl.gz"
    _write_sample_session(recording)

    server_url = f"ws://127.0.0.1:{live_gateway_port}/ingest"
    control_base = control_base_url(server_url)
    reader = SessionReader(recording)

    with connect(f"ws://127.0.0.1:{live_gateway_port}/live") as live_ws:
        counts = replay_once(reader, server_url, control_base, speed=0.0)
        # One imu tick -> one broadcast; the event alone broadcasts nothing (the
        # gateway's "event" case is a no-op today, see replay.py's docstring); one gps
        # fix -> one more broadcast. Read both, in order.
        frame_after_imu = json.loads(live_ws.recv())
        frame_after_gps = json.loads(live_ws.recv())

    assert counts == (1, 1, 1)
    # Before the recorded gps_off marker took effect, GPS was still on.
    assert frame_after_imu["gps_enabled"] is True
    # The recorded gps_off event replayed as a REAL POST /control/gps call, not just
    # bytes on /ingest -- this is what "indistinguishable in mechanics from live" means.
    assert frame_after_gps["gps_enabled"] is False
