"""The gateway: /healthz, /ingest, /live, tile serving, and the GPS toggle.

Spec: docs/BUILD_PLAN.md sections 6.1 and 6.8  |  OWNER: Harsh  |  MILESTONE: M4

/live's position is fused through a real `dr_core.fusion.Eskf` -- see
`services/gateway/hub.py` for exactly what is and is not wired into it. Most of these
tests exercise the wiring (timestamps preserved, frames broadcast, the toggle observed)
rather than fusion correctness, which is `tests/test_fusion.py`'s job; the two
IMU-driven tests below exist specifically to prove the real Eskf is in the loop and not
the old flat-earth placeholder, which could never move `psi_rad` off `0.0` or `cov` off
`eye(7)`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.gateway import create_app

# A GPS fix carrying a deliberately distinctive t_ns, so a restamped value cannot be
# confused with a coincidence.
_SAMPLE_GPS_FIX = {
    "type": "gps",
    "t_ns": 1_723_456_789_012_345,
    "lat_deg": 20.3535,
    "lon_deg": 85.8164,
    "accuracy_m": 5.0,
}


def test_healthz_is_up() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthz_reports_what_is_actually_loaded() -> None:
    """Five minutes before the demo the useful question is not "is it up" but "are the
    tiles and the model actually there". The probe answers that one."""
    with TestClient(create_app(tiles_path=Path("kiit.mbtiles"))) as client:
        body = client.get("/healthz").json()
    assert body["tiles_loaded"] is True
    assert body["model_loaded"] is False


def test_ingest_preserves_the_device_capture_timestamp() -> None:
    """The server must NEVER restamp on arrival.

    Restamping is invisible, plausible-looking, and destroys the alignment the entire
    timing subsystem exists to guarantee. Proven end to end: push a fix on /ingest,
    read the resulting frame off /live, and check its timestamp is the CAPTURE time,
    not whatever the gateway's clock read when the message arrived.
    """
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/live") as live_ws,
        client.websocket_connect("/ingest") as ingest_ws,
    ):
        ingest_ws.send_json(_SAMPLE_GPS_FIX)
        frame = live_ws.receive_json()
    # /live encodes t_ns as a decimal string -- docs/CONVENTIONS.md section 6.
    assert frame["t_ns"] == str(_SAMPLE_GPS_FIX["t_ns"])
    assert frame["state"]["t_ns"] == str(_SAMPLE_GPS_FIX["t_ns"])


def test_live_socket_broadcasts_telemetry_frames() -> None:
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/live") as live_ws,
        client.websocket_connect("/ingest") as ingest_ws,
    ):
        ingest_ws.send_json(_SAMPLE_GPS_FIX)
        frame = live_ws.receive_json()
    assert "state" in frame
    assert "nis" in frame


def test_live_frame_carries_the_session_origin_from_the_first_gps_fix() -> None:
    """The map (apps/web/src/map/TrackMap.tsx) cannot convert p_world back to
    lat/lon without this -- see docs/CONVENTIONS.md section 6 and map/enu.ts."""
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/live") as live_ws,
        client.websocket_connect("/ingest") as ingest_ws,
    ):
        ingest_ws.send_json(_SAMPLE_GPS_FIX)
        frame = live_ws.receive_json()
    assert frame["origin_lat_deg"] == _SAMPLE_GPS_FIX["lat_deg"]
    assert frame["origin_lon_deg"] == _SAMPLE_GPS_FIX["lon_deg"]


def test_gps_toggle_is_reflected_in_the_telemetry_stream() -> None:
    """The demo's central gesture. Judges must see cause and effect immediately."""
    with TestClient(create_app()) as client:
        assert client.post("/control/gps", json={"enabled": False}).status_code == 200
        with (
            client.websocket_connect("/live") as live_ws,
            client.websocket_connect("/ingest") as ingest_ws,
        ):
            ingest_ws.send_json(_SAMPLE_GPS_FIX)
            frame = live_ws.receive_json()
    assert frame["gps_enabled"] is False


def _imu_tick(t_ns: int, *, w_z: float = 0.0) -> dict[str, object]:
    """One stationary-looking accelerometer tick, sustained yaw rate optional."""
    return {"type": "imu", "t_ns": t_ns, "a": [0.0, 0.0, 9.81], "w": [0.0, 0.0, w_z], "m": None}


def test_imu_ticks_drive_the_real_eskf_not_a_frozen_placeholder() -> None:
    """The old placeholder hardcoded `psi_rad=0.0` and `cov=eye(7)` forever -- it had
    no way to move either. A sustained gyro yaw rate through the real Eskf's `predict`
    must move `psi_rad` off zero, proving the filter is actually integrating."""
    base_t_ns = 1_000_000_000_000
    dt_ns = 5_000_000  # 5 ms, 200 Hz
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/live") as live_ws,
        client.websocket_connect("/ingest") as ingest_ws,
    ):
        frame = None
        for i in range(50):
            ingest_ws.send_json(_imu_tick(base_t_ns + i * dt_ns, w_z=0.5))
            frame = live_ws.receive_json()
    assert frame is not None
    # First tick only sets the filter's clock (Eskf.predict's first-call special
    # case) -- the other 49 integrate roughly 0.5 rad/s * 0.245 s =~ 0.12 rad.
    assert frame["state"]["psi_rad"] != 0.0
    assert abs(frame["state"]["psi_rad"]) > 0.05
    assert frame["state"]["cov"] != [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)]


def test_stationary_imu_ticks_fire_zupt_and_zaru() -> None:
    """Standing still long enough must light the ZUPT/ZARU lamps -- the most legible
    moment in the demo (docs/DEMO_RUNBOOK.md's ten-second stop)."""
    base_t_ns = 2_000_000_000_000
    dt_ns = 5_000_000  # 5 ms, 200 Hz -- matches StationaryDetector's default rate_hz
    # StationaryDetector needs its 100-sample window (0.5 s) full, then 0.3 s more of
    # held-low variance before it fires -- 161 ticks (indices 0..160) is the first
    # tick where that has happened; see the analysis in this PR's description.
    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/live") as live_ws,
        client.websocket_connect("/ingest") as ingest_ws,
    ):
        frame = None
        for i in range(161):
            ingest_ws.send_json(_imu_tick(base_t_ns + i * dt_ns))
            frame = live_ws.receive_json()
    assert frame is not None
    assert frame["zupt_active"] is True
    assert frame["zaru_active"] is True


def test_replay_501s_until_a_golden_run_exists() -> None:
    """Honest about scope: replay needs a recorded session that does not exist yet."""
    with TestClient(create_app()) as client:
        response = client.post("/control/replay")
    assert response.status_code == 501


@pytest.fixture
def tiny_mbtiles(tmp_path: Path) -> Path:
    """A minimal, valid MBTiles file with exactly one tile at z=0/x=0/y=0 (XYZ).

    Built programmatically rather than committed as a binary fixture -- consistent
    with data/README.md: regenerate, do not store.
    """
    path = tmp_path / "tiny.mbtiles"
    # The smallest possible valid PNG: a 1x1 transparent pixel.
    one_pixel_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6360000002000100ffff03000006"
        "00057251b30000000049454e44ae426082"
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE tiles (zoom_level INT, tile_column INT, tile_row INT, tile_data BLOB)"
        )
        # XYZ (0,0,0) -> TMS row (2**0 - 1) - 0 = 0.
        conn.execute("INSERT INTO tiles VALUES (0, 0, 0, ?)", (one_pixel_png,))
    return path


def test_tiles_are_served_from_the_local_mbtiles(tiny_mbtiles: Path) -> None:
    """No venue internet in the loop, by construction. Watch the TMS y-flip:
    MBTiles rows count from the bottom, Leaflet counts from the top."""
    with TestClient(create_app(tiles_path=tiny_mbtiles)) as client:
        response = client.get("/tiles/0/0/0.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_missing_tile_is_a_404(tiny_mbtiles: Path) -> None:
    with TestClient(create_app(tiles_path=tiny_mbtiles)) as client:
        response = client.get("/tiles/9/999/999.png")
    assert response.status_code == 404


def test_tile_route_is_absent_without_a_tiles_path() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/tiles/0/0/0.png")
    assert response.status_code == 404
