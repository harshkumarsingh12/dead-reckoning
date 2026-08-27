"""The gateway. /healthz is real from day one; the rest is the M4 ledger.

Spec: docs/BUILD_PLAN.md sections 6.1 and 6.8  |  OWNER: Harsh  |  MILESTONE: M4
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.gateway import create_app


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


@pytest.mark.xfail(reason="M4 -- /ingest unimplemented (owner: Harsh)", strict=True)
def test_ingest_preserves_the_device_capture_timestamp() -> None:
    """The server must NEVER restamp on arrival.

    Restamping is invisible, plausible-looking, and destroys the alignment the entire
    timing subsystem exists to guarantee.
    """
    with TestClient(create_app()) as client, client.websocket_connect("/ingest") as ws:
        ws.send_json({"type": "imu", "t_ns": 123456789, "a": [0, 0, 9.8], "w": [0, 0, 0]})
        echoed = ws.receive_json()
    assert echoed["t_ns"] == 123456789


@pytest.mark.xfail(reason="M4 -- /live unimplemented (owner: Harsh)", strict=True)
def test_live_socket_broadcasts_telemetry_frames() -> None:
    with TestClient(create_app()) as client, client.websocket_connect("/live") as ws:
        frame = ws.receive_json()
    assert "state" in frame
    assert "nis" in frame


@pytest.mark.xfail(reason="M4 -- tile route unimplemented (owner: Harsh)", strict=True)
def test_tiles_are_served_from_the_local_mbtiles() -> None:
    """No venue internet in the loop, by construction. Watch the TMS y-flip:
    MBTiles rows count from the bottom, Leaflet counts from the top."""
    with TestClient(create_app(tiles_path=Path("tests/fixtures/tiny.mbtiles"))) as client:
        response = client.get("/tiles/16/48000/29000.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


@pytest.mark.xfail(reason="M4 -- GPS toggle unimplemented (owner: Harsh)", strict=True)
def test_gps_toggle_is_reflected_in_the_telemetry_stream() -> None:
    """The demo's central gesture. Judges must see cause and effect immediately."""
    with TestClient(create_app()) as client:
        assert client.post("/control/gps", json={"enabled": False}).status_code == 200
        with client.websocket_connect("/live") as ws:
            assert ws.receive_json()["gps_enabled"] is False
