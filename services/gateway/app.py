"""FastAPI application factory.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8

Routes:
    GET  /healthz                     liveness
    WS   /ingest                      phone pushes device-timestamped samples
    WS   /live                        browser subscribes to TelemetryFrame updates
    GET  /tiles/{z}/{x}/{y}.png       tiles from the local MBTiles, no internet
    POST /control/gps                 the demo's GPS-off toggle
    POST /control/replay              start the golden-run replay (#37, not yet)

`/ingest`, `/live`, `/control/gps`, and tile serving are real. `/live`'s position is a
placeholder GPS passthrough until the ESKF (M3) lands -- see `services/gateway/hub.py`
for exactly why and what replaces it. `/control/replay` still 501s: it needs a
recorded golden run that does not exist yet (#37).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from dr_core import __version__
from services.gateway.hub import Hub
from services.gateway.tiles import xyz_to_tms_row
from services.gateway.wire import decode_gps, decode_imu

if TYPE_CHECKING:
    from pathlib import Path

# Binds every interface on purpose: the phone reaches the laptop over its own
# hotspot, so a localhost-only bind makes the entire transport unusable.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def create_app(tiles_path: Path | None = None, model_path: Path | None = None) -> FastAPI:
    """Build the app.

    Args:
        tiles_path: local ``.mbtiles`` file. None disables the tile route, which is
            fine for development against an online basemap but NOT acceptable for the
            demo -- see the offline checklist in docs/DEMO_RUNBOOK.md.
        model_path: ONNX velocity model. None runs baselines only.
    """
    app = FastAPI(
        title="SIH26168 dead-reckoning gateway",
        version=__version__,
        docs_url="/docs",
    )
    hub = Hub()

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness probe. Reports whether the demo-critical assets are actually loaded.

        Returns the readiness of each piece rather than a bare "ok", because the
        question that matters five minutes before the demo is not "is it up" but
        "are the tiles and the model actually there".
        """
        return {
            "status": "ok",
            "version": __version__,
            "tiles_loaded": tiles_path is not None,
            "model_loaded": model_path is not None,
        }

    _register_ingest(app, hub)
    _register_live(app, hub)
    _register_tiles(app, tiles_path)
    _register_control(app, hub)

    return app


def _register_ingest(app: FastAPI, hub: Hub) -> None:
    """WS /ingest -- the phone's uplink.

    Samples arrive carrying their DEVICE capture timestamp. This handler must never
    restamp them: doing so would silently destroy the alignment the whole timing
    subsystem exists to guarantee (build plan section 5). Nothing here does -- every
    decoded record keeps the `t_ns` it arrived with, all the way to the /live frame.
    """

    @app.websocket("/ingest")
    async def ingest(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_json()
                match raw.get("type"):
                    case "imu":
                        decode_imu(raw)  # validated now; fused once M1-M3 land
                    case "gps":
                        await hub.on_gps(decode_gps(raw))
                    case "meta" | "event":
                        pass  # TODO(M1/M3): session context, ZUPT/ZARU markers
        except WebSocketDisconnect:
            pass


def _register_live(app: FastAPI, hub: Hub) -> None:
    """WS /live -- broadcasts TelemetryFrame to every connected browser."""

    @app.websocket("/live")
    async def live(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await hub.subscribe()
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(queue)


def _register_tiles(app: FastAPI, tiles_path: Path | None) -> None:
    """GET /tiles/{z}/{x}/{y}.png -- served from the local MBTiles.

    MBTiles is a SQLite container; a tile is one BLOB row keyed by (zoom, column, row).
    Rows are stored TMS-style (bottom-up); Leaflet/XYZ requests count rows top-down --
    `services.gateway.tiles.xyz_to_tms_row` is that fix, shared with the builder in
    `scripts/make_tiles.py` so the two can never disagree about which row is which.
    Getting it backwards produces a map that looks almost right, which is a worse bug
    than one that looks completely wrong.

    A sync (non-async) handler: FastAPI runs it in its worker threadpool automatically,
    which keeps a blocking `sqlite3` call off the event loop without extra plumbing.
    """
    if tiles_path is None:
        return

    @app.get("/tiles/{z}/{x}/{y}.png")
    def get_tile(z: int, x: int, y: int) -> Response:
        tms_row = xyz_to_tms_row(y, z)
        with sqlite3.connect(tiles_path) as conn:
            row = conn.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level = ? AND tile_column = ? "
                "AND tile_row = ?",
                (z, x, tms_row),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="tile not found")
        return Response(content=row[0], media_type="image/png")


def _register_control(app: FastAPI, hub: Hub) -> None:
    """POST /control/* -- GPS toggle and replay control, driven by the demo UI."""

    @app.post("/control/gps")
    async def control_gps(payload: dict[str, bool]) -> dict[str, bool]:
        hub.set_gps_enabled(bool(payload.get("enabled", True)))
        return {"enabled": hub.gps_enabled}

    @app.post("/control/replay")
    async def control_replay() -> None:
        # Needs a recorded golden run -- not yet available (#37, owner: Harsh).
        raise HTTPException(status_code=501, detail="replay not implemented yet (see #37)")
