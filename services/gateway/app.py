"""FastAPI application factory.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8

Routes:
    GET  /healthz                     liveness
    WS   /ingest                      phone pushes device-timestamped samples
    WS   /live                        browser subscribes to TelemetryFrame updates
    GET  /tiles/{z}/{x}/{y}.png       tiles from the local MBTiles, no internet
    GET  /reports/{run_id}/{file}     a report generate_report already wrote to disk
    POST /control/gps                 the demo's GPS-off toggle
    POST /control/replay              start the golden-run replay (#37, not yet)

`/ingest`, `/live`, `/control/gps`, tile serving, and report serving are real. `/live`'s
position now comes from a real `dr_core.fusion.Eskf` (see `services/gateway/hub.py`
for exactly what is and is not wired into it yet -- the learned-velocity and
magnetometer updates are still open). `/reports` serves files a human already
generated offline with `scripts/run_eval.py`; there is no route here that triggers
report generation from a live session yet -- see `services/gateway/reports.py` for why
that is not honestly buildable until a ground-truth loop exists. `/control/replay`
still 501s: it needs a recorded golden run that does not exist yet (#37).
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from dr_core import __version__
from services.gateway.hub import Hub
from services.gateway.reports import content_type_for, resolve_report_file
from services.gateway.tiles import xyz_to_tms_row
from services.gateway.wire import decode_event, decode_gps, decode_imu, decode_meta

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# Binds every interface on purpose: the phone reaches the laptop over its own
# hotspot, so a localhost-only bind makes the entire transport unusable.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def create_app(
    tiles_path: Path | None = None,
    model_path: Path | None = None,
    reports_dir: Path | None = None,
    record_dir: Path | None = None,
) -> FastAPI:
    """Build the app.

    Args:
        tiles_path: local ``.mbtiles`` file. None disables the tile route, which is
            fine for development against an online basemap but NOT acceptable for the
            demo -- see the offline checklist in docs/DEMO_RUNBOOK.md.
        model_path: ONNX velocity model. None runs baselines only.
        reports_dir: directory `generate_report` writes ``<run_id>/`` subfolders into.
            None disables the ``/reports`` route.
        record_dir: directory every live session gets mirrored into as it streams,
            named ``<session_id>.jsonl.gz`` (dr_core.io.SessionWriter's format). None
            disables recording entirely -- the default, so tests and casual runs don't
            silently leave files behind.
    """
    hub = Hub(record_dir=record_dir)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        # Ordinary websocket disconnects deliberately do NOT close the recording (see
        # _register_ingest) since a hotspot hiccup looks identical to one on the wire.
        # A graceful shutdown (Ctrl+C ending the demo) is the one signal this process
        # actually gets that the session is really over, so it's the one place the
        # gzip stream gets its proper close -- otherwise the file is missing its final
        # block and trailer, and a later SessionReader open raises EOFError.
        hub.close_recording()

    app = FastAPI(
        title="SIH26168 dead-reckoning gateway",
        version=__version__,
        docs_url="/docs",
        lifespan=_lifespan,
    )

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
            "reports_dir_configured": reports_dir is not None,
        }

    _register_ingest(app, hub)
    _register_live(app, hub)
    _register_tiles(app, tiles_path)
    _register_reports(app, reports_dir)
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
                        await hub.on_imu(decode_imu(raw))
                    case "gps":
                        await hub.on_gps(decode_gps(raw))
                    case "meta":
                        hub.on_meta(decode_meta(raw))
                    case "event":
                        hub.on_event(decode_event(raw))
        except WebSocketDisconnect:
            pass
        # Deliberately NOT closing the recording here. There is no "session ended"
        # message in the wire protocol -- a disconnect means exactly the same thing
        # whether it's a hotspot hiccup mid-walk (StreamClient.kt reconnects with a
        # fresh websocket in 500ms and resends the SAME meta) or the actual end of the
        # walk. Closing on every disconnect would truncate the file on every hiccup;
        # see Hub.on_meta's docstring. The recording closes when a genuinely new
        # session_id arrives, or the process shuts down gracefully (below).


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


def _register_reports(app: FastAPI, reports_dir: Path | None) -> None:
    """GET /reports/{run_id}/{filename} -- a file `generate_report` already wrote.

    Read-only and refuses anything outside `reports_dir` or not on the fixed list of
    filenames the report writer actually produces -- see `services/gateway/reports.py`
    for the path-traversal reasoning and for why there is no *trigger* route here yet.
    """
    if reports_dir is None:
        return

    @app.get("/reports/{run_id}/{filename}")
    def get_report_file(run_id: str, filename: str) -> FileResponse:
        path = resolve_report_file(reports_dir, run_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="report file not found")
        return FileResponse(path, media_type=content_type_for(filename))


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
