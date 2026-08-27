"""FastAPI application factory.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8

Routes:
    GET  /healthz                     liveness -- implemented, CI depends on it
    WS   /ingest                      phone pushes device-timestamped samples
    WS   /live                        browser subscribes to TelemetryFrame updates
    GET  /tiles/{z}/{x}/{y}.png       tiles from the local MBTiles, no internet
    POST /control/gps                 the demo's GPS-off toggle
    POST /control/replay              start the golden-run replay

Only /healthz is real so far. Everything else is a stub with its owner and milestone on
it, which is deliberate: this is a scaffold, and an owner should find an empty seat
rather than someone else's half-finished furniture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from dr_core import __version__

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

    _register_ingest(app)
    _register_live(app)
    _register_tiles(app, tiles_path)
    _register_control(app)

    return app


def _register_ingest(app: FastAPI) -> None:
    """WS /ingest -- the phone's uplink.

    Samples arrive carrying their DEVICE capture timestamp. The server must never
    restamp them on arrival: doing so silently destroys the alignment the whole timing
    subsystem exists to guarantee (build plan section 5).

    OWNER: Harsh  |  MILESTONE: M4
    """


def _register_live(app: FastAPI) -> None:
    """WS /live -- broadcasts TelemetryFrame to every connected browser.

    OWNER: Harsh  |  MILESTONE: M4
    """


def _register_tiles(app: FastAPI, tiles_path: Path | None) -> None:
    """GET /tiles/{z}/{x}/{y}.png -- served from the local MBTiles.

    MBTiles is a SQLite container; a tile is one BLOB row keyed by (zoom, column, row).
    Note the TMS y-flip: MBTiles rows count from the bottom, Leaflet counts from the
    top, and getting that backwards produces a map that looks almost right.

    OWNER: Harsh  |  MILESTONE: M4
    """


def _register_control(app: FastAPI) -> None:
    """POST /control/* -- GPS toggle and replay control, driven by the demo UI.

    OWNER: Harsh  |  MILESTONE: M4
    """
