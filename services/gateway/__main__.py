"""``python -m services.gateway`` -- start the live demo server.

OWNER: Harsh  |  MILESTONE: M4

Binds 0.0.0.0 on purpose: the phone reaches the laptop across its own hotspot, so
localhost-only would make the whole transport unusable. Nothing sensitive is served,
and the network is a two-device hotspot -- but see docs/DEMO_RUNBOOK.md for the
hardening Sristee owns before this runs on any shared network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from services.gateway.app import DEFAULT_HOST, DEFAULT_PORT, create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="services.gateway")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tiles", type=Path, default=None, help="path to .mbtiles")
    parser.add_argument("--model", type=Path, default=None, help="path to .onnx")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app = create_app(tiles_path=args.tiles, model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
