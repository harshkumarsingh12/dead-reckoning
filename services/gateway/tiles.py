"""Slippy-map tile math and MBTiles packaging.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8

Shared by two call sites on purpose:
  - ``services/gateway/app.py``'s ``/tiles`` route reads an existing MBTiles file.
  - ``scripts/make_tiles.py`` builds one.

Both need the exact same TMS row convention. Keeping it in one place is what stops the
route and the builder from silently disagreeing about which row is which -- the "map
looks almost right" bug the route's own docstring already warns about.
"""

from __future__ import annotations

import math
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# The whole file lives at this layer because it belongs equally to /tiles serving and
# to the builder. It never touches IMU or GPS data, so it has nothing to do with
# dr_core.preprocess -- this is plain geodesy for pixels, not sensor fusion.


def deg2tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Standard slippy-map (XYZ) tile containing a lat/lon at a given zoom.

    Reference: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int
) -> list[tuple[int, int, int]]:
    """Every XYZ tile (z, x, y) that covers the bounding box at one zoom level."""
    x0, y0 = deg2tile(max_lat, min_lon, zoom)  # top-left (max lat = north)
    x1, y1 = deg2tile(min_lat, max_lon, zoom)  # bottom-right
    return [
        (zoom, x, y)
        for x in range(min(x0, x1), max(x0, x1) + 1)
        for y in range(min(y0, y1), max(y0, y1) + 1)
    ]


def xyz_to_tms_row(y: int, z: int) -> int:
    """Flip an XYZ (top-down) row into the TMS (bottom-up) row MBTiles stores.

    The gateway's ``/tiles`` route applies the same flip on the way out; this is the
    one place both directions are defined, so they cannot drift apart.
    """
    # mypy: int.__pow__ types as Any (a negative exponent would return float);
    # z is always >= 0 here, so the int() below is just making that explicit.
    return int(2**z) - 1 - y


def build_mbtiles(
    out_path: Path,
    tiles: Iterable[tuple[int, int, int, bytes]],
    bounds: tuple[float, float, float, float],
    minzoom: int,
    maxzoom: int,
    name: str = "sih26168-demo",
    tile_format: str = "png",
) -> int:
    """Write a valid MBTiles file (the ``tiles`` table plus the required ``metadata``).

    Args:
        tiles: ``(z, x, y, png_bytes)`` in XYZ convention -- the row is flipped to TMS
            here, once, so every caller can think in the XYZ convention everyone
            actually uses (Leaflet, this codebase's own tile math).
        bounds: ``(min_lon, min_lat, max_lon, max_lat)``, for the metadata table.

    Returns:
        The number of tiles written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with sqlite3.connect(out_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tiles "
            "(zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles "
            "(zoom_level, tile_column, tile_row)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT)")
        conn.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("name", name),
                ("format", tile_format),
                ("bounds", ",".join(str(b) for b in bounds)),
                ("minzoom", str(minzoom)),
                ("maxzoom", str(maxzoom)),
                ("type", "baselayer"),
            ],
        )
        for z, x, y, data in tiles:
            conn.execute(
                "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?)",
                (z, x, xyz_to_tms_row(y, z), data),
            )
            count += 1
    return count
