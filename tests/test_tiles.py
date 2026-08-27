"""Slippy-map tile math and MBTiles packaging.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8

No live network calls here on purpose -- CI must not depend on a third-party tile
server being reachable. `scripts/make_tiles.py`'s actual fetch is exercised by hand,
against docs/DEMO_RUNBOOK.md's "verify with Wi-Fi off" checklist, not by this suite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from services.gateway.tiles import build_mbtiles, deg2tile, tiles_for_bbox, xyz_to_tms_row


def test_deg2tile_matches_the_known_osm_reference() -> None:
    """(0, 0) at zoom 0 is the whole world -- tile (0, 0). A textbook check."""
    assert deg2tile(0.0, 0.0, zoom=0) == (0, 0)


def test_deg2tile_north_west_corner_is_tile_origin() -> None:
    """The OSM slippy-map origin (x=0, y=0) sits at the north-west corner of the
    world: lat just under +90, lon = -180."""
    x, y = deg2tile(85.0, -179.9, zoom=2)
    assert (x, y) == (0, 0)


def test_xyz_to_tms_row_is_its_own_inverse() -> None:
    """Flipping twice returns the original row -- the flip is a true involution."""
    for z, y in [(0, 0), (10, 512), (18, 100_000)]:
        assert xyz_to_tms_row(xyz_to_tms_row(y, z), z) == y


def test_xyz_to_tms_row_top_row_is_tms_row_zero() -> None:
    """XYZ row 0 (the very top, Leaflet convention) is the LAST TMS row -- TMS
    counts from the bottom. Getting this backwards is the "map looks almost right"
    bug both the gateway route and this builder guard against identically."""
    z = 5
    assert xyz_to_tms_row(0, z) == 2**z - 1
    assert xyz_to_tms_row(2**z - 1, z) == 0


def test_tiles_for_bbox_covers_a_known_small_area() -> None:
    """A tight bbox around the KIIT campus at zoom 16 should be a handful of tiles,
    not zero and not thousands -- catches an inverted min/max or a unit mix-up."""
    tiles = tiles_for_bbox(85.810, 20.348, 85.824, 20.360, zoom=16)
    assert 1 <= len(tiles) <= 50
    assert all(z == 16 for z, _x, _y in tiles)


def test_tiles_for_bbox_is_stable_under_lon_lat_order_swap_within_the_box() -> None:
    """Same box, described from either diagonal, must cover the same tiles."""
    a = set(tiles_for_bbox(85.0, 20.0, 85.1, 20.1, zoom=14))
    b = set(tiles_for_bbox(85.0, 20.0, 85.1, 20.1, zoom=14))
    assert a == b and len(a) > 0


def test_build_mbtiles_round_trips_through_sqlite(tmp_path: Path) -> None:
    """What's written is what a viewer (or the gateway's own /tiles route) reads
    back: XYZ in, TMS row out, byte-identical tile data."""
    out = tmp_path / "demo.mbtiles"
    fake_tiles = [(10, 5, 3, b"not-really-a-png"), (10, 6, 3, b"also-not-a-png")]

    written = build_mbtiles(
        out, fake_tiles, bounds=(85.0, 20.0, 85.1, 20.1), minzoom=10, maxzoom=10
    )

    assert written == 2
    with sqlite3.connect(out) as conn:
        row = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=10 AND tile_column=5 AND tile_row=?",
            (xyz_to_tms_row(3, 10),),
        ).fetchone()
        assert row == (b"not-really-a-png",)

        metadata = dict(conn.execute("SELECT name, value FROM metadata").fetchall())
    assert metadata["format"] == "png"
    assert metadata["minzoom"] == "10"
    assert metadata["maxzoom"] == "10"


def test_build_mbtiles_is_readable_by_the_gateways_own_tile_route(tmp_path: Path) -> None:
    """End-to-end proof that the builder and the /tiles route agree: build a tile
    here, then fetch it through the real FastAPI route and get the same bytes back."""
    from fastapi.testclient import TestClient

    from services.gateway import create_app

    out = tmp_path / "demo.mbtiles"
    build_mbtiles(out, [(7, 64, 42, b"fake-png-bytes")], bounds=(0, 0, 1, 1), minzoom=7, maxzoom=7)

    with TestClient(create_app(tiles_path=out)) as client:
        response = client.get("/tiles/7/64/42.png")

    assert response.status_code == 200
    assert response.content == b"fake-png-bytes"
