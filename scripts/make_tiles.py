#!/usr/bin/env python
"""Build the offline MBTiles for the demo area.

OWNER: Harsh  |  MILESTONE: M4

Do this DAYS before the event, on a connection you trust, and verify the result with
the laptop's Wi-Fi physically switched off. "We will grab the tiles at the venue" is
how a demo dies (build plan risk register).

Output is gitignored -- hundreds of MB has no business in a repository. Keep the file
in the team drive and note its checksum in docs/DEMO_RUNBOOK.md.

    python scripts/make_tiles.py --bbox 85.810,20.348,85.824,20.360 --zoom 14-19 \
        --out tiles/kiit.mbtiles

Respects OSM's tile usage policy (https://operations.osmfoundation.org/policies/tiles/):
a descriptive User-Agent, a small delay between requests, and a hard cap on the total
tile count so a typo in --zoom cannot turn into an accidental bulk scrape of a free
public service. Override the cap with --max-tiles if you genuinely need a bigger area
-- and consider a paid tile provider first if you do.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

from services.gateway.tiles import build_mbtiles, tiles_for_bbox

DEFAULT_MAX_TILES = 2000
REQUEST_DELAY_S = 0.1
USER_AGENT = "dead-reckoning-tile-fetcher/0.1 (SIH 2026 hackathon demo prep)"
DEFAULT_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--bbox needs exactly 4 comma-separated values: min_lon,min_lat,max_lon,max_lat"
        )
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("--bbox must have min < max on both axes")
    return min_lon, min_lat, max_lon, max_lat


def parse_zoom_range(raw: str) -> range:
    if "-" in raw:
        lo, hi = raw.split("-", 1)
        return range(int(lo), int(hi) + 1)
    z = int(raw)
    return range(z, z + 1)


def fetch_tile(client: httpx.Client, z: int, x: int, y: int, url_template: str) -> bytes:
    response = client.get(url_template.format(z=z, x=x, y=y))
    response.raise_for_status()
    return response.content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--zoom", default="14-19", help="single level or a range like 14-19")
    parser.add_argument("--out", default="tiles/demo.mbtiles")
    parser.add_argument(
        "--source",
        default=DEFAULT_TILE_URL,
        help="tile URL template with {z}/{x}/{y}; respect the source's usage policy",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=DEFAULT_MAX_TILES,
        help=f"refuse without confirmation past this many tiles (default {DEFAULT_MAX_TILES})",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    try:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(args.bbox)
        zoom_levels = parse_zoom_range(args.zoom)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    all_tiles = [
        tile for z in zoom_levels for tile in tiles_for_bbox(min_lon, min_lat, max_lon, max_lat, z)
    ]
    print(f"{len(all_tiles)} tiles across zoom {zoom_levels.start}-{zoom_levels.stop - 1}")

    if len(all_tiles) > args.max_tiles and not args.yes:
        print(
            f"error: {len(all_tiles)} tiles exceeds --max-tiles ({args.max_tiles}). "
            "This is a courtesy cap against accidentally bulk-scraping a free tile "
            "server -- narrow --bbox/--zoom, raise --max-tiles deliberately, or use "
            "--yes if you have actually checked this against the source's usage policy.",
            file=sys.stderr,
        )
        return 2

    fetched: list[tuple[int, int, int, bytes]] = []
    failed = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=10.0) as client:
        for i, (z, x, y) in enumerate(all_tiles):
            try:
                data = fetch_tile(client, z, x, y, args.source)
                fetched.append((z, x, y, data))
            except httpx.HTTPError as exc:
                print(f"  skip {z}/{x}/{y}: {exc}", file=sys.stderr)
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(all_tiles)} fetched...")
            time.sleep(REQUEST_DELAY_S)

    written = build_mbtiles(
        Path(args.out),
        fetched,
        bounds=(min_lon, min_lat, max_lon, max_lat),
        minzoom=zoom_levels.start,
        maxzoom=zoom_levels.stop - 1,
    )
    size_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"wrote {written} tiles ({failed} failed) to {args.out} ({size_mb:.1f} MB)")
    print(
        "now verify offline: disconnect Wi-Fi, run the gateway with --tiles "
        f"{args.out}, and confirm the map still renders. See docs/DEMO_RUNBOOK.md."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
