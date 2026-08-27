/**
 * The map: estimated dot, uncertainty ellipse, true path, and the diverging baseline.
 *
 * OWNER: Tanmay  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8
 *
 * Tiles come from the gateway's /tiles route, backed by a local MBTiles file. Never
 * point this at an external tile host: the demo runs with the venue network unplugged,
 * and ci-web.yml fails the build if an external host appears in the bundle.
 *
 * Two rendering rules that matter on stage:
 *   1. Draw the LAGGED filter state as it is. Do not extrapolate to hide the 300 ms
 *      lag — extrapolation makes the dot cut corners, which reads as broken.
 *   2. Keep the raw-integration baseline on screen the whole time. Watching it spiral
 *      away is the most persuasive thing in the demo.
 *
 * Markers are Leaflet `circleMarker`s (plain SVG circles) rather than `marker`s with
 * the default icon on purpose -- Leaflet's default marker images resolve to paths
 * that break under a bundler unless separately patched, and a circle is all three of
 * these dots need to be.
 */
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'

import type { TelemetryFrame } from '../types'
import { tokens } from '../ui/tokens'
import { ellipseFromCov, ellipseLatLngPoints, enuToLatLng } from './enu'

export interface TrackMapProps {
  frame: TelemetryFrame | null
}

const DEFAULT_ZOOM = 17
const DOT_RADIUS_PX = 7

export function TrackMap({ frame }: TrackMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const hasCenteredRef = useRef(false)

  const estimateRef = useRef<L.CircleMarker | null>(null)
  const ellipseRef = useRef<L.Polygon | null>(null)
  const truthRef = useRef<L.CircleMarker | null>(null)
  const baselineRef = useRef<L.CircleMarker | null>(null)

  // Create the map once. Never recreated on a frame update -- only marker positions
  // move; the map instance and the operator's own pan/zoom stay put.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: [0, 0],
      zoom: DEFAULT_ZOOM,
      zoomControl: true,
      attributionControl: false,
    })
    L.tileLayer('/tiles/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map)

    estimateRef.current = L.circleMarker([0, 0], {
      radius: DOT_RADIUS_PX,
      color: tokens.color.estimate,
      fillColor: tokens.color.estimate,
      fillOpacity: 1,
    })
    ellipseRef.current = L.polygon([], {
      color: tokens.color.estimate,
      fillColor: tokens.color.ellipse,
      weight: 1,
      fillOpacity: 1,
    })
    truthRef.current = L.circleMarker([0, 0], {
      radius: DOT_RADIUS_PX - 1,
      color: tokens.color.truth,
      fillColor: tokens.color.truth,
      fillOpacity: 1,
    })
    baselineRef.current = L.circleMarker([0, 0], {
      radius: DOT_RADIUS_PX - 1,
      color: tokens.color.baseline,
      fillColor: tokens.color.baseline,
      fillOpacity: 0.9,
    })

    mapRef.current = map

    // Leaflet measures its container at construction time; a flex layout can still be
    // settling then, which produces a map that looks frozen at a wrong size.
    const resizeObserver = new ResizeObserver(() => map.invalidateSize())
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
      hasCenteredRef.current = false
    }
  }, [])

  // Move the markers whenever a new frame arrives. Requires an origin -- without one
  // there is no way to place local ENU metres on a real lat/lng basemap.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !frame || frame.origin_lat_deg === null || frame.origin_lon_deg === null) {
      return
    }
    const originLat = frame.origin_lat_deg
    const originLon = frame.origin_lon_deg

    const [estEast, estNorth] = frame.state.p_world
    const estimateLatLng = enuToLatLng(estEast, estNorth, originLat, originLon)
    estimateRef.current?.setLatLng(estimateLatLng).addTo(map)

    const ellipse = ellipseFromCov(frame.state.cov)
    const ellipsePoints = ellipseLatLngPoints(estEast, estNorth, ellipse, originLat, originLon)
    ellipseRef.current?.setLatLngs(ellipsePoints).addTo(map)

    if (frame.truth_p_world) {
      const [tEast, tNorth] = frame.truth_p_world
      truthRef.current?.setLatLng(enuToLatLng(tEast, tNorth, originLat, originLon)).addTo(map)
    } else {
      truthRef.current?.remove()
    }

    if (frame.baseline_p_world) {
      const [bEast, bNorth] = frame.baseline_p_world
      baselineRef.current
        ?.setLatLng(enuToLatLng(bEast, bNorth, originLat, originLon))
        .addTo(map)
    } else {
      baselineRef.current?.remove()
    }

    if (!hasCenteredRef.current) {
      map.setView(estimateLatLng, DEFAULT_ZOOM)
      hasCenteredRef.current = true
    }
  }, [frame])

  const waitingForOrigin =
    !frame || frame.origin_lat_deg === null || frame.origin_lon_deg === null

  return (
    <div className="track-map" data-testid="track-map">
      <div ref={containerRef} className="track-map-canvas" />
      {waitingForOrigin ? (
        <p className="track-map-overlay">Waiting for the first GPS fix to set the origin…</p>
      ) : null}
    </div>
  )
}
