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
 */
import type { TelemetryFrame } from '../types'

export interface TrackMapProps {
  frame: TelemetryFrame | null
}

export function TrackMap({ frame }: TrackMapProps) {
  // TODO(M4, Tanmay): Leaflet map, tileLayer('/tiles/{z}/{x}/{y}.png'), the estimated
  // marker, the 1-sigma ellipse from frame.state.cov, the truth polyline, and the
  // baseline marker.
  return (
    <div className="track-map" data-testid="track-map">
      {frame ? null : <p>Waiting for the first telemetry frame…</p>}
    </div>
  )
}
