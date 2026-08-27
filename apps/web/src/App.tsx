/**
 * Shell. Map on top, telemetry strip underneath.
 *
 * OWNER: Tanmay (layout + map), Akshit (strip + design system)
 * MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8
 *
 * Built against mock TelemetryFrame objects from day one. The UI does not wait for
 * the ESKF, and the ESKF does not wait for the UI — that is the whole point of the
 * frozen contract in types.ts.
 */
import { TrackMap } from './map/TrackMap'
import { TelemetryStrip } from './telemetry/TelemetryStrip'
import { useTelemetry } from './ws/useTelemetry'

export function App() {
  const { frame, connected } = useTelemetry('/live')

  return (
    <div className="app">
      <TrackMap frame={frame} />
      <TelemetryStrip frame={frame} connected={connected} />
    </div>
  )
}
