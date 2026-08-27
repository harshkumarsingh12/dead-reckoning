/**
 * The telemetry strip. This is what turns a nice demo into a credible one.
 *
 * OWNER: Akshit  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md section 6.8
 *
 * Always on screen, under the map:
 *   - per-channel NIS with its chi-square bounds
 *   - ZUPT / ZARU firing lamp
 *   - magnetometer gate accept/reject, with the specific rejection reason
 *   - the model's current 1-sigma
 *   - the current heading source
 *   - the live drift-% counter
 *
 * Judges ask "how do you know your uncertainty is honest?". This strip is the answer,
 * visible before they finish asking.
 */
import type { TelemetryFrame } from '../types'

export interface TelemetryStripProps {
  frame: TelemetryFrame | null
  connected: boolean
}

export function TelemetryStrip({ frame, connected }: TelemetryStripProps) {
  // TODO(M4, Akshit): NIS gauges against bounds, ZUPT/ZARU lamp, mag-gate indicator,
  // sigma readout, heading-source chip, drift-% counter.
  return (
    <div className="telemetry-strip" data-testid="telemetry-strip">
      <span>{connected ? 'live' : 'disconnected'}</span>
      {frame ? <span>{frame.drift_pct?.toFixed(2) ?? '—'}% drift</span> : null}
    </div>
  )
}
