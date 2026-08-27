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
 *
 * Every accept/reject/warn state below pairs colour with a dot glyph AND a text
 * label -- never colour alone (styles.css). Roughly one judge in twelve is
 * colour-blind, and low-contrast greys that look refined on a laptop vanish under a
 * projector in a bright hall.
 */
import type { ReactNode } from 'react'

import type { HeadingSource, MagGateVerdict, TelemetryFrame } from '../types'

export interface TelemetryStripProps {
  frame: TelemetryFrame | null
  connected: boolean
}

const HEADING_SOURCE_LABEL: Record<HeadingSource, string> = {
  gyro: 'Gyro (dead reckoning)',
  magnetometer: 'Magnetometer',
  velocity: 'Velocity update',
  zaru: 'ZARU (stationary)',
  gps_course: 'GPS course',
}

const MAG_VERDICT_LABEL: Record<MagGateVerdict, string> = {
  accepted: 'Accepted',
  rejected_magnitude: 'Rejected — magnitude',
  rejected_dip: 'Rejected — dip angle',
  rejected_innovation: 'Rejected — innovation',
}

type Variant = 'ok' | 'warn' | 'reject' | 'muted'

function Badge({ variant, children }: { variant: Variant; children: ReactNode }) {
  return (
    <span className={`badge badge--${variant}`}>
      <span className="badge-dot" aria-hidden="true" />
      {children}
    </span>
  )
}

function ConnectionBadge({ connected }: { connected: boolean }) {
  return (
    <Badge variant={connected ? 'ok' : 'reject'}>{connected ? 'Live' : 'Disconnected'}</Badge>
  )
}

function GpsBadge({ enabled }: { enabled: boolean }) {
  return <Badge variant={enabled ? 'ok' : 'warn'}>{enabled ? 'GPS on' : 'GPS off'}</Badge>
}

function DriftCounter({ driftPct }: { driftPct: number | null }) {
  return (
    <div className="drift-counter">
      <span className="strip-label">Drift</span>
      {driftPct === null ? (
        <span className="value value--pending">—</span>
      ) : (
        <span className="value">{driftPct.toFixed(1)}%</span>
      )}
    </div>
  )
}

function HeadingSourceBadge({ source }: { source: HeadingSource }) {
  // Only ZARU and the velocity update actively correct heading without GPS or a
  // trusted magnetometer; gyro-only means nothing has corrected it recently.
  const variant: Variant = source === 'gyro' ? 'warn' : 'ok'
  return <Badge variant={variant}>{HEADING_SOURCE_LABEL[source]}</Badge>
}

function MagGateBadge({ verdict }: { verdict: MagGateVerdict }) {
  return (
    <Badge variant={verdict === 'accepted' ? 'ok' : 'reject'}>{MAG_VERDICT_LABEL[verdict]}</Badge>
  )
}

function ZuptZaruLamps({ zupt, zaru }: { zupt: boolean; zaru: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
      <Badge variant={zupt ? 'ok' : 'muted'}>ZUPT {zupt ? 'firing' : 'idle'}</Badge>
      <Badge variant={zaru ? 'ok' : 'muted'}>ZARU {zaru ? 'firing' : 'idle'}</Badge>
    </div>
  )
}

/** One bar per measurement channel: value against its chi-square upper bound. */
function NisGauges({
  nis,
  bounds,
}: {
  nis: Record<string, number>
  bounds: Record<string, [number, number]>
}) {
  const entries = Object.entries(nis)
  if (entries.length === 0) {
    return <span className="strip-label">No channels reporting yet</span>
  }
  return (
    <div className="nis-gauges">
      {entries.map(([channel, value]) => {
        const upper = bounds[channel]?.[1]
        const inBounds = upper === undefined || value <= upper
        const fraction = upper && upper > 0 ? Math.min(value / upper, 1.5) / 1.5 : 0
        return (
          <div className="nis-gauge" key={channel}>
            <span className="strip-label">{channel}</span>
            <div className="nis-gauge-track">
              <div
                className={`nis-gauge-fill nis-gauge-fill--${inBounds ? 'ok' : 'warn'}`}
                style={{ transform: `scaleX(${fraction})` }}
              />
            </div>
            <span className="nis-gauge-value">
              {value.toFixed(2)} {inBounds ? '' : `(> ${upper?.toFixed(2)})`}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function TelemetryStrip({ frame, connected }: TelemetryStripProps) {
  return (
    <div className="telemetry-strip" data-testid="telemetry-strip">
      <div className="strip-group">
        <span className="strip-label">Link</span>
        <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
          <ConnectionBadge connected={connected} />
          {frame ? <GpsBadge enabled={frame.gps_enabled} /> : null}
        </div>
      </div>

      <DriftCounter driftPct={frame?.drift_pct ?? null} />

      {frame ? (
        <>
          <div className="strip-group">
            <span className="strip-label">Heading source</span>
            <HeadingSourceBadge source={frame.state.heading_source} />
          </div>

          <div className="strip-group">
            <span className="strip-label">Magnetometer gate</span>
            <MagGateBadge verdict={frame.mag_verdict} />
          </div>

          <div className="strip-group">
            <span className="strip-label">Stationary updates</span>
            <ZuptZaruLamps zupt={frame.zupt_active} zaru={frame.zaru_active} />
          </div>

          <div className="strip-group">
            <span className="strip-label">Model σ</span>
            <span>{frame.model_sigma_mps.toFixed(3)} m/s</span>
          </div>

          <div className="strip-group" style={{ flex: '1 1 auto' }}>
            <span className="strip-label">NIS (χ² consistency)</span>
            <NisGauges nis={frame.nis} bounds={frame.nis_bounds} />
          </div>
        </>
      ) : (
        <span className="strip-label">Waiting for the first telemetry frame…</span>
      )}
    </div>
  )
}
