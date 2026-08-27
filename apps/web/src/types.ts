/**
 * The TypeScript mirror of src/dr_core/types.py — the frozen contract, wire side.
 *
 * OWNER: shared. Changing this means changing types.py in the same PR, and telling
 * the group chat first. A mismatch between the two shows up as a silently missing
 * field on the telemetry strip rather than as an error.
 *
 * Conventions (docs/CONVENTIONS.md): timestamps are boot-monotonic nanoseconds and
 * arrive as strings because they exceed Number.MAX_SAFE_INTEGER; angles are radians;
 * positions are metres in the world ENU frame.
 */

export type HeadingSource =
  | 'gyro'
  | 'magnetometer'
  | 'velocity'
  | 'zaru'
  | 'gps_course'

export type MagGateVerdict =
  | 'accepted'
  | 'rejected_magnitude'
  | 'rejected_dip'
  | 'rejected_innovation'

export interface FilterState {
  /** Boot-monotonic nanoseconds, as a string: ns exceeds Number.MAX_SAFE_INTEGER. */
  t_ns: string
  /** [east, north] metres. */
  p_world: [number, number]
  v_world: [number, number]
  psi_rad: number
  gyro_bias_z: number
  scale: number
  /** Row-major 7x7 error-state covariance, ordered by ERROR_STATE_ORDER. */
  cov: number[][]
  heading_source: HeadingSource
}

export interface TelemetryFrame {
  t_ns: string
  state: FilterState
  /** Raw double integration — the second dot that visibly spirals away. */
  baseline_p_world: [number, number] | null
  truth_p_world: [number, number] | null
  /** Channel name to latest NIS. */
  nis: Record<string, number>
  /** Channel name to [lower, upper] chi-square bound. */
  nis_bounds: Record<string, [number, number]>
  zupt_active: boolean
  zaru_active: boolean
  mag_verdict: MagGateVerdict
  model_sigma_mps: number
  gps_enabled: boolean
  distance_travelled_m: number
  /** The headline number. Null before enough distance has been walked. */
  drift_pct: number | null
}
