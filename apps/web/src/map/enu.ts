/**
 * Local ENU metres <-> lat/lng, and the 1-sigma uncertainty ellipse from the
 * position sub-block of the error-state covariance.
 *
 * OWNER: Tanmay  |  MILESTONE: M4  |  Spec: docs/CONVENTIONS.md sections 1, 5
 *
 * `enuToLatLng` is the exact inverse of the flat-earth projection in
 * `services/gateway/hub.py`'s `_project_enu`. The two cannot literally share source
 * across the Python/TypeScript boundary, so keep them in lockstep by inspection if
 * either ever changes -- this is the one place on the web side that assumes the
 * gateway's specific equirectangular approximation.
 */

const EARTH_RADIUS_M = 6_371_000

export function enuToLatLng(
  eastM: number,
  northM: number,
  originLatDeg: number,
  originLonDeg: number,
): [number, number] {
  const originLatRad = (originLatDeg * Math.PI) / 180
  const lat = originLatDeg + (northM / EARTH_RADIUS_M) * (180 / Math.PI)
  const lon = originLonDeg + (eastM / (EARTH_RADIUS_M * Math.cos(originLatRad))) * (180 / Math.PI)
  return [lat, lon]
}

export interface EllipseParams {
  semiMajorM: number
  semiMinorM: number
  /** Radians, CCW from East -- the world-frame convention (docs/CONVENTIONS.md §1). */
  rotationRad: number
}

/**
 * Eigendecomposition of the 2x2 position sub-block of the 7x7 error-state covariance.
 * Index order is fixed by `ERROR_STATE_ORDER` = (dpx, dpy, ...) in dr_core/types.py,
 * so the position block is always rows/cols 0-1.
 */
export function ellipseFromCov(cov: number[][]): EllipseParams {
  const a = cov[0]?.[0] ?? 0
  const b = cov[0]?.[1] ?? 0
  const d = cov[1]?.[1] ?? 0

  const trace = a + d
  const det = a * d - b * b
  const discriminant = Math.max(trace * trace / 4 - det, 0)
  const term = Math.sqrt(discriminant)
  const lambda1 = trace / 2 + term // larger eigenvalue -> semi-major axis
  const lambda2 = trace / 2 - term

  // Eigenvector of a symmetric 2x2 [[a,b],[b,d]] for eigenvalue L is (b, L-a) when
  // b != 0 (derived from the first eigen-equation row); if b == 0 the matrix is
  // already diagonal and the major axis lies along whichever axis has the larger
  // variance.
  const rotationRad =
    Math.abs(b) < 1e-15 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(lambda1 - a, b)

  return {
    semiMajorM: Math.sqrt(Math.max(lambda1, 0)),
    semiMinorM: Math.sqrt(Math.max(lambda2, 0)),
    rotationRad,
  }
}

/** Points (lat, lng) tracing the 1-sigma ellipse, centred at (centerEastM, centerNorthM). */
export function ellipseLatLngPoints(
  centerEastM: number,
  centerNorthM: number,
  ellipse: EllipseParams,
  originLatDeg: number,
  originLonDeg: number,
  segments = 48,
): [number, number][] {
  const { semiMajorM, semiMinorM, rotationRad } = ellipse
  const cosR = Math.cos(rotationRad)
  const sinR = Math.sin(rotationRad)
  const points: [number, number][] = []
  for (let i = 0; i <= segments; i++) {
    const t = (2 * Math.PI * i) / segments
    const x = semiMajorM * Math.cos(t)
    const y = semiMinorM * Math.sin(t)
    const east = centerEastM + x * cosR - y * sinR
    const north = centerNorthM + x * sinR + y * cosR
    points.push(enuToLatLng(east, north, originLatDeg, originLonDeg))
  }
  return points
}
