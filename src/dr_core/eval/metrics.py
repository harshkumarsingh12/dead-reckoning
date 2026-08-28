"""ATE, RTE, drift, and the model calibration coverage test.

OWNER: Sikruti  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 8

Targets on a 100-300 m indoor loop:

    metric                      acceptable        strong
    drift (final / distance)    < 5%              < 2-3%
    RTE over 60 s               a few metres      1-2 m
    NIS / NEES                  within bounds     across carry positions
    model coverage at 1 sigma   ~68%              holds across carry positions
    inference per window        < 10 ms           on-device viable
    raw-integration baseline    > 100%            (contrast, not a target)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

from dr_core.types import Trajectory

NS_PER_S = 1_000_000_000


def resample_to(trajectory: Trajectory, t_ns: npt.NDArray[Any]) -> Trajectory:
    """Interpolate a trajectory onto a given timebase. Used by every metric above.

    Timestamps are int64 nanoseconds. Converted to float64 only for interpolation
    arithmetic, with the original int64 timebase preserved in the result.
    """
    src_t = trajectory.t_ns.astype(np.float64)
    dst_t = np.asarray(t_ns, dtype=np.float64)

    px = np.interp(dst_t, src_t, trajectory.p_world[:, 0])
    py = np.interp(dst_t, src_t, trajectory.p_world[:, 1])
    p_world = np.column_stack([px, py]).astype(np.float64)

    psi_rad: FloatArray | None = None
    if trajectory.psi_rad is not None:
        psi_unwrapped = np.unwrap(trajectory.psi_rad)
        psi_interp = np.interp(dst_t, src_t, psi_unwrapped)
        psi_rad = np.asarray((psi_interp + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)

    return Trajectory(
        t_ns=np.asarray(t_ns, dtype=np.int64),
        p_world=p_world,
        psi_rad=psi_rad,
        label=trajectory.label,
    )


def _umeyama_se2(src: Array, dst: Array) -> tuple[Array, Array]:
    """Closed-form SE(2) alignment (rotation + translation), Umeyama method.

    Returns (R, t) such that ``dst ≈ (R @ src.T).T + t``.
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance
    cov = dst_c.T @ src_c / len(src)
    u, _, vt = np.linalg.svd(cov)
    # Ensure a proper rotation (det = +1)
    d = np.array([1.0, np.linalg.det(u) * np.linalg.det(vt)])
    rot = u @ np.diag(d) @ vt
    trans = mu_dst - rot @ mu_src
    return rot, trans


def ate(estimate: Trajectory, truth: Trajectory, align: bool = True) -> float:
    """Absolute Trajectory Error: RMSE of position after alignment.

    Args:
        estimate: the estimated path.
        truth: ground truth. Resampled onto the estimate's timestamps internally.
        align: apply an SE(2) alignment first. Standard for ATE, but report the
            unaligned number too when the start point is genuinely known -- in this
            demo it is, since the walk starts on a marked spot.

    Returns:
        RMSE in metres.
    """
    truth_r = resample_to(truth, estimate.t_ns)
    est_p = estimate.p_world.copy()
    tru_p = truth_r.p_world.copy()

    if align and len(estimate) >= 2:
        rot, trans = _umeyama_se2(est_p, tru_p)
        est_p = (rot @ est_p.T).T + trans

    errors = np.linalg.norm(est_p - tru_p, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def rte(estimate: Trajectory, truth: Trajectory, window_s: float = 60.0) -> float:
    """Relative Trajectory Error over a fixed window.

    Reflects the drift RATE rather than accumulated error, so it stays comparable
    across runs of different lengths. This is the number to quote when comparing two
    models on differently sized loops.
    """
    truth_r = resample_to(truth, estimate.t_ns)
    window_ns = int(window_s * NS_PER_S)

    t = estimate.t_ns
    est_p = estimate.p_world
    tru_p = truth_r.p_world

    errors: list[float] = []
    for i in range(len(t)):
        t_end = t[i] + window_ns
        # Find the closest index at or past t_end
        j = int(np.searchsorted(t, t_end))
        if j >= len(t):
            break
        delta_est = est_p[j] - est_p[i]
        delta_tru = tru_p[j] - tru_p[i]
        errors.append(float(np.linalg.norm(delta_est - delta_tru)))

    if not errors:
        return 0.0
    return float(np.mean(errors))


def final_error(estimate: Trajectory, truth: Trajectory) -> float:
    """Distance between the estimated and true endpoints, metres.

    On a closed loop this is the loop-closure error -- the closing shot of the demo.
    """
    truth_r = resample_to(truth, estimate.t_ns)
    return float(np.linalg.norm(estimate.p_world[-1] - truth_r.p_world[-1]))


def drift_pct(estimate: Trajectory, truth: Trajectory) -> float:
    """final_error / distance travelled, as a percentage. The headline number.

    Distance is computed along the TRUTH path, not the estimate.
    """
    truth_r = resample_to(truth, estimate.t_ns)
    # Total distance along truth
    diffs = np.diff(truth_r.p_world, axis=0)
    distance = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if distance < 1e-9:
        return 0.0
    fe = float(np.linalg.norm(estimate.p_world[-1] - truth_r.p_world[-1]))
    return (fe / distance) * 100.0


def calibration_coverage(errors: Array, sigmas: Array, k: float = 1.0) -> float:
    """Fraction of held-out errors falling inside k sigma, per axis.

    MANDATORY before the model's covariance is allowed anywhere near the filter's R.
    A well-calibrated 1-sigma lands near 0.68. Materially below that means the model
    is over-confident, which silently poisons fusion and makes the on-screen ellipse
    indefensible; materially above means it is under-confident and the filter is
    ignoring information it has.

    ``errors`` and ``sigmas`` are one axis at a time (call twice for a 2D velocity,
    once per axis, per the docstring above) -- ``errors[i]`` is the signed residual
    for sample ``i`` and ``sigmas[i]`` is the model's claimed 1-sigma for that same
    sample, so shapes must match elementwise.
    """
    errors = np.asarray(errors, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    inside = np.abs(errors) <= (k * sigmas)
    return float(np.mean(inside))
