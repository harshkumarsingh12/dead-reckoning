"""The 2D error-state Kalman filter.

OWNER: Sikruti  |  MILESTONE: M3  |  Spec: docs/BUILD_PLAN.md section 7

Why error-state rather than a plain EKF: the nominal state is propagated with the full
nonlinear equations while the filter runs on a small error state where linearisation is
exact where it matters, namely heading. ZUPT and ZARU become clean pseudo-measurements
on the error state. In 2D the extra machinery costs very little, and "why ESKF?" is
itself a judging differentiator.

Nominal state:  p = (px, py) world ENU metres,  v = (vx, vy) world m/s,  psi radians.
Estimated parameters:  gyro yaw bias b_g,  per-session velocity scale s.
Error state:  [dpx, dpy, dvx, dvy, dpsi, db_g, ds]  -- 7 states, ordering fixed in
dr_core.types.ERROR_STATE_ORDER. The covariance is kept on the error state.

Done when: fused beats model-only on ATE and RTE, per-channel NIS sits within its
chi-square bounds, and a 10 s stop produces zero position creep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dr_core.fusion.gating import ChiSquareGate, NisLogger
from dr_core.types import (
    ERROR_STATE_DIM,
    FilterState,
    HeadingSource,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import GpsFix, MatN, Vec2, VelocityEstimate

    Array = npt.NDArray[np.float64]


def wrap_angle(rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True, slots=True)
class EskfConfig:
    """Process noise and gate levels. Tune against recorded runs, not by feel."""

    sigma_v_process: float = 0.5  # m/s per sqrt(s), constant-velocity model slack
    sigma_psi_process: float = 0.01  # rad per sqrt(s)
    sigma_bg_random_walk: float = 1e-4  # rad/s per sqrt(s)
    sigma_scale_random_walk: float = 1e-5  # per sqrt(s)
    sigma_zupt: float = 0.02  # m/s, tight -- it is a strong physical statement
    sigma_zaru: float = 0.005  # rad/s
    gate_level: float = 0.95  # chi-square level, every channel
    freeze_scale_without_gps: bool = True
    velocity_cov_scale: float = 1.0  # covariance inflation multiplier (s_cov >= 1.0)
    min_velocity_variance: float = 0.0  # (m/s)^2 minimum variance floor on diagonal of R


class Eskf:
    """Stateful 2D ESKF. One instance per session or replay."""

    def __init__(self, config: EskfConfig | None = None) -> None:
        self.config = config if config is not None else EskfConfig()

        # Nominal state:
        self._t_ns: int = 0
        self._p: Vec2 = np.zeros(2, dtype=np.float64)  # position world ENU, m
        self._v: Vec2 = np.zeros(2, dtype=np.float64)  # velocity world ENU, m/s
        self._psi: float = 0.0  # heading radians, 0 = East, CCW positive
        self._bg: float = 0.0  # gyro yaw bias, rad/s
        self._s: float = 1.0  # velocity scale, dimensionless ~ 1.0

        # Error-state covariance (7x7): [dpx, dpy, dvx, dvy, dpsi, db_g, ds]
        self._P: MatN = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.01, 0.01]).astype(np.float64)

        self._gps_enabled: bool = True
        self._heading_source: HeadingSource = HeadingSource.GYRO

        self._origin_lat: float | None = None
        self._origin_lon: float | None = None

        # Gating and NIS loggers
        self._vel_gate = ChiSquareGate(dof=2, level=self.config.gate_level)
        self._zupt_gate = ChiSquareGate(dof=2, level=self.config.gate_level)
        self._zaru_gate = ChiSquareGate(dof=1, level=self.config.gate_level)
        self._mag_gate = ChiSquareGate(dof=1, level=self.config.gate_level)
        self._gps_gate = ChiSquareGate(dof=2, level=self.config.gate_level)

        self._nis_logger = NisLogger(
            {
                "velocity": 2,
                "zupt": 2,
                "zaru": 1,
                "heading": 1,
                "gps": 2,
            }
        )

    # -------------------------------------------------------------- prediction

    def predict(self, t_ns: int, gyro_yaw_rate: float) -> None:
        """Propagate the nominal state and the error-state covariance to ``t_ns``.

        psi advances by (gyro_yaw - b_g) * dt; p advances by v * dt; v follows a
        constant-velocity model with process noise, because the learned velocity update
        is the anchor that actually corrects it.
        """
        if self._t_ns == 0:
            self._t_ns = t_ns
            return

        dt_s = (t_ns - self._t_ns) * 1e-9
        if dt_s <= 0:
            return

        # 1. Propagate nominal state
        self._psi += (gyro_yaw_rate - self._bg) * dt_s
        self._psi = wrap_angle(self._psi)
        self._p += self._v * dt_s
        # v, bg, s remain constant in predict step

        # 2. Error-state transition matrix F (7x7)
        # [dpx, dpy, dvx, dvy, dpsi, db_g, ds]
        mat_f = np.eye(ERROR_STATE_DIM, dtype=np.float64)
        mat_f[0, 2] = dt_s  # dpx += dvx * dt
        mat_f[1, 3] = dt_s  # dpy += dvy * dt
        mat_f[4, 5] = -dt_s  # dpsi += -db_g * dt

        # Process noise Q (7x7)
        q_scale = (
            0.0
            if (not self._gps_enabled and self.config.freeze_scale_without_gps)
            else (self.config.sigma_scale_random_walk**2) * dt_s
        )
        mat_q = np.diag(
            [
                0.0,
                0.0,
                (self.config.sigma_v_process**2) * dt_s,
                (self.config.sigma_v_process**2) * dt_s,
                (self.config.sigma_psi_process**2) * dt_s,
                (self.config.sigma_bg_random_walk**2) * dt_s,
                q_scale,
            ]
        ).astype(np.float64)

        # 3. Propagate covariance
        self._P = mat_f @ self._P @ mat_f.T + mat_q
        self._t_ns = t_ns

    # ------------------------------------------------------------ measurements

    def update_velocity(self, estimate: VelocityEstimate) -> bool:
        """Primary anchor: the learned velocity, fused IN THE DEVICE FRAME.

        Measurement model: h(x) = (1/s) * R(-psi) @ v_world.

        Fusing in the device frame rather than the world frame is the key wiring
        decision. Because h depends on psi, the Jacobian carries a dh/dpsi term, so
        every velocity update also corrects heading -- which is what stops heading
        drift from bending the whole path through turns. It carries a dh/ds term too,
        making the scale observable.

        R is the model's own NLL-trained covariance for this window, used directly.

        Returns:
            True if the update was accepted, False if the chi-square gate rejected it.
        """
        s = self._s
        psi = self._psi
        vx, vy = self._v[0], self._v[1]

        c, sn = float(np.cos(psi)), float(np.sin(psi))
        # R(-psi) = [[c, sn], [-sn, c]]
        # h(x) = (1/s) * R(-psi) @ v_world
        h_pred = np.array(
            [
                (c * vx + sn * vy) / s,
                (-sn * vx + c * vy) / s,
            ],
            dtype=np.float64,
        )

        # Innovation
        y = estimate.v_dev - h_pred

        # Measurement Jacobian H (2x7)
        # States: [dpx, dpy, dvx, dvy, dpsi, db_g, ds]
        mat_h = np.zeros((2, ERROR_STATE_DIM), dtype=np.float64)
        mat_h[0, 2] = c / s
        mat_h[0, 3] = sn / s
        mat_h[1, 2] = -sn / s
        mat_h[1, 3] = c / s

        # Heading sensitivity (dh/dpsi)
        mat_h[0, 4] = (-sn * vx + c * vy) / s
        mat_h[1, 4] = (-c * vx - sn * vy) / s

        # Scale sensitivity (dh/ds)
        if self._gps_enabled or not self.config.freeze_scale_without_gps:
            mat_h[0, 6] = -h_pred[0] / s
            mat_h[1, 6] = -h_pred[1] / s
        else:
            mat_h[0, 6] = 0.0
            mat_h[1, 6] = 0.0

        mat_r = estimate.cov.copy() * self.config.velocity_cov_scale
        if self.config.min_velocity_variance > 0.0:
            mat_r[0, 0] = max(float(mat_r[0, 0]), self.config.min_velocity_variance)
            mat_r[1, 1] = max(float(mat_r[1, 1]), self.config.min_velocity_variance)
        mat_s = mat_h @ self._P @ mat_h.T + mat_r

        accepted, nis = self._vel_gate.accept(y, mat_s)
        self._nis_logger.record("velocity", nis, accepted)

        if not accepted:
            return False

        # Kalman update
        mat_k = self._P @ mat_h.T @ np.linalg.inv(mat_s)
        dx = mat_k @ y

        # Joseph form covariance update
        mat_ikh = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k @ mat_h
        self._P = mat_ikh @ self._P @ mat_ikh.T + mat_k @ mat_r @ mat_k.T

        self._inject_and_reset(dx)
        self._heading_source = HeadingSource.VELOCITY
        return True

    def update_zupt(self, t_ns: int) -> bool:
        """Zero-velocity update. Measure v = 0 with small R.

        Physics-based and completely independent of the learned model, which is exactly
        why it is worth having: when the network is wrong, this still holds.
        """
        # Measurement z = [0, 0], h(x) = v_world = [vx, vy]
        y = -self._v.copy()

        mat_h = np.zeros((2, ERROR_STATE_DIM), dtype=np.float64)
        mat_h[0, 2] = 1.0
        mat_h[1, 3] = 1.0

        mat_r = np.diag([self.config.sigma_zupt**2, self.config.sigma_zupt**2]).astype(np.float64)
        mat_s = mat_h @ self._P @ mat_h.T + mat_r

        accepted, nis = self._zupt_gate.accept(y, mat_s)
        self._nis_logger.record("zupt", nis, accepted)

        if not accepted:
            return False

        mat_k = self._P @ mat_h.T @ np.linalg.inv(mat_s)
        dx = mat_k @ y

        mat_ikh = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k @ mat_h
        self._P = mat_ikh @ self._P @ mat_ikh.T + mat_k @ mat_r @ mat_k.T

        self._inject_and_reset(dx)
        return True

    def update_zaru(self, t_ns: int, gyro_yaw_rate: float) -> bool:
        """Zero angular-rate update. Measure (gyro_yaw - b_g) = 0, pinning the bias."""
        # Measurement z = 0, h(x) = gyro_yaw_rate - b_g
        y = -(gyro_yaw_rate - self._bg)

        mat_h = np.zeros((1, ERROR_STATE_DIM), dtype=np.float64)
        mat_h[0, 5] = -1.0

        mat_r = np.array([[self.config.sigma_zaru**2]], dtype=np.float64)
        mat_s = mat_h @ self._P @ mat_h.T + mat_r

        accepted, nis = self._zaru_gate.accept(np.array([y]), mat_s)
        self._nis_logger.record("zaru", nis, accepted)

        if not accepted:
            return False

        mat_k = self._P @ mat_h.T @ np.linalg.inv(mat_s)
        dx = (mat_k * y).flatten()

        mat_ikh = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k @ mat_h
        self._P = mat_ikh @ self._P @ mat_ikh.T + mat_k @ mat_r @ mat_k.T

        self._inject_and_reset(dx)
        self._heading_source = HeadingSource.ZARU
        return True

    def update_magnetometer(self, t_ns: int, heading_rad: float, sigma_rad: float) -> bool:
        """Heading update from an already triple-gated magnetometer reading."""
        y = wrap_angle(heading_rad - self._psi)

        mat_h = np.zeros((1, ERROR_STATE_DIM), dtype=np.float64)
        mat_h[0, 4] = 1.0

        mat_r = np.array([[sigma_rad**2]], dtype=np.float64)
        mat_s = mat_h @ self._P @ mat_h.T + mat_r

        accepted, nis = self._mag_gate.accept(np.array([y]), mat_s)
        self._nis_logger.record("heading", nis, accepted)

        if not accepted:
            return False

        mat_k = self._P @ mat_h.T @ np.linalg.inv(mat_s)
        dx = (mat_k * y).flatten()

        mat_ikh = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k @ mat_h
        self._P = mat_ikh @ self._P @ mat_ikh.T + mat_k @ mat_r @ mat_k.T

        self._inject_and_reset(dx)
        self._heading_source = HeadingSource.MAGNETOMETER
        return True

    def update_gps(self, fix: GpsFix) -> bool:
        """Position update, plus course-over-ground heading above a speed threshold.

        GPS is both the training label offline and an opportunistic reset online. It is
        also the only observation window in which the velocity scale ``s`` is
        separable from speed -- when GPS drops, ``s`` freezes (build plan 7.2).
        """
        if not self._gps_enabled:
            return False

        if self._origin_lat is None or self._origin_lon is None:
            self._origin_lat = fix.lat_deg
            self._origin_lon = fix.lon_deg

        # Local ENU coordinates relative to session origin
        lat_rad = np.deg2rad(self._origin_lat)
        d_lat = fix.lat_deg - self._origin_lat
        d_lon = fix.lon_deg - self._origin_lon

        # WGS84 approx: meters per degree
        m_per_deg_lat = 111132.92
        m_per_deg_lon = 111412.84 * np.cos(lat_rad)

        p_gps = np.array([d_lon * m_per_deg_lon, d_lat * m_per_deg_lat], dtype=np.float64)
        y = p_gps - self._p

        mat_h = np.zeros((2, ERROR_STATE_DIM), dtype=np.float64)
        mat_h[0, 0] = 1.0
        mat_h[1, 1] = 1.0

        r_pos = max(0.1, fix.accuracy_m)
        mat_r = np.diag([r_pos**2, r_pos**2]).astype(np.float64)
        mat_s = mat_h @ self._P @ mat_h.T + mat_r

        accepted, nis = self._gps_gate.accept(y, mat_s)
        self._nis_logger.record("gps", nis, accepted)

        if not accepted:
            return False

        mat_k = self._P @ mat_h.T @ np.linalg.inv(mat_s)
        dx = mat_k @ y

        mat_ikh = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k @ mat_h
        self._P = mat_ikh @ self._P @ mat_ikh.T + mat_k @ mat_r @ mat_k.T

        self._inject_and_reset(dx)

        # Course over ground heading update if moving at sufficient speed
        if fix.speed_mps is not None and fix.speed_mps > 1.0 and fix.course_rad is not None:
            y_course = wrap_angle(fix.course_rad - self._psi)
            mat_h_course = np.zeros((1, ERROR_STATE_DIM), dtype=np.float64)
            mat_h_course[0, 4] = 1.0
            mat_r_course = np.array([[0.1**2]], dtype=np.float64)
            mat_s_course = mat_h_course @ self._P @ mat_h_course.T + mat_r_course
            mat_k_course = self._P @ mat_h_course.T @ np.linalg.inv(mat_s_course)
            dx_course = (mat_k_course * y_course).flatten()
            mat_ikh_course = np.eye(ERROR_STATE_DIM, dtype=np.float64) - mat_k_course @ mat_h_course
            self._P = (
                mat_ikh_course @ self._P @ mat_ikh_course.T
                + mat_k_course @ mat_r_course @ mat_k_course.T
            )
            self._inject_and_reset(dx_course)
            self._heading_source = HeadingSource.GPS_COURSE

        return True

    # ----------------------------------------------------------------- plumbing

    def _inject_and_reset(self, dx: Array) -> None:
        """Fold the error state into the nominal state and zero it.

        With a scalar heading angle in 2D the covariance-reset Jacobian is essentially
        identity. Apply it anyway and say so, rather than leaving a reader wondering
        whether it was forgotten.
        """
        self._p += dx[0:2]
        self._v += dx[2:4]
        self._psi = wrap_angle(self._psi + float(dx[4]))
        self._bg += float(dx[5])
        self._s += float(dx[6])

        # Reset Jacobian G = I_7 for 2D error state
        mat_g = np.eye(ERROR_STATE_DIM, dtype=np.float64)
        self._P = mat_g @ self._P @ mat_g.T

    @property
    def state(self) -> FilterState:
        """Current nominal state and error-state covariance."""
        return FilterState(
            t_ns=self._t_ns,
            p_world=self._p.copy(),
            v_world=self._v.copy(),
            psi_rad=self._psi,
            gyro_bias_z=self._bg,
            scale=self._s,
            cov=self._P.copy(),
            heading_source=self._heading_source,
        )

    def set_gps_enabled(self, enabled: bool) -> None:
        """The demo's GPS-off toggle. Freezes the scale state when disabled."""
        self._gps_enabled = enabled
