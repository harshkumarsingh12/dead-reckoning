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

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import FilterState, GpsFix, VelocityEstimate

    Array = npt.NDArray[np.float64]


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


class Eskf:
    """Stateful 2D ESKF. One instance per session or replay."""

    def __init__(self, config: EskfConfig | None = None) -> None:
        raise NotImplementedError("M3 -- owner: Sikruti")

    # -------------------------------------------------------------- prediction

    def predict(self, t_ns: int, gyro_yaw_rate: float) -> None:
        """Propagate the nominal state and the error-state covariance to ``t_ns``.

        psi advances by (gyro_yaw - b_g) * dt; p advances by v * dt; v follows a
        constant-velocity model with process noise, because the learned velocity update
        is the anchor that actually corrects it.
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

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
        raise NotImplementedError("M3 -- owner: Sikruti")

    def update_zupt(self, t_ns: int) -> bool:
        """Zero-velocity update. Measure v = 0 with small R.

        Physics-based and completely independent of the learned model, which is exactly
        why it is worth having: when the network is wrong, this still holds.
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

    def update_zaru(self, t_ns: int, gyro_yaw_rate: float) -> bool:
        """Zero angular-rate update. Measure (gyro_yaw - b_g) = 0, pinning the bias."""
        raise NotImplementedError("M3 -- owner: Sikruti")

    def update_magnetometer(self, t_ns: int, heading_rad: float, sigma_rad: float) -> bool:
        """Heading update from an already triple-gated magnetometer reading."""
        raise NotImplementedError("M3 -- owner: Sikruti")

    def update_gps(self, fix: GpsFix) -> bool:
        """Position update, plus course-over-ground heading above a speed threshold.

        GPS is both the training label offline and an opportunistic reset online. It is
        also the only observation window in which the velocity scale ``s`` is
        separable from speed -- when GPS drops, ``s`` freezes (build plan 7.2).
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

    # ----------------------------------------------------------------- plumbing

    def _inject_and_reset(self, dx: Array) -> None:
        """Fold the error state into the nominal state and zero it.

        With a scalar heading angle in 2D the covariance-reset Jacobian is essentially
        identity. Apply it anyway and say so, rather than leaving a reader wondering
        whether it was forgotten.
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

    @property
    def state(self) -> FilterState:
        """Current nominal state and error-state covariance."""
        raise NotImplementedError("M3 -- owner: Sikruti")

    def set_gps_enabled(self, enabled: bool) -> None:
        """The demo's GPS-off toggle. Freezes the scale state when disabled."""
        raise NotImplementedError("M3 -- owner: Sikruti")
