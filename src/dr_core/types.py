"""THE FROZEN CONTRACT.

Every subsystem in this repository talks to every other subsystem through the types
defined here, and through nothing else. That is what lets six people build in parallel:
the web UI does not wait for the ESKF, and the ESKF does not wait for the model. Build
against these dataclasses and hand-written mock instances, then swap in real data.

CHANGING THIS FILE IS A TEAM DECISION. Announce it in the group chat and tag the
owners of every affected area (see CONTRIBUTING.md) before you open the PR. A silent
change here breaks four people at once.

Conventions enforced throughout (full statement in docs/CONVENTIONS.md):

  * Time      -- int64 nanoseconds in the device BOOT-MONOTONIC domain. Never float
                 seconds, never wall clock, never network-arrival time.
  * Frames    -- ``world`` is ENU (x=East, y=North, z=Up), metres.
                 ``body``  is the raw device frame.
                 ``dev``   is the gravity-aligned, heading-agnostic frame the learned
                           velocity model regresses into (build plan 6.4).
  * Angles    -- radians internally, everywhere. Degrees only at the UI boundary.
  * Units     -- strict SI: m, m/s, m/s^2, rad, rad/s, tesla.
  * Naming    -- a vector carries its frame: ``v_world``, ``v_dev``, ``a_body``. A bare
                 ``v`` is a review comment waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Vec2 = npt.NDArray[np.float64]  # shape (2,)
    Vec3 = npt.NDArray[np.float64]  # shape (3,)
    Mat2 = npt.NDArray[np.float64]  # shape (2, 2)
    MatN = npt.NDArray[np.float64]  # shape (n, n)
    Quat = npt.NDArray[np.float64]  # shape (4,), (w, x, y, z)
    FloatArray = npt.NDArray[np.float64]  # shape (n,) or (n, k)
    IntArray = npt.NDArray[np.int64]  # shape (n,)


# --------------------------------------------------------------------------- enums


class HeadingSource(StrEnum):
    """Which channel most recently corrected heading.

    Shown live on the telemetry strip. Judges ask "what happens when the magnetometer
    fails indoors?" and this is the on-screen answer (build plan 12, Q3).
    """

    GYRO = "gyro"  # dead reckoning only, no correction
    MAGNETOMETER = "magnetometer"  # mag heading passed the triple gate
    VELOCITY = "velocity"  # device-frame velocity update's d/dpsi term
    ZARU = "zaru"  # stationary, gyro bias pinned
    GPS_COURSE = "gps_course"  # course over ground, at sufficient speed


class MagGateVerdict(StrEnum):
    """Outcome of the magnetometer triple gate (build plan 6.3).

    A reading is accepted only if magnitude AND dip AND the chi-square innovation test
    all pass. Magnitude alone is a weak check: indoor disturbances routinely rotate the
    field while leaving its strength near normal.
    """

    ACCEPTED = "accepted"
    REJECTED_MAGNITUDE = "rejected_magnitude"
    REJECTED_DIP = "rejected_dip"
    REJECTED_INNOVATION = "rejected_innovation"


class CarryPosition(StrEnum):
    """How the phone was carried during a run. Drives augmentation and eval slicing."""

    HAND = "hand"
    POCKET = "pocket"
    BAG = "bag"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------- raw measurements


@dataclass(frozen=True, slots=True)
class ImuSample:
    """One synchronised inertial sample, stamped at capture on the device.

    Attributes:
        t_ns: boot-monotonic nanoseconds, assigned on the device at capture. Never
            assigned on network arrival -- that is the classic silent killer
            (build plan 5).
        a_body: specific force, m/s^2, raw device frame (gravity still present).
        w_body: angular rate, rad/s, raw device frame.
        m_body: magnetic field, tesla, raw device frame. None when no magnetometer
            sample arrived with this tick.
    """

    t_ns: int
    a_body: Vec3
    w_body: Vec3
    m_body: Vec3 | None = None


@dataclass(frozen=True, slots=True)
class GpsFix:
    """A GNSS fix, in the same clock domain as the IMU.

    Fixes arrive hundreds of milliseconds late, so ``t_ns`` is the CAPTURE time
    (``Location.getElapsedRealtimeNanos()`` on Android), not the arrival time. That is
    what lets the reorder buffer fuse it where it actually belongs on the timeline.
    """

    t_ns: int
    lat_deg: float
    lon_deg: float
    accuracy_m: float
    speed_mps: float | None = None
    course_rad: float | None = None  # course over ground, radians, ENU convention
    altitude_m: float | None = None


# ------------------------------------------------------------------- derived estimates


@dataclass(frozen=True, slots=True)
class OrientationEstimate:
    """AHRS output: which way is down, which way is forward."""

    t_ns: int
    q_world_body: Quat  # (w, x, y, z), rotates body -> world
    mag_verdict: MagGateVerdict = MagGateVerdict.REJECTED_INNOVATION


@dataclass(frozen=True, slots=True)
class VelocityEstimate:
    """The learned model's output: planar velocity WITH an honest covariance.

    The covariance is trained jointly under a Gaussian negative-log-likelihood
    objective, not bolted on after an MSE fit, and its calibration is verified by a
    coverage test on held-out trajectories (build plan 6.4). This matters: an
    over-confident covariance silently poisons the filter and makes the on-screen
    uncertainty ellipse indefensible under questioning.

    Attributes:
        v_dev: shape (2,), m/s, in the gravity-aligned heading-agnostic device frame.
            NOT world frame -- fusing in the device frame is what puts a d/dpsi term in
            the measurement Jacobian, so every velocity update also corrects heading
            (build plan 7.1).
        cov: shape (2, 2), m^2/s^2. Becomes R for the filter update directly.
    """

    t_ns: int
    v_dev: Vec2
    cov: Mat2

    @property
    def sigma_max(self) -> float:
        """Largest 1-sigma axis, m/s. What the telemetry strip displays."""
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.cov))))


# Error-state ordering is fixed and must not be permuted. Every Jacobian in
# dr_core.fusion indexes against these.
ERROR_STATE_ORDER: tuple[str, ...] = ("dpx", "dpy", "dvx", "dvy", "dpsi", "db_g", "ds")
ERROR_STATE_DIM: int = len(ERROR_STATE_ORDER)


@dataclass(frozen=True, slots=True)
class FilterState:
    """The ESKF nominal state plus its error-state covariance, at one instant."""

    t_ns: int
    p_world: Vec2  # position, m, ENU
    v_world: Vec2  # velocity, m/s, ENU
    psi_rad: float  # heading, radians, 0 = East, CCW positive
    gyro_bias_z: float  # rad/s
    scale: float  # per-session velocity scale `s`, dimensionless, ~1.0
    cov: MatN  # (7, 7) error-state covariance, ordered by ERROR_STATE_ORDER
    heading_source: HeadingSource = HeadingSource.GYRO


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A time-ordered path.

    Ground truth, an estimate, and a baseline all use this one shape, so the eval
    harness compares any two without special-casing.
    """

    t_ns: IntArray  # (n,) boot-monotonic nanoseconds
    p_world: FloatArray  # (n, 2) metres, ENU
    psi_rad: FloatArray | None = None  # (n,) radians
    label: str = "unnamed"

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


# ------------------------------------------------------------------------- telemetry


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """One frame pushed over the WebSocket to the live UI.

    This is the wire format between ``services/gateway`` and ``apps/web``. It carries
    everything the on-screen telemetry strip needs, which is what converts a nice demo
    into "these people know exactly what their filter is doing" during Q&A
    (build plan 6.8).
    """

    t_ns: int
    state: FilterState
    baseline_p_world: Vec2 | None = None  # raw double integration, the diverging dot
    truth_p_world: Vec2 | None = None  # known marked path, when available
    nis: dict[str, float] = field(default_factory=dict)  # channel -> latest NIS
    nis_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    zupt_active: bool = False
    zaru_active: bool = False
    mag_verdict: MagGateVerdict = MagGateVerdict.REJECTED_INNOVATION
    model_sigma_mps: float = 0.0
    gps_enabled: bool = True
    distance_travelled_m: float = 0.0
    drift_pct: float | None = None  # live headline number: error / distance
    # The world-frame origin, so the browser can convert p_world (local ENU metres)
    # back to lat/lon for the real basemap. None until the session's first GPS fix
    # sets an origin (see services/gateway/hub.py). Additive field -- every existing
    # TelemetryFrame call site keeps working unchanged.
    origin_lat_deg: float | None = None
    origin_lon_deg: float | None = None


# --------------------------------------------------------------------------- session


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Everything needed to interpret a recording, including how to fix its clock.

    ``boot_to_utc_offset_ns`` is estimated once at session start (re-estimated on
    detected drift) and is what lets a boot-monotonic recording be aligned against
    UTC-stamped GPS labels. Without it, training labels are silently misaligned by
    hundreds of milliseconds and nothing downstream can recover.
    """

    session_id: str
    device_model: str
    carry_position: CarryPosition = CarryPosition.UNKNOWN
    imu_rate_hz: float = 200.0
    boot_to_utc_offset_ns: int = 0
    origin_lat_deg: float | None = None  # ENU origin for the world frame
    origin_lon_deg: float | None = None
    gyro_bias_body: Vec3 | None = None  # from the stationary window at session start
    mag_hard_iron_body: Vec3 | None = None  # from the 10 s figure-8
    notes: str = ""
