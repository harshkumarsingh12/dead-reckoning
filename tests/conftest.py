"""Synthetic motion generators shared by every test.

These are real and implemented -- they are test infrastructure, not product code, and
the three frame invariants in test_frames.py are worthless without them.

Frames follow docs/CONVENTIONS.md: world is ENU, body is the raw device frame,
timestamps are int64 nanoseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from dr_core.types import CarryPosition, ImuSample, SessionMeta, Trajectory

GRAVITY = 9.80665
NS_PER_S = 1_000_000_000
RATE_HZ = 200.0


def _rotz(psi: float) -> np.ndarray:
    """Rotation about the world z axis by psi radians."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _synth(
    duration_s: float,
    a_world_fn: object,
    w_body_fn: object,
    psi_fn: object,
    rate_hz: float = RATE_HZ,
) -> list[ImuSample]:
    """Build an IMU stream from world-frame acceleration and body-frame angular rate.

    The device is held flat (no pitch or roll), so the body frame differs from world
    only by the yaw angle psi. Gravity is added in the body frame, exactly as a real
    accelerometer measures specific force.
    """
    n = int(duration_s * rate_hz)
    dt = 1.0 / rate_hz
    g_world = np.array([0.0, 0.0, GRAVITY])
    samples: list[ImuSample] = []
    for i in range(n):
        t = i * dt
        psi = float(psi_fn(t))  # type: ignore[operator]
        r_wb = _rotz(psi)
        a_world = np.asarray(a_world_fn(t), dtype=np.float64)  # type: ignore[operator]
        # Specific force in the body frame: measured accel plus the gravity reaction.
        a_body = r_wb.T @ (a_world + g_world)
        w_body = np.asarray(w_body_fn(t), dtype=np.float64)  # type: ignore[operator]
        samples.append(
            ImuSample(
                t_ns=round(t * NS_PER_S),
                a_body=a_body,
                w_body=w_body,
                m_body=r_wb.T @ np.array([20e-6, 0.0, -45e-6]),
            )
        )
    return samples


@pytest.fixture
def straight_line() -> tuple[list[ImuSample], Trajectory]:
    """20 s walking due East at 1.4 m/s after a 1 s acceleration ramp."""
    speed = 1.4
    ramp_s = 1.0
    duration = 20.0

    def a_world(t: float) -> list[float]:
        return [speed / ramp_s, 0.0, 0.0] if t < ramp_s else [0.0, 0.0, 0.0]

    samples = _synth(duration, a_world, lambda _t: [0.0, 0.0, 0.0], lambda _t: 0.0)

    t = np.arange(0.0, duration, 1.0 / RATE_HZ)
    x = np.where(t < ramp_s, 0.5 * (speed / ramp_s) * t**2, speed * (t - ramp_s / 2.0))
    truth = Trajectory(
        t_ns=(t * NS_PER_S).astype(np.int64),
        p_world=np.column_stack([x, np.zeros_like(x)]),
        psi_rad=np.zeros_like(t),
        label="straight_line",
    )
    return samples, truth


@pytest.fixture
def pure_turn() -> tuple[list[ImuSample], Trajectory]:
    """A 20 m radius circle walked at 1.4 m/s -- constant speed, constant yaw rate."""
    speed, radius = 1.4, 20.0
    omega = speed / radius
    duration = 2.0 * np.pi / omega

    def a_world(t: float) -> list[float]:
        # Centripetal acceleration, pointing at the circle centre.
        return [-speed * omega * np.cos(omega * t), -speed * omega * np.sin(omega * t), 0.0]

    samples = _synth(
        duration,
        a_world,
        lambda _t: [0.0, 0.0, omega],
        lambda t: omega * t,
    )

    t = np.arange(0.0, duration, 1.0 / RATE_HZ)
    truth = Trajectory(
        t_ns=(t * NS_PER_S).astype(np.int64),
        p_world=np.column_stack([radius * np.sin(omega * t), radius * (1 - np.cos(omega * t))]),
        psi_rad=omega * t,
        label="pure_turn",
    )
    return samples, truth


@pytest.fixture
def rotation_in_place() -> tuple[list[ImuSample], Trajectory]:
    """The phone spun about yaw while the person does not move.

    This is the invariant that catches the classic frame and lever-arm failure. Straight
    line and pure turn both pass under a wrong-frame implementation; this one does not.
    Position must stay exactly at the origin.
    """
    omega = np.pi / 2  # 90 deg/s
    duration = 8.0
    samples = _synth(
        duration,
        lambda _t: [0.0, 0.0, 0.0],
        lambda _t: [0.0, 0.0, omega],
        lambda t: omega * t,
    )
    t = np.arange(0.0, duration, 1.0 / RATE_HZ)
    truth = Trajectory(
        t_ns=(t * NS_PER_S).astype(np.int64),
        p_world=np.zeros((t.size, 2)),
        psi_rad=omega * t,
        label="rotation_in_place",
    )
    return samples, truth


@pytest.fixture
def stationary() -> list[ImuSample]:
    """60 s of a phone sitting perfectly still. Realistic sensor noise, no motion."""
    rng = np.random.default_rng(26168)
    n = int(60.0 * RATE_HZ)
    return [
        ImuSample(
            t_ns=int(i * NS_PER_S / RATE_HZ),
            a_body=np.array([0.0, 0.0, GRAVITY]) + rng.normal(0.0, 0.02, 3),
            w_body=rng.normal(0.0, 0.002, 3),
            m_body=np.array([20e-6, 0.0, -45e-6]) + rng.normal(0.0, 1e-7, 3),
        )
        for i in range(n)
    ]


@pytest.fixture
def session_meta() -> SessionMeta:
    return SessionMeta(
        session_id="test-0001",
        device_model="synthetic",
        carry_position=CarryPosition.HAND,
        imu_rate_hz=RATE_HZ,
        origin_lat_deg=20.3535,
        origin_lon_deg=85.8164,  # KIIT campus, the likely test loop
    )
