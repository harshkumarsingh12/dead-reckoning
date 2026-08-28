"""Round-trip tests for the session recording format (SessionWriter <-> SessionReader).

Spec: docs/BUILD_PLAN.md section 6.1  |  OWNER: Sristee  |  MILESTONE: M0

A recorded session and a live session must flow through the identical downstream
pipeline, so the writer and reader have to be exact inverses. This is the acceptance
test for that: everything written comes back -- same values, same order, same types.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pytest

from dr_core.io.session import SessionEvent, SessionReader, SessionWriter
from dr_core.types import CarryPosition, GpsFix, ImuSample, SessionMeta


def test_write_read_round_trips_every_record(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl.gz"

    gyro_bias = np.array([0.01, -0.02, 0.03])
    mag_hard_iron = np.array([1e-6, -2e-6, 3e-6])
    meta = SessionMeta(
        session_id="rt-0001",
        device_model="synthetic",
        carry_position=CarryPosition.POCKET,
        imu_rate_hz=200.0,
        boot_to_utc_offset_ns=1_700_000_000_000_000_000,
        origin_lat_deg=20.3535,
        origin_lon_deg=85.8164,
        gyro_bias_body=gyro_bias,
        mag_hard_iron_body=mag_hard_iron,
        notes="round-trip fixture",
    )

    a0 = np.array([0.1, 0.2, 9.8])
    w0 = np.array([0.01, 0.02, 0.03])
    m0 = np.array([20e-6, 0.0, -45e-6])
    imu0 = ImuSample(t_ns=1_000_000_000, a_body=a0, w_body=w0, m_body=m0)

    a1 = np.array([0.3, -0.1, 9.79])
    w1 = np.array([-0.01, 0.0, 0.02])
    imu1 = ImuSample(t_ns=1_005_000_000, a_body=a1, w_body=w1, m_body=None)

    gps = GpsFix(
        t_ns=1_010_000_000,
        lat_deg=20.3536,
        lon_deg=85.8165,
        accuracy_m=4.5,
        speed_mps=1.4,
        course_rad=0.7853981633974483,
        altitude_m=45.2,
    )

    event_t_ns = 1_012_000_000
    event_payload = {"corner": 2, "label": "NE"}

    with SessionWriter(path, meta) as w:
        w.write_imu(imu0)
        w.write_imu(imu1)
        w.write_gps(gps)
        w.write_event(event_t_ns, "corner_2", payload=event_payload)

    reader = SessionReader(path)

    # --- meta round-trips, including every optional field ---
    m = reader.meta
    assert isinstance(m, SessionMeta)
    assert m.session_id == meta.session_id
    assert m.device_model == meta.device_model
    assert m.carry_position == meta.carry_position
    assert m.imu_rate_hz == meta.imu_rate_hz
    assert m.boot_to_utc_offset_ns == meta.boot_to_utc_offset_ns
    assert isinstance(m.boot_to_utc_offset_ns, int)
    assert m.origin_lat_deg == meta.origin_lat_deg
    assert m.origin_lon_deg == meta.origin_lon_deg
    assert m.notes == meta.notes
    assert m.gyro_bias_body is not None
    assert m.mag_hard_iron_body is not None
    np.testing.assert_allclose(m.gyro_bias_body, gyro_bias)
    np.testing.assert_allclose(m.mag_hard_iron_body, mag_hard_iron)

    # --- records come back in capture order, with the right types ---
    records = list(reader)
    assert [rt for rt, _ in records] == ["imu", "imu", "gps", "event"]
    (t0, r0), (_t1, r1), (t2, r2), (t3, r3) = records

    # imu0 (m_body set)
    assert t0 == "imu"
    assert isinstance(r0, ImuSample)
    assert r0.t_ns == imu0.t_ns
    assert isinstance(r0.t_ns, int)
    np.testing.assert_allclose(r0.a_body, a0)
    np.testing.assert_allclose(r0.w_body, w0)
    assert r0.m_body is not None
    np.testing.assert_allclose(r0.m_body, m0)

    # imu1 (m_body None)
    assert isinstance(r1, ImuSample)
    assert r1.t_ns == imu1.t_ns
    np.testing.assert_allclose(r1.a_body, a1)
    np.testing.assert_allclose(r1.w_body, w1)
    assert r1.m_body is None

    # gps with all optional fields
    assert t2 == "gps"
    assert isinstance(r2, GpsFix)
    assert r2.t_ns == gps.t_ns
    assert isinstance(r2.t_ns, int)
    assert r2.lat_deg == gps.lat_deg
    assert r2.lon_deg == gps.lon_deg
    assert r2.accuracy_m == gps.accuracy_m
    assert r2.speed_mps == gps.speed_mps
    assert r2.course_rad == gps.course_rad
    assert r2.altitude_m == gps.altitude_m

    # event with payload
    assert t3 == "event"
    assert isinstance(r3, SessionEvent)
    assert r3.t_ns == event_t_ns
    assert isinstance(r3.t_ns, int)
    assert r3.name == "corner_2"
    assert r3.payload == event_payload


def test_t_ns_is_stored_on_the_wire_as_a_decimal_string(tmp_path: Path) -> None:
    """The wire contract (CONVENTIONS section 6): every t_ns is a JSON string, so a JS
    reader cannot silently round it past 2**53. This guards the reason we chose strings."""
    path = tmp_path / "wire.jsonl.gz"
    meta = SessionMeta(session_id="w", device_model="d")
    with SessionWriter(path, meta) as w:
        sample = ImuSample(t_ns=1_723_456_789_012_345_678, a_body=np.zeros(3), w_body=np.zeros(3))
        w.write_imu(sample)

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert all(isinstance(rec["t_ns"], str) for rec in lines if "t_ns" in rec)


def _t_ns(payload: object) -> int:
    assert isinstance(payload, ImuSample | GpsFix | SessionEvent)
    return payload.t_ns


def test_replay_at_speed_zero_matches_iteration_order(tmp_path: Path) -> None:
    """speed=0 yields exactly what __iter__ yields, in the same capture order."""
    path = tmp_path / "replay.jsonl.gz"
    meta = SessionMeta(session_id="rp", device_model="d")
    with SessionWriter(path, meta) as w:
        w.write_imu(ImuSample(t_ns=1_000, a_body=np.zeros(3), w_body=np.zeros(3)))
        w.write_gps(GpsFix(t_ns=2_000, lat_deg=1.0, lon_deg=2.0, accuracy_m=3.0))
        w.write_event(3_000, "tap")

    reader = SessionReader(path)
    plain = [(rt, _t_ns(p)) for rt, p in reader]
    replayed = [(rt, _t_ns(p)) for rt, p in reader.replay(speed=0)]
    assert replayed == plain
    assert replayed == [("imu", 1_000), ("gps", 2_000), ("event", 3_000)]


def test_replay_at_speed_zero_never_sleeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wide t_ns gap must NOT trigger time.sleep when speed=0."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    path = tmp_path / "nosleep.jsonl.gz"
    meta = SessionMeta(session_id="ns", device_model="d")
    with SessionWriter(path, meta) as w:
        w.write_imu(ImuSample(t_ns=1_000, a_body=np.zeros(3), w_body=np.zeros(3)))
        w.write_imu(ImuSample(t_ns=9_999_999_999, a_body=np.zeros(3), w_body=np.zeros(3)))

    consumed = list(SessionReader(path).replay(speed=0))
    assert len(consumed) == 2
    assert slept == []
