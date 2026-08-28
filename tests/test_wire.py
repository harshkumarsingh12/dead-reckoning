"""Round-trip tests for the /ingest wire (de)serialization.

Spec: docs/CONVENTIONS.md section 6  |  OWNER: Harsh  |  MILESTONE: M4

`encode_*_for_ingest` exists so `scripts/replay.py` can push a recorded session back
through /ingest byte-for-byte indistinguishable from the phone's own messages. The
guard here is that encode -> decode is the identity, for the same reason
`test_ingest_preserves_the_device_capture_timestamp` exists in test_gateway.py: a
timestamp or a field silently mangled in this round trip is invisible until the demo.
"""

from __future__ import annotations

import numpy as np

from dr_core.types import GpsFix, ImuSample
from services.gateway.wire import (
    decode_gps,
    decode_imu,
    encode_event_for_ingest,
    encode_gps_for_ingest,
    encode_imu_for_ingest,
)


def test_imu_round_trips_through_ingest_encoding() -> None:
    sample = ImuSample(
        t_ns=1_723_456_789_012_345,
        a_body=np.array([0.1, -0.2, 9.8]),
        w_body=np.array([0.01, 0.0, -0.02]),
        m_body=np.array([1e-5, 2e-5, -3e-5]),
    )
    decoded = decode_imu(encode_imu_for_ingest(sample))
    assert decoded.t_ns == sample.t_ns
    assert np.allclose(decoded.a_body, sample.a_body)
    assert np.allclose(decoded.w_body, sample.w_body)
    assert decoded.m_body is not None
    assert np.allclose(decoded.m_body, sample.m_body)


def test_imu_round_trip_preserves_a_missing_magnetometer_reading() -> None:
    """None means 'no fresh reading', not 'reused a stale one' (APP.md) -- must survive
    the round trip as None, not as a zero vector or anything else that looks like data."""
    sample = ImuSample(t_ns=1, a_body=np.zeros(3), w_body=np.zeros(3), m_body=None)
    decoded = decode_imu(encode_imu_for_ingest(sample))
    assert decoded.m_body is None


def test_gps_round_trips_through_ingest_encoding() -> None:
    fix = GpsFix(
        t_ns=1_723_456_789_012_345,
        lat_deg=20.3535,
        lon_deg=85.8164,
        accuracy_m=5.0,
        speed_mps=1.4,
        course_rad=1.5708,
        altitude_m=42.0,
    )
    decoded = decode_gps(encode_gps_for_ingest(fix))
    assert decoded == fix


def test_gps_round_trip_preserves_absent_optional_fields() -> None:
    fix = GpsFix(t_ns=1, lat_deg=0.0, lon_deg=0.0, accuracy_m=10.0)
    decoded = decode_gps(encode_gps_for_ingest(fix))
    assert decoded.speed_mps is None
    assert decoded.course_rad is None
    assert decoded.altitude_m is None


def test_imu_ingest_timestamp_is_a_plain_json_integer_not_a_string() -> None:
    """/ingest and /live deliberately disagree on this (docs/CONVENTIONS.md section 6)
    -- getting it backwards here would desync replay from what the phone actually sends."""
    wire = encode_imu_for_ingest(ImuSample(t_ns=5, a_body=np.zeros(3), w_body=np.zeros(3)))
    assert isinstance(wire["t_ns"], int)


def test_event_for_ingest_matches_the_documented_shape() -> None:
    """APP.md's wire format: {"type":"event","t_ns":...,"name":"tap"}."""
    assert encode_event_for_ingest(t_ns=42, name="tap") == {
        "type": "event",
        "t_ns": 42,
        "name": "tap",
    }
