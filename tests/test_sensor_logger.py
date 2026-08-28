"""Unit tests for the Sensor Logger -> session-format converter.

Spec: data/README.md  |  OWNER: Sumedha  |  MILESTONE: M2

The CSV column names, order and units here are copied verbatim from a REAL Sensor
Logger export (see dr_core.datasets.sensor_logger's module docstring for how each
mapping was verified) -- not invented. In particular: TotalAcceleration.csv is the
gravity-INCLUSIVE channel (Accelerometer.csv is not), and every axis CSV's column
order is time,seconds_elapsed,z,y,x.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from dr_core.datasets import load_own_recording
from dr_core.datasets.sensor_logger import convert
from dr_core.types import CarryPosition

NS_PER_S = 1_000_000_000

_TOTAL_ACCEL_CSV = """time,seconds_elapsed,z,y,x
{t0},0.0,9.8,0.1,0.2
{t1},0.01,9.7,0.2,0.3
{t2},0.02,9.9,0.0,0.1
{t3},0.03,9.8,0.1,0.2
{t4},0.04,9.7,0.2,0.1
"""

_GYRO_CSV = """time,seconds_elapsed,z,y,x
{t0},0.0,0.01,0.02,0.01
{t1},0.01,0.02,0.01,0.02
{t2},0.02,0.01,0.01,0.01
{t3},0.03,0.00,0.02,0.01
{t4},0.04,0.01,0.01,0.02
"""

_MAG_CSV = """time,seconds_elapsed,z,y,x
{t0},0.0,-42.0,2.0,-18.0
{t2},0.02,-42.5,2.5,-18.5
{t4},0.04,-41.5,1.5,-17.5
"""

_LOCATION_CSV = (
    "time,seconds_elapsed,mslAltitudeAccuracy,bearingAccuracy,speedAccuracy,"
    "verticalAccuracy,horizontalAccuracy,speed,altitudeAboveMeanSeaLevel,bearing,"
    "altitude,longitude,latitude\n"
    "{t0},0.0,,45,1.5,3.7,5.0,1.4,,0,50.0,85.8164,20.3535\n"
    "{t4},0.04,,45,1.5,3.7,5.0,1.4,,0,50.0,85.81645,20.35352\n"
)

_METADATA_CSV = (
    "version,device name,recording epoch time,recording time,recording timezone,"
    "platform,appVersion,device id,sensors,sampleRateMs,standardisation,platform version\n"
    "3,TestPhone,{t0},2026-08-28_13-21-44,Asia/Kolkata,android,1.64,test-device-id,"
    "Accelerometer|Gyroscope|Magnetometer|Location|TotalAcceleration,10|10|10|0|10,false,36"
)


def _write_export(tmp_path: Path, t0_ns: int = 1_787_923_304_000_000_000) -> Path:
    times = [t0_ns + i * 10_000_000 for i in range(5)]  # 10 ms apart -> 100 Hz

    session_dir = tmp_path / "export"
    session_dir.mkdir()
    (session_dir / "TotalAcceleration.csv").write_text(
        _TOTAL_ACCEL_CSV.format(t0=times[0], t1=times[1], t2=times[2], t3=times[3], t4=times[4])
    )
    (session_dir / "Gyroscope.csv").write_text(
        _GYRO_CSV.format(t0=times[0], t1=times[1], t2=times[2], t3=times[3], t4=times[4])
    )
    (session_dir / "Magnetometer.csv").write_text(
        _MAG_CSV.format(t0=times[0], t2=times[2], t4=times[4])
    )
    (session_dir / "Location.csv").write_text(_LOCATION_CSV.format(t0=times[0], t4=times[4]))
    (session_dir / "Metadata.csv").write_text(_METADATA_CSV.format(t0=times[0]))

    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in session_dir.iterdir():
            zf.write(f, arcname=f.name)
    return zip_path


def test_convert_reads_total_acceleration_not_accelerometer(tmp_path: Path) -> None:
    """The whole point of the module docstring's warning: a_body must come from the
    gravity-inclusive channel, verified here by checking its magnitude lands near
    9.8 m/s^2, not near zero the way gravity-removed data would."""
    zip_path = _write_export(tmp_path)
    out_path = tmp_path / "session.jsonl.gz"

    convert(zip_path, out_path, CarryPosition.HAND, session_id="test-session")
    recording = load_own_recording(out_path)

    assert len(recording.imu) == 5
    first = recording.imu[0]
    assert np.linalg.norm(first.a_body) == pytest.approx(9.82, abs=0.1)


def test_convert_uses_zyx_column_order_not_xyz(tmp_path: Path) -> None:
    """Sensor Logger's per-axis CSVs are time,seconds_elapsed,z,y,x -- getting this
    backwards silently swaps axes rather than erroring."""
    zip_path = _write_export(tmp_path)
    out_path = tmp_path / "session.jsonl.gz"

    convert(zip_path, out_path, CarryPosition.HAND, session_id="test-session")
    recording = load_own_recording(out_path)

    # Row 1 of the fixture: z=9.8, y=0.1, x=0.2 -> a_body must be [x, y, z] = [0.2, 0.1, 9.8]
    np.testing.assert_allclose(recording.imu[0].a_body, [0.2, 0.1, 9.8])


def test_convert_pairs_magnetometer_by_nearest_time(tmp_path: Path) -> None:
    """Magnetometer is sparser than accel/gyro; every accel sample should still find a
    reasonably close magnetometer reading and get microtesla converted to tesla."""
    zip_path = _write_export(tmp_path)
    out_path = tmp_path / "session.jsonl.gz"

    convert(zip_path, out_path, CarryPosition.HAND, session_id="test-session")
    recording = load_own_recording(out_path)

    assert all(s.m_body is not None for s in recording.imu)
    # Fixture row 1: mag z=-42.0 uT -> -42.0e-6 T
    assert recording.imu[0].m_body[2] == pytest.approx(-42.0e-6)


def test_convert_derives_gps_truth(tmp_path: Path) -> None:
    zip_path = _write_export(tmp_path)
    out_path = tmp_path / "session.jsonl.gz"

    convert(zip_path, out_path, CarryPosition.POCKET, session_id="test-session")
    recording = load_own_recording(out_path)

    assert recording.meta.carry_position == CarryPosition.POCKET
    assert len(recording.gps) == 2
    assert recording.truth is not None
    assert recording.truth.label == "gps_truth"
    # First fix sits on the origin -> world ENU (0, 0); the second moved slightly NE.
    np.testing.assert_allclose(recording.truth.p_world[0], [0.0, 0.0], atol=1e-6)
    assert recording.truth.p_world[1, 0] > 0.0  # moved east (lon increased)
    assert recording.truth.p_world[1, 1] > 0.0  # moved north (lat increased)


def test_convert_rejects_a_folder_missing_total_acceleration(tmp_path: Path) -> None:
    session_dir = tmp_path / "export"
    session_dir.mkdir()
    (session_dir / "Gyroscope.csv").write_text(_GYRO_CSV.format(t0=0, t1=1, t2=2, t3=3, t4=4))

    with pytest.raises(FileNotFoundError, match="TotalAcceleration"):
        convert(session_dir, tmp_path / "out.jsonl.gz", CarryPosition.HAND, session_id="s")
