"""Convert a Sensor Logger export into dr_core.io's session format.

OWNER: Sumedha  |  MILESTONE: M2  |  CLI: scripts/import_sensor_logger.py

Column mapping and units below were verified against real exports from this project's
own campus recordings -- not assumed from Sensor Logger's documentation, which does not
spell out the exact semantics precisely enough to trust blind:

  * ``a_body`` comes from ``TotalAcceleration.csv``, NOT ``Accelerometer.csv``. Sensor
    Logger's "Accelerometer" channel is gravity-REMOVED (its magnitude in a real sample
    was ~2 m/s^2); ``dr_core.types.ImuSample.a_body`` must be raw specific force WITH
    gravity (docs/CONVENTIONS.md), which is ``TotalAcceleration.csv`` (magnitude
    ~8-10 m/s^2, matching ``Gravity.csv``'s ~9.8 m/s^2 baseline). Using the wrong one
    would silently double-remove gravity downstream in ``align_gravity``.
  * Every per-axis CSV's column order is ``time,seconds_elapsed,z,y,x`` -- NOT x,y,z.
  * Magnetometer is microtesla; converted to tesla (x1e-6) for dr_core's convention.
  * The ``time`` column is Unix epoch nanoseconds, not dr_core's boot-monotonic domain
    (docs/CONVENTIONS.md section 3) -- a deliberate, documented simplification for
    RECORDED training data (not live fusion): every channel is stamped from the same
    app clock, so relative alignment across channels and against GPS is preserved,
    which is what window construction and label alignment actually need.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from dr_core.io import SessionWriter
from dr_core.types import CarryPosition, GpsFix, ImuSample, SessionMeta

_TOLERANCE_GYRO_NS = 20_000_000  # 20 ms -- gyro requested at the same rate as accel
_TOLERANCE_MAG_NS = 100_000_000  # 100 ms -- magnetometer delivers at a lower rate


class ConversionResult(NamedTuple):
    out_path: Path
    imu_samples: int
    imu_rate_hz: float
    gps_fixes: int


def _extract_if_zip(path: Path, workdir: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError(f"expected a Sensor Logger .zip export or extracted folder, got: {path}")
    extract_dir = workdir / path.stem
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _read_axis_csv(path: Path) -> pd.DataFrame:
    """Read one of Sensor Logger's ``time,seconds_elapsed,z,y,x`` per-axis CSVs."""
    df = pd.read_csv(path)
    missing = {"time", "x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")
    return df[["time", "x", "y", "z"]].sort_values("time").reset_index(drop=True)


def _device_model(session_dir: Path) -> str:
    meta_path = session_dir / "Metadata.csv"
    if not meta_path.exists():
        return "unknown"
    meta = pd.read_csv(meta_path)
    if "device name" in meta.columns and len(meta) > 0:
        return str(meta["device name"].iloc[0])
    return "unknown"


def _build_imu_samples(session_dir: Path) -> list[ImuSample]:
    accel_path = session_dir / "TotalAcceleration.csv"
    gyro_path = session_dir / "Gyroscope.csv"
    if not accel_path.exists() or not gyro_path.exists():
        raise FileNotFoundError(
            f"{session_dir}: need TotalAcceleration.csv and Gyroscope.csv (enable "
            "Accelerometer and Gyroscope in Sensor Logger's settings before recording)"
        )

    accel = _read_axis_csv(accel_path).rename(columns={"x": "ax", "y": "ay", "z": "az"})
    gyro = _read_axis_csv(gyro_path).rename(columns={"x": "gx", "y": "gy", "z": "gz"})

    merged = pd.merge_asof(
        accel, gyro, on="time", direction="nearest", tolerance=_TOLERANCE_GYRO_NS
    )
    merged = merged.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)

    mag_path = session_dir / "Magnetometer.csv"
    has_mag = mag_path.exists()
    if has_mag:
        mag = _read_axis_csv(mag_path).rename(columns={"x": "mx", "y": "my", "z": "mz"})
        merged = pd.merge_asof(
            merged, mag, on="time", direction="nearest", tolerance=_TOLERANCE_MAG_NS
        )

    samples: list[ImuSample] = []
    for row in merged.itertuples(index=False):
        m_body = None
        if has_mag and not (pd.isna(row.mx) or pd.isna(row.my) or pd.isna(row.mz)):
            m_body = np.array([row.mx, row.my, row.mz], dtype=np.float64) * 1e-6  # uT -> T
        samples.append(
            ImuSample(
                t_ns=int(row.time),
                a_body=np.array([row.ax, row.ay, row.az], dtype=np.float64),
                w_body=np.array([row.gx, row.gy, row.gz], dtype=np.float64),
                m_body=m_body,
            )
        )
    return samples


def _build_gps_fixes(session_dir: Path) -> list[GpsFix]:
    loc_path = session_dir / "Location.csv"
    if not loc_path.exists():
        return []
    loc = pd.read_csv(loc_path).sort_values("time").reset_index(drop=True)

    fixes: list[GpsFix] = []
    for row in loc.itertuples(index=False):
        lat = getattr(row, "latitude", None)
        lon = getattr(row, "longitude", None)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue
        accuracy = getattr(row, "horizontalAccuracy", None)
        speed = getattr(row, "speed", None)
        altitude = getattr(row, "altitude", None)
        fixes.append(
            GpsFix(
                t_ns=int(row.time),
                lat_deg=float(lat),
                lon_deg=float(lon),
                accuracy_m=(
                    float(accuracy) if accuracy is not None and not pd.isna(accuracy) else 5.0
                ),
                speed_mps=float(speed) if speed is not None and not pd.isna(speed) else None,
                altitude_m=(
                    float(altitude) if altitude is not None and not pd.isna(altitude) else None
                ),
            )
        )
    return fixes


def convert(
    input_path: Path, out_path: Path, carry_position: CarryPosition, session_id: str
) -> ConversionResult:
    """Convert one Sensor Logger export into a ``dr_core.io.SessionWriter`` file.

    Args:
        input_path: a Sensor Logger ``.zip`` export, or an already-extracted folder.
        out_path: where to write the ``.jsonl.gz`` session file.
        carry_position: how the phone was carried during this recording.
        session_id: identifier stored in the session's metadata.

    Raises:
        FileNotFoundError: the export is missing TotalAcceleration.csv or Gyroscope.csv.
        ValueError: fewer than two IMU samples could be aligned, or ``input_path`` is
            neither a directory nor a ``.zip`` file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = _extract_if_zip(input_path, Path(tmp))

        imu = _build_imu_samples(session_dir)
        if len(imu) < 2:
            raise ValueError(f"{input_path}: fewer than 2 aligned IMU samples, nothing to write")
        gps = _build_gps_fixes(session_dir)

        duration_s = (imu[-1].t_ns - imu[0].t_ns) / 1.0e9
        imu_rate_hz = (len(imu) - 1) / duration_s if duration_s > 0 else 200.0

        meta = SessionMeta(
            session_id=session_id,
            device_model=_device_model(session_dir),
            carry_position=carry_position,
            imu_rate_hz=imu_rate_hz,
            notes=f"imported from Sensor Logger export: {input_path.name}",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with SessionWriter(out_path, meta) as writer:
            for sample in imu:
                writer.write_imu(sample)
            for fix in gps:
                writer.write_gps(fix)

    return ConversionResult(
        out_path=out_path, imu_samples=len(imu), imu_rate_hz=imu_rate_hz, gps_fixes=len(gps)
    )
