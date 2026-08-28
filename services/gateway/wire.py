"""JSON wire (de)serialization for the gateway's two sockets.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/CONVENTIONS.md section 6

Two different timestamp encodings, deliberately:

  - **`/ingest`** (phone -> gateway): a plain JSON integer. Python's `json` module has
    arbitrary-precision integers, so a nanosecond boot-monotonic timestamp round-trips
    exactly. Matches the Android streamer's wire format
    (`apps/android/.../wire/Messages.kt`).
  - **`/live`** (gateway -> browser): a decimal STRING. `JSON.parse` in the browser
    loses precision above `Number.MAX_SAFE_INTEGER`, which nanoseconds-since-boot
    exceed after roughly 104 days of device uptime.

The `encode_*_for_ingest` functions are the mirror image of `decode_imu`/`decode_gps`:
they build the exact same `/ingest` shape from a `dr_core.types` dataclass instead of
parsing it. `scripts/replay.py` is the reason these exist -- pushing a recorded session
back through `/ingest` needs to produce bytes indistinguishable from the phone's own,
which is the whole point of a replay (build plan section 6.1: "indistinguishable in
mechanics from live").
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dr_core.types import GpsFix, ImuSample, TelemetryFrame


def decode_imu(raw: dict[str, Any]) -> ImuSample:
    """Parse one `{"type": "imu", ...}` /ingest message."""
    m = raw.get("m")
    return ImuSample(
        t_ns=int(raw["t_ns"]),
        a_body=np.asarray(raw["a"], dtype=np.float64),
        w_body=np.asarray(raw["w"], dtype=np.float64),
        m_body=np.asarray(m, dtype=np.float64) if m is not None else None,
    )


def decode_gps(raw: dict[str, Any]) -> GpsFix:
    """Parse one `{"type": "gps", ...}` /ingest message."""
    return GpsFix(
        t_ns=int(raw["t_ns"]),
        lat_deg=float(raw["lat_deg"]),
        lon_deg=float(raw["lon_deg"]),
        accuracy_m=float(raw["accuracy_m"]),
        speed_mps=raw.get("speed_mps"),
        course_rad=raw.get("course_rad"),
        altitude_m=raw.get("altitude_m"),
    )


def encode_imu_for_ingest(sample: ImuSample) -> dict[str, Any]:
    """Build an `{"type": "imu", ...}` /ingest message from a parsed sample."""
    return {
        "type": "imu",
        "t_ns": sample.t_ns,
        "a": [float(x) for x in sample.a_body],
        "w": [float(x) for x in sample.w_body],
        "m": [float(x) for x in sample.m_body] if sample.m_body is not None else None,
    }


def encode_gps_for_ingest(fix: GpsFix) -> dict[str, Any]:
    """Build an `{"type": "gps", ...}` /ingest message from a parsed fix."""
    return {
        "type": "gps",
        "t_ns": fix.t_ns,
        "lat_deg": fix.lat_deg,
        "lon_deg": fix.lon_deg,
        "accuracy_m": fix.accuracy_m,
        "speed_mps": fix.speed_mps,
        "course_rad": fix.course_rad,
        "altitude_m": fix.altitude_m,
    }


def encode_event_for_ingest(t_ns: int, name: str) -> dict[str, Any]:
    """Build an `{"type": "event", ...}` /ingest message -- calibration and demo markers."""
    return {"type": "event", "t_ns": t_ns, "name": name}


def encode_telemetry_frame(frame: TelemetryFrame) -> dict[str, Any]:
    """Serialize a TelemetryFrame for the /live socket, per docs/CONVENTIONS.md 6."""
    state = frame.state
    return {
        "t_ns": str(frame.t_ns),
        "state": {
            "t_ns": str(state.t_ns),
            "p_world": [float(state.p_world[0]), float(state.p_world[1])],
            "v_world": [float(state.v_world[0]), float(state.v_world[1])],
            "psi_rad": float(state.psi_rad),
            "gyro_bias_z": float(state.gyro_bias_z),
            "scale": float(state.scale),
            "cov": np.asarray(state.cov, dtype=np.float64).tolist(),
            "heading_source": state.heading_source.value,
        },
        "baseline_p_world": (
            [float(frame.baseline_p_world[0]), float(frame.baseline_p_world[1])]
            if frame.baseline_p_world is not None
            else None
        ),
        "truth_p_world": (
            [float(frame.truth_p_world[0]), float(frame.truth_p_world[1])]
            if frame.truth_p_world is not None
            else None
        ),
        "nis": dict(frame.nis),
        "nis_bounds": {k: list(v) for k, v in frame.nis_bounds.items()},
        "zupt_active": frame.zupt_active,
        "zaru_active": frame.zaru_active,
        "mag_verdict": frame.mag_verdict.value,
        "model_sigma_mps": frame.model_sigma_mps,
        "gps_enabled": frame.gps_enabled,
        "distance_travelled_m": frame.distance_travelled_m,
        "drift_pct": frame.drift_pct,
        "origin_lat_deg": frame.origin_lat_deg,
        "origin_lon_deg": frame.origin_lon_deg,
    }
