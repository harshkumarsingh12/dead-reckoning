"""Read and write session recordings (gzipped JSON Lines).

OWNER: Sristee  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 6.1

Wire format, one JSON object per line:

    {"type": "meta",  ...SessionMeta fields..., "schema_version": 1}   <- first line
    {"type": "imu",   "t_ns": "123", "a": [...], "w": [...], "m": [...]|null}
    {"type": "gps",   "t_ns": "123", "lat": ..., "lon": ..., "acc": ...}
    {"type": "event", "t_ns": "123", "name": "gps_off"|"zupt_marker"|"tap"}

Every ``*_ns`` timestamp crosses as a DECIMAL STRING, not a JSON number: nanoseconds
since boot exceed ``Number.MAX_SAFE_INTEGER`` in about 104 days of uptime and
``JSON.parse`` would silently round them (docs/CONVENTIONS.md section 6). Python parses
either form losslessly; the string is what keeps a JS reader honest.

The event records are what make the demo scriptable: the GPS-off toggle and the
sharp-motion alignment taps land in the same stream as the data, at the same
timestamps, so a replay reproduces the run exactly rather than approximately.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from dr_core.types import CarryPosition, GpsFix, ImuSample, SessionMeta

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy.typing as npt

    Vec3 = npt.NDArray[np.float64]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """A scripted demo marker recorded inline with the data.

    Names are free-form but conventionally one of: gps_off, gps_on, tap, zupt_marker,
    corner_N. The payload is optional structured detail for richer markers.
    """

    t_ns: int
    name: str
    payload: dict[str, object] | None = None


def _vec(v: Vec3) -> list[float]:
    """Serialise a 3-vector to a plain JSON list of floats."""
    return [float(x) for x in v]


def _vec_or_none(v: Vec3 | None) -> list[float] | None:
    return None if v is None else _vec(v)


def _arr_or_none(v: object) -> Vec3 | None:
    return None if v is None else np.asarray(v, dtype=np.float64)


def _parse_meta(rec: dict[str, Any]) -> SessionMeta:
    return SessionMeta(
        session_id=rec["session_id"],
        device_model=rec["device_model"],
        carry_position=CarryPosition(rec["carry_position"]),
        imu_rate_hz=rec["imu_rate_hz"],
        boot_to_utc_offset_ns=int(rec["boot_to_utc_offset_ns"]),
        origin_lat_deg=rec["origin_lat_deg"],
        origin_lon_deg=rec["origin_lon_deg"],
        gyro_bias_body=_arr_or_none(rec["gyro_bias_body"]),
        mag_hard_iron_body=_arr_or_none(rec["mag_hard_iron_body"]),
        notes=rec.get("notes", ""),
    )


def _parse_imu(rec: dict[str, Any]) -> ImuSample:
    return ImuSample(
        t_ns=int(rec["t_ns"]),
        a_body=np.asarray(rec["a"], dtype=np.float64),
        w_body=np.asarray(rec["w"], dtype=np.float64),
        m_body=_arr_or_none(rec["m"]),
    )


def _parse_gps(rec: dict[str, Any]) -> GpsFix:
    return GpsFix(
        t_ns=int(rec["t_ns"]),
        lat_deg=rec["lat"],
        lon_deg=rec["lon"],
        accuracy_m=rec["acc"],
        speed_mps=rec["speed"],
        course_rad=rec["course"],
        altitude_m=rec["alt"],
    )


def _parse_event(rec: dict[str, Any]) -> SessionEvent:
    return SessionEvent(t_ns=int(rec["t_ns"]), name=rec["name"], payload=rec.get("payload"))


class SessionWriter:
    """Append-only writer. Use as a context manager so the file is always closed.

    The meta header is written on ``__enter__``; the file is not opened before that, so
    writes outside the ``with`` block fail loudly rather than silently dropping records.

    Example:
        with SessionWriter(path, meta) as w:
            w.write_imu(sample)
    """

    def __init__(self, path: Path, meta: SessionMeta) -> None:
        self._path = path
        self._meta = meta
        self._fh: TextIO | None = None

    def _meta_record(self) -> dict[str, object]:
        m = self._meta
        return {
            "type": "meta",
            "schema_version": SCHEMA_VERSION,
            "session_id": m.session_id,
            "device_model": m.device_model,
            "carry_position": m.carry_position.value,
            "imu_rate_hz": m.imu_rate_hz,
            "boot_to_utc_offset_ns": str(m.boot_to_utc_offset_ns),
            "origin_lat_deg": m.origin_lat_deg,
            "origin_lon_deg": m.origin_lon_deg,
            "gyro_bias_body": _vec_or_none(m.gyro_bias_body),
            "mag_hard_iron_body": _vec_or_none(m.mag_hard_iron_body),
            "notes": m.notes,
        }

    def _write(self, record: dict[str, object]) -> None:
        if self._fh is None:
            raise RuntimeError("SessionWriter used outside its context manager")
        self._fh.write(json.dumps(record) + "\n")

    def write_imu(self, sample: ImuSample) -> None:
        self._write(
            {
                "type": "imu",
                "t_ns": str(sample.t_ns),
                "a": _vec(sample.a_body),
                "w": _vec(sample.w_body),
                "m": _vec_or_none(sample.m_body),
            }
        )

    def write_gps(self, fix: GpsFix) -> None:
        self._write(
            {
                "type": "gps",
                "t_ns": str(fix.t_ns),
                "lat": fix.lat_deg,
                "lon": fix.lon_deg,
                "acc": fix.accuracy_m,
                "speed": fix.speed_mps,
                "course": fix.course_rad,
                "alt": fix.altitude_m,
            }
        )

    def write_event(self, t_ns: int, name: str, payload: dict[str, object] | None = None) -> None:
        """Record a demo event -- gps_off, gps_on, tap, zupt_marker, corner_N."""
        record: dict[str, object] = {"type": "event", "t_ns": str(t_ns), "name": name}
        if payload is not None:
            record["payload"] = payload
        self._write(record)

    def __enter__(self) -> SessionWriter:
        self._fh = gzip.open(self._path, "wt", encoding="utf-8")
        self._write(self._meta_record())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class SessionReader:
    """Streaming reader. Never loads a whole recording into memory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            first = fh.readline()
        if not first.strip():
            raise ValueError(f"empty session file: {path}")
        header = json.loads(first)
        if header.get("type") != "meta":
            raise ValueError(f"first record is {header.get('type')!r}, expected 'meta'")
        version = header.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
        self._meta = _parse_meta(header)

    @property
    def meta(self) -> SessionMeta:
        """Parsed from the header line, available before iteration starts."""
        return self._meta

    def __iter__(self) -> Iterator[tuple[str, object]]:
        """Yield ``(record_type, parsed_record)`` in capture order."""
        with gzip.open(self._path, "rt", encoding="utf-8") as fh:
            fh.readline()  # skip the meta header
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rtype = rec["type"]
                if rtype == "imu":
                    yield ("imu", _parse_imu(rec))
                elif rtype == "gps":
                    yield ("gps", _parse_gps(rec))
                elif rtype == "event":
                    yield ("event", _parse_event(rec))
                else:
                    raise ValueError(f"unknown record type: {rtype!r}")

    def replay(self, speed: float = 1.0) -> Iterator[tuple[str, object]]:
        """Iterate paced by the capture timestamps, so a replay reproduces the cadence.

        Consecutive records are separated by ``(t_ns gap) / speed`` seconds. ``speed=0``
        yields as fast as possible with no sleeps, which is what the eval harness uses.
        """
        prev_ns: int | None = None
        for record in self:
            payload = record[1]
            if isinstance(payload, ImuSample | GpsFix | SessionEvent):
                t_ns = payload.t_ns
                if speed > 0 and prev_ns is not None:
                    delay_s = (t_ns - prev_ns) / 1e9 / speed
                    if delay_s > 0:
                        time.sleep(delay_s)
                prev_ns = t_ns
            yield record
