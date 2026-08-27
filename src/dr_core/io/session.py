"""Read and write session recordings (gzipped JSON Lines).

OWNER: Sristee  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 6.1

Wire format, one JSON object per line:

    {"type": "meta",  ...SessionMeta fields...}          <- always the first line
    {"type": "imu",   "t_ns": 123, "a": [...], "w": [...], "m": [...]|null}
    {"type": "gps",   "t_ns": 123, "lat": ..., "lon": ..., "acc": ...}
    {"type": "event", "t_ns": 123, "name": "gps_off"|"zupt_marker"|"tap"}

The event records are what make the demo scriptable: the GPS-off toggle and the
sharp-motion alignment taps land in the same stream as the data, at the same
timestamps, so a replay reproduces the run exactly rather than approximately.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dr_core.types import GpsFix, ImuSample, SessionMeta

SCHEMA_VERSION = 1


class SessionWriter:
    """Append-only writer. Use as a context manager so the file is always closed.

    Example:
        with SessionWriter(path, meta) as w:
            w.write_imu(sample)
    """

    def __init__(self, path: Path, meta: SessionMeta) -> None:
        raise NotImplementedError("M0 -- owner: Sristee")

    def write_imu(self, sample: ImuSample) -> None:
        raise NotImplementedError("M0 -- owner: Sristee")

    def write_gps(self, fix: GpsFix) -> None:
        raise NotImplementedError("M0 -- owner: Sristee")

    def write_event(self, t_ns: int, name: str, payload: dict[str, object] | None = None) -> None:
        """Record a demo event -- gps_off, gps_on, tap, zupt_marker, corner_N."""
        raise NotImplementedError("M0 -- owner: Sristee")

    def __enter__(self) -> SessionWriter:
        raise NotImplementedError("M0 -- owner: Sristee")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError("M0 -- owner: Sristee")


class SessionReader:
    """Streaming reader. Never loads a whole recording into memory."""

    def __init__(self, path: Path) -> None:
        raise NotImplementedError("M0 -- owner: Sristee")

    @property
    def meta(self) -> SessionMeta:
        """Parsed from the header line, available before iteration starts."""
        raise NotImplementedError("M0 -- owner: Sristee")

    def __iter__(self) -> Iterator[tuple[str, object]]:
        """Yield ``(record_type, parsed_record)`` in capture order."""
        raise NotImplementedError("M0 -- owner: Sristee")

    def replay(self, speed: float = 1.0) -> Iterator[tuple[str, object]]:
        """Iterate in real time, so a replay is paced like the original walk.

        ``speed=0`` runs as fast as possible, which is what the eval harness uses.
        """
        raise NotImplementedError("M0 -- owner: Sristee")
