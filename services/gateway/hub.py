"""In-process pub/sub between the phone's uplink and every connected browser.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 6.1, 6.8

## The placeholder this file centres on

There is no ESKF yet (M3, owner: Sikruti) and no AHRS or preprocessing yet (M1, owner:
Sristee). Until those land, `Hub.on_gps` below is a bare local-tangent-plane projection
of the raw GPS fix -- **not** a fused position estimate. It exists so the web map
(issue #38, owner: Tanmay) has a real, moving dot to build against today from an actual
phone, rather than only the synthetic mocks the frozen contract already allows for.

When the real ESKF lands, replace the body of `on_gps` with a call into
`dr_core.fusion.Eskf.update_gps` (and route IMU samples into `update_velocity` /
ZUPT / ZARU instead of the current `pass`). `_project_enu` and the distance/origin
bookkeeping in this class can be deleted outright at that point -- the ESKF owns all
of it internally.
"""

from __future__ import annotations

import asyncio
import contextlib
import math

import numpy as np

from dr_core.types import ERROR_STATE_DIM, FilterState, GpsFix, TelemetryFrame
from services.gateway.wire import encode_telemetry_frame

EARTH_RADIUS_M = 6_371_000.0
_QUEUE_MAXSIZE = 64


def _project_enu(fix: GpsFix, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    """Flat-earth local tangent-plane projection.

    Valid to a small fraction of a percent over a few kilometres, which covers any
    demo loop this project runs. This is plain geodesy, not a stand-in for the
    orientation- and gravity-aware ENU frame `dr_core.preprocess` will eventually own;
    it never touches IMU data and has nothing to do with AHRS.
    """
    origin_lat_rad = math.radians(origin_lat_deg)
    east = math.radians(fix.lon_deg - origin_lon_deg) * math.cos(origin_lat_rad) * EARTH_RADIUS_M
    north = math.radians(fix.lat_deg - origin_lat_deg) * EARTH_RADIUS_M
    return east, north


class Hub:
    """One instance per running app. Owns the GPS toggle and the /live fan-out."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()
        self.gps_enabled = True
        self._origin_deg: tuple[float, float] | None = None
        self._last_world = (0.0, 0.0)
        self._distance_m = 0.0

    async def subscribe(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._subscribers.discard(queue)

    def set_gps_enabled(self, enabled: bool) -> None:
        self.gps_enabled = enabled

    async def on_gps(self, fix: GpsFix) -> None:
        """Placeholder path: project the fix to a local frame and broadcast it.

        See the module docstring -- this is what M3 replaces wholesale.
        """
        if self._origin_deg is None:
            self._origin_deg = (fix.lat_deg, fix.lon_deg)

        if self.gps_enabled:
            east, north = _project_enu(fix, *self._origin_deg)
            step_m = math.hypot(east - self._last_world[0], north - self._last_world[1])
            self._distance_m += step_m
            self._last_world = (east, north)
        # else: hold the last known position. There is no IMU-only estimator wired in
        # here yet, so freezing the dot is the honest behaviour -- it says "we do not
        # yet track through a GPS drop", rather than quietly pretending to.

        state = FilterState(
            t_ns=fix.t_ns,
            p_world=np.array(self._last_world, dtype=np.float64),
            v_world=np.zeros(2),
            psi_rad=0.0,
            gyro_bias_z=0.0,
            scale=1.0,
            cov=np.eye(ERROR_STATE_DIM),
        )
        frame = TelemetryFrame(
            t_ns=fix.t_ns,
            state=state,
            gps_enabled=self.gps_enabled,
            distance_travelled_m=self._distance_m,
        )
        await self._broadcast(encode_telemetry_frame(frame))

    async def _broadcast(self, wire_frame: dict[str, object]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                # A slow browser must never back-pressure the phone's uplink -- drop
                # that one subscriber's oldest pending frame instead of blocking here.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(wire_frame)
