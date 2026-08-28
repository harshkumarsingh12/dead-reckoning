"""In-process pub/sub between the phone's uplink and every connected browser.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 6.1, 6.8

## The real ESKF, wired in

This used to be a flat-earth GPS passthrough (see git history on this file, or issue
#38 for why that placeholder existed). Now every IMU tick and GPS fix flows through a
real `dr_core.fusion.Eskf`: IMU ticks drive `predict` plus the stationary detector's
ZUPT/ZARU updates, GPS fixes drive `update_gps`. That means the dot now genuinely
dead-reckons through a GPS-off period via ZUPT/ZARU and gyro integration, rather than
freezing outright -- watch `state.psi_rad` and `state.cov` move, which the old
placeholder could never do (it hardcoded both).

**What is deliberately still not wired here, and why:**

- **No learned-velocity update.** M2 (the causal TCN) hasn't started -- it's blocked
  on RoNIN/OxIOD dataset access (see docs/ROADMAP.md). There is no `VelocityEstimate`
  to hand `Eskf.update_velocity` yet.
- **No magnetometer heading update.** The triple gate has an open, real bug (#59:
  `expected_dip_rad=0` rejects even a clean field) and there is no live calibration
  pipeline feeding `dr_core.ahrs.AhrsFilter` -- calibration is computed OFFLINE from a
  recording per `APP.md`, not from the live `/ingest` event markers. Wiring the
  magnetometer here today would fuse a gate that's known to reject good readings.
- **No reorder buffer.** `docs/ROADMAP.md`'s M3 table marks this "done", but that is
  the eval harness's usage (`dr_core.eval.cli`), not this file -- `ReorderBuffer` is
  never imported here. Messages are fused in arrival order. Flagging this rather than
  quietly leaving the roadmap's claim standing.
- **`gyro_yaw_rate` is raw `w_body[2]`, not orientation-rotated.** The correct value
  is the gyro rotated into the world frame via the AHRS quaternion, but that needs the
  same live calibration this file doesn't have access to yet (see above). Using the
  raw device-frame z-component is a standard "phone held roughly flat" approximation;
  it is measurably wrong when the phone is held vertically (a normal walking carry),
  and ZARU/GPS course-over-ground are what keep heading from running away between
  corrections in the meantime. Replace with `AhrsFilter.heading_rad`-derived rate once
  live calibration exists.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import numpy as np

from dr_core.fusion.eskf import Eskf
from dr_core.fusion.zupt import StationaryDetector
from dr_core.types import TelemetryFrame
from services.gateway.wire import encode_telemetry_frame

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import GpsFix, ImuSample

_QUEUE_MAXSIZE = 64


class Hub:
    """One instance per running app. Owns the Eskf, the GPS toggle, and the /live fan-out."""

    def __init__(self, imu_rate_hz: float = 200.0) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()
        self.gps_enabled = True

        self._eskf = Eskf()
        self._stationary = StationaryDetector(rate_hz=imu_rate_hz)

        self._origin_deg: tuple[float, float] | None = None
        self._last_p_world: npt.NDArray[np.float64] | None = None
        self._distance_m = 0.0

    async def subscribe(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._subscribers.discard(queue)

    def set_gps_enabled(self, enabled: bool) -> None:
        self.gps_enabled = enabled

    async def on_imu(self, sample: ImuSample) -> None:
        """Every accelerometer/gyroscope/magnetometer tick, combined on the phone.

        Predicts the filter forward, then feeds the stationary detector; a positive
        verdict fires both ZUPT (velocity is exactly zero) and ZARU (angular rate is
        exactly zero, pinning gyro bias) -- physics-based, independent of any model.
        """
        self._eskf.set_gps_enabled(self.gps_enabled)
        yaw_rate = float(sample.w_body[2])  # approximation -- see module docstring
        self._eskf.predict(sample.t_ns, yaw_rate)

        stationary = self._stationary.update(sample)
        if stationary:
            self._eskf.update_zupt(sample.t_ns)
            self._eskf.update_zaru(sample.t_ns, yaw_rate)

        await self._broadcast_state(sample.t_ns, zupt_active=stationary, zaru_active=stationary)

    async def on_gps(self, fix: GpsFix) -> None:
        """A GPS fix -- fused through the real Eskf, or held while GPS is toggled off.

        Predicts to the fix's capture time first (with an assumed-zero yaw rate for
        that instant, since no synchronised gyro reading exists for it): a measurement
        update should always be preceded by bringing the state to the same timestamp,
        and this is also what gives the filter's clock a starting value on a
        GPS-only session with no IMU ticks at all.
        """
        if self._origin_deg is None:
            self._origin_deg = (fix.lat_deg, fix.lon_deg)

        self._eskf.set_gps_enabled(self.gps_enabled)
        self._eskf.predict(fix.t_ns, 0.0)
        if self.gps_enabled:
            self._eskf.update_gps(fix)
        # else: hold the predicted-only state. IMU ticks keep dead-reckoning it via
        # on_imu regardless -- this is the whole point of wiring the Eskf in.

        await self._broadcast_state(fix.t_ns)

    async def _broadcast_state(
        self, t_ns: int, *, zupt_active: bool = False, zaru_active: bool = False
    ) -> None:
        state = self._eskf.state

        if self._last_p_world is not None:
            self._distance_m += float(np.linalg.norm(state.p_world - self._last_p_world))
        self._last_p_world = state.p_world

        frame = TelemetryFrame(
            t_ns=t_ns,
            state=state,
            gps_enabled=self.gps_enabled,
            distance_travelled_m=self._distance_m,
            zupt_active=zupt_active,
            zaru_active=zaru_active,
            origin_lat_deg=self._origin_deg[0] if self._origin_deg is not None else None,
            origin_lon_deg=self._origin_deg[1] if self._origin_deg is not None else None,
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
