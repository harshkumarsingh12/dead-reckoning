"""Stationary detection, which drives both ZUPT and ZARU.

OWNER: Sikruti  |  MILESTONE: M3  |  Spec: docs/BUILD_PLAN.md section 6.6

Standing still is free information and dead reckoning normally throws it away. Detect
it and you get two corrections at once: velocity is exactly zero (ZUPT) and angular
rate is exactly zero, which pins the gyro bias (ZARU).

It is also the most legible moment in the live demo. The presenter stops walking, the
lamp lights, the ellipse tightens, drift stops climbing. Judges see the mechanism work
rather than being told it does.

Done when: standing still for 10 s produces no position creep, the bias converges, and
the lamp fires on cue during the scripted arc.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dr_core.types import ImuSample


@dataclass(frozen=True, slots=True)
class StationaryConfig:
    """Thresholds for the detector.

    Tune on a recording that contains both a genuine stop and slow walking. The failure
    mode to avoid is firing during a slow shuffle, which pins the state to a position
    the person is actually leaving.
    """

    window_s: float = 0.5
    accel_var_threshold: float = 0.05  # (m/s^2)^2
    gyro_var_threshold: float = 0.01  # (rad/s)^2
    min_duration_s: float = 0.3  # must hold this long before firing


class StationaryDetector:
    """Sliding-window variance test over accelerometer and gyroscope."""

    def __init__(self, config: StationaryConfig | None = None, rate_hz: float = 200.0) -> None:
        self.config = config if config is not None else StationaryConfig()
        self.rate_hz = rate_hz
        self.window_size = max(2, round(self.config.window_s * rate_hz))
        self._a_norms: deque[float] = deque(maxlen=self.window_size)
        self._w_norms: deque[float] = deque(maxlen=self.window_size)

        self._candidate_start_ns: int | None = None
        self._current_t_ns: int = 0
        self._is_stationary: bool = False

    def update(self, sample: ImuSample) -> bool:
        """Feed one sample. Returns whether the device is currently stationary."""
        self._current_t_ns = sample.t_ns
        a_norm = float(np.linalg.norm(sample.a_body))
        w_norm = float(np.linalg.norm(sample.w_body))
        self._a_norms.append(a_norm)
        self._w_norms.append(w_norm)

        if len(self._a_norms) < self.window_size:
            self._is_stationary = False
            return False

        a_var = float(np.var(self._a_norms))
        w_var = float(np.var(self._w_norms))

        if a_var < self.config.accel_var_threshold and w_var < self.config.gyro_var_threshold:
            if self._candidate_start_ns is None:
                self._candidate_start_ns = sample.t_ns
            duration_s = (sample.t_ns - self._candidate_start_ns) * 1e-9
            self._is_stationary = duration_s >= self.config.min_duration_s
        else:
            self._candidate_start_ns = None
            self._is_stationary = False

        return self._is_stationary

    @property
    def is_stationary(self) -> bool:
        """Current verdict, without advancing the window."""
        return self._is_stationary

    @property
    def stationary_duration_s(self) -> float:
        """How long the current stationary episode has lasted. Zero if moving."""
        if self._is_stationary and self._candidate_start_ns is not None:
            return max(0.0, (self._current_t_ns - self._candidate_start_ns) * 1e-9)
        return 0.0
