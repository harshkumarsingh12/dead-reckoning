"""The magnetometer triple gate: magnitude AND dip AND innovation.

OWNER: Sristee  |  MILESTONE: M1  |  Spec: docs/BUILD_PLAN.md section 6.3

Why three checks and not one: indoor magnetic disturbances -- rebar, lift motors, door
frames -- frequently ROTATE the field while leaving its magnitude close to normal. A
magnitude-only check waves those straight through and the heading quietly bends. Adding
the dip (inclination) angle catches exactly that case, and the chi-square innovation
test in the filter catches whatever survives both.

This is a scored differentiator, not defensive plumbing: "what happens when the
magnetometer fails indoors?" is one of the four predictable judge questions
(build plan section 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dr_core.types import MagGateVerdict

if TYPE_CHECKING:
    import numpy.typing as npt

    Vec3 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MagGateConfig:
    """Tolerances for the three checks.

    Defaults are starting points, not gospel -- tune them against a recorded indoor
    walk and record the chosen values in the PR description.
    """

    magnitude_tolerance_frac: float = 0.20  # +/- 20% of the calibrated field strength
    dip_tolerance_rad: float = 0.175  # ~10 degrees
    innovation_chi2_level: float = 0.95  # gate level used by the filter


class MagGate:
    """Stateful gate. Tracks accept/reject counts for the telemetry strip."""

    def __init__(
        self,
        expected_magnitude_t: float,
        expected_dip_rad: float,
        config: MagGateConfig | None = None,
    ) -> None:
        self._expected_magnitude_t = expected_magnitude_t
        self._expected_dip_rad = expected_dip_rad
        self._config = config if config is not None else MagGateConfig()
        self._accepted = 0
        self._total = 0

    def check(self, m_body: Vec3, gravity_body: Vec3) -> MagGateVerdict:
        """Run the magnitude and dip checks on one calibrated magnetometer reading.

        The innovation check lives in the filter, since it needs the current state; a
        reading that passes here is handed on and may still be rejected there.

        Args:
            m_body: hard-iron-corrected field vector, tesla, device frame.
            gravity_body: the gravity direction in the same frame (points down), used to
                compute the dip angle between the field and the horizontal plane.

        Returns:
            ACCEPTED, or the specific reason for rejection -- specific because
            "rejected" alone tells you nothing when you are debugging on demo day.
        """
        self._total += 1

        magnitude = float(np.linalg.norm(m_body))
        if self._expected_magnitude_t > 0.0:
            frac_error = abs(magnitude - self._expected_magnitude_t) / self._expected_magnitude_t
            if frac_error > self._config.magnitude_tolerance_frac:
                return MagGateVerdict.REJECTED_MAGNITUDE

        g_norm = float(np.linalg.norm(gravity_body))
        if magnitude < 1e-15 or g_norm < 1e-15:
            # No usable field or no gravity reference: cannot judge dip, so reject on the
            # weaker check rather than wave it through (house rule: fail loudly).
            return MagGateVerdict.REJECTED_MAGNITUDE
        down = np.asarray(gravity_body, dtype=np.float64) / g_norm
        # Dip is the angle the field makes below the horizontal plane: positive when the
        # field tilts down, along gravity. sin(dip) = (m . down) / |m|.
        sin_dip = float(np.clip(np.dot(m_body, down) / magnitude, -1.0, 1.0))
        dip = float(np.arcsin(sin_dip))
        if abs(dip - self._expected_dip_rad) > self._config.dip_tolerance_rad:
            return MagGateVerdict.REJECTED_DIP

        self._accepted += 1
        return MagGateVerdict.ACCEPTED

    @property
    def accept_rate(self) -> float:
        """Fraction of readings accepted so far. Displayed live."""
        if self._total == 0:
            return 0.0
        return self._accepted / self._total
