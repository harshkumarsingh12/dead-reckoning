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

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dr_core.types import MagGateVerdict

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
        raise NotImplementedError("M1 -- owner: Sristee")

    def check(self, m_body: Vec3, gravity_body: Vec3) -> MagGateVerdict:
        """Run the magnitude and dip checks on one calibrated magnetometer reading.

        The innovation check lives in the filter, since it needs the current state; a
        reading that passes here is handed on and may still be rejected there.

        Args:
            m_body: hard-iron-corrected field vector, tesla, device frame.
            gravity_body: the gravity direction in the same frame, used to compute the
                dip angle between the field and the horizontal plane.

        Returns:
            ACCEPTED, or the specific reason for rejection -- specific because
            "rejected" alone tells you nothing when you are debugging on demo day.
        """
        raise NotImplementedError("M1 -- owner: Sristee")

    @property
    def accept_rate(self) -> float:
        """Fraction of readings accepted so far. Displayed live."""
        raise NotImplementedError("M1 -- owner: Sristee")
