"""Chi-square innovation gating and the NIS/NEES log.

OWNER: Sikruti  |  MILESTONE: M0 (logger) / M3 (gating)
Spec: docs/BUILD_PLAN.md sections 6.6 and 8

Two jobs, and they are different:

  * The GATE stops one bad sensor reading from corrupting the state. Every measurement
    channel passes it -- velocity, heading, GPS, no exceptions.
  * The LOG proves the filter's uncertainty is honest rather than decorative. NIS
    within its chi-square bounds is what makes the on-screen ellipse mean something,
    and it is the answer to "how do you know your uncertainty is honest?".

The logger is built at M0, long before the filter exists, because a consistency check
added after the fact tends to be tuned until it agrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]


class ChiSquareGate:
    """Rejects a measurement whose normalised innovation squared is implausible."""

    def __init__(self, dof: int, level: float = 0.95) -> None:
        """
        Args:
            dof: innovation dimension (2 for planar velocity, 1 for heading).
            level: gate level. 0.95 means roughly 5% of good measurements are
                rejected -- an accepted and deliberate cost.
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

    def accept(self, innovation: Array, innovation_cov: Array) -> tuple[bool, float]:
        """Test one innovation.

        Returns:
            (accepted, nis). The NIS is returned even on rejection, because a rejected
            measurement's NIS is the most informative number on the strip when
            something is going wrong.
        """
        raise NotImplementedError("M3 -- owner: Sikruti")

    @property
    def threshold(self) -> float:
        """The chi-square critical value for this gate."""
        raise NotImplementedError("M3 -- owner: Sikruti")


@dataclass(slots=True)
class ChannelStats:
    """Running NIS statistics for one measurement channel."""

    dof: int
    accepted: int = 0
    rejected: int = 0
    nis_history: list[float] = field(default_factory=list)

    @property
    def mean_nis(self) -> float:
        """Should sit near ``dof`` if the filter is consistent."""
        raise NotImplementedError("M0 -- owner: Sikruti")

    @property
    def bounds(self) -> tuple[float, float]:
        """Two-sided chi-square bounds on the average NIS, for the strip."""
        raise NotImplementedError("M0 -- owner: Sikruti")


class NisLogger:
    """Per-channel NIS accounting, live during the demo and saved after it."""

    def __init__(self, channels: dict[str, int]) -> None:
        """
        Args:
            channels: channel name -> innovation dimension, e.g.
                {"velocity": 2, "heading": 1, "gps": 2, "zupt": 2, "zaru": 1}.
        """
        raise NotImplementedError("M0 -- owner: Sikruti")

    def record(self, channel: str, nis: float, accepted: bool) -> None:
        """Log one gate decision."""
        raise NotImplementedError("M0 -- owner: Sikruti")

    def snapshot(self) -> dict[str, float]:
        """Latest NIS per channel, for TelemetryFrame.nis."""
        raise NotImplementedError("M0 -- owner: Sikruti")

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Chi-square bounds per channel, for TelemetryFrame.nis_bounds."""
        raise NotImplementedError("M0 -- owner: Sikruti")

    def is_consistent(self) -> dict[str, bool]:
        """Per channel: is the average NIS inside its bounds?

        This is the pass/fail the milestone exit criteria are written against.
        """
        raise NotImplementedError("M0 -- owner: Sikruti")


def nees(error: Array, covariance: Array) -> float:
    """Normalised estimation error squared, against ground truth.

    Only computable offline on recorded runs where truth is known. NIS is its live
    cousin and needs no truth.
    """
    raise NotImplementedError("M0 -- owner: Sikruti")
