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

import numpy as np
from scipy import stats

if TYPE_CHECKING:
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
        self._dof = dof
        self._level = level
        self._threshold = float(stats.chi2.ppf(level, df=dof))

    def accept(self, innovation: Array, innovation_cov: Array) -> tuple[bool, float]:
        """Test one innovation.

        Returns:
            (accepted, nis). The NIS is returned even on rejection, because a rejected
            measurement's NIS is the most informative number on the strip when
            something is going wrong.
        """
        y = np.atleast_1d(innovation)
        s = np.atleast_2d(innovation_cov)
        nis = float(y @ np.linalg.solve(s, y))
        accepted = bool(nis <= self._threshold)
        return accepted, nis

    @property
    def threshold(self) -> float:
        """The chi-square critical value for this gate."""
        return self._threshold


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
        if not self.nis_history:
            return float(self.dof)
        return float(np.mean(self.nis_history))

    @property
    def bounds(self) -> tuple[float, float]:
        """Two-sided chi-square bounds on the average NIS, for the strip."""
        n = len(self.nis_history)
        if n == 0:
            low = float(stats.chi2.ppf(0.025, df=self.dof))
            high = float(stats.chi2.ppf(0.975, df=self.dof))
            return low, high
        # Sum of N chi2(dof) variables is chi2(N * dof)
        low = float(stats.chi2.ppf(0.025, df=n * self.dof) / n)
        high = float(stats.chi2.ppf(0.975, df=n * self.dof) / n)
        return low, high


class NisLogger:
    """Per-channel NIS accounting, live during the demo and saved after it."""

    def __init__(self, channels: dict[str, int]) -> None:
        """
        Args:
            channels: channel name -> innovation dimension, e.g.
                {"velocity": 2, "heading": 1, "gps": 2, "zupt": 2, "zaru": 1}.
        """
        self._channels: dict[str, ChannelStats] = {
            ch: ChannelStats(dof=dof) for ch, dof in channels.items()
        }

    def record(self, channel: str, nis: float, accepted: bool) -> None:
        """Log one gate decision."""
        if channel in self._channels:
            st = self._channels[channel]
            st.nis_history.append(float(nis))
            if accepted:
                st.accepted += 1
            else:
                st.rejected += 1

    def snapshot(self) -> dict[str, float]:
        """Latest NIS per channel, for TelemetryFrame.nis."""
        return {
            ch: (st.nis_history[-1] if st.nis_history else float(st.dof))
            for ch, st in self._channels.items()
        }

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Chi-square bounds per channel, for TelemetryFrame.nis_bounds."""
        return {ch: st.bounds for ch, st in self._channels.items()}

    def is_consistent(self) -> dict[str, bool]:
        """Per channel: is the average NIS inside its bounds?

        This is the pass/fail the milestone exit criteria are written against.
        """
        res: dict[str, bool] = {}
        for ch, st in self._channels.items():
            if not st.nis_history:
                res[ch] = True
            else:
                low, high = st.bounds
                res[ch] = bool(low <= st.mean_nis <= high)
        return res


def nees(error: Array, covariance: Array) -> float:
    """Normalised estimation error squared, against ground truth.

    Only computable offline on recorded runs where truth is known. NIS is its live
    cousin and needs no truth.
    """
    e = np.atleast_1d(error)
    cov = np.atleast_2d(covariance)
    return float(e @ np.linalg.solve(cov, e))
