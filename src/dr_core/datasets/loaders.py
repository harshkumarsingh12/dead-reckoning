"""RoNIN, OxIOD, and our own recordings -- normalised to one in-memory shape.

OWNER: Sumedha  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dr_core.types import GpsFix, ImuSample, SessionMeta, Trajectory


@dataclass(frozen=True, slots=True)
class Recording:
    """One walk: the inputs, the ground truth, and enough metadata to interpret both."""

    meta: SessionMeta
    imu: Sequence[ImuSample]
    truth: Trajectory | None = None  # Vicon, GPS, or surveyed corner points
    gps: Sequence[GpsFix] = ()

    @property
    def duration_s(self) -> float:
        raise NotImplementedError("M0 -- owner: Sumedha")

    @property
    def distance_m(self) -> float:
        """Ground-truth path length. The denominator of every drift percentage."""
        raise NotImplementedError("M0 -- owner: Sumedha")


def load_ronin(root: Path, subjects: list[str] | None = None) -> list[Recording]:
    """Load RoNIN. Large multi-subject smartphone IMU with high-quality truth.

    Also the method template -- heading-agnostic velocity regression is RoNIN's idea and
    this project builds directly on it.

    Raises:
        FileNotFoundError: with the access-request URL in the message, because the
            first person to hit this will not have the data yet.
    """
    raise NotImplementedError("M0 -- owner: Sumedha")


def load_oxiod(root: Path, carry_positions: list[str] | None = None) -> list[Recording]:
    """Load OxIOD. Vicon ground truth across hand, pocket, bag and trolley.

    The carry-position variety is the reason to bother with it: it is what the
    carry-position robustness claim in the demo actually rests on.
    """
    raise NotImplementedError("M0 -- owner: Sumedha")


def load_own_recording(path: Path) -> Recording:
    """Load one of our own sessions, written by ``dr_core.io``.

    These are the domain-matched fine-tuning set and the demo course. Outdoor walks
    with strong GPS provide velocity and position labels; indoor loops with surveyed
    corners provide the evaluation truth.
    """
    raise NotImplementedError("M0 -- owner: Sumedha")


def split_by_trajectory(
    recordings: list[Recording],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 26168,
) -> tuple[list[Recording], list[Recording], list[Recording]]:
    """Split whole recordings into train / val / test.

    By trajectory, never by window. A window-level split leaks context between the sets
    and every metric afterwards is fiction. This function exists so nobody has to
    remember that at 3 a.m.

    Returns:
        (train, val, test).
    """
    raise NotImplementedError("M0 -- owner: Sumedha")
