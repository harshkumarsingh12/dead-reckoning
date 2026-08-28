"""RoNIN, OxIOD, and our own recordings -- normalised to one in-memory shape.

OWNER: Sumedha  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dr_core.types import GpsFix, ImuSample, SessionMeta, Trajectory

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

# WGS84 approximation for a local ENU projection, meters per degree at the origin
# latitude. Mirrors dr_core.fusion.eskf.Eskf.update_gps's conversion exactly -- GPS is
# a training label here and an opportunistic filter reset there, and both need the same
# lat/lon -> world ENU arithmetic or the label a model is trained against and the truth
# an evaluation run is scored against would silently disagree.
_M_PER_DEG_LAT = 111132.92


def _gps_to_world_enu(
    fixes: Sequence[GpsFix], origin_lat_deg: float, origin_lon_deg: float
) -> Array:
    lat_rad = np.deg2rad(origin_lat_deg)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad)
    return np.array(
        [
            [
                (fix.lon_deg - origin_lon_deg) * m_per_deg_lon,
                (fix.lat_deg - origin_lat_deg) * _M_PER_DEG_LAT,
            ]
            for fix in fixes
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class Recording:
    """One walk: the inputs, the ground truth, and enough metadata to interpret both."""

    meta: SessionMeta
    imu: Sequence[ImuSample]
    truth: Trajectory | None = None  # Vicon, GPS, or surveyed corner points
    gps: Sequence[GpsFix] = ()

    @property
    def duration_s(self) -> float:
        if len(self.imu) < 2:
            return 0.0
        return float(self.imu[-1].t_ns - self.imu[0].t_ns) / 1.0e9

    @property
    def distance_m(self) -> float:
        """Ground-truth path length. The denominator of every drift percentage."""
        if self.truth is None or len(self.truth) < 2:
            raise ValueError("no ground-truth trajectory (or fewer than 2 points) to measure")
        diffs = np.diff(self.truth.p_world, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


# Access-request details echoed into the FileNotFoundError below, kept in sync with
# scripts/fetch_datasets.py's DATASET_INFO so there is one place these live.
_ACCESS_URLS = {
    "ronin": "https://ronin.cs.sfu.ca/",
    "oxiod": "http://deepio.cs.ox.ac.uk/",
}


def load_ronin(root: Path, subjects: list[str] | None = None) -> list[Recording]:
    """Load RoNIN. Large multi-subject smartphone IMU with high-quality truth.

    Also the method template -- heading-agnostic velocity regression is RoNIN's idea and
    this project builds directly on it.

    NOT YET IMPLEMENTED beyond the access check: RoNIN's on-disk layout (HDF5 per
    subject/trial, with the IMU streams, orientation and Tango ground truth stored under
    specific dataset paths) needs to be parsed against real downloaded files to get
    right. This environment has no RoNIN access (issue #14, blocked) and no local copy
    to verify column names, units or axis order against -- guessing at that schema and
    being wrong would silently mislabel training data, which is worse than this stub
    (AGENTS.md: "prefer a stub with a clear NotImplementedError... over a plausible
    guess"). Implement the parse once access lands and a real file is in hand to check
    against.

    Raises:
        FileNotFoundError: with the access-request URL in the message, because the
            first person to hit this will not have the data yet.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"RoNIN root not found: {root}. Request access first: {_ACCESS_URLS['ronin']} "
            "(see data/README.md and scripts/fetch_datasets.py --dataset ronin --info)"
        )
    raise NotImplementedError(
        "M0 -- owner: Sumedha -- RoNIN access-check passes, but the on-disk parser "
        "still needs a real downloaded file to verify the schema against"
    )


def load_oxiod(root: Path, carry_positions: list[str] | None = None) -> list[Recording]:
    """Load OxIOD. Vicon ground truth across hand, pocket, bag and trolley.

    The carry-position variety is the reason to bother with it: it is what the
    carry-position robustness claim in the demo actually rests on.

    NOT YET IMPLEMENTED beyond the access check -- see load_ronin's docstring; the same
    reasoning applies to OxIOD's per-trial CSV layout.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"OxIOD root not found: {root}. Request access first: {_ACCESS_URLS['oxiod']} "
            "(see data/README.md and scripts/fetch_datasets.py --dataset oxiod --info)"
        )
    raise NotImplementedError(
        "M0 -- owner: Sumedha -- OxIOD access-check passes, but the on-disk parser "
        "still needs a real downloaded file to verify the schema against"
    )


def load_own_recording(path: Path) -> Recording:
    """Load one of our own sessions, written by ``dr_core.io``.

    These are the domain-matched fine-tuning set and the demo course. Outdoor walks
    with strong GPS provide velocity and position labels; indoor loops with surveyed
    corners provide the evaluation truth.

    Ground truth is derived from the recording's GPS fixes when present (projected to
    world ENU the same way ``dr_core.fusion.eskf`` does it, relative to the session's
    recorded origin, or the first fix if no origin was set). A GPS-free recording (an
    indoor loop with truth entered separately from surveyed corner points) comes back
    with ``truth=None`` -- the caller supplies truth for those.
    """
    from dr_core.io.session import SessionReader

    reader = SessionReader(path)
    meta = reader.meta

    imu: list[ImuSample] = []
    gps: list[GpsFix] = []
    for record_type, payload in reader:
        if record_type == "imu" and isinstance(payload, ImuSample):
            imu.append(payload)
        elif record_type == "gps" and isinstance(payload, GpsFix):
            gps.append(payload)

    truth: Trajectory | None = None
    if gps:
        origin_lat = meta.origin_lat_deg if meta.origin_lat_deg is not None else gps[0].lat_deg
        origin_lon = meta.origin_lon_deg if meta.origin_lon_deg is not None else gps[0].lon_deg
        truth = Trajectory(
            t_ns=np.array([fix.t_ns for fix in gps], dtype=np.int64),
            p_world=_gps_to_world_enu(gps, origin_lat, origin_lon),
            label="gps_truth",
        )

    return Recording(meta=meta, imu=imu, truth=truth, gps=gps)


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

    Raises:
        ValueError: if the fractions are not each in [0, 1] and summing to at most 1.
    """
    fracs_valid = (0.0 <= train_frac <= 1.0) and (0.0 <= val_frac <= 1.0)
    if not fracs_valid or train_frac + val_frac > 1.0:
        raise ValueError(
            f"invalid split: train_frac={train_frac}, val_frac={val_frac} "
            "(each must be in [0, 1] and sum to at most 1)"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recordings))
    shuffled = [recordings[i] for i in order]

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
