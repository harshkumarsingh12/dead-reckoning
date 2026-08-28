"""Turn Recordings into (window, target velocity) pairs. Shared by train and eval.

OWNER: Sumedha  |  MILESTONE: M2

Moved out of scripts/train.py so scripts/evaluate_model.py can build IDENTICAL windows
against held-out data -- the same shared-preprocessing discipline as
``dr_core.preprocess``, just one layer up: if training and evaluation build windows two
different ways, a good eval number stops meaning anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dr_core.ahrs import AhrsFilter, MagGate
from dr_core.datasets.loaders import Recording, load_own_recording, load_oxiod
from dr_core.models.tcn import IN_CHANNELS
from dr_core.preprocess import CalibrationResult, heading_rad, prepare_window, rotate_world_to_dev

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import OrientationEstimate, Trajectory

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]


def load_combined_recordings(
    data_dir: str | Path | None,
    oxiod_root: str | Path | None,
    oxiod_carry_positions: list[str] | None = None,
) -> list[Recording]:
    """Load our own recordings and/or OxIOD into one list, in a fixed, deterministic
    order -- train.py and evaluate_model.py must build this identically, since
    ``split_by_trajectory``'s seeded shuffle depends on the input order matching.

    Raises:
        FileNotFoundError: ``oxiod_root`` is given but does not exist.
    """
    recordings: list[Recording] = []
    if data_dir is not None:
        session_paths = sorted(Path(data_dir).rglob("*.jsonl.gz"))
        recordings.extend(load_own_recording(path) for path in session_paths)
    if oxiod_root is not None:
        recordings.extend(load_oxiod(Path(oxiod_root), carry_positions=oxiod_carry_positions))
    return recordings


def orientations_for_recording(recording: Recording, rate_hz: float) -> list[OrientationEstimate]:
    """Run the AHRS over one recording's raw IMU using its own session calibration.

    ``accel_bias_body`` has no source in ``SessionMeta`` -- the calibration ritual
    (data/README.md) only measures gyro bias and magnetometer hard iron, so zero is not
    a guess here, it is what the documented ritual actually calibrates. Same for a
    missing ``gyro_bias_body`` / ``mag_hard_iron_body``: treated as already zero-bias
    rather than invented.
    """
    meta = recording.meta
    calibration = CalibrationResult(
        gyro_bias_body=(meta.gyro_bias_body if meta.gyro_bias_body is not None else np.zeros(3)),
        accel_bias_body=np.zeros(3),
        mag_hard_iron_body=(
            meta.mag_hard_iron_body if meta.mag_hard_iron_body is not None else np.zeros(3)
        ),
    )
    mag_gate = MagGate(calibration.expected_field_strength_t, calibration.expected_dip_rad)
    ahrs = AhrsFilter(calibration, mag_gate, rate_hz=rate_hz)
    return [ahrs.update(sample) for sample in recording.imu]


def velocity_from_truth(truth: Trajectory) -> tuple[IntArray, Array]:
    """World-frame velocity between consecutive truth points, attributed to the END of
    each interval -- causal, matching a window's own "ends at now" convention."""
    t_s = truth.t_ns.astype(np.float64) / 1.0e9
    dt_s = np.diff(t_s)
    dp = np.diff(truth.p_world, axis=0)
    v_world = dp / dt_s[:, None]
    return truth.t_ns[1:], v_world


def windows_for_recording(
    recording: Recording,
    orientations: list[OrientationEstimate],
    window_s: float,
    hop_s: float,
    rate_hz: float,
) -> tuple[Array, Array, IntArray]:
    """Slide causal windows across one recording, paired with a device-frame target
    velocity interpolated from the ground truth at each window's end time.

    Returns:
        (windows, targets, t_ns) -- ``t_ns[i]`` is the capture time of the LAST sample
        in ``windows[i]``, the instant that window's prediction is attributed to.
    """
    window_samples = round(window_s * rate_hz)
    empty = (
        np.empty((0, IN_CHANNELS, window_samples)),
        np.empty((0, 2)),
        np.empty((0,), dtype=np.int64),
    )

    if recording.truth is None or len(recording.truth) < 2 or len(recording.imu) < 2:
        return empty

    t_ns = np.array([s.t_ns for s in recording.imu], dtype=np.int64)
    t_v_ns, v_world = velocity_from_truth(recording.truth)

    window_ns = int(window_s * 1.0e9)
    hop_ns = int(hop_s * 1.0e9)
    lookback_ns = int(window_s * 1.5 * 1.0e9)  # margin so resampling never runs short

    windows: list[Array] = []
    targets: list[Array] = []
    window_t_ns: list[int] = []

    t_end = t_ns[0] + window_ns
    while t_end <= t_ns[-1]:
        lo = int(np.searchsorted(t_ns, t_end - lookback_ns, side="left"))
        hi = int(np.searchsorted(t_ns, t_end, side="right"))
        if hi - lo < 2:
            t_end += hop_ns
            continue

        try:
            window = prepare_window(
                list(recording.imu[lo:hi]), orientations[lo:hi], rate_hz=rate_hz, window_s=window_s
            )
        except ValueError:
            t_end += hop_ns
            continue

        v_world_now = np.array(
            [np.interp(t_end, t_v_ns, v_world[:, 0]), np.interp(t_end, t_v_ns, v_world[:, 1])]
        )
        psi_now = heading_rad(np.asarray(orientations[hi - 1].q_world_body, dtype=np.float64))
        target = rotate_world_to_dev(v_world_now, psi_now)

        windows.append(window)
        targets.append(target)
        window_t_ns.append(int(t_ns[hi - 1]))
        t_end += hop_ns

    if not windows:
        return empty
    return np.stack(windows), np.stack(targets), np.array(window_t_ns, dtype=np.int64)


def build_dataset(
    recordings: list[Recording], window_s: float, hop_s: float, rate_hz: float
) -> tuple[Array, Array]:
    """Windows + targets pooled across many recordings, for training. Drops the
    per-window timestamp ``windows_for_recording`` returns -- pooled training doesn't
    need it, only per-recording evaluation does."""
    window_samples = round(window_s * rate_hz)
    all_windows: list[Array] = []
    all_targets: list[Array] = []
    for recording in recordings:
        if recording.truth is None:
            continue
        orientations = orientations_for_recording(recording, rate_hz=rate_hz)
        windows, targets, _t_ns = windows_for_recording(
            recording, orientations, window_s, hop_s, rate_hz
        )
        if windows.shape[0] > 0:
            all_windows.append(windows)
            all_targets.append(targets)

    if not all_windows:
        return np.empty((0, IN_CHANNELS, window_samples)), np.empty((0, 2))
    return np.concatenate(all_windows, axis=0), np.concatenate(all_targets, axis=0)
