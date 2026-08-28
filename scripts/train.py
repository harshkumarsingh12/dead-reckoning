#!/usr/bin/env python
"""Train the causal TCN velocity model. Needs the [ml] extra.

OWNER: Sumedha  |  MILESTONE: M2

    pip install -e ".[ml]"
    python scripts/train.py --data data/own --oxiod-data data/oxiod --epochs 60 --out models/tcn.pt

Imports dr_core.preprocess -- the SAME module the live pipeline imports. If you ever
find yourself preparing data any other way here, stop: that divergence is the exact
failure the shared module exists to prevent, and it will show up as an unexplained
live-demo underperformance rather than as an error.

``--data`` is a directory searched for our own recordings (``*.jsonl.gz``, written by
``dr_core.io.SessionWriter``). ``--oxiod-data`` is an OxIOD root (see
``dr_core.datasets.load_oxiod``). At least one of the two is required. RoNIN loading
(``dr_core.datasets.load_ronin``) is not implemented yet -- no real file to verify its
on-disk schema against (see ``src/dr_core/datasets/loaders.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dr_core.ahrs import AhrsFilter, MagGate
from dr_core.datasets import Recording, load_own_recording, load_oxiod, split_by_trajectory
from dr_core.models.tcn import IN_CHANNELS, augment_random_yaw, build_model, gaussian_nll_loss
from dr_core.preprocess import (
    DEFAULT_HOP_S,
    DEFAULT_RATE_HZ,
    CalibrationResult,
    heading_rad,
    prepare_window,
    rotate_world_to_dev,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from dr_core.types import OrientationEstimate, Trajectory

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]


def _orientations_for_recording(recording: Recording, rate_hz: float) -> list[OrientationEstimate]:
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


def _velocity_from_truth(truth: Trajectory) -> tuple[IntArray, Array]:
    """World-frame velocity between consecutive truth points, attributed to the END of
    each interval -- causal, matching a window's own "ends at now" convention."""
    t_s = truth.t_ns.astype(np.float64) / 1.0e9
    dt_s = np.diff(t_s)
    dp = np.diff(truth.p_world, axis=0)
    v_world = dp / dt_s[:, None]
    return truth.t_ns[1:], v_world


def _windows_for_recording(
    recording: Recording,
    orientations: list[OrientationEstimate],
    window_s: float,
    hop_s: float,
    rate_hz: float,
) -> tuple[Array, Array]:
    """Slide causal windows across one recording, paired with a device-frame target
    velocity interpolated from the ground truth at each window's end time."""
    window_samples = round(window_s * rate_hz)
    empty = (np.empty((0, IN_CHANNELS, window_samples)), np.empty((0, 2)))

    if recording.truth is None or len(recording.truth) < 2 or len(recording.imu) < 2:
        return empty

    t_ns = np.array([s.t_ns for s in recording.imu], dtype=np.int64)
    t_v_ns, v_world = _velocity_from_truth(recording.truth)

    window_ns = int(window_s * 1.0e9)
    hop_ns = int(hop_s * 1.0e9)
    lookback_ns = int(window_s * 1.5 * 1.0e9)  # margin so resampling never runs short

    windows: list[Array] = []
    targets: list[Array] = []

    t_end = t_ns[0] + window_ns
    while t_end <= t_ns[-1]:
        lo = int(np.searchsorted(t_ns, t_end - lookback_ns, side="left"))
        hi = int(np.searchsorted(t_ns, t_end, side="right"))
        if hi - lo < 2:
            t_end += hop_ns
            continue

        try:
            window = prepare_window(
                recording.imu[lo:hi], orientations[lo:hi], rate_hz=rate_hz, window_s=window_s
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
        t_end += hop_ns

    if not windows:
        return empty
    return np.stack(windows), np.stack(targets)


def _build_dataset(
    recordings: list[Recording], window_s: float, hop_s: float, rate_hz: float
) -> tuple[Array, Array]:
    window_samples = round(window_s * rate_hz)
    all_windows: list[Array] = []
    all_targets: list[Array] = []
    for recording in recordings:
        if recording.truth is None:
            continue
        orientations = _orientations_for_recording(recording, rate_hz=rate_hz)
        windows, targets = _windows_for_recording(recording, orientations, window_s, hop_s, rate_hz)
        if windows.shape[0] > 0:
            all_windows.append(windows)
            all_targets.append(targets)

    if not all_windows:
        return np.empty((0, IN_CHANNELS, window_samples)), np.empty((0, 2))
    return np.concatenate(all_windows, axis=0), np.concatenate(all_targets, axis=0)


class _WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """(window, target) pairs, with optional random-yaw augmentation applied per draw
    (a fresh rotation each epoch, rather than baked in once) -- see augment_random_yaw's
    docstring for why this is what actually enforces heading-agnosticism."""

    def __init__(self, windows: Array, targets: Array, augment: bool, seed: int) -> None:
        self._windows = windows
        self._targets = targets
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self._windows.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._windows[index]
        target = self._targets[index]
        if self._augment:
            window, target = augment_random_yaw(window, target, self._rng)
        return (
            torch.from_numpy(window.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    for windows, targets in loader:
        windows = windows.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            pred = model(windows)[:, :, -1]  # last (current) time step
            loss = gaussian_nll_loss(pred, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * windows.shape[0]
        total_n += windows.shape[0]
    return total_loss / total_n if total_n else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="directory of our own recordings (*.jsonl.gz)")
    parser.add_argument("--oxiod-data", default=None, help="OxIOD dataset root")
    parser.add_argument(
        "--oxiod-carry-positions",
        nargs="*",
        default=None,
        help="subset of OxIOD carry-position folders (default: all except the official test split)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--out", default="models/tcn.pt")
    parser.add_argument("--seed", type=int, default=26168)
    parser.add_argument(
        "--no-yaw-aug",
        action="store_true",
        help="disable random-yaw augmentation (ablation only -- it is what enforces "
        "heading-agnosticism, and without it the model fails from the pocket)",
    )
    args = parser.parse_args()

    if args.data is None and args.oxiod_data is None:
        print("train: pass --data, --oxiod-data, or both -- nothing to train on", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)

    recordings: list[Recording] = []

    if args.data is not None:
        data_root = Path(args.data)
        session_paths = sorted(data_root.rglob("*.jsonl.gz"))
        if not session_paths:
            print(f"train: no *.jsonl.gz recordings found under {data_root}", file=sys.stderr)
        recordings.extend(load_own_recording(path) for path in session_paths)
        print(f"train: loaded {len(session_paths)} of our own recordings from {data_root}")

    if args.oxiod_data is not None:
        try:
            oxiod_recordings = load_oxiod(
                Path(args.oxiod_data), carry_positions=args.oxiod_carry_positions
            )
        except FileNotFoundError as e:
            print(f"train: {e}", file=sys.stderr)
            return 2
        print(f"train: loaded {len(oxiod_recordings)} OxIOD recordings")
        recordings.extend(oxiod_recordings)

    with_truth = [r for r in recordings if r.truth is not None]
    if not with_truth:
        print(
            f"train: {len(recordings)} recording(s) loaded, but none carry ground "
            "truth -- nothing to train velocity labels against.",
            file=sys.stderr,
        )
        return 2

    train_recs, val_recs, _test_recs = split_by_trajectory(with_truth, seed=args.seed)
    print(
        f"train: {len(train_recs)} train / {len(val_recs)} val / {len(_test_recs)} test recordings"
    )

    x_train, y_train = _build_dataset(train_recs, args.window_s, DEFAULT_HOP_S, DEFAULT_RATE_HZ)
    x_val, y_val = _build_dataset(val_recs, args.window_s, DEFAULT_HOP_S, DEFAULT_RATE_HZ)
    if x_train.shape[0] == 0:
        print(
            "train: zero training windows were built -- recordings may be too short "
            f"for a {args.window_s:.2f} s window",
            file=sys.stderr,
        )
        return 2
    print(f"train: {x_train.shape[0]} train windows, {x_val.shape[0]} val windows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"train: using device {device} ({device_name})")
    pin_memory = device.type == "cuda"

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        _WindowDataset(x_train, y_train, augment=not args.no_yaw_aug, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        _WindowDataset(x_val, y_val, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device) if len(val_loader) else float("nan")
        print(f"epoch {epoch:3d}/{args.epochs}  train_nll={train_loss:.4f}  val_nll={val_loss:.4f}")

        if val_loss < best_val_loss or (len(val_loader) == 0 and epoch == args.epochs):
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_nll": val_loss,
                    "window_s": args.window_s,
                    "rate_hz": DEFAULT_RATE_HZ,
                },
                out_path,
            )

    print(f"train: best checkpoint written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
