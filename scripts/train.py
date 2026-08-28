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

Window-building (``dr_core.datasets.windowing``) is shared with
``scripts/evaluate_model.py`` -- training and evaluation must build windows the exact
same way, or a good eval number stops meaning anything.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dr_core.datasets import build_dataset, load_combined_recordings, split_by_trajectory
from dr_core.models.tcn import augment_random_yaw, build_model, gaussian_nll_loss
from dr_core.preprocess import DEFAULT_HOP_S, DEFAULT_RATE_HZ

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]


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
    parser.add_argument("--epochs", type=int, default=60, help="additional epochs to run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--out", default="models/tcn.pt")
    parser.add_argument("--seed", type=int, default=26168)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="checkpoint to resume from -- --epochs is how many MORE epochs to run "
        "beyond the checkpoint's own epoch count",
    )
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

    try:
        recordings = load_combined_recordings(
            args.data, args.oxiod_data, args.oxiod_carry_positions
        )
    except FileNotFoundError as e:
        print(f"train: {e}", file=sys.stderr)
        return 2
    print(f"train: loaded {len(recordings)} recordings")

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

    x_train, y_train = build_dataset(train_recs, args.window_s, DEFAULT_HOP_S, DEFAULT_RATE_HZ)
    x_val, y_val = build_dataset(val_recs, args.window_s, DEFAULT_HOP_S, DEFAULT_RATE_HZ)
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

    # Fingerprints the exact set of recordings behind val_loader, so a resumed run can
    # tell whether "best_val_loss" from the checkpoint is even comparable to this run's
    # val_nll. Different --data/--oxiod-data/--oxiod-carry-positions between the
    # original run and a resume produce a DIFFERENT split_by_trajectory result (the
    # seeded shuffle depends on the input list's length and order) -- comparing losses
    # across two different validation sets is comparing two different numbers that
    # happen to share a name.
    val_fingerprint = hashlib.sha256(
        "|".join(sorted(r.meta.session_id for r in val_recs)).encode()
    ).hexdigest()

    model = build_model().to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from is not None:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        if resume_ckpt.get("val_fingerprint") == val_fingerprint:
            best_val_loss = float(resume_ckpt.get("val_nll", float("inf")))
            print(
                f"train: resumed from {args.resume_from} (epoch {start_epoch}, "
                f"val_nll={best_val_loss:.4f}, same validation set -- comparable)"
            )
        else:
            print(
                f"train: resumed from {args.resume_from} (epoch {start_epoch}) -- "
                "validation set differs from that checkpoint's (different --data/"
                "--oxiod-data/--oxiod-carry-positions), so its val_nll is not "
                "comparable to this run's. Starting best_val_loss fresh at +inf "
                "instead of silently comparing two different numbers."
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device) if len(val_loader) else float("nan")
        print(f"epoch {epoch:3d}/{end_epoch}  train_nll={train_loss:.4f}  val_nll={val_loss:.4f}")

        # Always save the last epoch too, even if it never beat best_val_loss -- a
        # resume run must never finish with literally nothing written, whatever the
        # reason (a raised validation bar, an unlucky loss curve, or a bug we haven't
        # found yet).
        is_best = len(val_loader) > 0 and val_loss < best_val_loss
        is_last = epoch == end_epoch
        if is_best or is_last:
            if is_best:
                best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_nll": val_loss,
                    "window_s": args.window_s,
                    "rate_hz": DEFAULT_RATE_HZ,
                    "seed": args.seed,
                    "val_fingerprint": val_fingerprint,
                    "is_best": is_best,
                },
                out_path,
            )

    print(f"train: checkpoint written to {out_path} (best_val_nll={best_val_loss:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
