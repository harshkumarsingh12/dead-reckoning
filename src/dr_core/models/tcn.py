"""Causal TCN with a Gaussian-NLL covariance head. Training only -- needs [ml].

OWNER: Sumedha  |  MILESTONE: M2  |  Spec: docs/BUILD_PLAN.md section 6.4

Design points that are not negotiable without a spec change:

  * CAUSAL. No future context. A centred 1-2 s window estimates mid-window velocity and
    silently adds 0.5-1.0 s of delay, which is exactly the "dot lags the turn" failure
    that reads as weak on a live demo.
  * Output frame is the gravity-aligned, HEADING-AGNOSTIC device frame, not world.
    Combined with random-yaw augmentation this is what makes the model indifferent to
    how the phone is held.
  * The covariance head is trained JOINTLY under Gaussian NLL. An MSE fit with a
    variance bolted on afterwards produces numbers that look like uncertainty and are
    not, which then poisons the filter through R.

Done when: model-only integration beats the PDR baseline on held-out data at
single-digit-percent drift, the 1-sigma coverage test passes at roughly 68%, and one
window infers in under 10 ms.

Torch is imported lazily, inside each function, and nowhere at module level. That is
what lets this file -- and anything that imports its constants -- collect cleanly under
the default (non-``[ml]``) test run; see test_models.py's module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

# Input channels: gravity-aligned accel (3) + gyro (3).
IN_CHANNELS = 6
# Output: mean (2) + lower-triangular Cholesky factor of the 2x2 covariance (3).
OUT_DIM = 5

# Cholesky-factor floor, m/s. Keeps the loss from running away by driving predicted
# variance to zero on easy windows (see gaussian_nll_loss).
_DEFAULT_MIN_SIGMA = 1e-3


def build_model(
    in_channels: int = IN_CHANNELS,
    hidden: int = 64,
    levels: int = 6,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> Any:
    """Construct the causal TCN.

    Each level is a residual block of two dilated causal convolutions sharing one
    dilation ``2**level``, following Bai/Kolter/Koltun's TCN. Causality comes from
    padding only the left (past) side of the sequence and cropping the equivalent
    amount off the right after each conv, so output at time ``t`` never sees input at
    ``t' > t``.

    Receptive field is ``1 + 2 * (kernel_size - 1) * (2**levels - 1)`` samples; at
    200 Hz the defaults (kernel_size=3, levels=6) cover roughly 1.3 s, inside the
    1-2 s window the build plan calls for (section 6.4). The model still accepts any
    input length -- the receptive field is a capacity bound, not a requirement that
    every window be that long.

    Returns:
        A ``torch.nn.Module`` mapping ``(batch, in_channels, window_samples)`` to
        ``(batch, OUT_DIM, window_samples)``. The prediction for a causal window ending
        at "now" is the last time step: ``model(x)[:, :, -1]``.

    Raises:
        ImportError: if torch is not available. Install with: pip install -e ".[ml]"
    """
    import torch
    from torch import nn

    class _Chomp1d(nn.Module):
        """Drops the trailing ``chomp_size`` steps a symmetric same-padding conv added."""

        def __init__(self, chomp_size: int) -> None:
            super().__init__()
            self._chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._chomp_size == 0:
                return x
            return x[:, :, : -self._chomp_size].contiguous()

    class _CausalResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
                _Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
                _Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            self.out_relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            out: torch.Tensor = self.out_relu(y + res)
            return out

    class CausalTcn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[nn.Module] = []
            ch = in_channels
            for level in range(levels):
                dilation = 2**level
                blocks.append(_CausalResidualBlock(ch, hidden, kernel_size, dilation))
                ch = hidden
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Conv1d(hidden, OUT_DIM, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.head(self.blocks(x))
            return out

    return CausalTcn()


def gaussian_nll_loss(pred: Any, target: Any, min_sigma: float = 1e-3) -> Any:
    """Gaussian negative log-likelihood over the 2D velocity error.

    Trains the mean and the covariance together. ``min_sigma`` floors the predicted
    scale so the loss cannot run away by driving the variance to zero on easy windows.

    ``pred``'s last 3 columns are the lower-triangular Cholesky factor of the
    covariance, ``L = [[l11, 0], [l21, l22]]`` with ``Sigma = L @ L.T``: ``l11`` and
    ``l22`` pass through a floored softplus to stay strictly positive, ``l21`` is
    unconstrained. The residual is whitened by solving ``L z = r`` (forward
    substitution), so the loss is ``0.5 * z^T z + log(l11) + log(l22)`` plus the
    (constant, but kept for a comparable absolute NLL number) ``log(2 * pi)`` term.

    Args:
        pred: (batch, 5) -- mean (2) then the Cholesky factor entries (3).
        target: (batch, 2) -- ground-truth device-frame velocity.
        min_sigma: floor on the diagonal, m/s.
    """
    import torch
    import torch.nn.functional as functional

    mean = pred[..., 0:2]
    l11 = functional.softplus(pred[..., 2]) + min_sigma
    l21 = pred[..., 3]
    l22 = functional.softplus(pred[..., 4]) + min_sigma

    residual = target - mean
    r0 = residual[..., 0]
    r1 = residual[..., 1]

    # Forward substitution: L z = r.
    z0 = r0 / l11
    z1 = (r1 - l21 * z0) / l22

    quad = z0 * z0 + z1 * z1
    log_det_half = torch.log(l11) + torch.log(l22)  # 0.5 * log|Sigma|
    nll = 0.5 * quad + log_det_half + float(np.log(2.0 * np.pi))
    return nll.mean()


def augment_random_yaw(window: Array, target: Array, rng: Any) -> tuple[Array, Array]:
    """Rotate a training window and its label by a random yaw.

    This is what actually enforces heading-agnosticism. Without it the model quietly
    learns the yaw distribution of the training set and falls apart from the pocket.

    ``window`` is ``(IN_CHANNELS, window_samples)`` in the gravity-aligned device frame
    (rows 0:3 = accel, 3:6 = gyro; both are 3-vectors, so the rotation below is applied
    to their horizontal (x, y) components only -- the vertical (Up) axis is invariant
    under a yaw rotation). ``target`` is the (2,) planar device-frame velocity label,
    rotated by the same angle so the pair stays consistent.
    """
    theta = float(rng.uniform(-np.pi, np.pi))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    window_out = np.array(window, dtype=np.float64, copy=True)
    window_out[0:2, :] = rot @ window_out[0:2, :]  # accel x, y
    window_out[3:5, :] = rot @ window_out[3:5, :]  # gyro x, y

    target_out: Array = rot @ np.asarray(target, dtype=np.float64)
    return window_out, target_out


def export_onnx(model: Any, path: str, window_samples: int, quantize_int8: bool = True) -> None:
    """Export to ONNX, optionally int8-quantized.

    Exported EARLY and deliberately: it de-risks on-device inference and gives a real
    per-window latency number to quote, rather than a hope.

    The exported graph has a fixed input shape ``(1, IN_CHANNELS, window_samples)`` --
    the live pipeline always feeds exactly one window at a time, so there is nothing to
    gain from dynamic axes and something to lose (a slower, less optimizable graph).
    """
    import shutil
    import tempfile
    from pathlib import Path

    import torch

    model = model.eval()
    dummy = torch.randn(1, IN_CHANNELS, window_samples)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["window"],
        output_names=["prediction"],
        opset_version=18,
        # dynamo=True (the default) is torch's own recommended exporter as of the
        # pinned version; the legacy TorchScript exporter is now deprecated and would
        # turn into a hard error under this repo's filterwarnings config. Needs the
        # `onnxscript` package -- see the `ml` extra in pyproject.toml.
        dynamo=True,
    )

    if quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_path = Path(tmp_dir) / "quantized.onnx"
            quantize_dynamic(str(out_path), str(quantized_path), weight_type=QuantType.QInt8)
            shutil.move(str(quantized_path), str(out_path))
