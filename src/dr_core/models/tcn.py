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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

# Input channels: gravity-aligned accel (3) + gyro (3).
IN_CHANNELS = 6
# Output: mean (2) + lower-triangular Cholesky factor of the 2x2 covariance (3).
OUT_DIM = 5


def build_model(
    in_channels: int = IN_CHANNELS,
    hidden: int = 64,
    levels: int = 6,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> Any:
    """Construct the causal TCN.

    Receptive field is ``(kernel_size - 1) * (2**levels - 1) + 1`` samples; at 200 Hz
    the defaults cover roughly 1.9 s, matching the intended window.

    Returns:
        A ``torch.nn.Module``. Typed as Any so this module imports cleanly without the
        [ml] extra installed.

    Raises:
        ImportError: if torch is not available. Install with: pip install -e ".[ml]"
    """
    raise NotImplementedError("M2 -- owner: Sumedha")


def gaussian_nll_loss(pred: Any, target: Any, min_sigma: float = 1e-3) -> Any:
    """Gaussian negative log-likelihood over the 2D velocity error.

    Trains the mean and the covariance together. ``min_sigma`` floors the predicted
    scale so the loss cannot run away by driving the variance to zero on easy windows.

    Args:
        pred: (batch, 5) -- mean (2) then the Cholesky factor entries (3).
        target: (batch, 2) -- ground-truth device-frame velocity.
        min_sigma: floor on the diagonal, m/s.
    """
    raise NotImplementedError("M2 -- owner: Sumedha")


def augment_random_yaw(window: Array, target: Array, rng: Any) -> tuple[Array, Array]:
    """Rotate a training window and its label by a random yaw.

    This is what actually enforces heading-agnosticism. Without it the model quietly
    learns the yaw distribution of the training set and falls apart from the pocket.
    """
    raise NotImplementedError("M2 -- owner: Sumedha")


def export_onnx(model: Any, path: str, window_samples: int, quantize_int8: bool = True) -> None:
    """Export to ONNX, optionally int8-quantized.

    Exported EARLY and deliberately: it de-risks on-device inference and gives a real
    per-window latency number to quote, rather than a hope.
    """
    raise NotImplementedError("M2 -- owner: Sumedha")
