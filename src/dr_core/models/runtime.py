"""ONNX Runtime inference wrapper. This is what the live pipeline actually calls.

OWNER: Sumedha  |  MILESTONE: M2  |  Spec: docs/BUILD_PLAN.md section 6.4

Kept free of any torch import on purpose: the demo laptop and the eventual on-device
build only ever need onnxruntime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from dr_core.types import VelocityEstimate

    Array = npt.NDArray[np.float64]

INFERENCE_BUDGET_MS = 10.0  # stated target, laptop, per window (build plan section 8)


class VelocityModelRuntime:
    """Loads a ``.onnx`` velocity model and runs one causal window at a time."""

    def __init__(self, model_path: Path | str, warmup_iterations: int = 5) -> None:
        """Load the model and warm it up.

        The warm-up matters: the first ONNX Runtime call pays graph-optimisation cost,
        and taking that hit during the demo is a visible stutter on the first frame.
        """
        raise NotImplementedError("M2 -- owner: Sumedha")

    def predict(self, window: Array, t_ns: int) -> VelocityEstimate:
        """Run one window.

        Args:
            window: (IN_CHANNELS, window_samples), produced by
                ``dr_core.preprocess.prepare_window``. Never build this array by hand
                anywhere else -- that is the training/live mismatch the shared
                preprocessing module exists to prevent.
            t_ns: capture timestamp of the LAST sample in the window. The estimate is
                attributed to that instant, which is what makes a causal window worth
                having.

        Returns:
            A VelocityEstimate carrying the device-frame velocity and its covariance.
        """
        raise NotImplementedError("M2 -- owner: Sumedha")

    @property
    def last_inference_ms(self) -> float:
        """Wall time of the most recent predict() call. Displayed on the strip."""
        raise NotImplementedError("M2 -- owner: Sumedha")

    def benchmark(self, iterations: int = 200) -> float:
        """Median per-window inference time in ms. Must come in under the budget."""
        raise NotImplementedError("M2 -- owner: Sumedha")
