"""ONNX Runtime inference wrapper. This is what the live pipeline actually calls.

OWNER: Sumedha  |  MILESTONE: M2  |  Spec: docs/BUILD_PLAN.md section 6.4

Kept free of any torch import on purpose: the demo laptop and the eventual on-device
build only ever need onnxruntime.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from dr_core.models.tcn import IN_CHANNELS
from dr_core.types import VelocityEstimate

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

INFERENCE_BUDGET_MS = 10.0  # stated target, laptop, per window (build plan section 8)

# Cholesky-factor floor, m/s -- must match dr_core.models.tcn.gaussian_nll_loss so the
# live covariance matches what training actually optimised against.
_MIN_SIGMA = 1e-3


def _softplus(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Numerically stable: softplus(x) = max(x, 0) + log1p(exp(-|x|)).
    result: npt.NDArray[np.float64] = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    return result


def _cholesky_output_to_cov(raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Turn the model's 3 raw Cholesky entries into a 2x2 covariance matrix.

    Mirrors ``dr_core.models.tcn.gaussian_nll_loss`` exactly: ``l11``/``l22`` through a
    floored softplus, ``l21`` unconstrained, ``Sigma = L @ L.T``. Any divergence between
    this and the training-time parameterisation would silently poison the filter's R.
    """
    l11 = float(_softplus(raw[0:1])[0]) + _MIN_SIGMA
    l21 = float(raw[1])
    l22 = float(_softplus(raw[2:3])[0]) + _MIN_SIGMA
    lower = np.array([[l11, 0.0], [l21, l22]], dtype=np.float64)
    cov: npt.NDArray[np.float64] = lower @ lower.T
    return cov


class VelocityModelRuntime:
    """Loads a ``.onnx`` velocity model and runs one causal window at a time."""

    def __init__(self, model_path: Path | str, warmup_iterations: int = 5) -> None:
        """Load the model and warm it up.

        The warm-up matters: the first ONNX Runtime call pays graph-optimisation cost,
        and taking that hit during the demo is a visible stutter on the first frame.
        """
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        window_samples = input_shape[-1]
        if not isinstance(window_samples, int):
            # Dynamic axis on a graph we always export with a fixed shape -- fall back
            # to a reasonable default rather than guessing wrong silently.
            window_samples = 200
        self._window_samples = window_samples
        self._last_inference_ms = 0.0

        warmup_window = np.zeros((IN_CHANNELS, self._window_samples), dtype=np.float64)
        for _ in range(warmup_iterations):
            self._run(warmup_window)

    def _run(self, window: Array) -> npt.NDArray[np.float64]:
        x = np.asarray(window, dtype=np.float32).reshape(1, IN_CHANNELS, -1)
        start = time.perf_counter()
        (out,) = self._session.run(None, {self._input_name: x})
        self._last_inference_ms = (time.perf_counter() - start) * 1000.0
        # (1, OUT_DIM, window_samples) -> the last (current) time step, float64 for the
        # rest of the pipeline's strict-SI-double convention.
        return np.asarray(out[0, :, -1], dtype=np.float64)

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
        out = self._run(window)
        v_dev = out[0:2]
        cov = _cholesky_output_to_cov(out[2:5])
        return VelocityEstimate(t_ns=t_ns, v_dev=v_dev, cov=cov)

    @property
    def last_inference_ms(self) -> float:
        """Wall time of the most recent predict() call. Displayed on the strip."""
        return self._last_inference_ms

    def benchmark(self, iterations: int = 200) -> float:
        """Median per-window inference time in ms. Must come in under the budget."""
        window = np.zeros((IN_CHANNELS, self._window_samples), dtype=np.float64)
        timings_ms: list[float] = []
        for _ in range(iterations):
            self._run(window)
            timings_ms.append(self._last_inference_ms)
        return float(np.median(timings_ms))
