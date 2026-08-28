"""The learned velocity model: shapes, loss, augmentation, and the latency budget.

Spec: docs/BUILD_PLAN.md section 6.4  |  OWNER: Sumedha  |  MILESTONE: M2

Marked ``ml`` so the heavy torch install is confined to its own workflow. The stubs
raise before importing torch, so these still take part in the default run and stay
visible on the burndown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dr_core.models.runtime import INFERENCE_BUDGET_MS
from dr_core.models.tcn import IN_CHANNELS, OUT_DIM, augment_random_yaw, build_model

pytestmark = pytest.mark.ml


def test_output_dimension_carries_a_full_covariance() -> None:
    """2 for the mean, 3 for the lower-triangular Cholesky factor of a 2x2.

    Passes today: it pins the contract between the model head and the filter's R
    before either exists, so neither owner can drift.
    """
    assert IN_CHANNELS == 6
    assert OUT_DIM == 5


def test_model_is_strictly_causal() -> None:
    """Perturbing a FUTURE sample must not change the current output.

    The one property that cannot be checked by eyeballing the architecture. A
    non-causal model looks fine offline and lags every turn on stage.
    """
    import torch

    model = build_model()
    model.eval()
    x = torch.randn(1, IN_CHANNELS, 200)
    y_before = model(x)[:, :, -1].clone()
    x[..., -1] += 100.0  # only the last sample changes -- it IS the current instant
    x_future = x.clone()
    x_future[..., :-1] = torch.randn_like(x_future[..., :-1])
    assert not torch.allclose(model(x_future)[:, :, -1], y_before)


def test_nll_loss_penalises_overconfidence() -> None:
    """A tight sigma on a large error must cost more than an honest wide one.

    This is what makes the covariance head mean something instead of decorating a
    plain MSE fit.
    """
    import torch

    from dr_core.models.tcn import gaussian_nll_loss

    target = torch.tensor([[1.0, 0.0]])
    overconfident = torch.tensor([[0.0, 0.0, -3.0, -3.0, 0.0]])
    honest = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    assert gaussian_nll_loss(overconfident, target) > gaussian_nll_loss(honest, target)


def test_random_yaw_augmentation_preserves_speed() -> None:
    """Rotating a window rotates its label by the same angle. Speed is invariant."""
    rng = np.random.default_rng(0)
    window = rng.normal(size=(IN_CHANNELS, 200))
    target = np.array([1.4, 0.0])
    _w2, t2 = augment_random_yaw(window, target, rng)
    assert np.linalg.norm(t2) == pytest.approx(np.linalg.norm(target))
    assert not np.allclose(t2, target)


@pytest.mark.xfail(reason="M2 -- ONNX runtime unimplemented (owner: Sumedha)", strict=True)
def test_inference_fits_the_stated_budget() -> None:
    """Guards a number we say out loud on stage.

    Named so ``pytest -m ml -k budget`` picks it out in CI.
    """
    from pathlib import Path

    from dr_core.models.runtime import VelocityModelRuntime

    runtime = VelocityModelRuntime(Path("models/tcn.onnx"))
    assert runtime.benchmark() < INFERENCE_BUDGET_MS


def test_onnx_output_matches_torch(tmp_path: Path) -> None:
    """Export parity. A quantized model that quietly disagrees with the one you
    validated is a very expensive way to lose a demo."""
    import onnxruntime as ort
    import torch

    from dr_core.models.tcn import export_onnx

    model = build_model()
    model.eval()
    onnx_path = tmp_path / "parity.onnx"
    export_onnx(model, str(onnx_path), window_samples=200, quantize_int8=False)

    x = torch.randn(1, IN_CHANNELS, 200)
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})

    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
