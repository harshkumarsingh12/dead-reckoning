#!/usr/bin/env python
"""Export the trained model to ONNX, int8-quantized.

OWNER: Sumedha  |  MILESTONE: M2

Run this EARLY, on the first checkpoint that trains at all, not at the end. It
de-risks on-device inference and produces a real per-window latency number to quote
instead of a hope. Target is under 10 ms per window on the laptop.

    python scripts/export_onnx.py models/tcn.pt --out models/tcn.onnx --benchmark

Exit codes: 0 exported (and, with --benchmark, under the latency budget); 1 exported but
over the latency budget -- reported, not hidden (AGENTS.md); 2 bad usage.
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK = 0
EXIT_BUDGET_MISSED = 1
EXIT_USAGE = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="models/tcn.onnx")
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="report median inference ms")
    args = parser.parse_args()

    import torch

    from dr_core.models.tcn import build_model, export_onnx

    checkpoint_path = args.checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"export_onnx: no such checkpoint: {checkpoint_path}", file=sys.stderr)
        return EXIT_USAGE

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    export_onnx(
        model,
        args.out,
        window_samples=args.window_samples,
        quantize_int8=not args.no_quantize,
    )
    print(f"export_onnx: wrote {args.out}")

    if not args.benchmark:
        return EXIT_OK

    from dr_core.models.runtime import INFERENCE_BUDGET_MS, VelocityModelRuntime

    runtime = VelocityModelRuntime(args.out)
    median_ms = runtime.benchmark()
    print(f"export_onnx: median inference {median_ms:.3f} ms (budget {INFERENCE_BUDGET_MS} ms)")
    if median_ms >= INFERENCE_BUDGET_MS:
        print(
            f"export_onnx: OVER BUDGET by {median_ms - INFERENCE_BUDGET_MS:.3f} ms",
            file=sys.stderr,
        )
        return EXIT_BUDGET_MISSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
