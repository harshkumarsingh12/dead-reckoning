#!/usr/bin/env python
"""Turn a recording into a trajectory, plots, and numbers. The M0 deliverable.

OWNER: Sikruti  |  MILESTONE: M0

    python scripts/run_eval.py data/loops/corridor_01.jsonl.gz --model models/tcn.onnx

Thin by design -- it delegates to ``dr_core.eval.cli`` so the same code path is
reachable as the installed ``dr-eval`` command and is covered by tests.
"""

from __future__ import annotations

import sys

from dr_core.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
