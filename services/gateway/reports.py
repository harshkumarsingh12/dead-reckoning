"""Serves the files `dr_core.eval.report.generate_report` writes to disk.

OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 6.8, 8

`generate_report` (owner: Sikruti) writes `trajectory.png`, `error_time.png`,
`error_cdf.png`, `nis.png` and `report.json` into `<output_dir>/<run_id>/`. This module
is the read side of that layout: given a `reports_dir`, resolve `(run_id, filename)`
to a real path and refuse to serve anything outside it.

## What this deliberately does NOT do yet

There is no route here that *triggers* `generate_report` from a live gateway session.
Doing that honestly needs two things that do not exist yet:

  1. A real fused trajectory in the live path -- `/live` currently broadcasts the flat-
     earth GPS placeholder in `services/gateway/hub.py`, not ESKF output.
  2. A surveyed ground-truth trajectory for the demo loop -- `generate_report` requires
     ``truth`` as a real argument, and no loop has been surveyed yet (issue #22).

Wiring an "end run" button that calls `generate_report` with fabricated truth data
would produce numbers that look real and are not -- exactly what AGENTS.md's "do not
invent numbers" rule exists to prevent. So for now: this only serves reports that a
human already generated for real, offline, with `scripts/run_eval.py`. The live
auto-trigger is a follow-up once the two items above land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Only these are ever written by generate_report; anything else is refused rather than
# silently 404'd, so a typo'd filename is obviously a bug and not a missing report.
ALLOWED_FILENAMES = frozenset(
    {"trajectory.png", "error_time.png", "error_cdf.png", "nis.png", "report.json"}
)


def resolve_report_file(reports_dir: Path, run_id: str, filename: str) -> Path | None:
    """Resolve a `(run_id, filename)` pair to a real file inside `reports_dir`.

    Returns ``None`` -- never raises -- if the filename is not one `generate_report`
    ever writes, if the resolved path would escape `reports_dir` (a `run_id` like
    `../../etc` must not reach the filesystem outside it), or if the file simply does
    not exist yet.
    """
    if filename not in ALLOWED_FILENAMES:
        return None

    candidate = (reports_dir / run_id / filename).resolve()
    try:
        candidate.relative_to(reports_dir.resolve())
    except ValueError:
        return None  # escaped reports_dir -- e.g. run_id containing ".."

    if not candidate.is_file():
        return None
    return candidate


def content_type_for(filename: str) -> str:
    return "application/json" if filename.endswith(".json") else "image/png"
