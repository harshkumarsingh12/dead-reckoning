"""The session record format. One file per walk, readable by every part of the system.

OWNER: Sristee (backup: Harsh)  |  MILESTONE: M0
Spec: docs/BUILD_PLAN.md section 6.1

The format is deliberately boring: gzipped JSON Lines, one record per line, a header
line carrying SessionMeta. It is greppable, streamable, diffable, and does not need a
special tool at 2 a.m. when a recording looks wrong.

The important property is that a RECORDED session and a LIVE session flow through the
identical downstream pipeline. That is what makes the golden-run replay a genuine
fallback rather than a separate demo path that has never been tested.
"""

from dr_core.io.session import SessionReader, SessionWriter

__all__ = ["SessionReader", "SessionWriter"]
