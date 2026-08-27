"""The live demo server: phone in, filtered state out, tiles on the side.

OWNER: Harsh (backup: Tanmay)  |  MILESTONE: M4
Spec: docs/BUILD_PLAN.md sections 6.1 and 6.8

Everything here runs over the PHONE'S OWN HOTSPOT, with map tiles served from a local
MBTiles file. There is no venue internet in the loop at any point, by construction
rather than by luck -- see docs/DEMO_RUNBOOK.md.

Depends on ``dr_core``. ``dr_core`` must never depend on this; tests/test_contract.py
enforces the direction.
"""

from services.gateway.app import create_app

__all__ = ["create_app"]
