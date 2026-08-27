"""One clock domain, and a buffer that fuses late measurements where they belong.

OWNER: Sristee (backup: Harsh) -- see CONTRIBUTING.md
MILESTONE: M0

Timing is the most common invisible failure in sensor fusion. It is engineered out
here rather than hoped away. See docs/BUILD_PLAN.md section 5.
"""

from dr_core.timebase.clock import ClockMapper, verify_sharp_motion_alignment
from dr_core.timebase.reorder import ReorderBuffer

__all__ = ["ClockMapper", "ReorderBuffer", "verify_sharp_motion_alignment"]
