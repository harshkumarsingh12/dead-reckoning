"""A lagged-timeline reorder buffer, so late measurements fuse at their capture time.

OWNER: Sristee  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 5

GPS arrives hundreds of milliseconds late; the model has its own group delay; the
network adds jitter. Rather than pretending otherwise, the filter runs on a timeline
lagged ~300 ms behind wall clock. Measurements queue by CAPTURE timestamp and are
released in order once the lagged clock has passed them.

The cost is a 300 ms lag in the rendered dot. The benefit is that the dot does not cut
corners and no measurement is ever fused at the wrong instant. That trade is honest and
survives questioning; extrapolating to hide the lag does not.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_LAG_NS = 300_000_000  # 300 ms


class ReorderBuffer(Generic[T]):
    """Min-heap of measurements keyed by capture timestamp.

    Example:
        buf = ReorderBuffer[ImuSample](lag_ns=DEFAULT_LAG_NS)
        buf.push(sample.t_ns, sample)
        for t_ns, m in buf.drain(now_ns):
            filter.update(t_ns, m)
    """

    def __init__(self, lag_ns: int = DEFAULT_LAG_NS, max_size: int = 100_000) -> None:
        """Create an empty buffer.

        Args:
            lag_ns: how far behind wall clock the released timeline runs.
            max_size: hard cap; exceeding it means the consumer has stalled and is a
                bug worth surfacing loudly rather than silently growing memory.
        """
        self._lag_ns = lag_ns
        self._max_size = max_size
        raise NotImplementedError("M0 -- owner: Sristee")

    def push(self, t_ns: int, item: T) -> None:
        """Queue a measurement at its capture time.

        Raises:
            ValueError: if t_ns is older than the last released timestamp -- that
                measurement can no longer be fused in order and dropping it silently
                would hide a real timing bug.
        """
        raise NotImplementedError("M0 -- owner: Sristee")

    def drain(self, now_ns: int) -> list[tuple[int, T]]:
        """Release everything captured at or before ``now_ns - lag_ns``, in order."""
        raise NotImplementedError("M0 -- owner: Sristee")

    def __len__(self) -> int:
        raise NotImplementedError("M0 -- owner: Sristee")
