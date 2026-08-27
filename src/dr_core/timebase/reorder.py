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

import heapq
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
        # (t_ns, seq, item); seq breaks ties so two same-timestamp items order by
        # arrival and T is never compared.
        self._heap: list[tuple[int, int, T]] = []
        self._seq = 0
        # Highest timestamp released so far; None until the first drain.
        self._last_released_ns: int | None = None

    def push(self, t_ns: int, item: T) -> None:
        """Queue a measurement at its capture time.

        Raises:
            ValueError: if t_ns is older than the last released timestamp -- that
                measurement can no longer be fused in order and dropping it silently
                would hide a real timing bug.
        """
        if self._last_released_ns is not None and t_ns < self._last_released_ns:
            raise ValueError(
                f"measurement at {t_ns} ns is older than the last released "
                f"timestamp {self._last_released_ns} ns and can no longer be fused in order"
            )
        if len(self._heap) >= self._max_size:
            raise ValueError(f"reorder buffer is full ({self._max_size}); the consumer has stalled")
        heapq.heappush(self._heap, (t_ns, self._seq, item))
        self._seq += 1

    def drain(self, now_ns: int) -> list[tuple[int, T]]:
        """Release everything captured at or before ``now_ns - lag_ns``, in order."""
        cutoff = now_ns - self._lag_ns
        released: list[tuple[int, T]] = []
        while self._heap and self._heap[0][0] <= cutoff:
            t_ns, _seq, item = heapq.heappop(self._heap)
            released.append((t_ns, item))
            self._last_released_ns = t_ns
        return released

    def __len__(self) -> int:
        return len(self._heap)
