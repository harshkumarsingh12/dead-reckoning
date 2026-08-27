"""Clock-domain and reorder-buffer behaviour.

Spec: docs/BUILD_PLAN.md section 5  |  OWNER: Sristee  |  MILESTONE: M0

Timing bugs are silent. They do not crash, they do not raise, they just make every
downstream number slightly and unaccountably worse. These tests are the tripwire.
"""

from __future__ import annotations

import pytest

from dr_core.timebase.clock import NS_PER_S, ClockMapper, estimate_offset
from dr_core.timebase.reorder import DEFAULT_LAG_NS, ReorderBuffer

pytestmark = pytest.mark.timing


# --------------------------------------------------------------- passes today


def test_clock_mapper_round_trips() -> None:
    m = ClockMapper(offset_ns=1_700_000_000 * NS_PER_S)
    boot = 12_345_678_901
    assert m.to_boot_ns(m.to_utc_ns(boot)) == boot


def test_clock_mapper_offset_is_additive() -> None:
    m = ClockMapper(offset_ns=-500)
    assert m.to_utc_ns(1_000) == 500


def test_default_lag_is_300_ms() -> None:
    """The filter runs on a timeline 300 ms behind wall clock so late GPS still fuses
    at its capture time. Changing this is a spec change, not a tuning knob."""
    assert DEFAULT_LAG_NS == 300_000_000


# ------------------------------------------------------------ the M0 ledger


def test_offset_estimation_recovers_a_known_shift() -> None:
    true_offset = 1_700_000_000 * NS_PER_S
    boot = [i * 100_000_000 for i in range(20)]
    utc = [b + true_offset for b in boot]
    mapper = estimate_offset(boot, utc)
    assert mapper.offset_ns == pytest.approx(true_offset, abs=1_000_000)


@pytest.mark.xfail(reason="M0 -- ReorderBuffer unimplemented (owner: Sristee)", strict=True)
def test_reorder_buffer_releases_in_capture_order() -> None:
    """Measurements pushed out of order come out in order.

    This is the whole point: a GPS fix that arrives 400 ms late must be fused where it
    belongs on the timeline, not where it happened to show up.
    """
    buf: ReorderBuffer[str] = ReorderBuffer(lag_ns=DEFAULT_LAG_NS)
    buf.push(300_000_000, "gps_late")
    buf.push(100_000_000, "imu_a")
    buf.push(200_000_000, "imu_b")
    released = buf.drain(now_ns=1_000_000_000)
    assert [item for _t, item in released] == ["imu_a", "imu_b", "gps_late"]


@pytest.mark.xfail(reason="M0 -- ReorderBuffer unimplemented (owner: Sristee)", strict=True)
def test_reorder_buffer_holds_back_the_lag_window() -> None:
    """Nothing inside the lag window is released -- something later may still arrive."""
    buf: ReorderBuffer[str] = ReorderBuffer(lag_ns=DEFAULT_LAG_NS)
    buf.push(900_000_000, "recent")
    assert buf.drain(now_ns=1_000_000_000) == []


@pytest.mark.xfail(reason="M0 -- ReorderBuffer unimplemented (owner: Sristee)", strict=True)
def test_reorder_buffer_rejects_a_measurement_that_is_already_too_late() -> None:
    """Loudly, not silently. A dropped late measurement is a real timing bug and
    swallowing it here is how it stays undiagnosed until demo day."""
    buf: ReorderBuffer[str] = ReorderBuffer(lag_ns=DEFAULT_LAG_NS)
    buf.push(500_000_000, "ok")
    buf.drain(now_ns=1_000_000_000)
    with pytest.raises(ValueError, match="older than"):
        buf.push(400_000_000, "far too late")
