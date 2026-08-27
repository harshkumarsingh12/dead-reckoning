"""Boot-monotonic to UTC mapping, estimated once per session.

OWNER: Sristee  |  MILESTONE: M0  |  Spec: docs/BUILD_PLAN.md section 5

On Android, IMU ``SensorEvent.timestamp`` is nanoseconds in the boot-monotonic domain
(``elapsedRealtimeNanos``). GPS fixes carry a UTC time but also expose
``Location.getElapsedRealtimeNanos()`` in that same boot domain. Everything downstream
lives in boot-monotonic; UTC exists only so recordings can be aligned with external
labels.
"""

from __future__ import annotations

from dataclasses import dataclass

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ClockMapper:
    """Maps the device boot-monotonic domain onto UTC.

    Attributes:
        offset_ns: utc_ns = boot_ns + offset_ns.
        residual_std_ns: spread of the samples the offset was fitted from. A large
            value means the estimate is untrustworthy and should be re-taken.
    """

    offset_ns: int
    residual_std_ns: float = 0.0

    def to_utc_ns(self, boot_ns: int) -> int:
        """Convert a boot-monotonic timestamp to UTC nanoseconds."""
        return boot_ns + self.offset_ns

    def to_boot_ns(self, utc_ns: int) -> int:
        """Convert a UTC timestamp to the boot-monotonic domain."""
        return utc_ns - self.offset_ns


def estimate_offset(
    boot_ns: list[int],
    utc_ns: list[int],
) -> ClockMapper:
    """Fit the boot-to-UTC offset from paired observations taken at session start.

    Args:
        boot_ns: boot-monotonic capture timestamps.
        utc_ns: the corresponding UTC timestamps for the same events.

    Returns:
        A ClockMapper carrying the fitted offset and its residual spread.

    Raises:
        ValueError: if the two sequences differ in length or are empty.
    """
    raise NotImplementedError("M0 -- owner: Sristee")


def verify_sharp_motion_alignment(
    streams: dict[str, list[tuple[int, float]]],
    tolerance_ns: int = 20_000_000,
) -> dict[str, int]:
    """The sharp-motion-event test (build plan section 5).

    A distinct physical event -- a firm tap or a stomp -- must appear at the same
    instant across every aligned stream. This is the check that catches a clock-domain
    mismatch before it silently corrupts a whole day of training labels.

    Args:
        streams: name -> list of (t_ns, magnitude) samples.
        tolerance_ns: how far apart the detected peaks may be and still pass.
            Default 20 ms, roughly four samples at 200 Hz.

    Returns:
        name -> detected peak t_ns, for every stream.

    Raises:
        AssertionError: if any two detected peaks are further apart than the tolerance.
    """
    raise NotImplementedError("M0 -- owner: Sristee")
