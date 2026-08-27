package `in`.sih26168.imustreamer.wire

import kotlinx.serialization.Serializable

/**
 * The wire format spoken to the gateway's `/ingest` socket.
 *
 * Mirrors the on-disk session record schema in `dr_core.io.session` (see
 * docs/CONVENTIONS.md section 6): one JSON object per message, a `type` tag, a
 * capture timestamp, and nothing else magic.
 *
 * `t_ns` is a plain JSON integer here, not the decimal-string encoding used for the
 * gateway-to-browser `/live` stream. Python's `json` module has arbitrary-precision
 * integers, so nanosecond timestamps round-trip exactly; the string workaround in
 * CONVENTIONS.md section 6 is specifically for `JSON.parse` in the browser, which this
 * link never touches.
 */
@Serializable
data class ImuMessage(
    val type: String = "imu",
    /** Boot-monotonic nanoseconds, straight from [android.hardware.SensorEvent.timestamp]. */
    val t_ns: Long,
    /** [x, y, z] specific force, m/s^2, raw device frame. */
    val a: List<Double>,
    /** [x, y, z] angular rate, rad/s, raw device frame. */
    val w: List<Double>,
    /** [x, y, z] magnetic field, tesla, raw device frame. Null if no reading arrived this tick. */
    val m: List<Double>? = null,
)

@Serializable
data class GpsMessage(
    val type: String = "gps",
    /** Boot-monotonic nanoseconds, from [android.location.Location.getElapsedRealtimeNanos] —
     *  NOT [android.location.Location.getTime], which is wall-clock UTC and arrives late. */
    val t_ns: Long,
    val lat_deg: Double,
    val lon_deg: Double,
    val accuracy_m: Double,
    val speed_mps: Double? = null,
    val course_rad: Double? = null,
    val altitude_m: Double? = null,
)

@Serializable
data class EventMessage(
    val type: String = "event",
    val t_ns: Long,
    /** e.g. "tap" (clock alignment), "calib_still_start", "calib_figure8_end", "gps_off". */
    val name: String,
)

@Serializable
data class SessionMetaMessage(
    val type: String = "meta",
    val session_id: String,
    val device_model: String,
    val carry_position: String,
    val imu_rate_hz: Double,
    /** utc_ns = boot_ns + offset. Estimated once at session start; see [in.sih26168.imustreamer.SensorStreamService.bootToUtcOffsetNs]. */
    val boot_to_utc_offset_ns: Long,
    val origin_lat_deg: Double? = null,
    val origin_lon_deg: Double? = null,
)
