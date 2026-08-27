package `in`.sih26168.imustreamer

import android.app.Service
import android.content.Intent
import android.hardware.SensorEvent
import android.location.Location
import android.os.IBinder

/**
 * Samples the IMU and GPS and streams them to the gateway over the hotspot.
 *
 * OWNER: Harsh  |  MILESTONE: M4  |  Spec: docs/BUILD_PLAN.md sections 5 and 6.1
 *
 * A foreground service, not an Activity task, because sampling has to survive the
 * screen turning off mid-walk. A demo that dies because the phone locked itself is a
 * demo that dies.
 *
 * ## The one rule
 *
 * Timestamps are assigned by the SENSOR, at capture, in the boot-monotonic domain.
 * They are never assigned at send time, never at receive time, and never converted to
 * wall clock on the way out.
 *
 *   - [SensorEvent.timestamp] is already nanoseconds in the `elapsedRealtimeNanos`
 *     domain. Forward it untouched.
 *   - [Location.getElapsedRealtimeNanos] puts a GPS fix in that SAME domain, which is
 *     what makes a fix that arrives 400 ms late still fusable at the instant it was
 *     actually taken. Use it in preference to [Location.getTime].
 *
 * Restamping anywhere along this path is invisible, looks completely reasonable in
 * review, and quietly destroys the alignment the entire timing subsystem exists to
 * guarantee. If you are unsure whether a change restamps, it does; ask.
 */
class SensorStreamService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // TODO(M4, Harsh): start the foreground notification, register accelerometer,
        // gyroscope and magnetometer at SENSOR_DELAY_FASTEST, request location
        // updates, and open the WebSocket.
        return START_STICKY
    }

    /**
     * Estimates the boot-monotonic to UTC offset once, at session start.
     *
     * Sent in the session header so a recording can be aligned against UTC-stamped
     * labels later. Re-estimate if a drift is detected; do not recompute per sample,
     * which would inject jitter into the very thing being stabilised.
     */
    private fun bootToUtcOffsetNs(): Long {
        // TODO(M4, Harsh): System.currentTimeMillis() * 1_000_000 - elapsedRealtimeNanos(),
        // sampled a handful of times and taking the median.
        return 0L
    }

    /**
     * Records a sharp-motion marker at the current instant.
     *
     * The presenter taps the phone firmly at session start; the resulting spike must
     * appear at the same instant in every aligned stream. Costs nothing to record and
     * is the difference between a diagnosable timing bug and an undiagnosable one.
     */
    fun markSharpMotionEvent() {
        // TODO(M4, Harsh): emit an {"type":"event","name":"tap"} record.
    }
}
