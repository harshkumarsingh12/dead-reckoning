package `in`.sih26168.imustreamer

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import `in`.sih26168.imustreamer.net.StreamClient
import `in`.sih26168.imustreamer.wire.EventMessage
import `in`.sih26168.imustreamer.wire.GpsMessage
import `in`.sih26168.imustreamer.wire.ImuMessage
import `in`.sih26168.imustreamer.wire.SessionMetaMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import java.util.UUID
import kotlin.math.roundToLong

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
 *     domain. Forwarded untouched as [ImuMessage.t_ns].
 *   - [Location.getElapsedRealtimeNanos] puts a GPS fix in that SAME domain, which is
 *     what makes a fix that arrives 400 ms late still fusable at the instant it was
 *     actually taken. Used in preference to [Location.getTime].
 *
 * ## Combining three sensors into one sample
 *
 * Accelerometer, gyroscope and magnetometer deliver as independent events, not
 * synchronised triples. The accelerometer tick drives the combined [ImuMessage]; the
 * most recent gyroscope and magnetometer readings are latched onto it. A magnetometer
 * reading older than [STALE_MAG_NS] is dropped from the message (sent as `null`)
 * rather than latched forever -- this is what lets [dr_core.types.ImuSample.m_body]
 * be `None` mean "no fresh reading", not "reused a minute-old one".
 */
class SensorStreamService : Service(), SensorEventListener, LocationListener {

    companion object {
        const val ACTION_START = "in.sih26168.imustreamer.action.START"
        const val ACTION_STOP = "in.sih26168.imustreamer.action.STOP"
        const val ACTION_MARK_EVENT = "in.sih26168.imustreamer.action.MARK_EVENT"
        const val EXTRA_SERVER_URL = "server_url"
        const val EXTRA_CARRY_POSITION = "carry_position"
        const val EXTRA_EVENT_NAME = "event_name"

        private const val NOTIFICATION_CHANNEL_ID = "imu_streaming"
        private const val NOTIFICATION_ID = 1
        private const val STALE_MAG_NS = 500_000_000L // half a second

        /**
         * Single-service simplification: there is only ever one instance of this
         * service, so a companion-held StateFlow is observed by the Activity without
         * the ceremony of a bound-service Binder. Would need revisiting if the app
         * ever ran more than one streaming session at a time.
         */
        val connected = MutableStateFlow(false)
        val achievedRateHz = MutableStateFlow(0.0)
        val samplesSent = MutableStateFlow(0L)
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var streamClient: StreamClient
    private lateinit var sensorManager: SensorManager
    private var locationManager: LocationManager? = null

    private var gyro = doubleArrayOf(0.0, 0.0, 0.0)
    private var mag: DoubleArray? = null
    private var magAtNs: Long = 0L

    private var tickCount = 0
    private var tickWindowStartNs = 0L
    private var bootToUtcOffsetNs = 0L

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopStreaming()
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_MARK_EVENT -> {
                // A plain startService (never startForegroundService) call from an
                // Activity that is itself in the foreground -- no new 5-second
                // startForeground obligation is created by this branch.
                intent.getStringExtra(EXTRA_EVENT_NAME)?.let(::markEvent)
                return START_STICKY
            }
            else -> {
                val serverUrl = intent?.getStringExtra(EXTRA_SERVER_URL) ?: return START_NOT_STICKY
                val carryPosition = intent.getStringExtra(EXTRA_CARRY_POSITION) ?: "unknown"
                startForeground(NOTIFICATION_ID, buildNotification("Starting..."))
                startStreaming(serverUrl, carryPosition)
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopStreaming()
        scope.cancel()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ lifecycle

    private fun startStreaming(serverUrl: String, carryPosition: String) {
        bootToUtcOffsetNs = estimateBootToUtcOffsetNs()

        streamClient = StreamClient(serverUrl, scope)
        streamClient.start(
            SessionMetaMessage(
                session_id = UUID.randomUUID().toString(),
                device_model = android.os.Build.MODEL,
                carry_position = carryPosition,
                imu_rate_hz = SensorManager.SENSOR_DELAY_FASTEST.toDouble(),
                boot_to_utc_offset_ns = bootToUtcOffsetNs,
            ),
        )
        scope.launch { streamClient.connected.collect { connected.value = it } }

        listOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE, Sensor.TYPE_MAGNETIC_FIELD)
            .mapNotNull(sensorManager::getDefaultSensor)
            .forEach { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }

        locationManager = getSystemService(LocationManager::class.java)?.also { lm ->
            try {
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, this)
            } catch (_: SecurityException) {
                // No location permission. IMU-only streaming still works -- the whole
                // point of this project is tracking through exactly this condition.
            }
        }

        tickWindowStartNs = SystemClock.elapsedRealtimeNanos()
        updateNotification("Streaming to $serverUrl")
    }

    private fun stopStreaming() {
        sensorManager.unregisterListener(this)
        locationManager?.removeUpdates(this)
        if (::streamClient.isInitialized) streamClient.close()
        connected.value = false
        achievedRateHz.value = 0.0
    }

    // --------------------------------------------------------------- SensorEventListener

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> gyro = doubleArrayOf(
                event.values[0].toDouble(),
                event.values[1].toDouble(),
                event.values[2].toDouble(),
            )
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // uT -> T. Android reports microtesla; dr_core.types requires tesla
                // (docs/CONVENTIONS.md section 2) -- converted once, at this boundary.
                mag = doubleArrayOf(
                    event.values[0] * 1e-6,
                    event.values[1] * 1e-6,
                    event.values[2] * 1e-6,
                )
                magAtNs = event.timestamp
            }
            Sensor.TYPE_ACCELEROMETER -> onAccelerometerTick(event)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun onAccelerometerTick(event: SensorEvent) {
        val tNs = event.timestamp
        val freshMag = mag?.takeIf { tNs - magAtNs <= STALE_MAG_NS }

        streamClient.send(
            ImuMessage(
                t_ns = tNs,
                a = listOf(
                    event.values[0].toDouble(),
                    event.values[1].toDouble(),
                    event.values[2].toDouble(),
                ),
                w = gyro.toList(),
                m = freshMag?.toList(),
            ),
        )

        tickCount++
        val elapsed = tNs - tickWindowStartNs
        if (elapsed >= 1_000_000_000L) {
            achievedRateHz.value = tickCount * 1_000_000_000.0 / elapsed
            samplesSent.value += tickCount
            tickCount = 0
            tickWindowStartNs = tNs
            updateNotification("Streaming — ${achievedRateHz.value.roundToLong()} Hz")
        }
    }

    // ------------------------------------------------------------------- LocationListener

    override fun onLocationChanged(location: Location) {
        streamClient.send(
            GpsMessage(
                // getElapsedRealtimeNanos, NOT getTime -- see the class doc.
                t_ns = location.elapsedRealtimeNanos,
                lat_deg = location.latitude,
                lon_deg = location.longitude,
                accuracy_m = location.accuracy.toDouble(),
                speed_mps = location.speed.takeIf { location.hasSpeed() }?.toDouble(),
                course_rad = location.bearing.takeIf { location.hasBearing() }
                    ?.let { Math.toRadians((90.0 - it + 360.0) % 360.0) },
                altitude_m = location.altitude.takeIf { location.hasAltitude() },
            ),
        )
    }

    // Deprecated since API 29 but still declared abstract pre-Q; minSdk 26 needs it.
    @Suppress("DEPRECATION")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit

    // ------------------------------------------------------------------------ markers

    /**
     * Records a named event at the current instant -- the sharp-motion clock-alignment
     * tap, a calibration phase boundary, or a manual waypoint marker.
     *
     * Costs nothing to record and is the difference between a diagnosable timing bug
     * and an undiagnosable one (docs/BUILD_PLAN.md section 5).
     */
    fun markEvent(name: String) {
        if (!::streamClient.isInitialized) return
        streamClient.send(EventMessage(t_ns = SystemClock.elapsedRealtimeNanos(), name = name))
    }

    /**
     * Estimates the boot-monotonic to UTC offset once, at session start, by sampling
     * both clocks a handful of times and taking the median -- a single sample would
     * carry whatever scheduling jitter happened to land on that one call.
     */
    private fun estimateBootToUtcOffsetNs(): Long {
        val samples = LongArray(7) {
            System.currentTimeMillis() * 1_000_000L - SystemClock.elapsedRealtimeNanos()
        }
        samples.sort()
        return samples[samples.size / 2]
    }

    // --------------------------------------------------------------------- notification

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            getString(R.string.app_name),
            NotificationManager.IMPORTANCE_LOW,
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        val stopIntent = Intent(this, SensorStreamService::class.java).setAction(ACTION_STOP)
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent, PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setOngoing(true)
            .addAction(0, getString(R.string.stop), stopPendingIntent)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, buildNotification(text))
    }
}
