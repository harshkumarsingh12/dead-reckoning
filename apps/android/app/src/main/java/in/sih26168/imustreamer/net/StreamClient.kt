package `in`.sih26168.imustreamer.net

import `in`.sih26168.imustreamer.wire.EventMessage
import `in`.sih26168.imustreamer.wire.GpsMessage
import `in`.sih26168.imustreamer.wire.ImuMessage
import `in`.sih26168.imustreamer.wire.SessionMetaMessage
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.send
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

private const val RECONNECT_DELAY_MS = 500L

/**
 * The uplink to the gateway's `/ingest` socket, over the phone's own hotspot.
 *
 * OWNER: Harsh  |  Spec: docs/BUILD_PLAN.md sections 5, 6.1
 *
 * Reconnects with a fixed backoff on drop -- the hotspot will hiccup at least once
 * during a real walk, and this has to keep sampling regardless. A message queued
 * during a dropped connection can be lost if it was already pulled off the channel by
 * the failed send; for high-rate IMU samples that is an acceptable loss, but GPS fixes
 * and event markers are rare enough that losing one is worth noticing.
 *
 * Deliberately outside the reorder buffer's concern: this class only moves bytes. It
 * must never touch a timestamp -- every message here already carries its capture time,
 * assigned by the sensor or by [android.location.Location.getElapsedRealtimeNanos].
 */
class StreamClient(private val serverUrl: String, private val scope: CoroutineScope) {

    private val json = Json { encodeDefaults = true }
    private val client = HttpClient(OkHttp) { install(WebSockets) }
    private val pendingFrames = Channel<String>(capacity = 4096)

    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected

    private var loopJob: Job? = null

    fun start(meta: SessionMetaMessage) {
        if (loopJob?.isActive == true) return
        val header = json.encodeToString(meta)
        loopJob = scope.launch {
            while (isActive) {
                try {
                    client.webSocket(serverUrl) {
                        _connected.value = true
                        send(Frame.Text(header))
                        // Manual receive() rather than a `for`/consumeEach loop: both of
                        // those cancel the channel on exit, which would make it
                        // permanently unusable across a reconnect.
                        while (true) {
                            send(Frame.Text(pendingFrames.receive()))
                        }
                    }
                } catch (_: Exception) {
                    // Connection dropped or never opened. Fall through to the delay
                    // below and try again -- there is no operator action to take here,
                    // the demo just has to keep sampling through a hotspot hiccup.
                } finally {
                    _connected.value = false
                }
                delay(RECONNECT_DELAY_MS)
            }
        }
    }

    fun stop() {
        loopJob?.cancel()
        loopJob = null
        _connected.value = false
    }

    fun send(message: ImuMessage) = enqueue(json.encodeToString(message))
    fun send(message: GpsMessage) = enqueue(json.encodeToString(message))
    fun send(message: EventMessage) = enqueue(json.encodeToString(message))

    private fun enqueue(text: String) {
        pendingFrames.trySend(text)
    }

    fun close() {
        stop()
        pendingFrames.close()
        client.close()
    }
}
