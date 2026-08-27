package `in`.sih26168.imustreamer

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.os.Build
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import `in`.sih26168.imustreamer.databinding.ActivityMainBinding
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val PREFS_NAME = "imu_streamer"
private const val PREF_SERVER_URL = "server_url"

/**
 * Operator screen: point at the gateway, calibrate, start, mark events.
 *
 * OWNER: Harsh  |  MILESTONE: M4
 *
 * Deliberately minimal. The demo's visuals live in the web UI on the laptop; this
 * screen only has to be usable at arm's length while walking, which is why every
 * control is oversized and the status line is the only thing to read.
 *
 * The calibration ritual, in order, run once streaming has started:
 *   1. Phone still ~5 s            -> gyro bias
 *   2. Figure-8 sweep ~10 s        -> magnetometer hard iron
 *   3. One firm tap                -> the sharp-motion clock-alignment marker
 *
 * This screen only marks the phase boundaries as timestamped events; the actual bias
 * and hard-iron numbers are computed offline from the recording by
 * `dr_core.preprocess.calibrate` (owner: Sristee). Recomputing them on-device here
 * would be a second implementation of that math to keep in sync with the first --
 * exactly the training/live divergence the shared preprocessing module exists to rule
 * out.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: SharedPreferences
    private var isStreaming = false

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            if (granted[Manifest.permission.ACCESS_FINE_LOCATION] == true) {
                beginStreaming()
            } else {
                binding.status.text = getString(R.string.permission_denied)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        binding.serverUrl.setText(
            prefs.getString(PREF_SERVER_URL, "") ?: "",
        )

        binding.startStopButton.setOnClickListener { onStartStopClicked() }
        binding.calibrateButton.setOnClickListener { runCalibrationRitual() }
        binding.markEventButton.setOnClickListener {
            sendMarkEvent("manual_marker")
        }

        binding.status.text = getString(R.string.app_name)
        observeServiceState()
    }

    // ------------------------------------------------------------------- start/stop

    private fun onStartStopClicked() {
        if (isStreaming) {
            stopStreaming()
            return
        }

        val url = binding.serverUrl.text?.toString()?.trim().orEmpty()
        if (url.isEmpty()) {
            binding.status.text = getString(R.string.enter_server_url)
            return
        }
        prefs.edit().putString(PREF_SERVER_URL, url).apply()

        val permissionsNeeded = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        val missing = permissionsNeeded.filter {
            ContextCompat.checkSelfPermission(this, it) != android.content.pm.PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            beginStreaming()
        } else {
            requestPermissions.launch(missing.toTypedArray())
        }
    }

    private fun beginStreaming() {
        val url = binding.serverUrl.text?.toString()?.trim().orEmpty()
        val intent = Intent(this, SensorStreamService::class.java).apply {
            action = SensorStreamService.ACTION_START
            putExtra(SensorStreamService.EXTRA_SERVER_URL, url)
            putExtra(SensorStreamService.EXTRA_CARRY_POSITION, "hand")
        }
        ContextCompat.startForegroundService(this, intent)
        isStreaming = true
        binding.startStopButton.setText(R.string.stop)
    }

    private fun stopStreaming() {
        val intent = Intent(this, SensorStreamService::class.java).apply {
            action = SensorStreamService.ACTION_STOP
        }
        startService(intent)
        isStreaming = false
        binding.startStopButton.setText(R.string.start)
    }

    // ------------------------------------------------------------------- calibration

    private fun runCalibrationRitual() {
        if (!isStreaming) {
            binding.status.text = getString(R.string.start_before_calibrating)
            return
        }
        binding.calibrateButton.isEnabled = false
        lifecycleScope.launch {
            sendMarkEvent("calib_still_start")
            binding.status.text = getString(R.string.calib_hold_still)
            delay(5_000)
            sendMarkEvent("calib_still_end")

            sendMarkEvent("calib_figure8_start")
            binding.status.text = getString(R.string.calib_figure_eight)
            delay(10_000)
            sendMarkEvent("calib_figure8_end")

            binding.status.text = getString(R.string.calib_tap_now)
            delay(1_000)
            sendMarkEvent("tap")

            binding.status.text = getString(R.string.calib_done)
            binding.calibrateButton.isEnabled = true
        }
    }

    private fun sendMarkEvent(name: String) {
        val intent = Intent(this, SensorStreamService::class.java).apply {
            action = SensorStreamService.ACTION_MARK_EVENT
            putExtra(SensorStreamService.EXTRA_EVENT_NAME, name)
        }
        startService(intent)
    }

    // --------------------------------------------------------------------- live state

    private fun observeServiceState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    SensorStreamService.connected.collect { connected ->
                        if (isStreaming) {
                            binding.status.text = if (connected) {
                                getString(R.string.status_connected)
                            } else {
                                getString(R.string.status_reconnecting)
                            }
                        }
                    }
                }
                launch {
                    SensorStreamService.achievedRateHz.collect { hz ->
                        if (isStreaming && hz > 0) {
                            binding.status.text = getString(R.string.status_rate, hz)
                        }
                    }
                }
            }
        }
    }
}
