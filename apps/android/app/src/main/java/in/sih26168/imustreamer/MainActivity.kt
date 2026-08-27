package `in`.sih26168.imustreamer

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import `in`.sih26168.imustreamer.databinding.ActivityMainBinding

/**
 * Operator screen: point at the gateway, calibrate, start, mark events.
 *
 * OWNER: Harsh  |  MILESTONE: M4
 *
 * Deliberately minimal. The demo's visuals live in the web UI on the laptop; this
 * screen only has to be usable at arm's length while walking, which is why every
 * control is oversized and the status line is the only thing to read.
 *
 * The calibration ritual, in order, before every recording:
 *   1. Phone still ~5 s            -> gyro bias
 *   2. Figure-8 sweep ~10 s        -> magnetometer hard iron
 *   3. One firm tap                -> the sharp-motion clock-alignment marker
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // TODO(M4, Harsh): request location + notification permissions, persist the
        // server URL, bind the buttons to SensorStreamService, and surface the
        // connection state and live sample rate on binding.status. Showing the actual
        // achieved rate matters -- phones silently throttle sensors under thermal load
        // and finding that out during the demo is too late.
        binding.status.text = getString(R.string.app_name)
    }
}
