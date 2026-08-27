# APP.md — Android IMU streamer

**Path:** `apps/android/`  |  **Owner:** Harsh (backup: Akshit)  |  **Milestone:** M4
**Package:** `in.sih26168.imustreamer`  |  **Spec:** docs/BUILD_PLAN.md §5, §6.1

## What this app is, and is not

This is **not** a data-collection app. Training data for the learned velocity model
comes from off-the-shelf loggers (Sensor Logger / Phyphox) — see `data/README.md` and
the note in `docs/BUILD_PLAN.md` §3 ("do not build an app just to collect").

This app has exactly one job: during the **live demo**, stream the phone's
accelerometer, gyroscope, magnetometer and GPS to the gateway over the phone's own
hotspot, in real time, with every sample carrying its true capture timestamp. It is the
uplink half of the system in `docs/ARCHITECTURE.md`'s diagram — everything downstream
(preprocessing, AHRS, the model, the ESKF) depends on what this app gets right about
timing, and nothing else.

## The one rule

**Every timestamp is assigned by the sensor, at capture, in the boot-monotonic domain.
Never at send time, never at receive time, never converted to wall clock.**

- `SensorEvent.timestamp` is already nanoseconds in the `elapsedRealtimeNanos` domain —
  forwarded untouched into the wire message.
- `Location.getElapsedRealtimeNanos()` puts a GPS fix in that *same* domain, which is
  what lets a fix that arrives 400 ms late still be fused at the instant it was actually
  taken. Used in preference to `Location.getTime()` (wall-clock UTC, and later).

Restamping anywhere along this path is invisible, looks completely reasonable in code
review, and quietly destroys the alignment the entire timing subsystem in
`dr_core.timebase` exists to guarantee. See `docs/CONVENTIONS.md` §3.

## Requirements

| | |
|---|---|
| JDK | **21** for CI (Temurin). 22 also builds clean locally — verified. **Not** 25: Gradle 8.13 supports up to Java 23, and Android Studio's bundled JBR has moved to 25. |
| Android SDK | `compileSdk` / `targetSdk` 36, `minSdk` 26 |
| Gradle | 8.13, via the committed wrapper (`./gradlew`) |
| Physical requirement | A real device. Accelerometer/gyroscope/magnetometer/GPS are declared `<uses-feature>`; an emulator without simulated sensors will not produce useful data. |

### First-time setup

```bash
cd apps/android
cp local.properties.example local.properties   # then edit the SDK path
```

`local.properties` needs its colon **escaped** — `.properties` files treat `:` and `\`
specially:

```properties
sdk.dir=C\:/Users/you/AppData/Local/Android/Sdk
```

An unescaped colon builds fine but fails `lintDebug` with `PropertyEscape`.

### Build and verify

```bash
./gradlew :app:assembleDebug   # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:lintDebug
```

Or open `apps/android/` — **not the repo root** — directly in Android Studio. Opening
the root finds no Gradle project (this is a monorepo; Python, TypeScript and Kotlin sit
side by side) and Gradle sync never fires.

## How it works

```
MainActivity                    SensorStreamService (foreground)         StreamClient
──────────────                  ──────────────────────────────           ────────────
permission gate                 SensorManager: accel/gyro/mag            Ktor WS client
persists gateway URL      ──►    @ SENSOR_DELAY_FASTEST           ──►    /ingest uplink
Start / Stop / Calibrate         LocationManager: GPS_PROVIDER            500 ms reconnect
                                 combines into ImuMessage / GpsMessage    backoff, always
                                 on each accelerometer tick               retries
```

### `MainActivity`

The operator screen. Deliberately minimal — the demo's actual visuals live in the web
UI on the laptop; this screen only has to be usable at arm's length while walking.

- **Permission gate.** `ACCESS_FINE_LOCATION` (required before the foreground service
  can even start with `foregroundServiceType="location"` on API 34+) and, on API 33+,
  `POST_NOTIFICATIONS`.
- **Persists the gateway URL** in `SharedPreferences` so it survives a restart.
- **Start / Stop** toggles the foreground service via `Intent` actions
  (`ACTION_START` / `ACTION_STOP`) — there is no bound-service Binder; a companion-held
  `MutableStateFlow` (there is only ever one instance of this service) is observed
  instead, which avoids the ceremony for an app this size.
- **Calibrate** runs the guided ritual, driving it entirely with timestamped markers
  sent as `ACTION_MARK_EVENT` commands:

  | Step | Duration | Event name | What it's for |
  |---|---|---|---|
  | Hold still | 5 s | `calib_still_start` / `calib_still_end` | gyro bias |
  | Figure-8 sweep | 10 s | `calib_figure8_start` / `calib_figure8_end` | magnetometer hard iron |
  | Firm tap | — | `tap` | clock-alignment verification (§ below) |

  **The app only marks phase boundaries — it does not compute the calibration itself.**
  That math lives in `dr_core.preprocess.calibrate` (owner: Sristee) and is computed
  offline from the recording. Recomputing it on-device here would be a second
  implementation of the same math, which is exactly the training/live divergence
  `dr_core.preprocess` exists to rule out.
- **Mark event** sends a generic `manual_marker` — for a known waypoint or corner
  during a walk.

### `SensorStreamService`

A **foreground service**, not tied to the Activity's lifecycle, because sampling has to
survive the screen turning off mid-walk. A demo that dies because the phone locked
itself is a demo that dies.

- Registers accelerometer, gyroscope and magnetometer at `SENSOR_DELAY_FASTEST`.
- The three sensors deliver as **independent events**, not synchronised triples. The
  **accelerometer tick drives the combined sample**: the most recent gyroscope reading
  is latched onto it, and the most recent magnetometer reading too — *unless* it is
  older than 500 ms, in which case it is sent as `null`. This is what lets
  `dr_core.types.ImuSample.m_body` mean "no fresh reading", not "reused a minute-old
  one".
- Converts magnetometer µT → tesla at this boundary (Android reports µT;
  `dr_core.types` requires tesla per `docs/CONVENTIONS.md` §2).
- Converts GPS bearing (clockwise from North) to `psi_rad` (CCW from East) at this
  boundary, per the same convention doc.
- Estimates the boot-to-UTC offset **once**, at session start, as the median of seven
  paired samples — a single sample would carry whatever scheduling jitter happened to
  land on that one call.
- Shows the **achieved** sample rate in the persistent notification, not the requested
  one. Phones silently throttle sensors under thermal load; finding that out during the
  demo is too late.

### `StreamClient`

Owns the WebSocket to `/ingest`. Reconnects with a fixed 500 ms backoff — a real walk
will hit at least one hotspot hiccup, and this has to keep sampling through it
regardless. A message queued during a drop can be lost if it was already pulled off the
internal channel by the failed send; for 100–200 Hz IMU samples that is an acceptable
loss, but it is why GPS fixes and event markers matter more per-message than any single
IMU tick.

Deliberately dumb: it only moves bytes and never touches a timestamp. Every message
handed to it already carries its capture time.

## Wire format

Sent to the gateway's `/ingest` socket as JSON text frames, one object per message —
see `apps/android/.../wire/Messages.kt` and `docs/CONVENTIONS.md` §6.

```jsonc
// First message of every session:
{"type":"meta","session_id":"…","device_model":"Pixel 7","carry_position":"hand",
 "imu_rate_hz":200,"boot_to_utc_offset_ns":1734000000000000000}

// Then, per accelerometer tick:
{"type":"imu","t_ns":12345678901234,"a":[0.1,0.0,9.8],"w":[0.0,0.0,0.01],"m":null}

// Whenever GPS_PROVIDER reports:
{"type":"gps","t_ns":12345678901234,"lat_deg":20.3535,"lon_deg":85.8164,"accuracy_m":5.0}

// Calibration and manual markers:
{"type":"event","t_ns":12345678901234,"name":"tap"}
```

`t_ns` is a **plain JSON integer** here — not the decimal-string encoding used on the
gateway's browser-facing `/live` socket. Python's `json` module has arbitrary-precision
integers, so a nanosecond timestamp round-trips exactly without the workaround
JavaScript needs.

## Permissions, explained

| Permission | Why |
|---|---|
| `ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION` | The GPS channel — both a training label (offline) and an opportunistic reset (online). |
| `INTERNET`, `ACCESS_NETWORK_STATE` | The WebSocket uplink to the gateway. |
| `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_LOCATION` | Required on API 29+/34+ to run a foreground service whose job is location-adjacent sampling. |
| `WAKE_LOCK` | Implied by the foreground service; keeps sampling alive with the screen off. |
| `POST_NOTIFICATIONS` | So the persistent "streaming — N Hz" notification actually shows on API 33+. Not required to *start* the service, only to display it. |

`network_security_config.xml` opts into cleartext (`ws://`, not `wss://`) but **only**
for RFC1918 private ranges and `localhost` — the app can never be talked into sending
IMU traffic to an arbitrary cleartext host on the internet, even though cleartext is
allowed at all.

## Current status

| Piece | Status |
|---|---|
| Sensor sampling, combining, wire encoding | ✅ implemented |
| GPS, boot-to-UTC offset, event markers | ✅ implemented |
| WebSocket uplink with reconnect | ✅ implemented |
| Permission gating, guided calibration UI | ✅ implemented |
| `./gradlew :app:assembleDebug` / `:app:lintDebug` | ✅ green locally (JDK 22) and in CI (Temurin 21) |
| **On-device sensor behaviour** | ❌ **not verified** — no physical device or emulator sensor injection available in the environment this was built in |
| **Actual handshake with a running gateway** | ❌ **not verified** for the same reason |
| Launcher icon, colour scheme | ❌ not started (owner: Akshit) |

If you're picking this up: the code compiles and lints clean, but has never actually
talked to a phone's sensors or to a live gateway. Treat the first real device test as a
genuine first run, not a formality.

## Related

- `docs/CONVENTIONS.md` §3, §5, §6 — clocks, naming, wire format
- `docs/ARCHITECTURE.md` — where this fits in the full chain
- `services/gateway/` (see `WEB.md`'s sibling section, and `services/gateway/app.py`) —
  the other end of `/ingest`
- Issue #34 (this app), #37 (golden-run replay — needs a recording this app produces)
