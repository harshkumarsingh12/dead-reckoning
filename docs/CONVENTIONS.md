# Conventions

**One written convention, agreed once, followed everywhere.** Most trajectory bugs live
in frames, units and clocks. They do not crash and they do not raise — they produce a
path that is confidently wrong. This page is the cheapest insurance in the project.

If code and this page disagree, this page wins and the code is a bug.

---

## 1. Coordinate frames

Three frames. Every vector in the codebase belongs to exactly one of them, and says so
in its name.

| Frame | Suffix | Definition |
|---|---|---|
| **World** | `_world` | ENU. **x = East, y = North, z = Up.** Metres. Origin is the session's start point (`SessionMeta.origin_lat_deg` / `origin_lon_deg`). |
| **Body** | `_body` | The raw device frame, exactly as the Android sensor reports it. x right across the screen, y up the screen, z out of the screen. |
| **Device-aligned** | `_dev` | Gravity-aligned but **heading-agnostic**: z is true Up, the horizontal axes are an arbitrary but consistent rotation about it. This is the frame the learned velocity model regresses into. |

### Why `_dev` exists

It is the RoNIN trick, and it is what makes the model indifferent to how the phone is
held. Because the frame carries no absolute heading, a model trained with random-yaw
augmentation cannot learn "the user usually faces this way". Combined with fusing the
velocity in this frame — which puts a `dh/dpsi` term in the measurement Jacobian — it is
also how heading stays observable without the magnetometer.

### Heading

`psi` is the rotation from World to the device's forward direction:

```
psi = 0      facing East
psi = pi/2   facing North          (counter-clockwise positive, right-handed about Up)
```

This is the **maths convention, not the compass convention.** A compass bearing is
clockwise from North. Converting:

```python
psi_rad = np.deg2rad(90.0 - bearing_deg)
```

Get this backwards and the path is mirrored, which looks plausible for a straight line
and obviously wrong on a loop. That is precisely why `test_pure_turn_closes_the_circle`
exists.

### Rotations

`R(psi)` rotates **from device-aligned to world**:

```
v_world = R(psi) @ v_dev
v_dev   = R(-psi) @ v_world     # equivalently R(psi).T
```

Quaternions are `(w, x, y, z)`, scalar first, and `q_world_body` rotates body to world.
`scipy.spatial.transform.Rotation` uses scalar-**last** `(x, y, z, w)`. Convert at the
boundary; do not silently pass one where the other is expected.

---

## 2. Units

Strict SI, everywhere, with no exceptions and no "just this once".

| Quantity | Unit | Notes |
|---|---|---|
| Length | metre | |
| Velocity | m/s | |
| Acceleration | m/s² | Includes gravity until it is explicitly removed. |
| Angle | **radian** | Degrees exist only in the UI layer and in lat/lon. |
| Angular rate | rad/s | |
| Magnetic field | **tesla** | Android reports µT. Convert at the sensor boundary, once. |
| Time | **int64 nanoseconds** | See below. |
| Frequency | Hz | |

Degrees appear in exactly two places: geographic coordinates (`lat_deg`, `lon_deg`) and
text rendered for a human. Both carry a `_deg` suffix. Anything without a suffix is
radians.

---

## 3. Time

**Every timestamp in this system is `int64` nanoseconds in the device boot-monotonic
domain.** Not float seconds. Not milliseconds. Not wall clock.

- **`int`, not `float`.** A float64 has 53 bits of mantissa; nanoseconds since boot
  exceed that within a few months of uptime, and the resulting quantisation is a
  sub-millisecond error that no test will notice and every metric will feel.
- **Boot-monotonic, not UTC.** Wall clock jumps — NTP corrections, timezone changes,
  the user's clock app. A monotonic clock does not.
- **Assigned at capture, on the device.** `SensorEvent.timestamp` is already in this
  domain. `Location.getElapsedRealtimeNanos()` puts a GPS fix in the same domain, which
  is why a fix that arrives 400 ms late can still be fused at the instant it was taken.

### Never restamp

Assigning a timestamp on network arrival, on queue insertion, or on write is invisible,
looks entirely reasonable in review, and destroys the alignment the whole timing
subsystem exists to guarantee. If you are unsure whether a change restamps, assume it
does and ask.

### UTC

`SessionMeta.boot_to_utc_offset_ns` maps the boot domain onto UTC:

```python
utc_ns = boot_ns + offset_ns
```

Estimated **once** at session start (re-estimated only if drift is detected). It exists
so recordings can be aligned against externally timestamped labels. Nothing in the live
pipeline uses UTC for anything.

### The lagged timeline

The filter runs ~300 ms behind wall clock, with measurements released from a reorder
buffer in capture order. The rendered dot is therefore 300 ms old. **That is correct and
deliberate.** Do not extrapolate to hide it — extrapolation makes the dot cut corners,
which reads as broken to anyone watching, and is dishonest besides.

---

## 4. Naming

A vector's name carries its frame. Always.

```python
v_world  # world ENU velocity
v_dev  # device-aligned velocity — what the model outputs
a_body  # raw accelerometer, device frame
w_body  # raw gyroscope, device frame
psi_rad  # heading
p_world  # position
```

A bare `v` or `a` is a review comment waiting to happen. Other conventions:

| Pattern | Meaning |
|---|---|
| `*_ns` | int64 nanoseconds, boot-monotonic |
| `*_s` | float seconds — durations and thresholds only, never absolute times |
| `*_rad`, `*_deg` | angle, with its unit |
| `*_m`, `*_mps` | metres, metres per second |
| `sigma_*` | one standard deviation |
| `cov` | a covariance matrix, never a standard deviation |
| `R` | measurement noise covariance (Kalman convention) |
| `Q` | process noise covariance |
| `H` | measurement Jacobian |
| `dx` | error state, ordered by `ERROR_STATE_ORDER` |

Python is `snake_case`; TypeScript is `camelCase` except for wire fields, which mirror
the Python names exactly so the two contracts can be diffed by eye.

---

## 5. The error state

Fixed ordering, defined once in `dr_core.types.ERROR_STATE_ORDER`:

```
index   0    1    2    3    4     5     6
name    dpx  dpy  dvx  dvy  dpsi  db_g  ds
```

Every Jacobian indexes against it. Permuting it produces a filter that runs, converges
to something, and is wrong. `test_error_state_ordering_is_fixed` pins it.

---

## 6. Wire format

WebSocket frames are JSON, field names identical to the Python dataclass fields.

**Timestamps cross the wire as decimal strings.** `Number.MAX_SAFE_INTEGER` is about
9.007e15; nanoseconds since boot pass that in roughly 104 days of uptime, and
`JSON.parse` silently rounds. A string is ugly and correct.

```json
{
  "t_ns": "1723456789012345678",
  "state": { "p_world": [12.4, -3.1], "psi_rad": 1.5708 }
}
```

Enums cross as their lowercase string values (`"rejected_dip"`), never as integers —
they have to be debuggable from a browser console at 2 a.m.

---

## 7. Randomness

Seed is `26168` — the problem statement number — everywhere a seed is needed. Use
`np.random.default_rng(seed)`, never the global `np.random.*` functions: global state
makes a run reproducible only until someone adds a call above yours.

---

## 8. When you find a violation

Fix it, and add the test that would have caught it. If the convention itself is wrong,
change **this page** in the same PR and say so in the description. A convention that is
quietly worked around in one file has stopped being a convention.
