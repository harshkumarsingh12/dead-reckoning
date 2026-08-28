# Build Plan v2
## GPS-Denied Pedestrian Dead Reckoning (Theme: Space Technology)
### Integrated: original plan + principal engineering review (27 Aug 2026)

---

## 0. Scope and assumptions

- **Target scenario:** a pedestrian carrying a smartphone (hand or pocket), tracked in 2D (position + heading) through GPS dropouts: tunnels, indoors, underground, urban canyon.
- **Output:** a live map showing the estimated position dot moving with GPS off, a live drift/error readout, a live filter-health telemetry strip, and a post-run error report, all working **with zero venue internet**.
- **If the intended target is vehicular** (car in a tunnel), the motion model, ZUPT logic, and data plan change materially. This plan assumes pedestrian.

---

## 1. Design principles (why this is not flimsy)

Dead reckoning fails in specific, known ways. Each one is addressed structurally.

1. **No double integration.** Integrating raw acceleration twice produces error that grows faster than linearly and meters of drift within seconds. A learned model regresses **bounded-error velocity** directly from IMU windows (the RoNIN / TLIO approach). This is the load-bearing anti-drift decision.
2. **The heading loop is closed through the velocity update.** The learned velocity is fused **in the device-aligned frame**, so the measurement Jacobian contains a heading term and every velocity update corrects heading, especially during turns. Heading is not left dependent on the magnetometer alone.
3. **Orientation comes first.** A dedicated AHRS layer produces a clean orientation estimate (which way is down, which way is forward) before anything downstream runs.
4. **Uncertainty is proven honest, not decorative.** The model's covariance head is trained with a Gaussian NLL objective and verified by calibration coverage on held-out data; the filter's consistency is checked with NIS/NEES and chi-square innovation gating on every measurement channel. The on-screen ellipse means something and survives judge questioning.
5. **The clocks are owned.** All sensors are mapped onto one time domain at session start, and late-arriving measurements are fused at their true capture time through a reorder buffer. Time bugs are the classic silent killer; here they are engineered out, not hoped away.
6. **The demo survives a dead network.** Offline map tiles, phone-hotspot transport, and a recorded golden-run replay through the identical pipeline. No dependency on venue internet or Wi-Fi.
7. **Measured, not vibed.** An evaluation harness (ATE / RTE / drift %, plus consistency checks) exists from milestone zero, so "strong prototype" is a number on a known loop.

---

## 2. Architecture

```
  Smartphone sensors                        (per-sample, 100-200 Hz)
  accel | gyro | mag | GPS
        |
        | timestamped ON DEVICE in the boot-monotonic clock domain,
        | mapped once per session to UTC (see section 5)
        v
  [1] SHARED preprocessing + calibration module
      (single Python package imported by BOTH training and live paths)
      resample, units, gravity alignment, gyro-bias capture,
      magnetometer hard-iron calibration (10 s figure-8)
        |
        v
  [2] AHRS orientation  (Madgwick/Mahony: gyro+accel+mag -> quaternion)
        |                              |
        | gravity-aligned IMU window   | heading rate + mag heading (gated)
        v                              |
  [3] Learned velocity model  ---------+
      causal TCN: window ending at t, hop <= 200 ms
      outputs: device-frame planar velocity + NLL-trained covariance
      runtime: ONNX (int8), laptop now, on-device stretch
        |
        v
  [4] Reorder buffer (~300 ms lagged timeline)
      late GPS / velocity measurements inserted at capture time
        |
        v
  [5] 2D Error-State Kalman Filter (ESKF)
      nominal: p, v, psi   error state: dp, dv, dpsi, db_g, ds
      updates: device-frame velocity (with d/dpsi term), ZUPT, ZARU,
               gated magnetometer heading, GPS (position + scale learning)
      every update chi-square gated; NIS logged per channel
        |
        v
  [6] Trajectory  ----> [7] Map matching (stretch: snap to OSM path graph)
        |
        v
  [8] Live demo system (fully offline)
      Leaflet + pre-cached MBTiles served locally, phone-hotspot WebSocket,
      estimated dot + uncertainty ellipse + GPS-off toggle + drift readout,
      telemetry strip (NIS, ZUPT/ZARU lamp, mag gate, model sigma, heading source),
      auto post-run report, golden-run replay fallback
```

The spine is layers 1 to 5 plus 8. Map matching and on-device inference are gated stretch work.

---

## 3. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | modelling, filter, eval |
| IMU data collection | Sensor Logger / Phyphox | off-the-shelf for training data; do not build an app just to collect |
| Live-demo app | native Android (Kotlin) or Flutter | must expose sensor timestamps and stream over WebSocket |
| Numerics | NumPy, SciPy | |
| Deep learning | PyTorch (train), **ONNX Runtime (inference)** | export early, int8-quantized; gives a laptop inference budget number and de-risks on-device |
| AHRS | `imufusion` (Madgwick) or `ahrs` lib | do not hand-roll unless forced |
| Filter | **custom 2D ESKF in NumPy** (review estimate: ~150 LOC) + NIS/NEES logger module | FilterPy acceptable for early prototyping only |
| Public datasets | RoNIN, OxIOD | request access early; approval can take time |
| Map tiles | **pre-cached MBTiles served locally** (MBTiles is a SQLite container; serve via a small FastAPI endpoint or tileserver) | offline-first, no venue internet |
| Transport | **WebSocket over the phone's own hotspot** (FastAPI server on laptop) | plus recorded-session replay fallback |
| Live UI | Leaflet.js, React front end | matches team stack |
| Map matching (stretch) | OSMnx + `leuvenmapmatching` | HMM/Viterbi matching |

---

## 4. Data plan

Data is the make-or-break. Start day one, in parallel with everything else.

**Public datasets (stand on these, do not train from scratch):**
- **RoNIN**: large multi-subject smartphone IMU dataset with high-quality ground-truth trajectories; also the method template (heading-agnostic velocity regression). Check access terms and request the full set immediately.
- **OxIOD**: smartphone IMU with Vicon motion-capture ground truth across multiple carry positions (hand, pocket, bag, trolley). Excellent for carry-position robustness.

**Your own data (for the demo domain and fine-tuning):**
- Walks **outdoors with strong GPS**; GPS-derived velocity and position are the training labels. Fine-tune on these, then run with GPS off.
- Several **known indoor loops** (a corridor rectangle with measured corner points and a marked start). These are your evaluation ground truth and your demo course.
- Vary carry position (hand, pocket), walking speed, and phones.

**Discipline that prevents silent failures:**
- **One shared preprocessing module.** Resampling, unit conversion, and gravity alignment live in a single Python package imported by both the training pipeline and the live pipeline. Any mismatch between how training data and live data are prepared silently degrades the model; sharing the code makes the mismatch impossible.
- **Coordinate frames:** one written convention (body frame vs world ENU). Rotate body-frame data to world using the AHRS orientation. Most trajectory bugs live here.
- **Units and rate:** normalize sample rate and units identically for datasets and the live phone, via the shared module.
- **Splits:** split **by trajectory**, never by window, or metrics lie through leakage.
- **Augmentation (in training):** random yaw rotations (enforces the heading-agnostic frame), carry-position mixing across RoNIN and OxIOD, and speed diversity.

---

## 5. Time, clocks, and latency (dedicated subsystem)

This section exists because timing is the most common invisible failure in sensor fusion, and because a laggy demo dot is a visible one.

**One clock, mapped once.**
- On Android, IMU `SensorEvent.timestamp` is **nanoseconds in the boot-monotonic domain** (`elapsedRealtimeNanos`). GPS fixes carry a UTC time and also arrive **hundreds of milliseconds late**; usefully, `Location.getElapsedRealtimeNanos()` gives the fix in the same boot-monotonic domain.
- Stamp **every** sample on the device, in the boot-monotonic domain, at capture (never on network arrival). Estimate the boot-domain-to-UTC offset once at session start (re-estimate if drift is detected) so recordings can be aligned with GPS labels.
- **Verification:** the sharp-motion-event test. A distinct physical event (a firm tap or stomp) must appear at the same instant across aligned streams.

**Latency sources and the fix.**
- GPS latency: hundreds of ms. Model group delay: a non-causal 1 to 2 s window effectively estimates mid-window velocity, adding roughly 0.5 to 1 s of delay. Network jitter adds more.
- **Model fix:** a **causal** network (no future context) with the window ending at the current instant and a hop of at most 200 ms cuts effective model delay to roughly 0.2 s.
- **Filter fix:** run the ESKF on a timeline lagged by ~300 ms behind wall clock, with a **reorder buffer**. Measurements queue by capture timestamp; the filter processes everything up to (now minus 300 ms), so late GPS and velocity measurements are fused **at their capture time**, in order.
- **Render fix:** decouple rendering from capture. The UI draws the lagged-but-smooth filter state. This is honest (no fake extrapolation) and visually clean; the dot does not cut corners.

---

## 6. Subsystem specifications

Each lists purpose, approach, the robustness point, and a concrete "done when."

### 6.1 Data logging and streaming
- **Purpose:** clean, timestamped IMU (+GPS) both offline (training) and live (demo).
- **Approach:** off-the-shelf loggers for training data. For the live demo, a thin app that streams device-timestamped samples over WebSocket **to a FastAPI server on the laptop, over the phone's own hotspot** (no venue network in the loop).
- **Replay fallback:** a recorded "golden run" session that plays back through the **identical** live pipeline (same server, same filter, same UI), clearly labelled as a replay. The demo survives any network condition.
- **Robustness:** timestamps assigned at capture in the boot-monotonic domain (section 5).
- **Done when:** a recorded session and a live session flow through the same downstream pipeline unchanged, and the replay is indistinguishable in mechanics from live.

### 6.2 Preprocessing and calibration (the shared module)
- **Purpose:** turn raw streams into calibrated, world-referenced inputs, identically for training and live.
- **Approach:** gyro bias from a stationary window at session start; accel bias/scale; **magnetometer hard-iron calibration from a 10-second figure-8** at session start. Resample to a fixed rate, light low-pass, gravity handling via orientation. All of it in one importable package.
- **Done when:** a stationary phone reports near-zero world-frame velocity over 60 s, and the training and live paths import the same functions.

### 6.3 AHRS orientation
- **Purpose:** device orientation (down direction + heading).
- **Approach:** Madgwick or Mahony complementary filter fusing gyro (rate), accel (gravity, fixes pitch/roll), magnetometer (yaw). Quaternion out.
- **Robustness (upgraded magnetometer gate):** a magnetometer reading is accepted only if **all three** hold: (a) field magnitude within tolerance of the calibrated expectation, (b) **dip/inclination angle** within tolerance (indoor disturbances often rotate the field at near-normal magnitude, so magnitude alone is a weak check), and (c) the resulting heading innovation passes the chi-square gate in the filter.
- **Done when:** orientation is stable over minutes of walking with turns, and a deliberately introduced magnet visibly triggers rejection.

### 6.4 Learned velocity model (the ML core)
- **Purpose:** bounded-error planar velocity from IMU, replacing double integration.
- **Architecture:** a **causal TCN** (temporal convolutional network with no future context), input window of about 1 to 2 s of gravity-aligned accel + gyro ending at the current instant, hop of at most 200 ms. (A strictly causal LSTM is an acceptable variant and matches the problem statement's wording; the causal TCN is preferred for latency and training stability.)
- **Output:** 2D velocity in a device-orientation-aligned, heading-agnostic frame (RoNIN's key trick, making the model robust to how the phone is held), **plus a covariance**.
- **Loss:** **Gaussian negative log-likelihood** over the velocity error, so the covariance head is trained jointly and meaningfully, not bolted on after an MSE fit.
- **Calibration verification (mandatory):** on held-out trajectories, empirical coverage of roughly 68% of errors within 1 sigma (per axis), or NEES within statistical bounds. An uncalibrated covariance silently poisons the filter; this test is what makes "learned uncertainty feeds R" true.
- **Augmentation:** random yaw rotations, carry-position mixing, identical rate/units normalization via the shared preprocessing module.
- **Runtime:** export to **ONNX early**, int8-quantized. Record the per-window inference time as a stated budget (target: under 10 ms per window on the laptop).
- **Done when:** integrating the model's velocity alone (no filter) beats the PDR baseline on held-out data with single-digit-percent drift, the coverage test passes, and inference is under budget.

### 6.5 ESKF fusion
See section 7 for the explicit wiring.
- **Done when:** fused beats model-only on ATE/RTE, NIS is within bounds on all channels, and a 10 s stop produces zero position creep.

### 6.6 ZUPT, ZARU, and gating
- **Purpose:** kill drift during pauses, pin gyro bias, and reject bad measurements everywhere.
- **Approach:** stationary detection by low accel/gyro variance over a short window. On detection: **ZUPT** (velocity measured as zero) and **ZARU** (angular rate measured as zero, pinning the gyro bias). Independently, **every** measurement channel (velocity, heading, GPS) passes a **chi-square innovation gate at the 95% level**, with per-channel NIS logged continuously.
- **Robustness:** ZUPT/ZARU are physics-based corrections independent of the learned model; the gates stop any single bad sensor from corrupting the state.
- **Done when:** standing still 10 s produces no creep, bias converges, and the NIS log shows gates firing on injected outliers.

### 6.7 Map matching (stretch)
- **Purpose:** constrain the path to real walkable routes in the urban case.
- **Approach:** HMM/Viterbi matching against an OSM path graph (OSMnx + leuvenmapmatching).
- **Gate:** only after the core chain hits its metrics. Demo enhancer, not a dependency.

### 6.8 Live demo system
- **Purpose:** the winning demo, engineered to be un-killable.
- **Approach:** Leaflet map fed by the filter over the hotspot WebSocket, drawing tiles from **pre-cached MBTiles served locally**. On screen: estimated dot, known true path, uncertainty ellipse, GPS-off toggle, live drift-% counter, and the raw-integration baseline as a second, visibly diverging dot.
- **Telemetry strip (under the map):** per-channel NIS with chi-square bounds, a ZUPT/ZARU firing lamp, magnetometer gate accept/reject indicator, the model's current 1-sigma, and the current heading source (mag / ZARU / GPS). This strip is what converts "nice demo" into "these people know exactly what their filter is doing" during Q&A.
- **Post-run report:** the moment the walk ends, auto-generate an error-vs-time strip chart and an error CDF, plus the loop-closure error in metres and drift %.
- **Done when:** the full demo executes with venue internet unplugged and the dot does not visibly lag turns.

---

## 7. The ESKF wiring (explicit)

This is where designs usually go vague. It does not here. Reference: Sola, "Quaternion kinematics for the error-state Kalman filter" (the 2D case here is a simplification of that machinery).

**Why error-state:** the nominal state is propagated with the full nonlinear equations, while the filter runs on a small error state where linearization is exact where it matters (heading). ZUPT/ZARU become clean pseudo-measurements on the error state, and "why ESKF?" is itself a judging differentiator. In 2D the extra math cost is small.

**Nominal state:** position `p = (px, py)` (world ENU, m), velocity `v = (vx, vy)` (world, m/s), heading `psi`.
**Estimated parameters:** gyro yaw bias `b_g`, per-session velocity scale `s`.
**Error state:** `dx = [dp(2), dv(2), dpsi, db_g, ds]` (7 states), with the covariance kept on `dx`.

**Prediction (each step, dt):**
- `psi += (gyro_yaw - b_g) * dt`
- `p += v * dt`
- `v`: constant-velocity model with process noise (the velocity update is the anchor that corrects it)
- `b_g`, `s`: random walk / near-constant
- propagate the error-state covariance with the Jacobian F and process noise Q

**Measurement updates (every one chi-square gated at 95%, NIS logged per channel):**

1. **Learned velocity, fused in the device-aligned frame (primary anchor).**
   Measurement: `z = v_dev` (the network's device-frame planar velocity).
   Measurement model: `h(x) = (1/s) * R(-psi) * v_world`.
   Because `h` depends on `psi`, the Jacobian H contains a `d h / d psi` term, so **every velocity update also corrects heading**; heading drift stops bending the path during turns. It likewise contains a `d h / d s` term for the scale.
   Measurement noise R = the network's NLL-trained covariance for this window.
   (If a world-frame update were kept instead, R would have to be inflated by the heading-uncertainty term; the device-frame form is simpler and strictly better.)

2. **Per-session velocity scale `s`.** Learned-inertial models carry per-user/per-gait scale bias. While GPS is present, GPS position/velocity makes `s` observable and it converges; **when GPS drops, freeze `s`** (scale and speed are no longer separable). One extra state, several percent of drift removed.

3. **ZUPT.** Stationary detector fires: measure `v = 0`, small R.

4. **ZARU.** Stationary: measure `gyro_yaw - b_g = 0`, pinning the bias.

5. **Magnetometer heading.** Yaw measurement, accepted only under the triple gate (magnitude AND dip AND innovation, section 6.3), after hard-iron calibration.

6. **GPS.** When momentarily available: position update, heading correction from course-over-ground at sufficient speed, and the observation window in which `s` is learned. GPS is both a training label (offline) and an opportunistic reset (online).

**After every accepted update:** inject the error into the nominal state, reset `dx` to zero, and apply the covariance reset (with a scalar heading angle in 2D, the reset Jacobian is essentially identity; state it and move on).

**All updates flow through the ~300 ms reorder buffer** (section 5), so late measurements fuse at capture time, in order.

---

## 8. Consistency and evaluation

Built at milestone 0. Two families of numbers: accuracy and honesty.

**Accuracy:**
- **ATE** (Absolute Trajectory Error): RMSE of estimated vs ground-truth positions after alignment.
- **RTE** (Relative Trajectory Error): error over a fixed window (e.g. 60 s); reflects drift rate.
- **Final position / loop-closure error** and **drift %** = final error / distance travelled. The headline demo number.
- **Baselines always plotted alongside:** raw double integration (exceeds 100%, spirals) and simple PDR (step count x step length + heading). The contrast is the story.

**Honesty (filter and model consistency):**
- **NIS** per measurement channel, logged live, expected within chi-square bounds.
- **NEES** against ground truth on recorded runs.
- **Model calibration coverage:** ~68% of held-out velocity errors within 1 sigma.

**Targets on a 100 to 300 m indoor loop:**

| Metric | Acceptable | Strong (stretch) |
|---|---|---|
| Drift (final error / distance) | < 5% | < 2 to 3% |
| RTE over 60 s | a few metres | ~1 to 2 m |
| NIS / NEES | within bounds on all channels | same, across carry positions |
| Model coverage at 1 sigma | ~68% (calibrated) | holds across carry positions |
| Inference per window | < 10 ms (laptop, ONNX) | on-device viable |
| Raw-integration baseline | > 100% (contrast) | n/a |

---

## 9. Test suite (the cheap insurance)

- **Frame unit tests on synthetic data:** straight-line motion, pure turn, and **rotation-in-place** (the phone spun about yaw while the position is stationary must produce zero velocity). The rotation-in-place case catches the classic frame/lever-arm failure the first two cannot.
- **Stationary creep test:** 60 s still, near-zero world velocity; 10 s stop mid-walk, zero position creep.
- **Magnet test:** a deliberately introduced magnet must trip the magnetometer triple gate, visibly, without corrupting heading.
- **Timing alignment test:** the sharp-motion event appears at the same instant across all aligned streams.
- **Outlier injection:** synthetic bad GPS/velocity samples must be rejected by the chi-square gates and show up in the NIS log.
- **Calibration coverage test:** part of the standard eval run.

---

## 10. Phased milestones

Ordered to produce an end-to-end trajectory early, then improve. Demoable path by M1, the full hardened demo by M4.

| Milestone | Deliverable | Done when | Target | Status |
|---|---|---|---|---|
| **M0 — Harness + time** | Eval harness (ATE/RTE/drift) + NIS/NEES tooling + clock-offset mapper + frame unit tests including rotation-in-place; RoNIN/OxIOD access requested and loading | One command turns a recording into a trajectory plot, error numbers, and a timing-alignment check, all passing | Harness and clocks trusted | ✅ Done (Sikruti, Sristee, Harsh) |
| **M1 — Baselines + AHRS** | AHRS orientation; PDR and raw-integration baselines; mag hard-iron calibration + magnitude/dip/innovation triple gate | End-to-end on a test loop; mag gate rejects an induced disturbance (magnet test) | PDR < 10%; gate demonstrably fires | In progress (Sristee) |
| **M2 — Learned velocity** | Causal TCN with NLL-trained covariance; yaw + carry augmentation; shared train/live preprocessing; ONNX export | Model-only beats PDR on held-out data AND coverage is calibrated (~68% in 1 sigma); inference < 10 ms | Drift < 5%, calibrated sigma | In progress (Sumedha) |
| **M3 — ESKF fusion** | 2D ESKF: device-frame velocity update (with the heading term), ZUPT/ZARU, scale state s, chi-square gating, 300 ms reorder buffer | Fused beats model-only on ATE/RTE; NIS in bounds on all channels; 10 s stop gives zero creep | Drift < 3 to 5%; NIS consistent | ✅ Done (Sikruti) |
| **M4 — Hardened live demo** | Offline MBTiles, hotspot transport, golden-run replay fallback, telemetry strip, latency-compensated render, scripted 3-minute arc | Full demo executes with venue internet unplugged; the dot does not visibly lag turns | Headline number holds live, offline | In progress (Harsh, Tanmay, Akshit) |
| **M5 — Stretch** | On-device ONNX inference; OSM map matching; multi-carry live demo | Runs without a laptop; urban path snaps to routes | Bonus polish | Pending |

---

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Coordinate-frame bugs | trajectory silently wrong | frame unit tests incl. rotation-in-place; one written convention |
| Clock-domain mismatch (IMU boot clock vs GPS UTC, GPS latency) | mislabeled training data, corrupted fusion | one-clock mapping at session start; sharp-motion verification; reorder buffer |
| Heading drift | whole path bends | device-frame velocity update (heading term) + ZARU + gated mag + GPS course + map matching |
| Rotated-field magnetic disturbance indoors | bad heading at near-normal field strength | triple gate: magnitude AND dip AND innovation; hard-iron figure-8 |
| Over-confident model covariance | silently poisons fusion, indefensible ellipse | NLL training + coverage test + NIS monitoring |
| Laggy / corner-cutting demo dot | visibly weak live demo | causal TCN (hop <= 200 ms) + lagged-timeline filter + decoupled render |
| Venue internet/Wi-Fi dead or hostile | demo dies | offline MBTiles + phone hotspot + golden-run replay through the identical pipeline |
| Per-user velocity scale bias | consistent over/under-shoot | scale state s, learned under GPS, frozen without |
| Dataset-to-phone domain gap | model underperforms live | shared preprocessing module; fine-tune on own data |
| Carry-position mismatch | model fails from pocket | heading-agnostic frame + yaw/carry augmentation |
| Data leakage | metrics lie | split by trajectory, never by window |
| Dataset access delay | training blocked | request RoNIN/OxIOD access at M0, day one |
| Scope creep | nothing fully works | map matching + on-device are gated stretch, not dependencies |

---

## 12. Demo script and judging strategy

**The scripted 3-minute arc (visible cause and effect):**
1. Walk a marked loop; **toggle GPS off at a marked corner**. The uncertainty ellipse starts growing on screen.
2. **Stop for 10 seconds** mid-walk. The ZUPT lamp fires, the ellipse tightens, drift freezes. Judges see the mechanism work.
3. **Switch the phone from hand to pocket** while walking. The track holds (carry robustness, courtesy of the heading-agnostic model and augmentation).
4. **Return to the marked start.** The on-screen loop-closure error in metres and the drift % are the closing shot, with the raw-integration baseline dot having spiralled off in parallel the whole time.

**Always on screen:** the telemetry strip (per-channel NIS with bounds, ZUPT/ZARU lamp, mag gate accept/reject, model 1-sigma, heading source) and the live drift-% counter.

**The moment the walk ends:** the auto-generated post-run panel (error-vs-time strip chart + error CDF). Keep one backup slide with the ATE/RTE/drift table against the PDR and raw-integration baselines.

**Network-proof by construction:** offline tiles, hotspot link, golden-run replay. Rehearse the replay path so switching to it is seamless if anything misbehaves.

**Pre-briefed Q&A (rehearse crisp answers to the predictable four):**
1. *Why not just double-integrate the accelerometer?* Error compounds faster than linearly; bounded-error learned velocity does not.
2. *Why EKF + neural network rather than end-to-end LSTM?* The filter fuses independent physical constraints (ZUPT, heading, GPS) with principled uncertainty, degrades gracefully, and is inspectable; the network does the one thing networks are best at.
3. *What happens when the magnetometer fails indoors?* The triple gate rejects it; heading remains observable through the device-frame velocity update's heading term, ZARU pins the gyro bias, and GPS course corrects on reacquire.
4. *How do you know your uncertainty is honest?* NLL-trained covariance, verified 1-sigma coverage on held-out data, and live NIS within chi-square bounds, visible on the telemetry strip.

---

## 13. Minimum winning demo, and stretch

**Minimum winning demo (protect at all costs):**
- Known marked loop, GPS toggled off mid-walk, fully offline stack.
- Estimated dot tracks the real path on the local-tile map, without visible lag.
- ZUPT stop moment lands on cue; ellipse behaviour is visibly meaningful.
- On-screen loop-closure error and drift % stay small; raw-integration baseline spirals alongside.
- Golden-run replay standing by.

**Stretch, in priority order:**
1. Map matching for the urban route-snapping story.
2. Full on-device inference (no laptop in the loop), enabled by the early ONNX export.
3. Multiple carry positions demoed back to back.
4. Uncertainty ellipse snap-on-reacquire moment (toggle GPS back on at the end).

---

## 14. References (method sources)

- Herath, Yan, Furukawa. *RoNIN: Robust Neural Inertial Navigation in the Wild* (ICRA 2020). Heading-agnostic learned velocity regression; dataset.
- Liu et al. *TLIO: Tight Learned Inertial Odometry* (IEEE RA-L 2020). Learned displacement + covariance (NLL) fused in an EKF; the template for this plan's fusion.
- Chen et al. *OxIOD: The Dataset for Deep Inertial Odometry*. Multi-carry smartphone IMU with Vicon ground truth.
- Chen et al. *IONet: Learning to Cure the Curse of Drift in Inertial Odometry* (AAAI 2018). Early learned inertial odometry.
- Sola. *Quaternion kinematics for the error-state Kalman filter* (2017). ESKF formulation.
- Madgwick. *An efficient orientation filter for inertial and inertial/magnetic sensor arrays* (2010). AHRS.

---

*Bottom line: keep the original plan's discipline; wire heading into the velocity update, prove the covariance is honest, own the clocks, and make the demo survive a dead network. The core chain done cleanly, with real and provably honest numbers, beats six half-built features.*
