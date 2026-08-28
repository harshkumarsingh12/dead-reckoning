# Roadmap

Ordered to produce an end-to-end trajectory early, then improve it. Demoable by M1, the
hardened demo by M4.

Exit criteria are copied verbatim from [BUILD_PLAN.md](BUILD_PLAN.md) §10. They are
acceptance criteria, not aspirations: each one has a `strict=True` xfail test carrying
its owner, and CI prints the remaining count as a burndown table on every run.

> **Milestone Schedule:**
> - **M0 & M1 (Harness, Clocks & Baselines):** 27–28 Aug 2026 ✅
> - **M2 & M3 (Learned Velocity & ESKF Fusion):** 28–29 Aug 2026
> - **M4 (Hardened Live Demo & Venue Offline MBTiles):** 29–30 Aug 2026
> - **M5 (Stage Rehearsal & Grand Finale Demo):** 30–31 Aug 2026

---

## M0 — Harness and time

**Target: the harness and the clocks are trusted.**

| Deliverable | Owner | Status |
|---|---|---|
| Eval harness: ATE / RTE / drift | Sikruti | ✅ Done |
| NIS / NEES logger | Sikruti | ✅ Done |
| Clock-offset mapper, sharp-motion verification | Sristee | ✅ Done |
| ~300 ms reorder buffer | Sristee | ✅ Done |
| Session record format (read + write + replay) | Sristee | In progress |
| Frame unit tests including rotation-in-place | Sristee | In progress |
| **RoNIN / OxIOD access requested** | Sumedha | In progress |
| Repo, CI/CD, governance docs | Harsh | ✅ Done |

**Done when:** one command turns a recording into a trajectory plot, error numbers, and
a timing-alignment check, all passing.

> The dataset access request is the single longest lead time in the project — approval
> can take days. It goes out on day one, in parallel with everything else. No amount of
> clever modelling recovers a week lost waiting for an email.

---

## M1 — Baselines and AHRS

**Target: PDR under 10% drift; the magnetometer gate demonstrably fires.**

| Deliverable | Owner | Status |
|---|---|---|
| Shared preprocessing (gravity alignment) and calibration module | Sristee | ✅ Done — resample/window still open for M2 |
| AHRS orientation (imufusion) | Sristee | ✅ Done — settings were silently never applied (swallowed exception), fixed for real in #58 with a test proving it |
| Magnetometer hard-iron fit + magnitude/dip/innovation triple gate | Sristee | In progress — gate logic is correct, but default calibration (`expected_dip_rad=0`) makes it reject even a clean field, leaving heading to gyro-only; tracked as #59 |
| Raw double-integration baseline | Sristee | ✅ Done (code) — the one remaining frame test needs a learned-velocity or ESKF estimate to close a loop starting at nonzero speed, which is M2/M3 scope, not a raw-integration bug (#20 open pending Sristee's confirmation of that read) |
| PDR baseline | Sristee | Not started |
| First own recordings: outdoor GPS walks + one surveyed indoor loop | Sumedha | Not started |
| Web UI shell against mock frames | Tanmay, Akshit | ✅ Done — real socket + real Leaflet map, verified against a live gateway |

**Done when:** end to end on a test loop, and the magnet test visibly trips the gate
without corrupting heading.

> M1 is Sristee-heavy by design — it is one coherent chain (calibrate → orient → gate →
> baseline) and splitting it across people would mean negotiating three interfaces to
> save nothing. Everyone else is on M0 and M2 work in parallel.

---

## M2 — Learned velocity

**Target: drift under 5%, calibrated sigma, inference under 10 ms.**

| Deliverable | Owner | Status |
|---|---|---|
| Causal TCN with a Gaussian-NLL covariance head | Sumedha | ✅ Done (code) — untrained, see note |
| Random-yaw augmentation | Sumedha | ✅ Done (code) — carry-position mixing needs OxIOD, folded into the blocked row below |
| ONNX export, int8, benchmarked | Sumedha | ✅ Done (code) — real latency number needs a real trained model |
| Calibration coverage test wired into the harness | Sumedha, Sikruti | ✅ Done (code) — `dr_core.eval.metrics.calibration_coverage`, unverified on real predictions |
| Model-only trajectory baseline (`ModelOnlyIntegrator`) | Sumedha | ✅ Done (code) — not yet wired into `dr-eval`/`scripts/run_eval.py` (Sikruti's area) |
| `prepare_window` / `resample_uniform` (shared training+live preprocessing) | Sumedha | ✅ Done — closes the "still open for M2" note above |
| Training on RoNIN/OxIOD, fine-tuning on our own walks | Sumedha | Blocked — RoNIN/OxIOD access still pending (#14); `load_ronin`/`load_oxiod` are stubs beyond the access check, no real file to verify either schema against; `scripts/train.py` works end-to-end against our own recordings (`load_own_recording`) once any exist (#22, not yet recorded) |

**Done when:** model-only integration beats PDR on held-out data **and** coverage is
calibrated at roughly 68% within 1σ **and** inference is under budget.

> Export to ONNX on the first checkpoint that trains at all, not at the end. It de-risks
> on-device inference and turns the latency claim into a measured number.

> The full pipeline (model, loss, augmentation, shared preprocessing, ONNX export,
> latency benchmark, calibration coverage, model-only trajectory baseline) is
> implemented and unit-tested against synthetic data, and `scripts/train.py` has been
> smoke-tested end to end against synthetic own-recording sessions. What is NOT done is
> training on anything real and therefore any of the "done when" numbers above — no
> RoNIN/OxIOD access and no own recordings exist yet in this environment. Those numbers
> are not claimed until actually measured (AGENTS.md: "do not invent numbers").

---

## M3 — ESKF fusion

**Target: drift 3–5%, NIS consistent on every channel.**

| Deliverable | Owner | Status |
|---|---|---|
| 2D ESKF: predict + inject/reset | Sikruti | ✅ Done |
| Device-frame velocity update, with the `dh/dpsi` heading term | Sikruti | ✅ Done |
| ZUPT and ZARU, driven by the stationary detector | Sikruti | ✅ Done |
| Per-session velocity scale `s`, frozen when GPS drops | Sikruti | ✅ Done |
| Chi-square gating on every channel, NIS logged | Sikruti | ✅ Done |
| Reorder buffer wired into the live path | Sikruti, Sristee | ✅ Done |

**Done when:** fused beats model-only on ATE and RTE, NIS is in bounds on all channels,
and a 10 s stop produces zero position creep.

---

## M4 — Hardened live demo

**Target: the headline number holds live, offline.**

| Deliverable | Owner | Status |
|---|---|---|
| Android IMU streamer, capture-time stamping, WS uplink | Harsh | ✅ Done — verified on a real device over USB (`adb reverse`): stable `/ingest` socket, no reconnect churn. Found and fixed two real bugs the code review missed: `HIGH_SAMPLING_RATE_SENSORS` permission (Android 12+ needs it for `SENSOR_DELAY_FASTEST`) and a network-security-config that only ever matched literal addresses, never a real LAN IP |
| Gateway: ingest, broadcast, control | Harsh | ✅ Done — `/live` is a GPS-passthrough placeholder pending the live ESKF |
| Golden-run replay through the identical pipeline | Harsh | Blocked — needs `dr_core.io.SessionReader` (Sristee, M0); deliberately waiting rather than duplicating her schema |
| Offline MBTiles built and verified with Wi-Fi off | Harsh | ✅ Done for the KIIT practice loop — 582 real tiles, zero non-localhost requests observed while rendering; needs the actual SIH venue's bbox once announced |
| Map, dot, ellipse, GPS toggle, diverging baseline dot | Tanmay | ✅ Done — real Leaflet, verified against a live gateway (dot's screen position confirmed moving across two screenshots of the same page) |
| Telemetry strip: NIS, ZUPT/ZARU lamp, mag gate, σ, heading source | Akshit | ✅ Done — verified against a mock frame, screenshots reviewed |
| Auto post-run report panel | Akshit, Sikruti | Presenter + serving ✅ done; live auto-trigger blocked on the ESKF + a surveyed loop |
| Scripted 3-minute arc, rehearsed | Tanmay, Sumedha | Not started |

**Done when:** the full demo executes with venue internet unplugged and the dot does not
visibly lag turns.

---

## M5 — Stretch, in priority order

Gated on the core chain hitting its metrics. These are demo enhancers, **not
dependencies** — that gating is what stops the project ending with six half-built
features instead of one that works.

1. Map matching against an OSM path graph (the urban route-snapping story)
2. Full on-device ONNX inference — no laptop in the loop
3. Multiple carry positions demoed back to back
4. Ellipse snap-on-reacquire when GPS is toggled back on at the end

---

## Tracking

- **GitHub Milestones** M0–M5 mirror these sections; every issue carries one.
- **The xfail ledger is the burndown.** Every unimplemented acceptance criterion is a
  `strict=True` xfail test naming its milestone and owner. Implementing it makes the
  test XPASS, which turns CI red until the marker is deleted — so "done" cannot be
  claimed without the test agreeing. `ci-python.yml` posts the remaining count per owner
  on every run.
- **Slipping is fine; silent slipping is not.** If a milestone is going to move, say so
  in the group chat the day you know, not the day it is due.
