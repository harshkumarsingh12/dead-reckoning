# Roadmap

Ordered to produce an end-to-end trajectory early, then improve it. Demoable by M1, the
hardened demo by M4.

Exit criteria are copied verbatim from [BUILD_PLAN.md](BUILD_PLAN.md) §10. They are
acceptance criteria, not aspirations: each one has a `strict=True` xfail test carrying
its owner, and CI prints the remaining count as a burndown table on every run.

> **Dates are not filled in.** Set them from the actual SIH internal-hackathon and
> grand-finale dates and commit that in the same PR — a roadmap with no dates on it
> stops being a plan and becomes a wish list.

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
| Causal TCN with a Gaussian-NLL covariance head | Sumedha | ✅ Done — trained (epoch 113, seed 26168, OxIOD handheld+pocket) |
| Random-yaw augmentation | Sumedha | ✅ Done — used throughout the real training run |
| ONNX export, int8, benchmarked | Sumedha | ✅ Done — real checkpoint exported, `models/tcn.onnx` (LFS), median 9.4–9.7 ms over 5 runs, under the 10 ms budget |
| Calibration coverage test wired into the harness | Sumedha, Sikruti | ✅ Done (code), verified on real predictions — 0.53/0.56 (OxIOD), 0.29/0.42 (campus), below the ~0.68 target on both, see `docs/EVALUATION.md` |
| Model-only trajectory baseline (`ModelOnlyIntegrator`) | Sumedha | ✅ Done (code), exercised for real by `scripts/evaluate_model.py` — not yet wired into `dr-eval`/`scripts/run_eval.py` (Sikruti's area) |
| `prepare_window` / `resample_uniform` (shared training+live preprocessing) | Sumedha | ✅ Done — closes the "still open for M2" note above |
| Training on OxIOD, fine-tuning on our own walks | Sumedha | ✅ Trained on real OxIOD (8.55 h). Our own campus recordings (20 sessions, Sensor Logger) are evaluated held-out but **not yet folded into training** — a resume attempt hit a validation-set-mismatch bug (fixed in #80, unreviewed) before it could be re-run; RoNIN remains untried (`load_ronin` is still an access-check-only stub, #14) |

**Done when:** model-only integration beats PDR on held-out data **and** coverage is
calibrated at roughly 68% within 1σ **and** inference is under budget.

> Export to ONNX on the first checkpoint that trains at all, not at the end. It de-risks
> on-device inference and turns the latency claim into a measured number.

> **Real, measured results (2026-08-29, checkpoint epoch 113):** on OxIOD's own held-out
> trials, model-only integration clearly beats PDR — 0.57% mean drift vs PDR's 9.21%,
> winning 6/6 recordings. On our own held-out campus recordings (a domain the model never
> trained on), it beats PDR on 6/9, with the one recording long enough for drift-% to be a
> meaningful metric (`block_C`, 330.8 m) won clearly: 28.0% vs PDR's 70.9%. Inference is
> under the 10 ms budget for real, on the target laptop. Calibration coverage is measured
> but under the ~0.68 target on both domains (overconfident, more so out-of-domain). Full
> numbers, config, and honesty notes: `docs/EVALUATION.md`. Not claimed: single-digit
> drift on our own real-world data, or calibration at target on either domain — see that
> doc's "Not claimed" line (AGENTS.md: "do not invent numbers").

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
