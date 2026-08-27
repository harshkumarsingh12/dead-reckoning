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

| Deliverable | Owner |
|---|---|
| Eval harness: ATE / RTE / drift | Sikruti |
| NIS / NEES logger | Sikruti |
| Clock-offset mapper, sharp-motion verification | Sristee |
| ~300 ms reorder buffer | Sristee |
| Session record format (read + write + replay) | Sristee |
| Frame unit tests including rotation-in-place | Sristee |
| **RoNIN / OxIOD access requested** | Sumedha |
| Repo, CI/CD, governance docs | Harsh |

**Done when:** one command turns a recording into a trajectory plot, error numbers, and
a timing-alignment check, all passing.

> The dataset access request is the single longest lead time in the project — approval
> can take days. It goes out on day one, in parallel with everything else. No amount of
> clever modelling recovers a week lost waiting for an email.

---

## M1 — Baselines and AHRS

**Target: PDR under 10% drift; the magnetometer gate demonstrably fires.**

| Deliverable | Owner |
|---|---|
| Shared preprocessing and calibration module | Sristee |
| AHRS orientation (Madgwick) | Sristee |
| Magnetometer hard-iron fit + magnitude/dip/innovation triple gate | Sristee |
| Raw double-integration baseline | Sristee |
| PDR baseline | Sristee |
| First own recordings: outdoor GPS walks + one surveyed indoor loop | Sumedha |
| Web UI shell against mock frames | Tanmay, Akshit |

**Done when:** end to end on a test loop, and the magnet test visibly trips the gate
without corrupting heading.

> M1 is Sristee-heavy by design — it is one coherent chain (calibrate → orient → gate →
> baseline) and splitting it across people would mean negotiating three interfaces to
> save nothing. Everyone else is on M0 and M2 work in parallel.

---

## M2 — Learned velocity

**Target: drift under 5%, calibrated sigma, inference under 10 ms.**

| Deliverable | Owner |
|---|---|
| Causal TCN with a Gaussian-NLL covariance head | Sumedha |
| Random-yaw and carry-position augmentation | Sumedha |
| Training on RoNIN/OxIOD, fine-tuning on our own walks | Sumedha |
| ONNX export, int8, benchmarked | Sumedha |
| Calibration coverage test wired into the harness | Sumedha, Sikruti |

**Done when:** model-only integration beats PDR on held-out data **and** coverage is
calibrated at roughly 68% within 1σ **and** inference is under budget.

> Export to ONNX on the first checkpoint that trains at all, not at the end. It de-risks
> on-device inference and turns the latency claim into a measured number.

---

## M3 — ESKF fusion

**Target: drift 3–5%, NIS consistent on every channel.**

| Deliverable | Owner |
|---|---|
| 2D ESKF: predict + inject/reset | Sikruti |
| Device-frame velocity update, with the `dh/dpsi` heading term | Sikruti |
| ZUPT and ZARU, driven by the stationary detector | Sikruti |
| Per-session velocity scale `s`, frozen when GPS drops | Sikruti |
| Chi-square gating on every channel, NIS logged | Sikruti |
| Reorder buffer wired into the live path | Sikruti, Sristee |

**Done when:** fused beats model-only on ATE and RTE, NIS is in bounds on all channels,
and a 10 s stop produces zero position creep.

---

## M4 — Hardened live demo

**Target: the headline number holds live, offline.**

| Deliverable | Owner |
|---|---|
| Android IMU streamer, capture-time stamping, WS uplink | Harsh |
| Gateway: ingest, broadcast, control, replay | Harsh |
| Offline MBTiles built and verified with Wi-Fi off | Harsh |
| Golden-run replay through the identical pipeline | Harsh |
| Map, dot, ellipse, GPS toggle, diverging baseline dot | Tanmay |
| Telemetry strip: NIS, ZUPT/ZARU lamp, mag gate, σ, heading source | Akshit |
| Auto post-run report panel | Akshit, Sikruti |
| Scripted 3-minute arc, rehearsed | Tanmay, Sumedha |

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
