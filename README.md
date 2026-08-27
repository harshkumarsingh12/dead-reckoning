# Dead reckoning without GPS

**Smart India Hackathon 2026 · Problem statement SIH26168 · ISRO**

Keeping an accurate position estimate when the satellites are gone — in a tunnel, in a
basement, in an urban canyon — using nothing but the phone already in your pocket.

[![CI · Python](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-python.yml/badge.svg)](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-python.yml)
[![CI · Web](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-web.yml/badge.svg)](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-web.yml)
[![CI · Android](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-android.yml/badge.svg)](https://github.com/harshkumarsingh12/dead-reckoning/actions/workflows/ci-android.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The problem, and why the obvious answer fails

GPS drops. The phone has an accelerometer and a gyroscope, so integrate the
acceleration twice and you have position. Everyone tries this once.

It does not work. Integrating raw acceleration compounds error faster than linearly —
**metres of drift within seconds**, and it spirals. That failure is not a bug to be
fixed with better filtering; it is what double integration does.

## The approach

Do not integrate acceleration. **Regress velocity directly.**

A small neural network reads a one-second window of inertial data and outputs planar
velocity, along with an honest estimate of how wrong it probably is. Velocity error
stays bounded, so position error grows roughly linearly instead of explosively. This is
the RoNIN / TLIO line of work, and it is the load-bearing decision in the whole design.

Around that sits a 2D **error-state Kalman filter** that fuses the learned velocity with
things physics guarantees:

- **Zero-velocity updates.** You stopped walking, so your velocity is exactly zero. Free,
  exact, and completely independent of the model.
- **Zero angular-rate updates.** You are standing still, so the gyroscope is reading
  pure bias. Measure it and pin it.
- **A triple-gated magnetometer.** Accepted only if field magnitude, dip angle, *and* the
  innovation test all pass. Indoor disturbances rotate the field while leaving its
  strength normal, so a magnitude check alone waves them straight through.
- **GPS, when it briefly appears.** Both a training label offline and an opportunistic
  reset online.

The learned velocity is fused **in the device frame**, not the world frame. That detail
is what puts a heading term in the measurement Jacobian, so every velocity update also
corrects heading — which is why the path does not bend through turns even with the
magnetometer switched off.

---

## Architecture

```
  phone sensors ──► preprocess ──► AHRS ──► learned velocity ──► reorder buffer
  (100–200 Hz)      (SHARED by     Madgwick   causal TCN,          ~300 ms lag,
   stamped at       training and   + mag      NLL covariance,      fuse at capture
   capture, boot-   live — one     triple     ONNX int8            time
   monotonic ns)    code path)     gate
                                                       │
                                                       ▼
                                          2D error-state Kalman filter
                                          ZUPT · ZARU · scale state · χ² gating
                                                       │
                                    ┌──────────────────┴──────────────────┐
                                    ▼                                     ▼
                          evaluation harness                    offline live UI
                          ATE · RTE · drift %                   Leaflet + local
                          NIS · NEES · coverage                 MBTiles, no internet
```

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the complete engineering
plan is [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md), which is the spec this repo is held
against.

---

## Targets

On a 100–300 m indoor loop with surveyed corner points.

| Metric | Acceptable | Strong |
|---|---|---|
| Drift (final error / distance) | < 5% | < 2–3% |
| RTE over 60 s | a few metres | 1–2 m |
| NIS / NEES | within bounds, all channels | same, across carry positions |
| Model coverage @ 1σ | ~68% | holds across carry positions |
| Inference per window | < 10 ms | on-device viable |
| Raw double integration | > 100% | *(the contrast, not a target)* |

Definitions and the running results log: [docs/EVALUATION.md](docs/EVALUATION.md).

**The baselines are always plotted alongside.** Raw double integration spiralling off
the map as a second dot, live, is the most persuasive thing on screen — it shows the
problem being solved rather than asserting it was.

---

## What makes this hard to break

Each of these addresses a specific, known way dead reckoning fails.

| Failure mode | What we do about it |
|---|---|
| Double-integration drift | Bounded-error learned velocity, not integration |
| Heading drift bending the path | Device-frame velocity update carries a `dh/dpsi` term |
| Magnetic disturbance indoors | Magnitude **and** dip **and** innovation gate |
| Over-confident uncertainty | NLL-trained covariance, 1σ coverage test, live NIS |
| Clock-domain mismatch | One clock mapped once, sharp-motion verification, reorder buffer |
| Laggy or corner-cutting dot | Causal model, lagged timeline, render decoupled from capture |
| Coordinate-frame bugs | Three synthetic invariants in CI, including rotation-in-place |
| Dead venue Wi-Fi | Offline tiles, phone hotspot, golden-run replay |
| Scope creep | Map matching and on-device inference are **gated stretch**, not dependencies |

---

## Quickstart

Python **3.11** specifically — see [CONTRIBUTING.md](CONTRIBUTING.md) for why.

```bash
conda create -n sih26168 -c conda-forge --override-channels python=3.11 -y
conda activate sih26168

git clone https://github.com/harshkumarsingh12/dead-reckoning.git
cd dead-reckoning
pip install -e ".[dev]"
pre-commit install

pytest -q          # 17 passed, 29 xfailed  — the xfails are the work remaining
```

Run the live stack:

```bash
make serve                                  # gateway on 0.0.0.0:8000
cd apps/web && npm ci && npm run dev        # UI at http://localhost:5173
```

Evaluate a recording:

```bash
python scripts/run_eval.py data/loops/corridor_01.jsonl.gz --model models/tcn.onnx --no-gps
```

---

## Repository map

```
src/dr_core/        the shared library — imported by BOTH training and live
  types.py            ★ the frozen contract
  timebase/           clock mapping, ~300 ms reorder buffer
  preprocess/         ★ the one preprocessing path
  ahrs/               orientation + magnetometer triple gate
  models/             causal TCN, NLL covariance, ONNX runtime
  fusion/             2D ESKF, ZUPT/ZARU, χ² gating, NIS/NEES
  baselines/          raw double integration, PDR
  datasets/           RoNIN / OxIOD / our own recordings
  eval/               ATE, RTE, drift, coverage, post-run report
  io/                 session record format

services/gateway/   FastAPI: WS ingest, local MBTiles, replay, live broadcast
apps/android/       Kotlin IMU streamer — capture-time stamping over the hotspot (APP.md)
apps/web/           React + Leaflet — the dot, the ellipse, the telemetry strip (WEB.md)
scripts/            thin CLI entry points
tests/              frame invariants, timing, gating, the frozen contract
docs/               the plan, the conventions, the runbook
```

The Android app and the web UI each have their own doc going deeper than this map does:
[APP.md](APP.md) (requirements, wire format, permissions, current status) and
[WEB.md](WEB.md) (same, for the browser side).

Two structural guarantees, both enforced by tests rather than by good intentions:

- **`dr_core` never imports from `services/` or `apps/`.** Dependencies point inward.
- **Nothing on the live path imports torch.** Training uses the `[ml]` extra; the demo
  laptop does not need a 2 GB download.

---

## Development status

Scaffolded. The structure, the contract, the tests and the pipeline are in place; the
algorithms are being built by their owners.

Every unimplemented acceptance criterion is a real test, marked `xfail(strict=True)`
with its milestone and owner in the reason string. Implementing the feature makes the
test pass, which turns CI **red** until the marker is removed — so nothing can be
silently claimed as done. CI posts the remaining count per owner on every run.

Milestones and exit criteria: [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Team

| | Area |
|---|---|
| **Harsh Kumar Singh** ([@harshkumarsingh12](https://github.com/harshkumarsingh12)) | Android app, gateway, CI/CD, release |
| **Sristee Shrivastava** ([@srshriv](https://github.com/srshriv)) | Time and clocks, preprocessing, AHRS, baselines, transport security |
| **Sumedha** ([@sumedhag28](https://github.com/sumedhag28)) | Learned velocity model, datasets, results |
| **Sikruti Mahapatra** ([@hoursgotviral-dev](https://github.com/hoursgotviral-dev)) | ESKF fusion, gating, evaluation harness |
| **Tanmay** ([@7tanmay7](https://github.com/7tanmay7)) | Map UI, socket client, demo delivery |
| **Akshit** ([@Akshit19-05](https://github.com/Akshit19-05)) | Telemetry strip, design system, deck |

Full ownership table with paths: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## References

The method is not invented here, and saying so is the point — it stands on published
work rather than on a good feeling.

- Herath, Yan, Furukawa. *RoNIN: Robust Neural Inertial Navigation in the Wild*, ICRA 2020 — heading-agnostic learned velocity regression, and the dataset.
- Liu et al. *TLIO: Tight Learned Inertial Odometry*, IEEE RA-L 2020 — learned displacement with an NLL-trained covariance, fused in an EKF. The template for our fusion.
- Chen et al. *OxIOD: The Dataset for Deep Inertial Odometry* — multi-carry smartphone IMU with Vicon ground truth.
- Chen et al. *IONet: Learning to Cure the Curse of Drift in Inertial Odometry*, AAAI 2018.
- Solà. *Quaternion kinematics for the error-state Kalman filter*, 2017 — the ESKF formulation.
- Madgwick. *An efficient orientation filter for inertial and inertial/magnetic sensor arrays*, 2010 — the AHRS.

---

## License

MIT — see [LICENSE](LICENSE).
