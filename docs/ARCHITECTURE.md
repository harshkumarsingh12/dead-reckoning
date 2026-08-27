# Architecture

The full design rationale is in [BUILD_PLAN.md](BUILD_PLAN.md), which is the spec. This
page is the map: what the layers are, which way the dependencies point, and where each
person's work lives.

---

## The chain

```
  Smartphone                                            apps/android/  (Kotlin)
  accelerometer · gyroscope · magnetometer · GPS        100–200 Hz
        │
        │  stamped ON DEVICE, at capture, in the boot-monotonic domain
        │  (never on arrival — see CONVENTIONS.md §3)
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [1] dr_core.preprocess          SHARED by training AND live      │
  │      resample · units · gravity alignment · gyro bias ·           │
  │      magnetometer hard-iron calibration (10 s figure-8)           │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [2] dr_core.ahrs                Madgwick + magnetometer gate     │
  │      gyro + accel + gated mag  →  quaternion                      │
  └───────────────────────────────────────────────────────────────────┘
        │                                    │
        │ gravity-aligned window             │ heading rate, gated mag heading
        ▼                                    │
  ┌───────────────────────────────────────┐  │
  │  [3] dr_core.models                   │  │
  │      causal TCN, window ≤ 2 s,        │  │
  │      hop ≤ 200 ms                     │  │
  │      → v_dev + NLL-trained covariance │  │
  │      runtime: ONNX int8               │  │
  └───────────────────────────────────────┘  │
        │                                    │
        ▼                                    ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [4] dr_core.timebase.ReorderBuffer      ~300 ms lagged timeline  │
  │      late GPS and velocity inserted at CAPTURE time               │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [5] dr_core.fusion              2D error-state Kalman filter     │
  │      nominal:  p, v, psi        error: dp, dv, dpsi, db_g, ds     │
  │      updates:  device-frame velocity (carries dh/dpsi) · ZUPT ·   │
  │                ZARU · gated mag heading · GPS (+ scale learning)  │
  │      every update chi-square gated, per-channel NIS logged        │
  └───────────────────────────────────────────────────────────────────┘
        │
        ├──────────────► dr_core.eval        ATE · RTE · drift · NEES · coverage
        │                                    post-run report, auto-generated
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [6] services/gateway            FastAPI, fully offline           │
  │      WS /ingest · WS /live · GET /tiles (local MBTiles) ·         │
  │      POST /control/gps · replay                                   │
  └───────────────────────────────────────────────────────────────────┘
        │  phone hotspot, no venue network anywhere in the loop
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  [7] apps/web                    React + Leaflet                  │
  │      estimated dot · uncertainty ellipse · true path ·            │
  │      raw-integration baseline (visibly diverging) ·               │
  │      GPS-off toggle · live drift % · telemetry strip              │
  └───────────────────────────────────────────────────────────────────┘

  Stretch, gated on the core chain hitting its metrics:
      map matching (OSMnx + leuvenmapmatching) · on-device ONNX inference
```

Layers 1–5 plus 6–7 are the spine. Map matching and on-device inference are **gated
stretch work**, not dependencies — that is a scope decision, and it is what stops the
project ending with six half-built features.

---

## Dependency direction

```
apps/web ──────► services/gateway ──────► dr_core ──────► numpy · scipy · onnxruntime
apps/android ──►                                 ▲
                                                 │
scripts/ ────────────────────────────────────────┘
```

**`dr_core` never imports from `services/` or `apps/`.** Enforced, not requested:
`tests/test_contract.py::test_dr_core_never_imports_services_or_apps` walks the AST of
every module under `src/dr_core` and fails the build on a violation.

The reason is concrete. `dr_core` is imported by the training pipeline, the eval
harness, and the live gateway. The moment it reaches back out to the gateway, training
grows a FastAPI dependency, the demo laptop's install grows a training dependency, and
the shared-preprocessing guarantee starts to rot.

A second rule, enforced the same way: **no module on the live path may import torch at
top level.** Training uses `[ml]`; the demo laptop installs the default extra. A stray
top-level `import torch` in `dr_core.models.runtime` would stop the gateway starting on
the one machine that matters.

---

## The two guarantees

### One shared preprocessing path

`dr_core.preprocess` is imported by `scripts/train.py` **and** by the gateway. There is
no second implementation to drift from.

This is the difference between a bug you can find and one you cannot. If training
resampled at 200 Hz and live ran at 100 Hz, nothing errors — the model just quietly
underperforms in the demo, with no stack trace and no obvious cause, and you lose an
evening to it.

### One frozen contract

`src/dr_core/types.py` and its TypeScript mirror `apps/web/src/types.ts` define every
structure that crosses a subsystem boundary. Everyone builds against those dataclasses
and hand-written mocks; real data is swapped in when it lands.

This is what lets six people work at once. The web UI does not wait for the ESKF. The
ESKF does not wait for the model. The model does not wait for the app.

Changing either file is a team decision, announced before the PR. CODEOWNERS makes both
require the repo owner's review on top of whoever else is touched.

---

## Where each subsystem lives

| Layer | Path | Owner | Milestone |
|---|---|---|---|
| Time, reorder buffer, session IO | `src/dr_core/timebase/`, `io/` | Sristee | M0 |
| Preprocessing, calibration | `src/dr_core/preprocess/` | Sristee | M1 |
| AHRS, magnetometer triple gate | `src/dr_core/ahrs/` | Sristee | M1 |
| Baselines | `src/dr_core/baselines/` | Sristee | M1 |
| Learned velocity model | `src/dr_core/models/` | Sumedha | M2 |
| Datasets | `src/dr_core/datasets/` | Sumedha | M0 |
| ESKF, ZUPT/ZARU, gating | `src/dr_core/fusion/` | Sikruti | M3 |
| Evaluation harness | `src/dr_core/eval/` | Sikruti | M0 |
| Gateway | `services/gateway/` | Harsh | M4 |
| Android streamer | `apps/android/` | Harsh | M4 |
| Map UI, socket client | `apps/web/src/map/`, `ws/` | Tanmay | M4 |
| Telemetry strip, design | `apps/web/src/telemetry/`, `ui/` | Akshit | M4 |

---

## Why an error-state filter

The nominal state is propagated with the full nonlinear equations; the filter runs on a
small error state where linearisation is exact where it matters — heading. ZUPT and ZARU
become clean pseudo-measurements on the error state rather than awkward special cases.

In 2D the extra machinery costs very little. And "why ESKF rather than a plain EKF?" is
a question worth being able to answer in one sentence when a judge asks it.

Reference: Solà, *Quaternion kinematics for the error-state Kalman filter* (2017). The
2D case here is a simplification of that machinery.

---

## Why the velocity update is fused in the device frame

The single most important wiring decision in the filter, and the easiest one to get
subtly wrong.

The measurement model is `h(x) = (1/s) · R(-psi) · v_world`. Because `h` depends on
`psi`, the Jacobian carries a `dh/dpsi` term — **so every velocity update also corrects
heading.** Heading drift therefore stops bending the path through turns, and heading
stays observable even when the magnetometer is being rejected indoors.

The world-frame alternative works, but `R` then has to be inflated by the
heading-uncertainty term, and heading gains nothing from the update. The device-frame
form is simpler and strictly better.

`test_device_frame_velocity_update_corrects_heading` is the guard: it starts the filter
with a deliberate heading error, feeds consistent velocity, and requires convergence.
Wired in the world frame, that test cannot pass.
