# data/golden/

The ONE recording this project deliberately commits (see `data/README.md` and
`.gitignore` for why everything else under `data/` is not). This is the demo's
fallback: if anything looks wrong in the first 20 seconds of a live run, switch to
replaying this file through the identical pipeline (`scripts/replay.py`) rather than
debugging on stage — see `docs/DEMO_RUNBOOK.md`.

## `campus25_loop_01.jsonl.gz`

Recorded live on 2026-08-29 at KIIT Campus 25 (the CS campus), through the real
Android app → gateway `/ingest` → `--record-dir` path (no synthetic data).

| | |
|---|---|
| Device | V2321, carried in hand |
| Duration | 140.2 s |
| Samples | 59,144 IMU, 18 GPS |
| Loop closure | start and end GPS fixes are 9.9 m apart |
| **SHA-256** | `08ac031b09feab61a5396e4b0d58e23d036a315a707f4ccc86c8f3d409e91e57` |

**Known gap:** no calibration events (`calib_still_start`, `calib_figure8_start`,
`tap`) were recorded — the calibration ritual wasn't run before this walk. That means
the sharp-motion clock-alignment check can't be verified on this recording, and there
is no gyro-bias / hard-iron calibration data in it. Neither blocks replay (which only
needs real IMU/GPS timing), but both matter if this recording is ever used for offline
calibration or training. Two other takes from the same session were discarded: one had
zero GPS fixes (a firewall-blocked attempt, fixed mid-session — see the inbound rule
needed for `/ingest` to be reachable from another device), the other closed its loop
to a wider 16.7 m gap.
