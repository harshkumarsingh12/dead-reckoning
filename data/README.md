# data/

**Nothing in this directory is committed** except this file and one small golden run.
Everything else is gitignored — see `.gitattributes` for why (LFS free tier is 1 GB,
and blowing it mid-hackathon is a self-inflicted outage).

## Layout

```
data/
├── ronin/      public dataset — scripts/fetch_datasets.py --dataset ronin
├── oxiod/      public dataset — scripts/fetch_datasets.py --dataset oxiod
├── own/        our own recordings, written by dr_core.io.SessionWriter
│   ├── outdoor/    strong-GPS walks — GPS supplies the training labels
│   └── loops/      surveyed indoor loops — the evaluation truth and demo course
└── golden/     ONE small recording, committed, used as the demo replay fallback
```

## Getting the public datasets

Both need an **access request that can take days**. Send both on day one.

```bash
python scripts/fetch_datasets.py --dataset ronin --info
python scripts/fetch_datasets.py --dataset oxiod --info
```

## Recording our own

Owner: Sumedha (data plan), Harsh (the app).

- **Outdoor, strong GPS** — the fine-tuning set. GPS position and velocity are the
  labels. Walk with GPS on; run with GPS off.
- **Indoor loops** — the evaluation set. Pick a corridor rectangle, physically measure
  the corner points, mark the start. Those measurements are your ground truth; a loop
  with no measured corners produces numbers nobody can defend.
- **Vary**: carry position (hand, pocket), walking speed, and phone. A model tuned on
  one person holding one phone one way is a model that fails on stage.

Every session begins with the calibration ritual, in this order:

1. Phone **still** for ~5 s (gyro bias).
2. **Figure-8** sweep for ~10 s (magnetometer hard iron).
3. A firm **tap or stomp** (the sharp-motion event that verifies clock alignment).

Skipping step 3 costs nothing today and makes a timing bug undiagnosable later.
