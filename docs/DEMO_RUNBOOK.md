# Demo runbook

The demo is engineered to be un-killable. Everything here exists because a specific
thing has killed a specific hackathon demo before.

Owners: **Tanmay** (runbook, delivery) · **Sumedha** (numbers, Q&A) · **Harsh** (the
stack on the day).

---

## The one-line principle

**There is no venue network in the loop, anywhere.** Not for tiles, not for transport,
not for fonts. The laptop's Wi-Fi is switched off during the demo, deliberately, and
everything still works. That is a property of the build, not a hope.

---

## T-minus one week

- [ ] Build the offline tiles: `python scripts/make_tiles.py --bbox ... --out tiles/kiit.mbtiles`
- [ ] **Verify them with the laptop's Wi-Fi physically off.** Tiles that were silently
      served from a browser cache during development will be blank on the day.
- [ ] Record the golden run on the actual demo loop, with the full calibration ritual
      and a clean GPS-off toggle. Commit it under `data/golden/`.
- [ ] Note the tiles file's SHA-256 here, so a corrupt copy is detectable:
      `TODO: <checksum>`
- [ ] Survey the demo loop: measure the corners, mark the start on the floor. Those
      measurements are the ground truth behind the number you will quote.

## T-minus one day

- [ ] Full dry run, end to end, on the real loop, with Wi-Fi off.
- [ ] Full dry run of the **replay path**. Rehearse switching to it until it is boring.
- [ ] Charge everything. The phone streams sensors at 200 Hz with the screen on; budget
      accordingly and bring a power bank.
- [ ] Second phone flashed with the same APK, as a spare.
- [ ] Export the backup slide: ATE / RTE / drift against both baselines.

## T-minus one hour

- [ ] Laptop Wi-Fi **off**. Phone hotspot on. Laptop joined to the hotspot.
- [ ] `python -m services.gateway --tiles tiles/kiit.mbtiles --model models/tcn.onnx`
- [ ] `curl http://127.0.0.1:8000/healthz` → `tiles_loaded: true`, `model_loaded: true`.
      Do not skip this. It is the difference between finding out now and finding out on
      stage.
- [ ] Open the UI, confirm tiles render and the socket connects.
- [ ] Run the calibration ritual on the phone: still 5 s → figure-8 10 s → firm tap.
- [ ] Walk one lap as a warm-up and check the drift number is in the expected range.

---

## The scripted three-minute arc

Cause and effect, visible each time. Do not narrate what the judges can see; let the
screen do it and explain the mechanism underneath.

**1 · Walk the marked loop. Toggle GPS off at a marked corner.**
The uncertainty ellipse starts growing. Say what it means once: *"from here on, nothing
but the phone's own motion sensors."*

**2 · Stop for ten seconds mid-walk.**
The ZUPT lamp fires, the ellipse tightens, the drift counter freezes. This is the most
legible moment in the demo — the mechanism working, on screen, on cue. Give it the
silence to land.

**3 · Move the phone from hand to pocket, while walking.**
The track holds. That is carry-position robustness, and it comes from the
heading-agnostic model frame plus augmentation — worth one sentence, no more.

**4 · Return to the marked start.**
The loop-closure error in metres and the drift % are the closing shot, with the
raw-integration baseline dot having spiralled off the map in parallel the entire time.

**Always on screen throughout:** the telemetry strip — per-channel NIS with bounds, the
ZUPT/ZARU lamp, the magnetometer gate verdict, the model's current 1σ, the heading
source — and the live drift-% counter.

**The moment the walk ends:** the auto-generated post-run panel appears by itself
(error-vs-time strip chart, error CDF, loop-closure error, drift %). It is worth more
than any slide precisely because it visibly was not prepared in advance.

---

## When something goes wrong

Rehearse these. A calm recovery reads as competence; a scramble reads as a broken
project.

| Symptom | Do this |
|---|---|
| Socket drops mid-walk | The UI reconnects by itself. Keep walking, keep talking. |
| Phone loses the hotspot | Stop, reconnect, restart the run. Do not debug on stage. |
| Anything at all looks wrong in the first 20 s | Switch to the golden-run replay. Say plainly that it is a replay — it goes through the identical pipeline, and saying so is both true and more impressive than pretending. |
| Tiles blank | The map is scenery; the numbers are the result. Carry on and point at the drift counter. |
| Laptop dies | Backup slide with the ATE / RTE / drift table against both baselines. |

**Never say "it worked five minutes ago".** Switch to the fallback and keep the story
moving.

---

## Pre-briefed Q&A

Four predictable questions. Rehearse crisp answers; do not read them.

**"Why not just double-integrate the accelerometer?"**
Error compounds faster than linearly — metres within seconds. That is the red dot
spiralling off the screen, and it is on the map deliberately. A learned model regresses
velocity directly, so the error stays bounded.

**"Why an EKF plus a neural network, rather than end-to-end deep learning?"**
The filter fuses independent physical constraints — zero velocity when stopped, heading,
GPS when it appears — with principled uncertainty. It degrades gracefully and it is
inspectable, which is what the telemetry strip is showing you. The network does the one
thing networks are genuinely best at: turning a raw IMU window into a velocity estimate.

**"What happens when the magnetometer fails indoors?"**
The triple gate rejects it — magnitude, dip angle, and the innovation test. Magnitude
alone is not enough, because indoor disturbances rotate the field while leaving its
strength near normal. Heading stays observable through the velocity update's heading
term, ZARU pins the gyro bias while you are stopped, and GPS course corrects on
reacquire. You can watch the gate reject in real time on the strip.

**"How do you know your uncertainty is honest?"**
The covariance is trained under a Gaussian NLL objective, verified by a 1σ coverage test
on held-out trajectories, and monitored live as NIS against its chi-square bounds — the
strip on screen right now. The ellipse is not decoration.

---

## What we will not claim

Deciding this in advance is what keeps you out of trouble under questioning.

- Not a vehicular system. This is pedestrian dead reckoning; the motion model and ZUPT
  logic would change materially for a car in a tunnel.
- Not tested at scale. Our own recordings are a handful of walks on one campus.
- Map matching (if demoed) is a **stretch** feature and is presented as one.
- The drift number is from **our** loop with **our** phones. Say which loop and which
  phone every time you quote it.

An honest boundary, stated first, is worth more than a claim a judge gets to puncture.
