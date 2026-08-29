# Evaluation

Measured, not vibed. "Strong prototype" is a number on a known loop.

The harness is built at **M0**, before there is anything to evaluate. That ordering is
deliberate: a scoring method written after you have seen the score tends to get tuned
until it agrees with the score.

Owner: **Sikruti** (metrics, consistency) and **Sumedha** (model calibration).

---

## Running it

```bash
python scripts/run_eval.py data/loops/corridor_01.jsonl.gz --model models/tcn.onnx --no-gps
```

Writes to `reports/<run_id>/`:

| File | What |
|---|---|
| `trajectory.png` | Estimate, truth, and every baseline on one set of axes |
| `error_time.png` | Error vs time strip chart |
| `error_cdf.png` | Error CDF |
| `nis.png` | Per-channel NIS against its chi-square bounds |
| `report.json` | The machine-readable `RunReport` |

Exit codes are a contract, so CI can depend on them: `0` every target met, `1` a target
missed, `2` bad usage.

---

## Metrics

### Accuracy

| Metric | Definition |
|---|---|
| **ATE** | Absolute Trajectory Error. RMSE of estimated vs true position after SE(2) alignment. Report the **unaligned** number too — the demo starts on a physically marked spot, so the start point is genuinely known and alignment flatters the result. |
| **RTE** | Relative Trajectory Error over a fixed 60 s window. Reflects the drift *rate*, so it stays comparable across runs of different lengths. Quote this when comparing two models on differently sized loops. |
| **Final error** | Distance between the estimated and true endpoints. On a closed loop this is the loop-closure error — the closing shot of the demo. |
| **Drift %** | `final_error / distance_travelled`. **The headline number.** |

### Honesty

Accuracy alone is half the story. A filter can be accurate and still be lying about how
confident it is, and a judge who asks "how do you know your uncertainty is honest?"
deserves a real answer.

| Metric | Definition | Expected |
|---|---|---|
| **NIS** | Normalised Innovation Squared, per measurement channel. Needs no ground truth, so it runs live. | Mean near the channel's degrees of freedom, inside the two-sided chi-square bounds |
| **NEES** | Normalised Estimation Error Squared against truth. Offline only. | Within bounds |
| **Coverage @ 1σ** | Fraction of held-out velocity errors inside the model's own 1σ, per axis. | ~0.68 |

**Coverage is a gate, not a report line.** The model's covariance becomes the filter's
`R` directly. An over-confident covariance silently poisons fusion and makes the
on-screen ellipse indefensible; an under-confident one throws away information the model
actually has. The covariance is not allowed near the filter until this test passes.

### Baselines, always plotted

Never report a number alone.

| Baseline | Expected | Why it is there |
|---|---|---|
| **Raw double integration** | **> 100%** drift, spirals | Shows the problem being solved rather than asserting it was. Kept deliberately naive — making it "a bit better" is dishonest framing and weakens the contrast. |
| **PDR** (step × stride + heading) | < 10% | The honest classical alternative. This is the number the learned model has to beat before anyone can claim the ML is earning its place. |

If PDR alone hits target on our loop, that is a real and slightly uncomfortable finding.
Report it. A team that noticed and said so is more credible than one that did not check.

---

## Targets

On a 100–300 m indoor loop with surveyed corner points.

| Metric | Acceptable | Strong |
|---|---|---|
| Drift (final error / distance) | < 5% | < 2–3% |
| RTE over 60 s | a few metres | 1–2 m |
| NIS / NEES | within bounds on all channels | same, across carry positions |
| Model coverage @ 1σ | ~68% | holds across carry positions |
| Inference per window | < 10 ms (laptop, ONNX) | on-device viable |
| Raw-integration baseline | > 100% | (contrast, not a target) |

---

## Results log

Append a row per tagged milestone. Never delete a row — a result that got worse is the
most useful row in the table, and quietly dropping it is how a team stops learning from
its own history.

| Date | Tag | Loop | Config | Drift % | ATE (m) | RTE 60 s (m) | Coverage @1σ | NIS OK | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-29 | M2-v1 | OxIOD held-out, 6 trials (handheld+pocket) | `tcn.pt` epoch 113, seed 26168 | 0.57 mean / 0.49 median | 2.09 | 2.91 | x=0.53 y=0.56 | n/a (model-only) | PDR on the same 6: drift 9.21% mean, ATE 15.51 m, RTE 15.81 m. Model beats PDR 6/6. `reports/m2_eval_epoch113.json`. |
| 2026-08-29 | M2-v1 | Our own campus recordings, all held-out, 9 (GPS-quality-filtered from 21) | same checkpoint -- **no campus data in this training run** | 73.83 mean / 77.42 median | 14.10 | 7.81 | x=0.29 y=0.42 | n/a (model-only) | PDR: drift 87.19% mean, ATE 9.79 m, RTE 9.42 m. Model beats PDR 6/9; the one substantial walk (`block_C`, 330.8 m) is the clearest signal: model 28.0% vs PDR 70.9%. Most excluded recordings were short (<60 m) with a poor origin GPS fix (>30 m accuracy) -- see `scripts/evaluate_model.py --max-origin-gps-accuracy-m`. `reports/m2_eval_campus_all.json`. |

### M2-v1, in plain terms

Trained on OxIOD (handheld + pocket) only -- the training run intended to also include our
own campus recordings hit a data-loading issue on the training machine and was not
re-run before this snapshot; see `docs/ROADMAP.md` for the current plan on that.

- **On OxIOD's own held-out trials** (data from the same source as most of training):
  model-only integration clearly beats PDR -- 0.57% mean drift vs PDR's 9.21%, model
  wins all 6/6 recordings. Calibration coverage (0.53 / 0.56) is close to the ~0.68
  target but on the overconfident side, not yet there.
- **On our own campus recordings** (a domain the model never trained on at all): model
  beats PDR on 6 of 9 held-out recordings. The headline drift-% numbers (73.83% mean)
  look worse than the "<5%" target, but that average is dominated by very short walks
  (most under 60 m) where drift-% is a noisy metric regardless of which method is used
  -- PDR does just as badly on the same short recordings (87.19% mean). The one walk
  long enough for the metric to be meaningful (`block_C`, 330.8 m) shows the model
  winning clearly: 28.0% vs PDR's 70.9%. Calibration (0.29 / 0.42) is further from
  target here, consistent with evaluating on a domain the model never saw in training.
- **Latency**: exported to ONNX (int8) and benchmarked on the actual laptop this repo's
  budget target refers to (not the Colab GPU training machine) -- median **9.4-9.7 ms**
  across 5 repeated runs of 300 iterations each, consistently under the 10 ms budget
  but with a thinner margin than the architecture's first (untrained-weights) benchmark
  suggested. `pytest -m ml -k budget` passes for real against the committed
  `models/tcn.onnx`.
- **Not claimed:** a single-digit-percent drift number on our own real-world data, or
  calibration at the 0.68 target on either domain. Both are honest gaps, not hidden
  ones -- see `docs/ROADMAP.md`'s M2 table for what's open and why.

---

## Rules that keep the numbers real

1. **Split by trajectory, never by window.** Windows from one walk share so much context
   that a window-level split leaks the answer and every number afterwards is fiction.
   `dr_core.datasets.split_by_trajectory` exists so nobody has to remember this at 3 a.m.
2. **Held-out means held-out.** The test loop is not tuned against. If you looked at test
   results and then changed a hyperparameter, you were using a validation set — say so.
3. **Report the config with the number.** Carry position, phone, whether GPS was on, and
   the commit. A drift figure without those is not reproducible and therefore is not a
   result.
4. **The demo number is the honest number.** Whatever appears on screen during the live
   run is what gets quoted. Not the best of five runs.
