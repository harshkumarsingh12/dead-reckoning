# Agent rules

Constitution for any AI coding agent working in this repository — Claude Code, Copilot,
Cursor, or a human pasting into a chat window. Everything here applies to humans too;
none of it is AI-specific except where it says so.

Read this, then [docs/CONVENTIONS.md](docs/CONVENTIONS.md), before writing a line.

---

## Non-negotiable

1. **Never make a failing test pass by weakening it.** No deleted assertions, no
   loosened tolerances, no `.skip`, no widened `pytest.approx`, no `try/except` around a
   failure. Every assertion in `tests/` is an acceptance criterion copied from
   `docs/BUILD_PLAN.md`. If a test is wrong, say so and stop — do not quietly fix it.

2. **Never remove a `strict=True` xfail marker without implementing the thing.** The
   xfail ledger is this project's burndown. Deleting a marker to make CI green is
   claiming a milestone you did not reach, and CI reports the count per owner, so it
   will be noticed.

3. **`src/dr_core/types.py` is frozen.** So is its mirror `apps/web/src/types.ts`.
   Changing either breaks several people at once. Announce it in the group chat, tag the
   affected owners, and change both sides in the same PR.

4. **Training and live import the same preprocessing.** If you write a resample, a unit
   conversion, or a gravity alignment anywhere outside `dr_core.preprocess`, you have
   created the exact bug that module exists to prevent — and it will surface as an
   unexplained live-demo underperformance, not as an error.

5. **Never restamp a timestamp.** Every timestamp is assigned by the sensor, at capture,
   in the boot-monotonic domain. Not on arrival, not on write, not on send. This is
   invisible when you get it wrong and it destroys everything downstream.

6. **Never commit secrets, datasets, tiles, or checkpoints.** They are gitignored and
   the pre-commit hook plus `pr-hygiene.yml` will block them. Once a large blob is in
   the history, every clone pays for it forever.

7. **No AI attribution in git.** Do not add `Co-Authored-By` trailers naming an AI tool,
   do not mention Claude / Copilot / any model in a commit message or PR body, and do
   not set an AI as author or committer. Commits are authored by the human who ran the
   agent. This is a hard requirement, not a style preference.

8. **`docs/BUILD_PLAN.md` is the arbiter.** Behaviour that diverges from the spec means
   the spec changes in the same PR, with the reason stated. Do not change behaviour and
   leave the plan behind.

---

## Working style

- **Read before writing.** Check `src/dr_core/types.py` before inventing a shape, and
  check whether the function you are about to write already exists as a stub with an
  owner's name on it.
- **Stay in your area.** The ownership table is in [CONTRIBUTING.md](CONTRIBUTING.md).
  Need a change outside it — ask the owner; do not edit it yourself.
- **Small, reviewable steps.** Commit every time something works, not when everything
  works. A clean progressive history is explicitly scored; a single end-of-day dump is
  not.
- **Physics beats parameters.** ZUPT, ZARU and the frame invariants hold regardless of
  what the model does. When a learned component and a physical constraint disagree,
  suspect the learned component first.
- **When uncertain, fail loudly.** Raise, log, keep the gate closed. A silently dropped
  measurement or a swallowed exception is how a demo-day bug becomes undiagnosable.
- **Do not invent numbers.** If you have not run the eval, say you have not run the
  eval. Reporting a plausible-looking drift figure that was never measured is worse than
  reporting nothing.

---

## Conventions

Full statement in [docs/CONVENTIONS.md](docs/CONVENTIONS.md). The short version:

- Python 3.11, ruff-formatted, `mypy --strict` clean. Type every signature.
- Timestamps are `int64` **nanoseconds**, boot-monotonic. Never float seconds.
- Angles are **radians**. Degrees only for lat/lon and for text a human reads.
- Vectors carry their frame in the name: `v_world`, `v_dev`, `a_body`. A bare `v` gets
  a review comment.
- World frame is ENU: x East, y North, z Up. `psi = 0` faces East, CCW positive.
- Error-state ordering is fixed: `[dpx, dpy, dvx, dvy, dpsi, db_g, ds]`.
- Conventional commits: `feat:` `fix:` `docs:` `test:` `chore:` `refactor:` `perf:`
  `ci:`. The PR title is checked by CI because squash-merge uses it as the commit
  message.
- Random seed is `26168`, via `np.random.default_rng`, never the global functions.

---

## Definition of done

Before opening a PR:

```bash
ruff check . && ruff format --check . && mypy && pytest -q
```

Plus: new behaviour has a test; if you implemented an acceptance criterion you removed
its xfail marker; if behaviour diverges from the build plan, the plan changed too.

---

## Things that look fine and are not

A short list of the mistakes most likely to be made confidently in this codebase.

| Looks reasonable | Why it is wrong |
|---|---|
| Stamping a sample when the server receives it | Destroys clock alignment. Silent. |
| Extrapolating the dot forward to hide the 300 ms lag | Makes it cut corners; reads as broken, and is dishonest |
| Fusing the learned velocity in the world frame | Loses the `dh/dpsi` term, so heading stops being corrected and the path bends through turns |
| Splitting train/test by window | Leaks context between the sets; every metric afterwards is fiction |
| Accepting a magnetometer reading on magnitude alone | Indoor disturbances rotate the field at near-normal strength |
| Letting the velocity scale `s` adapt while GPS is off | Scale and speed are not separable without GPS |
| `import torch` at the top of a live-path module | The demo laptop does not install the `[ml]` extra; the gateway stops starting |
| Making the raw-integration baseline "a bit better" | It is there to show the problem. A tuned strawman is dishonest framing |
| Adding a launcher-icon placeholder, a mock data file, a scratch notebook | Someone owns that; leave the seat empty |

---

## If you are an AI agent

- **Do not touch files outside the area you were asked about.** This repo is split
  across six owners on purpose; a helpful drive-by fix in someone else's module creates
  a merge conflict they did not ask for.
- **Do not write to `tests/` to make your own change pass.** Propose the test change and
  explain it. A test written by the same pass that broke it is not evidence.
- **Say what you did not verify.** "I did not run the Android build" is useful.
  "Everything works" when you ran nothing is not.
- **Prefer a stub with a clear `NotImplementedError` and an owner tag over a plausible
  guess.** An empty seat is honest; a half-right implementation with someone else's name
  implicitly on it is not.
