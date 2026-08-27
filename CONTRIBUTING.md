# Contributing

Six people, four languages, one deadline. This document exists so nobody blocks anybody
and nobody's work gets clobbered. Read it once before your first commit.

---

## Setup

### Python (everyone needs this)

**Python 3.11 specifically.** Not 3.12, not 3.13. `torch` and `onnxruntime` wheels are
unreliable above 3.11, and everyone must share one interpreter or the shared
preprocessing guarantee stops meaning anything.

```bash
conda create -n sih26168 -c conda-forge --override-channels python=3.11 -y
conda activate sih26168

git clone https://github.com/harshkumarsingh12/dead-reckoning.git
cd dead-reckoning

pip install -e ".[dev]"
pre-commit install

pytest -q          # should be green before you change anything
```

> The `-c conda-forge --override-channels` is not optional decoration — Anaconda's
> default channels require accepting their Terms of Service, and conda-forge does not.

A green first run looks like **`17 passed, 29 xfailed`**. The xfails are the work
remaining; see [The xfail ledger](#the-xfail-ledger) below.

Everyday commands, identical on every platform:

```bash
make test            # Windows:  .\tasks.ps1 test
make all             #           .\tasks.ps1 all      -- what CI runs
make frames          #           .\tasks.ps1 frames   -- the coordinate invariants
```

### Web (Tanmay, Akshit)

```bash
cd apps/web
npm ci
npm run dev          # needs the gateway running: make serve
```

Node 22+.

### Android (Harsh)

```bash
cd apps/android
cp local.properties.example local.properties   # then edit the SDK path
./gradlew :app:assembleDebug
```

**JDK 21.** Not 25 — Gradle does not support it yet and the error it produces is obscure
enough to cost an afternoon. Android Studio's bundled JBR has moved to 25, so point
`JAVA_HOME` at a real Temurin 21 install.

Note the escaped colon in `local.properties` (`C\:/Users/...`). Android lint fails the
build if you get it wrong, which is at least an honest failure.

---

## Who owns what

Work is split so that six people can be in the repo at once without touching the same
files. **Stay inside your area.** If you need a change outside it, ask the owner rather
than editing it yourself — a drive-by fix in someone else's module creates a merge
conflict they did not ask for.

| # | Area | Paths | Primary | Backup |
|---|---|---|---|---|
| 1 | Repo infra, CI/CD, release | `.github/`, `Makefile`, `tasks.ps1`, `pyproject.toml` | **Harsh** | Sristee |
| 2 | Android IMU streamer | `apps/android/` | **Harsh** | Akshit |
| 3 | Gateway: ingest, tiles, replay | `services/gateway/` | **Harsh** | Tanmay |
| 4 | Time: clocks, reorder buffer, session IO | `src/dr_core/timebase/`, `io/` | **Sristee** | Harsh |
| 5 | Preprocessing, calibration, AHRS, mag gate | `src/dr_core/preprocess/`, `ahrs/` | **Sristee** | Sumedha |
| 6 | Baselines: raw integration, PDR | `src/dr_core/baselines/` | **Sristee** | Sikruti |
| 7 | Transport hardening, secret scanning | `pr-hygiene.yml`, gateway auth | **Sristee** | Harsh |
| 8 | Learned velocity model, ONNX | `src/dr_core/models/`, `scripts/train.py` | **Sumedha** | Sikruti |
| 9 | Datasets, data plan | `src/dr_core/datasets/`, `data/` | **Sumedha** | Sristee |
| 10 | ESKF, ZUPT/ZARU, gating | `src/dr_core/fusion/` | **Sikruti** | Sristee |
| 11 | Eval harness, metrics, report | `src/dr_core/eval/`, `scripts/run_eval.py` | **Sikruti** | Sumedha |
| 12 | Map, dot, ellipse, socket client | `apps/web/src/map/`, `ws/` | **Tanmay** | Akshit |
| 13 | Telemetry strip | `apps/web/src/telemetry/` | **Akshit** | Tanmay |
| 14 | Design system, report UI, deck | `apps/web/src/ui/`, `docs/` visuals | **Akshit** | Tanmay |
| 15 | Demo runbook, rehearsal, Q&A | `docs/DEMO_RUNBOOK.md`, the deck | **Tanmay** + **Sumedha** | everyone |

`.github/CODEOWNERS` encodes this, so GitHub requests the right reviewer automatically.

**Two splits that are not obvious, and why:**

- **Sikruti has both the eval harness (M0) and the ESKF (M3).** NIS and NEES live in
  both, so one person owning both means no interface to negotiate — and M0 doubles as a
  warm-up for M3.
- **Sristee has the baselines.** PDR needs her AHRS heading and her step detection.
  Splitting them would create a contract between two files one person could just own.

---

## The frozen contract

`src/dr_core/types.py` and its TypeScript mirror `apps/web/src/types.ts` define every
structure that crosses a subsystem boundary.

**Build against them, with hand-written mocks, from day one.** The web UI does not wait
for the ESKF. The ESKF does not wait for the model. That is the entire point.

Changing either file is a team decision: announce it in the group chat, tag the affected
owners, and change **both sides in the same PR**. A silent change there breaks four
people at once, and it breaks them as a missing field rather than as an error.

---

## The xfail ledger

Every unimplemented acceptance criterion already has a test, marked like this:

```python
@pytest.mark.xfail(reason="M3 -- ESKF unimplemented (owner: Sikruti)", strict=True)
def test_ten_second_stop_produces_zero_position_creep(): ...
```

The assertions are real, copied from the "done when" clauses in the build plan.
`strict=True` means that **when you implement the feature, the test starts passing, and
CI goes red until you delete the marker.**

So the workflow is: implement → the test XPASSes → remove the `@pytest.mark.xfail` line
→ green. You cannot claim a milestone without the test agreeing, and you cannot
accidentally leave a finished feature marked as pending.

CI prints the remaining count per milestone and owner on every run. That table is the
project's burndown chart.

**Deleting a marker without implementing the thing is claiming a milestone you did not
reach.** It will be noticed.

---

## Branching and pull requests

```bash
git switch -c feat/eskf-velocity-update      # feat/ fix/ docs/ chore/ test/ refactor/
# ... work, committing as you go ...
git push -u origin feat/eskf-velocity-update
gh pr create --fill
```

- **Never push directly to `main`.** It is protected; even a one-line fix goes through a
  PR. That includes Harsh.
- **One PR, one concern.** A PR touching four areas is four PRs.
- **CI must be green.** A red pipeline on `main` is a scored failure for the whole team.
- **One approval to merge.** Anyone can approve; do not wait on a specific person, ping
  the group chat.
- Rebase if your branch has drifted: `git pull --rebase origin main`.

### Commits

Conventional commits, imperative mood. Squash-merge uses the **PR title** as the commit
message, so CI checks that title's format.

```
feat: fuse the learned velocity in the device frame
fix: stop restamping GPS fixes on socket receive
docs: write the frame convention down
test: cover rotation-in-place
chore: pin numpy below 2.1
ci: split the torch install into its own workflow
```

**Commit continuously.** A clean progressive history is explicitly scored; a single
end-of-day dump scores badly. Commit every time something works, not when everything
works.

**No AI attribution.** No `Co-Authored-By` naming a model, no mention of an AI tool in a
message or PR body. Commits are authored by the person who wrote or ran them.

---

## Definition of done

Before you open a PR:

- [ ] `pytest` passes locally, and you did not weaken a test to get there
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `mypy` is clean
- [ ] `pytest -m frames` passes
- [ ] New behaviour has a test
- [ ] If you implemented an acceptance criterion, its xfail marker is gone
- [ ] If behaviour diverges from `docs/BUILD_PLAN.md`, the plan changed too
- [ ] Anything touching the model, filter or eval carries **before/after numbers** in
      the PR body — "it feels better" is not a result
- [ ] You stayed in your area, or the owner approved the crossing
- [ ] No secrets, no datasets, no `.mbtiles`, no checkpoints

---

## House rules

These come from [AGENTS.md](AGENTS.md) and apply to humans exactly as much as to AI
tools:

1. **Never make a failing test pass by weakening it.** If a test is wrong, say so in the
   PR and stop.
2. **Never restamp a timestamp.** Capture time, on the device, always.
3. **Training and live import the same preprocessing.** No exceptions.
4. **Physics beats parameters.** When the model and a physical constraint disagree,
   suspect the model.
5. **When uncertain, fail loudly.** A silently dropped measurement is a demo-day bug you
   will never find.
6. **Do not invent numbers.** If you did not run the eval, say so.

---

## When you're stuck

Say so in the group chat within ten minutes. The build window is short and an hour lost
to silent debugging is an hour the team does not get back.

Useful things to include: what you ran, what you expected, what happened, and which
layer you suspect (`timebase` / `preprocess` / `ahrs` / `models` / `fusion` / `gateway`
/ `web`). Guessing the layer is fine — it routes the question faster than leaving it
blank.
