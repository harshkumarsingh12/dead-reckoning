## What and why

<!-- One or two sentences. What changed, and which build-plan section or milestone it
     serves. Link the issue: Closes #NN -->

## Definition of done

<!-- Delete any line that genuinely does not apply, and say why in a comment. -->

- [ ] `pytest` passes locally, and I did not weaken a test to get there
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `mypy` is clean
- [ ] `pytest -m frames` passes
- [ ] New behaviour has a test
- [ ] If I implemented an acceptance criterion, I removed its `strict=True` xfail
      marker (leaving it makes CI red with XPASS — that is the design)
- [ ] If behaviour diverges from `docs/BUILD_PLAN.md`, the doc changed too
- [ ] I stayed inside my area, or the owner approved the crossing
- [ ] No secrets, no datasets, no `.mbtiles`, no checkpoints

## Numbers

<!-- Anything touching the model, filter, or eval: paste the before/after.
     "It feels better" is not a result.

     | metric        | before | after |
     |---------------|--------|-------|
     | drift %       |        |       |
     | ATE (m)       |        |       |
     | RTE 60 s (m)  |        |       |
     | NIS in bounds |        |       |
-->

## Contract impact

<!-- Did you touch src/dr_core/types.py or docs/CONVENTIONS.md? If so, say who you
     told in the group chat and when. A silent change there breaks four people. -->

None.
