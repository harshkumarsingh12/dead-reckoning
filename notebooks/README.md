# notebooks/

Scratch space. **Notebook files are gitignored** — only this README is tracked.

Notebooks are for looking at data, not for holding logic. Anything that another part
of the system depends on belongs in `dr_core`, where it can be imported, typed and
tested. A notebook cell that quietly reimplements preprocessing is exactly the
training/live divergence the shared module exists to prevent.

If you want a notebook reviewed, export the figure and put the reasoning in the PR.
