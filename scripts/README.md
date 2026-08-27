# scripts/

Thin command-line entry points. All real logic lives in `dr_core` so it can be tested;
these files only parse arguments and call in.

| Script | Owner | Milestone | Purpose |
|---|---|---|---|
| `fetch_datasets.py` | Sumedha | M0 | Download and lay out RoNIN / OxIOD |
| `run_eval.py` | Sikruti | M0 | Recording to trajectory, plots and numbers |
| `train.py` | Sumedha | M2 | Train the causal TCN (needs the `[ml]` extra) |
| `export_onnx.py` | Sumedha | M2 | Export and int8-quantize the trained model |
| `replay.py` | Harsh | M4 | Push a golden run through the live pipeline |
| `make_tiles.py` | Harsh | M4 | Build the offline MBTiles for the demo area |
