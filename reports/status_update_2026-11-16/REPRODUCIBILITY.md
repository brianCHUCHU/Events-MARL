# Portable report artifacts

`evaluation_curves.json`, `status_results.csv` and `analysis.json` contain the
compact numeric evidence for the figures and report. The complete local
trajectories and model checkpoints are intentionally not committed.

To redraw the figures, install matplotlib and numpy in a separate analysis
environment and run `python plot_results.py` from this directory. To regenerate
the detailed analysis from retained raw local runs, use
`python analyze_results.py --batch /path/to/status_update_batch`; its sibling
`batch.log` supplies the original batch wall-clock lifetime. The script writes
updated analysis products in this report directory.

See `docs/query-experiment-metrics.md` and the frozen configs in each local batch
for full training and metric conventions. See `README.txt` for the self-contained
LaTeX status-update source. The report describes the completed fixed/event
pilots; newer learned-gate runs are documented separately.
