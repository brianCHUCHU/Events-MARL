# Portable report artifacts

The current LaTeX report covers twelve completed seed-42 conditions, including
two zero-penalty learned-gate pilots on Pressure Plate. `report_table_data.csv`
contains the current tables' values, checked against all twelve runs' raw
evaluation logs. Query counts use each environment's common final round;
positive peak rounds identify the earliest highest-success checkpoint.

`evaluation_curves.json`, `status_results.csv`, `analysis.json`, `STATUS_REPORT.md`
and the earlier figures preserve the original ten fixed/event baseline runs.
Their peak-query columns retain the historical definition and must not be
mistaken for the new report's final-query column. The two additional gates'
curves, trace checks and analysis are in `../gate_lambda0_2026-11-16/`.
The complete local trajectories and model checkpoints are not committed.

To redraw the figures, install matplotlib and numpy in a separate analysis
environment and run `python plot_results.py` from this directory. To regenerate
the detailed analysis from retained raw local runs, use
`python analyze_results.py --batch /path/to/status_update_batch`; its sibling
`batch.log` supplies the original batch wall-clock lifetime. The script writes
updated analysis products in this report directory.

See `docs/query-experiment-metrics.md` and the frozen configs in each local batch
for full training and metric conventions. See `README.txt` for the self-contained
LaTeX status-update source. The AI disclosure is embedded directly in the main
TeX file; no separate disclosure input is required.
The source ZIP includes every required TeX and style file.
