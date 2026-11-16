# Pressure Plate gate pilots, seed 42

Both full-budget runs were launched concurrently after the consistency checks.
They are now **paused at the user request** in favor of the lambda-zero
primary comparison. See `pause_manifest.json`; the old launch manifest is
historical. The existing stopped processes retain their in-memory state.
`launch_manifest.json` is a timestamped launch observation, not a live status file.
The gate and coverage share the original VMAS simultaneous-goal task and each
train for 200 rounds (5.12M simulator steps). See
[the protocol](../../docs/learned-gate-pilot.md) for objectives and evaluation.

- `verification_summary.json`: test scope.
- `integrity_checks.json`: four short training/evaluation checks.
- `parameter_checks.json`: exact initialization and serial/parallel comparisons.
- `parallel_benchmark.json`: four-round timing check; not task performance.

Full local logs, frozen source, traces, configs and checkpoints are under
`runs/gate_pilot_2026-11-16/batch/` (ignored by Git). After completion the runner
writes `summary.csv` and `seed_summary.csv`; ongoing job state is in `status.json`.
No learned-gate performance claim is made in these launch artifacts.
