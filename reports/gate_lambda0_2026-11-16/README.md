# Pressure Plate gate pilots with lambda = 0

Primary comparison against the November 16 status-update baselines. Both
centralized gate variants train from scratch on seed 42, with zero query
penalty. The actor/generator retain the baseline task-reward objective.

Each condition uses 200 rounds, 128 environments, a 200-step horizon, and
1,280 evaluation episodes every 10 rounds. Original simultaneous team-goal
completion and the existing VMAS environment, backbone, PPO settings and
mandatory initial query are retained. Coverage exploration is used only in
training; evaluation uses the learned gate alone.

The source/configuration and full numeric artifacts are retained locally under
`runs/gate_lambda0_2026-11-16/batch/`. This is a new batch; no penalized policy
is reused. See [the full protocol](../../docs/learned-gate-pilot.md).

`integrity_checks.json` and `verification_summary.json` describe execution
checks, not convergence results. A launch manifest records startup only.
After completion the local runner writes `summary.csv` and `seed_summary.csv`.
