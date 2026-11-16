# Status update: Pressure Plate and Navigation, seed 42

All 10 fresh-training runs completed, with no failed runs. Actual elapsed batch time was approximately **6 hours 47 minutes**. At most one Pressure Plate and one Navigation process ran concurrently. Every evaluation checkpoint uses 1,280 explicitly seeded episodes, with the same evaluation population across checkpoints and schedules.

Numbers below are **highest observed evaluation success**, selected across checkpoints from one training seed. They are not independent test scores or estimates of stability across training seeds. Success requires all agents to complete the task.

## Peak performance and timesteps

Valid steps exclude post-terminal padding that is masked out of PPO. Simulator steps include all 128 vectorized slots. Both counts are environment steps, not per-agent steps. M = one million. Peak time is accumulated training plus evaluation time through the chosen checkpoint and excludes interpreter/model initialization; full process wall time includes startup and concurrent resource contention.

### Pressure Plate

| Schedule | Peak success | Peak round | Valid steps at peak | Simulator steps at peak | Final success | Train + eval to peak | Full process wall | Queries/episode at peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand-coded event | 99.30% | 130 | 3.204M | 3.328M | 18.98% | 20.0 min | 30.2 min | 3.13 |
| K=38 | 0.00% | — | — | — | 0.00% | — | 25.0 min | 6.00 |
| K=76 | 100.00% | 120 | 2.879M | 3.072M | 0.00% | 15.0 min | 24.4 min | 2.00 |
| K=152 | 0.00% | — | — | — | 0.00% | — | 25.7 min | 2.00 |
| Single query | 0.00% | — | — | — | 0.00% | — | 24.8 min | 1.00 |

Peak query counts are team refreshes for Pressure Plate (all three agents refresh together), and per-agent adapter queries for Navigation. Counts include the initial query. Zero-peak schedules never achieved a successful team episode at any evaluation checkpoint; the untrained round-zero tie is not reported as a learned peak.

### Navigation

| Schedule | Peak success | Peak round | Valid steps at peak | Simulator steps at peak | Final success | Train + eval to peak | Full process wall | Queries/episode at peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hand-coded event | 85.16% | 150 | 2.752M | 3.840M | 76.56% | 36.3 min | 83.0 min | 1.77 |
| K=25 | 85.94% | 175 | 3.417M | 4.480M | 74.84% | 32.8 min | 74.1 min | 5.04 |
| K=50 | 82.11% | 50 | 1.182M | 1.280M | 63.28% | 9.5 min | 73.4 min | 2.78 |
| K=100 | 86.72% | 200 | 3.711M | 5.120M | 75.08% | 36.7 min | 73.1 min | 1.61 |
| Single query | 79.77% | 250 | 4.529M | 6.400M | 74.14% | 66.2 min | 103.3 min | 1.00 |

Peak query counts are team refreshes for Pressure Plate (all three agents refresh together), and per-agent adapter queries for Navigation. Counts include the initial query. Zero-peak schedules never achieved a successful team episode at any evaluation checkpoint; the untrained round-zero tie is not reported as a learned peak.

## Navigation individual-goal metric

The team metric requires all four goals. The best team checkpoint (K=100) has 96.56% individual-goal collection, versus 86.72% full-team success. These are different metrics.

| Schedule | Individual goals at team-peak checkpoint | Individual goals at final checkpoint |
|---|---:|---:|
| event | 96.05% | 93.59% |
| fixed_25 | 96.21% | 93.03% |
| fixed_50 | 95.18% | 89.45% |
| fixed_100 | 96.56% | 93.07% |
| single | 94.55% | 92.93% |

## Degradation findings

### Pressure Plate: goal retention and simultaneous completion

- Hand-coded event reached 99.30% at round 130, remained above 97% at rounds 140 and 150, then fell to 18.98% at the final round. Mean return was essentially unchanged: 306.09 at the peak versus 306.13 at the end.
- Fixed K=76 reached 100% at round 120 and remained at 100% through round 180. It fell to 9.38% at round 190 and 0% at round 200. Mean return changed only from 307.66 to 305.24.
- In all 40 sampled final episodes for K=76, all three agents had visited the goal, but the team never occupied it simultaneously. Every sampled episode contained a goal exit. At the peak, all 40 samples completed.
- In all 40 sampled final hand-coded episodes, all three agents had also visited the goal. Thirty-two of the 40 failed to complete; all 40 contained a goal exit. Goal exits increased from 1 event in the peak sample to 111 in the final sample.
- In the full paired 1,280-episode evaluation, K=76 changed from 1,280 successes to zero on the same initial conditions. For the event method, 1,031 previously successful episodes became failures and 3 previously failed episodes became successes.

The direct observed failure is reaching the goal and then leaving before teammates finish. The frozen scenario pays a shared first-visit bonus once per arriving agent, while `done()` requires all agents to be in the goal at the same time. There is no separate completion bonus in this reward branch. This mismatch, together with stage-dependent shaping, is a strong candidate explanation for why reward stays high while task success collapses. The critic continues to explain shaped returns well (last-ten-round explained variance approximately 0.95 for event and 0.90 for K=76), so these curves do not look like a simple critic/NaN explosion. A reward/control ablation is still needed to establish the optimizer-level causal explanation.

The detailed state evidence samples 40 of 1,280 episodes per checkpoint. It supports a concrete behavioral diagnosis; it does not prove that all 1,280 failed episodes followed identical trajectories. Whole-population success, reward and milestone rates are recorded separately.

K=38 and K=152 did not learn team completion in this seed. In their final 40-episode samples, two agents visited the goal in every episode and most teams remained at stage 1. Single-query samples had only one goal visitor each. These are learning/coordination failures, distinct from the late loss of a previously successful policy.

### Navigation: most missing goals were never detected

- Peak-to-final team-success decreases range from 5.63 points (single query) to 18.83 points (K=50). Hand-coded event decreases by 8.59 points; K=100 decreases by 11.64 points.
- For K=50, individual goal collection falls from 95.18% to 89.45%. Targets that were detected but not collected account for only approximately 0.45% and 0.43% of agent episodes, respectively. Most of the additional misses therefore occur before target detection.
- For hand-coded event, individual goal collection falls from 96.05% to 93.59%; missed detection also explains most of the decline. Performance drops across all four agents rather than only one fixed agent.
- PPO clipping, gradient and replay diagnostics remain finite. Maximum pre-update rollout/replay log-probability MSE across all ten runs is 1.89e-13. The Navigation NMD reference estimate and its scale change substantially during learning; this is a candidate influence, not a demonstrated cause. A controlled NMD ablation would be needed for attribution.

## Integrity checks and limits

All 200 peak/final evaluation trace files passed checks of success accounting, fixed/single/event query counts and exclusion of queries after termination. Saved peak/final initial-observation samples match exactly. Pressure Plate first-visit bonus sums match latched goal visits. No additional training or changes to the frozen experiment algorithms were made while analyzing results.

These data demonstrate that both tasks can be learned under the corrected pipeline. Fixed intervals are strong baselines; this single-seed comparison does not establish a hand-coded-event advantage. Differences from earlier seed-42 batches must be interpreted with their different RNG/evaluation/NMD protocols in mind.

## Figures and reproducibility

- `learning_curves.png` / `.pdf`: equal simulator-step budget curves.
- `learning_curves_valid_steps.png` / `.pdf`: valid learning-step curves.
- `pressure_plate_degradation.png` / `.pdf`: success collapse while return stays high.
- `navigation_failure_breakdown.png` / `.pdf`: missed-detection versus post-detection failure.
- `status_results.csv`: peak/final scores, both step counts, timing, queries and checkpoint paths.
- `analysis.json`, `trace_checks.json`, `timing_and_checks.json`: full numerical evidence.
- `*_sampled_peak_final.json`: paired sampled episode diagnostics.
- `analyze_results.py`, `plot_results.py`: reproducible analysis scripts. Plotting dependencies are isolated in `analysis/plot_venv`; the training runtime was not modified.

## Suggested status-update wording

> We completed fresh one-seed comparisons of hand-coded, periodic and reset-only adapter queries in Pressure Plate and Navigation. Peak team success reached 99.3%/100% for hand-coded/fixed-K policies in Pressure Plate, and 85.2%/86.7% in Navigation. Late training degradation remains substantial. Pressure Plate traces show agents reaching the goal but failing to remain there together, while shaped return remains high, motivating a controlled investigation of reward–completion alignment. Fixed intervals remain competitive; benefits of learned event gating have not yet been tested.

