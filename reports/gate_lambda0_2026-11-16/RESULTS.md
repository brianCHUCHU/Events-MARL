# Completed Pressure Plate pilots: zero query penalty

Both seed-42 runs completed all 200 training rounds without an execution
failure. Each used 5.12M simulator steps, which were also 5.12M valid learning
steps because neither run completed a training episode. Evaluations used the
same 1,280 seeded resets every 10 rounds as the status-update baselines.
The two processes ran concurrently and took 1,453.1 and 1,463.5 seconds,
respectively: approximately **24 minutes 24 seconds for the pair**.

| Condition | Highest observed success | Final success | Final queries / episode | Final team return |
|---|---:|---:|---:|---:|
| Handcrafted event | 99.30% | 18.98% | 3.09 | 306.13 |
| Fixed K=38 | 0.00% | 0.00% | 6.00 | 210.97 |
| Fixed K=76 | 100.00% | 0.00% | 3.00 | 305.24 |
| Fixed K=152 | 0.00% | 0.00% | 2.00 | 197.68 |
| Initialization only | 0.00% | 0.00% | 1.00 | 115.12 |
| Learned gate, lambda=0 | 0.00% | 0.00% | 117.39 | 123.45 |
| Coverage gate, lambda=0 | 0.00% | 0.00% | 106.71 | 199.01 |

Queries are team refreshes, each updating all three agents, including the
initial query. This table uses **final-checkpoint** query counts for every
method. The original status-update tables instead used queries at the
earliest peak; for example, K=76 used two queries at its successful peak and
three at its unsuccessful final checkpoint. Different episode lengths account
for that difference. Conditions with no successful evaluation have no
meaningful positive-success peak round; the new comparison leaves it empty
rather than treating the round-zero tie as a learned peak.

## What the new runs show

Neither gate achieved any team success at any of the 21 evaluation checkpoints.
Neither had a successful training episode. This differs from the event/K=76
late degradation: the learned conditions never first acquired full completion.

Both gates retained frequent refreshes. Final evaluation means were 117.39
and 106.71 per 200-step episode, compared with approximately 3 for the two
strong reference schedules at their final checkpoints. Coverage improved raw
return and partial progress relative to the plain gate, but did not solve the
team task. Zero query penalty provides no explicit pressure to economize on
queries; whether excessive refresh frequency causes the coordination failure
needs a matched-frequency control, not just this comparison.

At the final checkpoint, sampled state trajectories cover 40 of the 1,280
evaluation episodes per method:

- Plain gate: 27 episodes had one agent ever visit the goal, 11 had two, and
  two had none. No sample had three visitors. Final task stages were 1 in 19
  episodes and 2 in 21 episodes.
- Coverage gate: all 40 episodes had exactly two agents ever visit the goal,
  and all ended in task stage 1. No sample had three visitors.

These samples show incomplete coordination, rather than all three reaching
the goal and then failing to stay together as in the earlier successful
policies' degraded final samples. They do not establish every episode's
trajectory or a causal explanation of why the gate failed to learn.

## Integrity and interpretation

All 420 evaluation task/gate trace pairs passed query-count, initialization,
termination-mask, success-accounting and metric-reconstruction checks.
Coverage exploration was disabled in evaluation. Every logged query penalty
was zero. Maximum pre-update actor likelihood replay MSE was 1.67e-13; gate
replay MSE was zero. This checks important implementation invariants, not the
absence of every possible modeling or optimization issue.

The environment, original task rewards, actor/generator architecture, PPO
settings and evaluation population match the status-update protocol. Added
gate optimization and generator batching remain method-specific work. Wall
times across batches are not a controlled speed comparison because the earlier
Pressure Plate runs shared the machine with Navigation, whereas these two
Pressure Plate runs shared it with one another.

All results use one training seed. They show failure of these basic gate
settings in this pilot; they do not establish that learned query policies or
coverage exploration are generally ineffective. The cost-0.01 runs remain
paused and are not included here.

## Artifacts

- `comparison.csv`: all seven settings with a consistent final-query definition.
- `results.json`: complete evaluation curves, checks and sampled failure counts.
- `comparison.png` and `.pdf`: success, raw return and query-frequency curves
  for the two gates and the two strong reference schedules.
- `analyze_results.py`: rerun using the separate plotting environment with
  `python analyze_results.py --plot`; optional `--batch` and `--baseline-batch`
  locate retained local raw data.

Full checkpoints, frozen source and numeric trajectories remain in
`runs/gate_lambda0_2026-11-16/batch/`; they are not committed to Git.
