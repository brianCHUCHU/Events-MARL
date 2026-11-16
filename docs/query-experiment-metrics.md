# Query-schedule experiments and diagnostics

## Audited protocol, version 2

The current audited comparison is Ours on **Pressure Plate and Navigation**,
using `research_metrics=True`. This flag also selects transition-token replay
and the controlled query schedules; do not disable it for these comparisons.
The legacy non-research event/replay path and full PS/DiCo/CASH/Wind Flocking
training are outside this audit's end-to-end coverage.

Version 2 uses deterministic `tanh(mean)` evaluation, explicit VMAS reset seeds
`evaluation.seed + batch` (default starts at 20261114), and restores VMAS,
Torch, NumPy and Python RNG state after evaluation, including on exceptions.
Evaluation scores do not choose training checkpoints, stop training, or change
optimizer settings. Changing the periodic evaluation interval must preserve
the final trained parameters for an otherwise identical seeded run.

Navigation NMD now generates reference adapters with their recorded **full
team LiDAR context**, rather than copies of one agent's LiDAR. Its reference
observations exclude post-terminal padding. It remains a reference estimate
on **pre-tanh Gaussian means**, using initial contexts and previous valid
rollout observations. Target 0.007 is not a guarantee of measured deployment
action diversity. These corrections affect Navigation training and require
fresh runs for results under the corrected protocol.
The first training round has no previous reference observations, so it records
NMD as unavailable and retains initial scale 1.0. It must not treat a missing
measurement as zero diversity and adjust the controller from that invented value.

Pressure Plate's original scenario fixes every ground robot's maximum speed
at 1.0 and discards the capability argument. Its four main method configs now
declare `[1.0, 1.0, 1.0]` to match the actual physics; the earlier declared
`[0.8, 1.0, 1.2]` never controlled the scenario. Unsupported speed settings now
raise instead of being ignored. This correction preserves the original
homogeneous physics. Old resolved configs should be replayed with their frozen
source; their speed declarations must not be interpreted as measured speeds.
Navigation uses the declared capability prefix for its actual team size and
rejects incomplete fixed capabilities instead of substituting a pattern.

Every new research run writes `logs/environment.json` with effective initial
agent properties and protocol details. New metrics and traces carry
`metrics_schema_version=2` and `trace_schema_version=2`, respectively. Original
run artifacts and frozen sources are preserved and must be labeled as the
earlier protocol rather than pooled with corrected runs.

Physical generator work is separate from logical query counts. In Navigation,
refreshing four agents in one team currently sends four full-team contexts to
the generator, generates sixteen adapters, and deploys four. The trace's
`physical_query_batch_size` now counts input team contexts, while
`generated_agent_adapters_at_step` and `deployed_agent_adapters_at_step` record
both adapter counts. Setup, refresh, NMD reference, and PPO regeneration work
are logged separately. Older traces counted distinct affected teams in the
physical batch field and therefore understated this work.

The PPO likelihood calculation uses the policy's configured minimum standard
deviation. Minibatches must contain complete teams and cover all rows; invalid
divisibility raises. Disabling minibatches performs one full-batch update per
epoch. Return-normalization statistics respect the valid-transition mask
(normalization remains disabled in both main experiment configs). Non-finite
research policy outputs, rewards, losses, or gradient diagnostics stop the run;
the first PPO update also checks rollout/replay log-probability agreement.

Run focused tests and short integration checks before a new full batch:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/validate_query_pipeline.py --out runs/a_new_validation_directory
```

The short validator uses two training rounds, all five schedules in each
environment, a second Navigation seed, full-batch updates, and two evaluation
frequency controls. It freezes its own source/configs. These checks establish
execution and invariants, not convergence or final task performance.

## Original batch setup and available measures

For the 2026-11-16 status-update batch, both environments run all five schedules
from scratch on seed 42. The summary records `peak_completion_rate`, its round,
valid environment steps, physical simulator steps, cumulative training plus
evaluation seconds, and the matching evaluated checkpoint. The full curve and
final result remain available. Label the peak as the **highest observed
evaluation success**, rather than a fresh independent test result. Its cumulative
time excludes interpreter/model startup; process wall time is recorded separately.

```bash
python scripts/run_query_experiments.py --scenario pressure_plate_navigation \
  --seeds 42 --workers 2 --out runs/a_new_status_batch
```

This combined mode permits at most one process per environment. Pressure Plate
evaluates every 10 rounds and Navigation every 25. A short benchmark on the M4
found that two simultaneous Navigation runs reduced throughput, whereas a mixed
Pressure Plate/Navigation pair improved elapsed time. The worker count is an
execution setting and does not change model/rollout budgets. When one environment
finishes its queue, the other continues with one process. Every batch freezes its
source and config. `--seeds`, `--schedules` (one environment), and `--eval-interval`
can select smaller new batches; resume uses the frozen plan.

The batch runner trains independent actor and generator weights for the two paper seeds
`0` and `42`. For Pressure Plate it trains the hand-coded event, fixed 38,
76, and 152 step schedules, and a single query at reset. Wind Flocking uses
the supplied diversity-target cycle, with a scheduled change every 50 steps.
Its returns and formation measures are reported separately from Pressure Plate
completion. The three fixed intervals were selected in the earlier seed-0
pilot and are held fixed here. The Wind Flocking evaluation applies the target
cycle during rollouts.

## Collecting metrics

Run from the repository root with the author's requirements installed in an
isolated Python environment.

```bash
python scripts/run_query_experiments.py --scenario pressure_plate --out runs/my_new_batch
```

Omit `--scenario` to include both environments. For a new batch the output
directory must be new. Use a new directory for a rerun with code fixes so
it has its own frozen source snapshot. To continue an existing batch for one environment,
stop its old launcher first and pass `--resume --scenario pressure_plate`.
Each run uses a fresh Python process and the
batch saves a source snapshot, config, and dependency versions. Logs and model
weights are local. For one run, use the upstream training CLI with
`--set research_metrics=True`. A positive `model.query_interval` selects a
fixed Pressure Plate schedule. Set `model.adaptive_hypernetwork=False` for the
Pressure Plate reset-only schedule.

## Measures captured

| Question | Evidence |
|---|---|
| How quickly does it learn? | Completion rate and raw mean-team return at common evaluation rounds, cumulative training environment and agent steps, normalized curve area, first observed 50% completion, wall time to that observed evaluation, and final performance. An empty threshold result means the schedule was never observed at 50%. |
| Does performance drop or vary across seeds? | Unpooled per-seed learning curves, outcome per evaluation episode, final values, mean and largest adjacent drop across the last three completion evaluations. Treat the training seed as the independent replicate. |
| What does a query cost? | Separate training, rollout, update, evaluation and trace time, process CPU seconds, peak resident memory, parameter counts, logical team queries, rollout generator batch counts, setup batch counts and PPO recomputations. Batch sizes are logged separately from semantic query counts. CPU seconds are observed process compute. |
| Which part of the task fails? | Pressure Plate first open, first use of each plate, first simultaneous plate use, first arrival at goal, and team completion. Wind Flocking mean shielding of smaller agents, separation, and energy, wind, center and formation reward components by episode and target segment. |
| Where does the generator receive training signal? | All refresh times and numbers of queried teams in training; full per-team query masks and milestones in evaluation; sampled current observations and active generator tokens in training and evaluation. |
| Did a new adapter actually change policy behavior? | Sampled LoRA factors at reset and refresh, active scale and tokens, executed actions and policy distribution means and standard deviations. Paired checkpoints and the saved observations support shared-observation policy-distance probes. |
| Is the update consistent or unstable? | Actor, generator and critic gradient norms, losses, entropy, clip fraction, critic explained variance, and mean squared rollout/recomputed log-probability difference before the first PPO update. |
| Does Wind Flocking track requested diversity? | Requested target and actual scale by step, upstream NMD estimate and EMA, and realized geometry/reward components. The upstream NMD estimator uses reference contexts. The saved deployment traces support separate distance probes on the actual adapters. |

## Trace conventions

- A training round is a vectorized rollout and its PPO updates. At default
  size, the simulator advances 25,600 environment slots. In the Pressure Plate
  Ours trainer, each environment's first completion is terminal. Its later
  padded slots contribute no queries, returns, or PPO updates. Logs distinguish
  the cumulative steps available to learning (`training_environment_steps`)
  from physical simulator steps (`simulator_training_environment_steps`).
  PPO replays do not add environment steps. Evaluation environments and RNG
  streams are isolated from training.
- Pressure Plate's shared arrival bonus is paid once per agent per episode.
  Team completion still requires all three agents to be at the goal together.
- Team query counts include one initial generation per environment. Evaluation
  counts stop on termination. `physical_query_batch_size` records batch size
  by step and covers refreshes. Initial batches are counted separately.
- Compressed numerical traces contain no pickled arrays. `sample_*` arrays
  keep four environments per recorded rollout, with observations taken before
  each action and rewards, milestones and task state taken afterward.
  `extra_query_at_step` is indexed from zero. First milestone steps use the
  number of transitions completed and `-1` for unreached milestones.
- Each evaluation has the checkpoint from that training round. Training traces
  are sampled at the first round and at later evaluation rounds, before the PPO
  update. An aggregate timing row is written for every training round.
- For this comparison, Pressure Plate refreshes retain whole-team attention.
  PPO reconstructs the recorded transition's tokens and, in Wind Flocking,
  its active diversity scaling. This changes training semantics relative to
  the upstream refresh path and the earlier seed-0 pilot.
- Opt-in centralized learned-gate and coverage pilots now record gate inputs,
  probabilities, query age, decision masks, PPO diagnostics and query cost.
  See `docs/learned-gate-pilot.md`. These are separate new runs; the earlier
  fixed/event pilot results contain no learned-gate measurements.

## Regression checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Navigation comparison

The Navigation batch trains the event trigger, fixed intervals 25, 50 and 100,
and reset-only generation on seeds 0 and 42. Each run retains the original
400 training rounds, 128 parallel environments and 200-step horizon. Evaluation
runs every 25 rounds, on the same 1,280 episodes. The original NMD control and
target 0.007 are shared across schedules. The fixed intervals are a frequency
sweep selected before Navigation runs.

```bash
python scripts/run_query_experiments.py --scenario navigation --out runs/new_navigation_batch
```

The Navigation scenario implements individual food collection. Each agent has a
matching food, and the team completes when all four have collected theirs. Report
both the percentage of individual goals collected and team completion.
Query costs use agent-adapter regenerations, including one at reset for every
agent. The trace also records physical generator calls and their sizes.

The event trigger refreshes only an agent that first detects its matching food.
It runs after the observation is received and before the next action. A fixed
schedule refreshes all agents every K actions. Reset-only generation retains
its initial adapters throughout the episode. Every schedule uses the same
full-team attention context. Each agent's saved generation context is retained
separately until its next refresh, so PPO reconstructs its deployed adapters
with the same teammates and scale.

Numerical traces retain each agent's query mask, first detection and collection
times, sampled current observations, active local LiDAR token, original
full-team generation context, action distribution and adapter snapshots.
These support detection-to-query and detection-to-arrival delay analysis.
`mean_team_queries` counts steps at which any agent in a team was refreshed.
Use `mean_agent_queries` and `training_agent_queries` for the Navigation
cost comparison.

The factory omits the unsupported `fixed_food_positions=False` argument.
Navigation's training and evaluation both use the existing randomized resets.
Episode termination masks stop later data from entering PPO. First-update
log-probability consistency is checked with real rollouts before launch.
