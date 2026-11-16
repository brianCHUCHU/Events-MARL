# Learned gate and coverage: Pressure Plate pilot

This adds two fresh-training conditions to the corrected author VMAS pipeline.
It borrows the centralized query-policy and exploration-mixture idea from
`Qz07/event-free-MARL` at commit `f76b6d8`, implemented here in Flax/JAX.
The teammate's custom discrete grid environment is not used.

## Conditions and comparison

Both conditions train seed **42**, 200 rounds, 128 environments and a 200-step
horizon: **5.12M simulator environment steps** per condition. The actor,
rank-8 final-layer adapters, transformer generator, three PPO epochs and eight
minibatches retain the preceding Pressure Plate pilot settings. All three
robots must occupy the goal simultaneously; first visits alone do not complete
the task. Raw environment rewards and physical dynamics are unchanged. NMD is
off, as in the paper's Pressure Plate configuration.

- `learned_gate`: a centralized MLP reads the concatenated current observations
  and the normalized time since the last query. Two 128-unit tanh layers feed a
  Bernoulli query head and a team value head. One query refreshes all three
  agents; otherwise the deployed adapters and generation contexts persist.
- `coverage_gate`: the same model and initialization, with training probability
  `p_behavior = 0.8 * sigmoid(logit) + 0.2 * 0.5`. Both rollout likelihoods and
  PPO ratios use this mixture. This is random coverage exploration, not a
  learned coverage estimator or an oracle event label.

Both receive one mandatory initial generation. Reset is excluded from learned
decisions and query cost. The query head starts at probability 0.5; actor,
generator and critic initialization preserve the baseline seed streams. Gate
sampling and shuffling have separate streams. The gate has 26,626 parameters
with the current observation dimensions, learning rate 3e-4, gradient norm limit
0.5 and entropy coefficient 0.01. These are basic gate baselines: neither feeds
the cached adapter/context explicitly into the gate.

## Objective and evaluation

The gate, actor and generator learn from the same reward: the existing clipped
task reward minus **0.01 per additional team query**. The cost is charged to
each agent's identical shared reward; team averaging therefore charges 0.01,
not 0.03. Raw rewards, returns, completion and query counts remain separately
logged. The existing GAE return/advantage clipping remains active. This is a
query-penalized method objective, so fixed/event baselines optimize a different
cost-free objective. A cost sweep or zero-cost control would be a later ablation.
Unlike the teammate prototype, we do not give the actor an unpenalized reward
while training the gate on the penalized reward.

Every 10 rounds, and before training, evaluate 1,280 episodes with the same
reset seeds as the earlier baseline pilot. Actor actions are `tanh(mean)`;
the gate is a seeded Bernoulli sample from its learned policy. Coverage's extra
exploration is disabled for evaluation. Retain the full curve, peak checkpoint
and final checkpoint so late degradation is visible. The highest observed
evaluation score is checkpoint-selected, not an independent test estimate.
One seed cannot establish across-seed stability or method superiority.

To limit JAX compilation shapes, generator refresh batches are rounded up to
multiples of 32 team contexts (capped at 128), padding with duplicates of an
already queried context. Only the actual queried teams receive adapters.
Logical queries, physical input contexts, generated adapters and deployed
adapters are counted separately, including padding. PPO reconstructs the
actual retained contexts; padding is not extra training data.

## Logs and validation

The existing task traces retain milestones, goal entries/exits, sampled states,
actions and adapters. `gate_training_traces/` and `gate_evaluation_traces/`
retain all teams' probabilities, query decisions, ages, validity masks, old
values and log probabilities, plus inputs for the first four teams. Trace
collection follows the existing training/evaluation schedule. Mixture source
labels are not sampled: exploration is represented exactly by its marginal
Bernoulli probability and saved epsilon. Gate entropy can be reconstructed
from that probability. Per-round metrics include gate losses, gradient norm,
clipping, pre-update replay error, query rate, query cost and update time.
Gate parameters are included in evaluated, periodic and final checkpoints.
As with the existing trainer, these parameter checkpoints do not implement
exact optimizer-state resume. Only load trusted local checkpoints.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/validate_learned_gate.py --out runs/new_gate_checks
python scripts/benchmark_learned_gate.py --out runs/new_gate_benchmark --workers 2
python scripts/run_query_experiments.py --scenario pressure_plate --seeds 42 \
  --schedules learned_gate coverage_gate --workers 2 --out runs/new_gate_batch
```

The short validator checks real environment execution, initial generation,
query accounting, actor and gate likelihood replay, gate learning, absence of
coverage noise in evaluation, and exact final actor/generator/critic/gate
parameter invariance when evaluation frequency changes. Unit tests also cover
mixture gradients, padded PPO rows and retained adapter-context replay.
Short checks establish implementation invariants, not convergence.

The batch runner freezes source/configs and records each process's wall/CPU
time and memory. Raw run directories remain local; the research branch includes
compact reports and verification evidence. Both conditions are additional
pilots; no new performance claim is made before they finish.
