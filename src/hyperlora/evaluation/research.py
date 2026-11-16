"""Evaluate the deployment schedule and retain replayable diagnostics."""

from __future__ import annotations

from time import perf_counter, process_time

import jax.numpy as jnp
import numpy as np
import torch

from ..diversity.control import alpha_from_ema, episode_target
from ..envs.capabilities import initial_capabilities
from ..envs.factory import make_env
from ..logging.diagnostics import RolloutTrace
from ..seeding import isolated_vmas_rng
from .actions import deterministic_action
from .quantitative import _agent_capabilities, _diversity_scaling, eval_capabilities


def evaluate(config, adapter, models, hn_apply, policy_outputs, nmd_ema, rounds, trace_root,
             training_steps):
    wall_started, cpu_started = perf_counter(), process_time()
    diagnostic_wall = 0.0
    num_envs, num_agents = config.env.num_envs, config.env.num_agents
    horizon = config.evaluation.max_eval_steps
    is_pressure_plate = config.env.scenario == "pressure_plate"
    is_navigation = config.env.scenario == "navigation"
    episodic_success = is_pressure_plate or is_navigation
    fixed = eval_capabilities(config, num_agents)
    capabilities = _agent_capabilities(config, fixed, num_agents)
    rollout_numbers = list(range(config.evaluation.num_eval_episodes))
    episode_data = []
    initial_queries = len(rollout_numbers) * num_envs
    rollout_wall_seconds = 0.0
    simulator_environment_steps = 0

    # A common evaluation population across methods and rounds, with an
    # RNG stream isolated from training.
    with isolated_vmas_rng():
        for batch in rollout_numbers:
            env = make_env(config.env, initial_capabilities(config.env))
            if is_pressure_plate:
                env.scenario.eval_mode = True
            if hasattr(env.scenario, "update_agent_capabilities"):
                env.scenario.update_agent_capabilities(capabilities)
            obs = env.reset(seed=config.evaluation.seed + batch)
            events = adapter.make_events(config, num_envs, num_agents, models.obs_dim, 0, nmd_ema)
            if config.query_gate.mode != "disabled":
                from ..training.query_gate import LearnedPressurePlateEvents

                events = LearnedPressurePlateEvents(config, num_envs, num_agents,
                    models.gate_state, 0, evaluation=True, eval_batch=batch)
            events.prime(obs)
            target = episode_target(config, 0)
            tokens = adapter.build_tokens(config, env, obs, models.obs_dim, num_envs,
                                          num_agents, num_agents, target)
            alpha = _diversity_scaling(config, nmd_ema)
            if not is_pressure_plate and config.diversity.use_diversity_control:
                alpha = alpha_from_ema(target, nmd_ema, config.diversity)
            adapters = hn_apply(tokens, alpha)
            trace = RolloutTrace(config.env.scenario, num_envs, num_agents, horizon, batch * 4)
            trace.initial_adapters(adapters)
            active = torch.ones(num_envs, dtype=torch.bool)

            simulation_started = perf_counter()
            for step in range(horizon):
                if episodic_success:
                    refresh = events.before_step(env, obs, tokens, step, active_envs=active)
                else:
                    refresh = events.before_step(env, obs, tokens, step)
                if refresh is not None and len(refresh.indices):
                    tokens = refresh.tokens
                    if refresh.alpha is not None:
                        alpha = refresh.alpha
                    updated = hn_apply(refresh.inputs, alpha if refresh.scaling is None
                                       else refresh.scaling)
                    if refresh.generated_indices is not None:
                        updated = {key: matrix[refresh.generated_indices] for key, matrix in updated.items()}
                    adapters = {
                        key: weights.at[refresh.indices].set(updated[key])
                        for key, weights in adapters.items()
                    }
                flat_obs = torch.stack(obs, dim=1).reshape(num_envs * num_agents, models.obs_dim)
                mean, std = policy_outputs(models.policy_state.params, jnp.asarray(flat_obs.numpy()),
                                           adapters)
                if not np.isfinite(np.asarray(mean)).all() or not np.isfinite(np.asarray(std)).all():
                    raise FloatingPointError(f"Non-finite evaluation policy output at batch {batch}, step {step}")
                actions = np.asarray(deterministic_action(mean))
                torch_actions = list(torch.tensor(
                    np.where(np.asarray(active)[:, None, None],
                             actions.reshape(num_envs, num_agents, -1), 0.0),
                    dtype=torch.float32,
                ).unbind(dim=1))
                next_obs, rewards, dones, _ = env.step(torch_actions)
                diagnostic_started = perf_counter()
                trace.step(env, obs, torch_actions, rewards, dones, tokens, refresh, adapters,
                           mean, std, active, alpha)
                diagnostic_wall += perf_counter() - diagnostic_started
                events.after_step(env, next_obs, tokens, step)
                events.commit()
                obs = next_obs
                active &= ~dones.bool()
                if not bool(active.any()):
                    break
            simulator_environment_steps += trace.step_index * num_envs
            rollout_wall_seconds += perf_counter() - simulation_started
            episode_data.append(trace.episode_arrays())
            trace.save(trace_root / f"round_{rounds:04d}" / f"batch_{batch:02d}.npz")
            if config.query_gate.mode != "disabled":
                events.save(trace_root.parent / "gate_evaluation_traces" /
                            f"round_{rounds:04d}" / f"batch_{batch:02d}.npz")

    results = {key: np.concatenate([batch[key] for batch in episode_data], axis=-1)
               for key in ("team_return", "length", "completed", "team_queries")}
    queries = results["team_queries"]
    metrics = {
        "training_environment_steps": training_steps,
        "simulator_training_environment_steps": rounds * num_envs * config.ppo.rollout_steps,
        "training_round": rounds,
        "metrics_schema_version": 2,
        "eval_seed_start": config.evaluation.seed,
        "evaluation_action_transform": "tanh_mean",
        "evaluation_rng_isolated": True,
        "total_episodes": len(queries),
        "completion_rate": float(results["completed"].mean() * 100) if episodic_success else None,
        "completed_episodes": int(results["completed"].sum()) if episodic_success else None,
        "avg_reward": float(results["team_return"].mean()),
        "std_reward": float(results["team_return"].std()),
        "avg_episode_length": float(results["length"].mean()),
        "team_queries": int(queries.sum()),
        "team_initial_queries": initial_queries,
        "mean_team_queries": float(queries.mean()),
        "std_team_queries": float(queries.std()),
        "generator_batches": len(rollout_numbers) + sum(
            np.count_nonzero(batch["physical_query_batch_size"]) for batch in episode_data),
        "generator_input_contexts": initial_queries + sum(
            int(batch["physical_query_batch_size"].sum()) for batch in episode_data),
        "generator_generated_adapters": initial_queries * num_agents + sum(
            int(batch["generated_agent_adapters_at_step"].sum()) for batch in episode_data),
        "generator_deployed_adapters": initial_queries * num_agents + sum(
            int(batch["deployed_agent_adapters_at_step"].sum()) for batch in episode_data),
        "simulator_environment_steps": simulator_environment_steps,
        "evaluation_wall_seconds": perf_counter() - wall_started,
        "evaluation_cpu_seconds": process_time() - cpu_started,
        "rollout_wall_seconds": rollout_wall_seconds,
        "rollout_diagnostics_wall_seconds": diagnostic_wall,
    }
    if is_navigation:
        per_agent = np.concatenate([batch["agent_queries"] for batch in episode_data], axis=0)
        collected = np.concatenate([batch["agent_first_collected_step"] for batch in episode_data], axis=0)
        metrics.update(
            query_count_unit="agent",
            mean_agent_queries=float(per_agent.mean()),
            agent_queries=int(per_agent.sum()),
            goal_completion_percent=float((collected >= 0).mean() * 100),
        )
    if config.query_gate.mode != "disabled":
        metrics.update(query_gate_mode=config.query_gate.mode,
                       evaluation_gate_policy="seeded_bernoulli_learned_policy",
                       evaluation_coverage_epsilon=0.0)
    for key in episode_data[0]:
        if key.startswith("first_"):
            steps = np.concatenate([batch[key] for batch in episode_data])
            reached = steps >= 0
            metrics[f"{key}_rate"] = float(reached.mean())
            metrics[f"mean_{key}"] = float(steps[reached].mean()) if reached.any() else None
        elif key.startswith("mean_"):
            metrics[key] = float(np.concatenate([batch[key] for batch in episode_data]).mean())
    return metrics
