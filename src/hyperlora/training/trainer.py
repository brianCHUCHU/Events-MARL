from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from ..algorithms.ours import make_train_step
from ..algorithms.cash import make_train_step as make_cash_train_step, sequence_minibatches
from ..algorithms.dico import make_ppo_update as make_dico_ppo_update
from ..algorithms.ps import make_train_step as make_ps_train_step, zero_adapters
from ..config import Config
from ..diversity.control import (alpha_broadcast, alpha_from_ema, alpha_mean_over_envs,
                                 episode_target, update_ema)
from ..diversity.nmd import buffer_entries, calculate_nmd
from ..diversity.sampler import build_sampler
from ..envs.adapters import get_adapter
from ..envs.capabilities import initial_capabilities, sample_capabilities, sample_team_size
from ..envs.events.base import NoEvents
from ..envs.tokens import HypernetworkInputs, capability_context
from ..evaluation import actions as eval_actions
from ..evaluation.quantitative import run_quantitative_evaluation
from ..envs.curriculum import STAGE_KWARGS, football_stage
from ..envs.factory import make_env, team_size
from ..logging.diagnostics import RolloutTrace, environment_manifest
from ..logging.jsonl import RunLogger
from ..diversity.snd_dico import calculate_snd_dico, dico_scaling
from ..models import CentralizedCritic, Hypernetwork, LoRAPolicy
from ..models.critic import CentralizedCriticDeepSets
from ..models.cash_policy import CASHPolicy
from ..models.critic import CentralizedCriticRNN
from ..models.dico_policy import DiCoDeepSetsPolicy, DiCoPolicy
from ..seeding import seed_globals
from .batching import make_indices, ps_minibatch_indices
from .checkpointing import save_checkpoint
from .gae import compute_gae, normalize_returns
from .jitted import make_dico_jitted, make_jitted
from .rollout import collect_rollout, collect_rollout_recurrent


def _initial_buffer_entries(tokens, fields: tuple, num_envs: int, active_agents: int) -> list:
    return buffer_entries(tokens, fields, num_envs, active_agents)


def _nmd_observations(trajectory, num_agents: int) -> list:
    if not trajectory.active_envs:
        return trajectory.obs
    active = np.repeat(np.asarray(trajectory.active_envs).reshape(-1), num_agents)
    return [jnp.concatenate(trajectory.obs, axis=0)[active]]


def _token_batches(token_rows: list, tokens, num_batches: int, num_envs: int,
                   num_agents: int, use_transition_tokens=False) -> dict:
    batch = {}
    varying = ("context", "lidar", "agent_positions", "env_context")
    if use_transition_tokens:
        varying += ("task", "target_nmd")
    for name in varying:
        if token_rows[0][name] is None:
            batch[name] = None
            continue
        flat = jnp.concatenate([row[name] for row in token_rows], axis=0)
        batch[name] = flat.reshape(num_batches * num_envs, num_agents, -1)
    if "generation_lidar" in token_rows[0]:
        batch["generation_lidar"] = jnp.concatenate(
            [row["generation_lidar"] for row in token_rows], axis=0)
    for name in ("task", "target_nmd"):
        if name in batch:
            continue
        value = getattr(tokens, name)
        batch[name] = None if value is None else jnp.tile(
            value[None, :, :, :], (num_batches, 1, 1, 1)).reshape(
                num_batches * num_envs, num_agents, -1)
    return batch


@dataclass
class Models:
    policy: Any
    hypernetwork: Any
    critic: Any
    policy_state: Any
    hn_state: Any
    critic_state: Any
    rng: Any
    obs_dim: int
    action_dim: int
    gate_state: Any = None


def _train_state(model, params, learning_rate: float, max_grad_norm: float, optimizer):
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate, b1=optimizer.beta1, b2=optimizer.beta2,
                   eps=optimizer.eps),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def token_dims(config: Config, obs_dim: int) -> dict[str, int]:
    from ..envs.tokens import lidar_dim, pressure_plate_env_context_dim

    model, env = config.model, config.env
    cap = 1 if model.use_capability_context else 0

    if env.scenario == "navigation":
        context = 2 * cap
    elif env.scenario == "dispersion":
        context = cap + (model.positional_encoding_dim if model.use_positional_context
                         else (env.max_agents if model.use_onehot_context else 0))
    elif env.scenario == "reverse_transport":
        context = 2 * cap
    elif env.scenario == "wind_flocking":
        context = 2 * cap
    elif env.scenario == "pressure_plate":
        context = 0
    elif env.scenario == "football":
        context = 5 * cap
    else:
        raise ValueError(f"token widths are not defined for scenario {env.scenario!r}")

    if env.scenario in ("dispersion", "pressure_plate", "football"):
        lidar = 0
    else:
        lidar = lidar_dim(obs_dim) if model.use_lidar_context else 0

    if env.scenario == "reverse_transport":
        env_context = 3
    elif env.scenario == "pressure_plate":
        env_context = pressure_plate_env_context_dim(model.env_context, env.max_agents)
    else:
        env_context = 0
    if env_context == 0 and model.env_context and env.scenario != "pressure_plate":
        raise ValueError(f"model.env_context is only read for pressure_plate, not {env.scenario!r}")

    return {
        "context": context,
        "lidar": lidar,
        "agent_position": model.agent_position_dim if model.use_agent_position_context else 0,
        "target_nmd": model.target_nmd_dim or 0,
        "task": model.task_embed_dim,
        "env_context": env_context,
    }


def build_models(config: Config, obs_dim: int, action_dim: int, num_agents: int) -> Models:
    model, opt = config.model, config.optimizer
    dims = token_dims(config, obs_dim)
    ldim = dims["lidar"]

    policy = LoRAPolicy(
        hidden_dims=tuple(model.policy_hidden_dims),
        action_dim=action_dim,
        lora_mode=model.lora_mode,
        log_std_max=model.log_std_max,
        min_std=model.min_std,
        discrete_actions=False,
    )
    hypernetwork = Hypernetwork(
        policy_dims={"obs_dim": obs_dim, "hidden_dims": list(model.policy_hidden_dims),
                     "action_dim": action_dim, "lora_rank": model.lora_rank},
        context_dim=dims["context"],
        task_embed_dim=dims["task"],
        lidar_dim=ldim,
        food_position_dim=0,
        agent_position_dim=dims["agent_position"],
        target_nmd_dim=dims["target_nmd"],
        env_context_dim=dims["env_context"],
        max_agents=config.env.max_agents,
        transformer_dim=model.transformer_dim,
        transformer_heads=model.transformer_heads,
        transformer_layers=model.transformer_layers,
        lora_mode=model.lora_mode,
        scaling_factor=model.lora_scaling_factor,
        use_cross_agent_attention=model.use_cross_agent_attention,
    )
    critic = CentralizedCritic(
        hidden_dim=model.critic_hidden_dim,
        num_layers=model.critic_num_layers,
        num_agents=num_agents,
        use_shared_value=model.use_shared_critic,
    )

    rng = jax.random.PRNGKey(config.seed if config.research_metrics else 0)
    rng, policy_rng, hn_rng, critic_rng = jax.random.split(rng, 4)

    dummy_obs = jnp.ones((1, obs_dim))
    dummy_adapters = {}
    input_dim = obs_dim
    for i, output_dim in enumerate(model.policy_hidden_dims):
        dummy_adapters[f"A{i + 1}"] = jnp.ones((1, model.lora_rank, input_dim))
        dummy_adapters[f"B{i + 1}"] = jnp.ones((1, output_dim, model.lora_rank))
        input_dim = output_dim
    final_idx = len(model.policy_hidden_dims) + 1
    dummy_adapters[f"A{final_idx}"] = jnp.ones(
        (1, model.lora_rank, model.policy_hidden_dims[-1]))
    dummy_adapters[f"B{final_idx}"] = jnp.ones((1, action_dim, model.lora_rank))

    policy_params = policy.init(policy_rng, dummy_obs, dummy_adapters)["params"]
    dummy = lambda dim: (jnp.ones((1, config.env.max_agents, dim)) if dim > 0 else None)
    hn_params = hypernetwork.init(
        hn_rng,
        jnp.ones((1, config.env.max_agents, dims["task"])),
        jnp.ones((1, config.env.max_agents, dims["context"])),
        dummy(ldim),
        None,
        dummy(dims["agent_position"]),
        dummy(dims["target_nmd"]),
        dummy(dims["env_context"]),
        None,
    )["params"]
    critic_params = critic.init(
        critic_rng, jnp.ones((1, obs_dim * num_agents)))["params"]

    return Models(
        policy=policy, hypernetwork=hypernetwork, critic=critic,
        policy_state=_train_state(policy, policy_params, opt.learning_rate,
                                  config.ppo.max_grad_norm, opt),
        hn_state=_train_state(hypernetwork, hn_params, opt.hn_learning_rate,
                              config.ppo.hypernetwork_grad_norm, opt),
        critic_state=_train_state(critic, critic_params, opt.critic_learning_rate,
                                  config.ppo.critic_grad_norm, opt),
        rng=rng, obs_dim=obs_dim, action_dim=action_dim,
    )


def policy_obs_dim(config: Config, obs_dim: int) -> int:
    if config.method in ("ps", "cash"):
        return obs_dim + token_dims(config, obs_dim)["env_context"]
    if (config.method == "dico" and config.env.scenario == "dispersion"
            and config.model.use_capability_context):
        return obs_dim + 1
    return obs_dim


def build_ps_models(config: Config, obs_dim: int, action_dim: int, num_agents: int) -> Models:
    model, opt = config.model, config.optimizer
    input_dim = policy_obs_dim(config, obs_dim)

    policy = LoRAPolicy(
        hidden_dims=tuple(model.policy_hidden_dims),
        action_dim=action_dim,
        log_std_max=model.log_std_max,
        min_std=model.min_std,
        discrete_actions=False,
    )
    critic = CentralizedCritic(
        hidden_dim=model.critic_hidden_dim,
        num_layers=model.critic_num_layers,
        num_agents=num_agents,
        use_shared_value=model.use_shared_critic,
    )

    rng = jax.random.PRNGKey(0)
    rng, policy_rng, _hn_rng, critic_rng = jax.random.split(rng, 4)
    policy_params = policy.init(
        policy_rng, jnp.ones((1, input_dim)),
        zero_adapters(1, input_dim, tuple(model.policy_hidden_dims), action_dim))["params"]
    critic_params = critic.init(critic_rng, jnp.ones((1, obs_dim * num_agents)))["params"]

    return Models(
        policy=policy, hypernetwork=None, critic=critic,
        policy_state=_train_state(policy, policy_params, opt.learning_rate,
                                  config.ppo.max_grad_norm, opt),
        hn_state=None,
        critic_state=_train_state(critic, critic_params, opt.critic_learning_rate,
                                  config.ppo.critic_grad_norm, opt),
        rng=rng, obs_dim=obs_dim, action_dim=action_dim,
    )


def _maybe_evaluate(config: Config, episode: int, env, logger, *, num_agents: int,
                    **kwargs) -> dict | None:
    evaluation = config.evaluation
    if not (evaluation.eval_interval and episode > 0
            and episode % evaluation.eval_interval == 0
            and episode >= evaluation.eval_start_episode):
        return None

    scenario = getattr(env, "scenario", None)
    eval_mode = config.env.scenario == "pressure_plate" and hasattr(scenario, "eval_mode")
    if eval_mode:
        scenario.eval_mode = True
    try:
        metrics = run_quantitative_evaluation(config=config, env=env, num_agents=num_agents,
                                              **kwargs)
    finally:
        if eval_mode:
            scenario.eval_mode = False

    if config.method == "ours":
        env.reset()
        pattern = {"speed": [[0.5, 1.5][i % 2] for i in range(num_agents)],
                   "lidar_range": [0.5] * num_agents}
        if scenario is not None and hasattr(scenario, "update_agent_capabilities"):
            scenario.update_agent_capabilities(pattern)

    logger.log({"episode": episode, "eval": True, **metrics})
    return metrics


def _action_dim(config: Config, env) -> int:
    if config.env.scenario == "football":
        return env.action_dim
    return env.get_agent_action_size(env.agents[0])


def _curriculum_kwargs(config: Config, episode: int):
    if config.env.scenario != "football" or config.method == "cash":
        return None, None
    kwargs = config.env.scenario_kwargs
    stage = football_stage(episode,
                           int(kwargs.get("curriculum_stage1_episodes", 500)),
                           int(kwargs.get("curriculum_stage2_episodes", 500)))
    return stage, STAGE_KWARGS[stage]


def train(config: Config, log_dir: Path | None = None, checkpoint_dir: Path | None = None):
    jax.config.update("jax_threefry_partitionable", config.jax_threefry_partitionable)
    if config.method == "ps":
        return train_ps(config, log_dir, checkpoint_dir)
    if config.method == "dico":
        return train_dico(config, log_dir, checkpoint_dir)
    if config.method == "cash":
        return train_cash(config, log_dir, checkpoint_dir)
    if config.method != "ours":
        raise NotImplementedError(f"method {config.method!r} is not supported")
    return train_ours(config, log_dir, checkpoint_dir)


def train_ours(config: Config, log_dir: Path | None = None, checkpoint_dir: Path | None = None):
    np_rng = seed_globals(config.seed)
    log_dir = Path(log_dir or config.logging.log_dir)
    checkpoint_dir = Path(checkpoint_dir or config.logging.checkpoint_dir)
    logger = RunLogger(log_dir)

    adapter = get_adapter(config.env.scenario)
    num_agents = team_size(config.env)
    env = make_env(config.env, initial_capabilities(config.env))
    obs = env.reset()
    if config.research_metrics:
        (log_dir / "environment.json").write_text(
            json.dumps(environment_manifest(config, env), indent=2, allow_nan=False) + "\n")
    obs_dim = obs[0].shape[-1]
    action_dim = _action_dim(config, env)

    models = build_models(config, obs_dim, action_dim, num_agents)
    gate_enabled = config.query_gate.mode != "disabled"
    if gate_enabled:
        from .query_gate import QueryGate, LearnedPressurePlateEvents, update_gate

        gate = QueryGate(config.query_gate.hidden_dim)
        gate_params = gate.init(jax.random.fold_in(jax.random.PRNGKey(config.seed), 9000),
                                jnp.ones((1, obs_dim*num_agents+1)))["params"]
        models.gate_state = _train_state(gate, gate_params, config.query_gate.learning_rate,
                                         config.ppo.max_grad_norm, config.optimizer)
    train_step = make_train_step(models.policy, models.hypernetwork, models.critic,
                                 config.ppo.adapter_regularization_coef)
    get_static_adapters, get_actions, get_value = make_jitted(
        models.policy, models.hypernetwork, models.critic)
    hn_apply = lambda inputs, scaling: get_static_adapters(
        models.hn_state.params, *inputs.as_args(), scaling)

    eval_actions_fn, eval_tokens_fn, eval_after_step_fn = eval_actions.ours(
        models, config, adapter, num_agents, obs_dim, hn_apply)

    diversity = config.diversity
    nmd_ema = diversity.target_nmd
    target_sampler = build_sampler(diversity.target_nmd_sampler)
    target_per_env = None
    previous_trajectory_obs = None
    value_clip_range = config.ppo.ppo_clip_epsilon if config.ppo.clip_value_loss else None

    def research_evaluate(rounds, environment_steps):
        from ..evaluation.research import evaluate

        metrics = evaluate(config, adapter, models, hn_apply, policy_outputs, nmd_ema,
                           rounds, log_dir / "evaluation_traces", environment_steps)
        logger.log({"episode": rounds - 1, "eval": True, **metrics})
        save_checkpoint(checkpoint_dir / f"evaluated_round_{rounds:04d}.npz",
                        policy_params=models.policy_state.params, critic_params=models.critic_state.params,
                        hn_params=models.hn_state.params, episode=rounds, nmd_ema=nmd_ema,
                        gate_params=None if not gate_enabled else models.gate_state.params)

    if config.research_metrics:
        @jax.jit
        def policy_outputs(params, observations, adapters):
            mean, log_std = models.policy.apply({"params": params}, observations, adapters)
            return mean, jnp.maximum(jnp.exp(log_std), models.policy.min_std)

        research_evaluate(0, 0)

    curriculum_stage = None
    environment_steps = 0
    for episode in range(config.ppo.num_episodes):
        started, cpu_started = time.perf_counter(), time.process_time()
        buffer: list = []

        stage, stage_kwargs = _curriculum_kwargs(config, episode)
        if stage is not None and stage != curriculum_stage:
            env = make_env(config.env, initial_capabilities(config.env), curriculum=stage_kwargs)
            curriculum_stage = stage

        current_num_agents = sample_team_size(config.env, np_rng)
        capabilities = sample_capabilities(config.env, current_num_agents, np_rng)
        adapter.apply_capabilities(config, env, capabilities, current_num_agents)
        obs = env.reset()

        target_nmd = episode_target(config, episode)
        target_token_value = target_nmd
        if target_sampler is not None:
            target_per_env = target_sampler.per_env(config.env.num_envs, np_rng)
            target_token_value = target_per_env
            target_nmd = float(target_per_env.mean())
        tokens = adapter.build_tokens(config, env, obs, obs_dim, config.env.num_envs, num_agents,
                                      current_num_agents, target_token_value)
        adapters = hn_apply(tokens, 1.0)
        buffer.extend(_initial_buffer_entries(
            tokens, adapter.buffer_token_fields, config.env.num_envs,
            current_num_agents if config.env.mixed_agent_training else num_agents))

        nmd = None
        nmd_input_contexts = 0
        if diversity.use_adapter_nmd and buffer and previous_trajectory_obs is not None:
            nmd_input_contexts = min(len(buffer), diversity.adapter_nmd_sample_size)
            nmd = adapter.nmd(
                buffer=buffer, config=config, hn_params=models.hn_state.params,
                policy_params=models.policy_state.params, hypernetwork=models.hypernetwork,
                policy=models.policy, num_agents=num_agents,
                rng_key=jax.random.PRNGKey(episode * 1000 + 42),
                trajectory_obs=previous_trajectory_obs,
            )
        alpha = 1.0
        if diversity.use_diversity_control and nmd is not None and np.isfinite(nmd):
            nmd_ema = update_ema(nmd_ema, nmd, diversity.nmd_moving_average_coef)
            alpha = alpha_mean_over_envs(
                alpha_from_ema(target_nmd, nmd_ema, diversity),
                config.env.num_envs)
            adapters = hn_apply(
                tokens, alpha_broadcast(alpha, config.env.num_envs, num_agents))

        events = adapter.make_events(config, config.env.num_envs, num_agents, obs_dim,
                                     episode, nmd_ema)
        if gate_enabled:
            events = LearnedPressurePlateEvents(config, config.env.num_envs, num_agents,
                                                 models.gate_state, episode)
        events.prime(obs)

        num_envs = config.env.num_envs
        rounds = episode + 1
        final_round = rounds == config.ppo.num_episodes
        scheduled_eval = (config.evaluation.eval_interval
                          and rounds % config.evaluation.eval_interval == 0
                          and rounds >= config.evaluation.eval_start_episode)
        trace = None
        if config.research_metrics and (episode == 0 or scheduled_eval or final_round):
            trace = RolloutTrace(config.env.scenario, num_envs, num_agents,
                                 config.ppo.rollout_steps, episode * 4)
        rollout_started = time.perf_counter()
        result = collect_rollout(
            env=env, obs=obs, policy_params=models.policy_state.params,
            critic_params=models.critic_state.params, adapters=adapters, hn_apply=hn_apply,
            tokens=tokens, events=events, get_actions=get_actions, get_value=get_value,
            config=config, num_agents=num_agents, current_num_agents=current_num_agents,
            episode=episode, alpha=alpha, buffer=buffer, trace=trace,
        )
        traj, obs, tokens = result.trajectory, result.obs, result.tokens
        alpha = result.alpha
        previous_trajectory_obs = _nmd_observations(traj, num_agents)
        rollout_finished = time.perf_counter()
        valid_steps = np.asarray(traj.active_envs) if traj.active_envs else None
        flat_steps = None if valid_steps is None else valid_steps.reshape(-1)
        simulator_steps = ppo_steps = config.ppo.rollout_steps * num_envs
        if flat_steps is not None:
            ppo_steps = int(flat_steps.sum())
        environment_steps += ppo_steps

        learning_rewards = traj.rewards
        gate_metrics = {}
        if gate_enabled:
            gate_data = events.arrays()
            cost = config.query_gate.query_cost * gate_data["query"][..., None]
            # Preserve baseline clipping scale; charge refresh cost after raw
            # reward clipping so a large arrival bonus cannot erase its cost.
            learning_rewards = (np.clip(np.asarray(traj.rewards).reshape(
                config.ppo.rollout_steps, num_envs, num_agents), -10, 10) - cost)
            gate_update_started = time.perf_counter()
            models.gate_state, gate_metrics = update_gate(
                config, models.gate_state, events, learning_rewards.mean(-1), traj.dones, episode)
            gate_metrics.update(gate_ppo_wall_seconds=time.perf_counter()-gate_update_started,
                                gate_total_query_cost=float(cost.sum()),
                                gate_valid_decisions=int(gate_data["valid"].sum()))

        advantages, returns, old_values = compute_gae(
            learning_rewards, [np.asarray(v) for v in traj.values], np.asarray(traj.dones),
            gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
            reward_scale=config.ppo.reward_scale, num_envs=config.env.num_envs,
            num_agents=num_agents, current_num_agents=current_num_agents,
            advantage_clip_range=config.ppo.advantage_clip_range,
            mixed_agent_training=config.env.mixed_agent_training, valid_steps=valid_steps,
        )

        obs_batch = jnp.concatenate(traj.obs, axis=0)
        global_state_batch = jnp.concatenate(traj.global_states, axis=0)
        if config.ppo.clip_global_state:
            global_state_batch = jnp.clip(global_state_batch, config.ppo.clip_global_state_min,
                                          config.ppo.clip_global_state_max)
        actions_batch = jnp.concatenate(traj.actions, axis=0)
        old_log_probs_batch = jnp.clip(
            jnp.nan_to_num(jnp.concatenate(traj.log_probs, axis=0), nan=-100.0, posinf=-100.0,
                           neginf=-100.0), -100.0, 100.0)
        advantages_batch = jnp.array(advantages)
        returns_batch = jnp.array(returns)
        if config.ppo.normalize_returns:
            valid_agents = None if flat_steps is None else jnp.asarray(np.repeat(flat_steps, num_agents))
            returns_batch, old_values = normalize_returns(returns_batch, old_values, valid_agents)
        old_values_batch = jnp.array(old_values)

        num_batches = obs_batch.shape[0] // (config.env.num_envs * num_agents)
        token_batch = _token_batches(traj.token_rows, tokens, num_batches, config.env.num_envs,
                                     num_agents, use_transition_tokens=config.research_metrics)
        action_scales = (jnp.repeat(jnp.asarray(traj.adapter_scales), num_envs)
                         if config.research_metrics else None)
        mask_batch = None
        if tokens.mask is not None:
            expanded = jnp.tile(tokens.mask, (config.env.num_envs, 1, 1, 1))
            mask_batch = jnp.tile(expanded[None, ...], (num_batches, 1, 1, 1, 1)).reshape(
                num_batches * config.env.num_envs, 1, num_agents, num_agents)

        entropy_coef = max(config.ppo.min_entropy_coef,
                           config.ppo.entropy_coef * (config.ppo.entropy_decay ** episode))

        loss_info = None
        across_epochs = None
        first_minibatch_log_prob_mse = None
        ppo_updates = 0
        ppo_input_contexts = 0
        num_minibatches = config.ppo.num_minibatches if config.ppo.use_minibatches else 1
        for _epoch in range(config.ppo.ppo_epochs):
            shuffle_rng = None
            if config.ppo.use_minibatches and config.ppo.shuffle_minibatches:
                models.rng, shuffle_rng = jax.random.split(models.rng)
            indices = make_indices(
                obs_batch.shape[0], num_batches * config.env.num_envs, num_agents,
                num_minibatches, config.ppo.use_minibatches and config.ppo.shuffle_minibatches,
                shuffle_rng)

            accumulated = None
            epoch_updates = 0
            for mb in range(num_minibatches):
                rows, env_rows = indices.slice(mb)
                batch_valid = None
                if flat_steps is not None:
                    selected = flat_steps[np.asarray(env_rows)]
                    if not selected.any():
                        continue
                    batch_valid = jnp.asarray(selected)
                policy_state, hn_state, critic_state, loss_info = train_step(
                    models.policy_state, models.hn_state, models.critic_state,
                    obs_batch[rows], global_state_batch[env_rows], actions_batch[rows],
                    old_log_probs_batch[rows], advantages_batch[rows], returns_batch[rows],
                    *(None if token_batch[name] is None else token_batch[name][env_rows]
                      for name in ("task", "context", "lidar", "agent_positions", "target_nmd",
                                   "env_context")),
                    None if mask_batch is None else mask_batch[env_rows],
                    config.ppo.ppo_clip_epsilon, entropy_coef, config.ppo.value_loss_coef,
                    num_agents, current_num_agents,
                    alpha if action_scales is None else action_scales[env_rows],
                    old_values_batch[rows],
                    value_clip_range=value_clip_range, valid_steps=batch_valid,
                    generation_lidar=(token_batch["generation_lidar"][rows]
                                      if "generation_lidar" in token_batch else None),
                )
                if config.research_metrics:
                    invalid = [key for key, value in loss_info.items()
                               if not np.isfinite(np.asarray(value)).all()]
                    if invalid:
                        raise FloatingPointError(f"Non-finite PPO diagnostics at round {rounds}: {invalid}")
                    if first_minibatch_log_prob_mse is None and float(loss_info["approx_kl"]) > 1e-6:
                        raise RuntimeError(f"Rollout/PPO log-prob mismatch before update at round {rounds}: "
                                           f"{float(loss_info['approx_kl'])}")
                models.policy_state, models.hn_state, models.critic_state = (
                    policy_state, hn_state, critic_state)
                epoch_updates += 1
                ppo_input_contexts += len(rows) if "generation_lidar" in token_batch else len(env_rows)
                if config.research_metrics and first_minibatch_log_prob_mse is None:
                    first_minibatch_log_prob_mse = loss_info["approx_kl"]
                accumulated = (loss_info if accumulated is None
                               else {k: accumulated[k] + v for k, v in loss_info.items()})
            ppo_updates += epoch_updates
            loss_info = {k: v / epoch_updates for k, v in accumulated.items()}
            if config.research_metrics:
                across_epochs = (loss_info if across_epochs is None else
                                 {key: across_epochs[key] + value for key, value in loss_info.items()})

        instrumentation = {}
        if config.research_metrics:
            loss_info = {key: value / config.ppo.ppo_epochs for key, value in across_epochs.items()}
            loss_info["pre_update_log_prob_mse"] = first_minibatch_log_prob_mse
            jax.block_until_ready(loss_info)
            update_finished = time.perf_counter()
            teams_per_batch = [(step, len(set(np.asarray(idx) // num_agents)))
                               for step, idx in traj.requery_events]
            instrumentation = {
                "training_round": rounds,
                "metrics_schema_version": 2,
                "training_environment_steps": environment_steps,
                "training_agent_steps": environment_steps * num_agents,
                "simulator_training_environment_steps": rounds * simulator_steps,
                "masked_post_terminal_env_steps": simulator_steps - ppo_steps,
                "pre_rollout_wall_seconds": rollout_started - started,
                "rollout_wall_seconds": rollout_finished - rollout_started,
                "update_wall_seconds": update_finished - rollout_finished,
                "training_cpu_seconds": time.process_time() - cpu_started,
                "team_initial_queries": num_envs,
                "team_requeries": sum(count for _, count in teams_per_batch),
                "generator_rollout_batches": 1 + len(teams_per_batch),
                "generator_setup_batches": 1 + int(diversity.use_diversity_control
                                                    and nmd is not None and np.isfinite(nmd)),
                "generator_ppo_batches": ppo_updates,
                "generator_setup_input_contexts": num_envs * (1 + int(
                    diversity.use_diversity_control and nmd is not None and np.isfinite(nmd))),
                "generator_refresh_input_contexts": sum(c for c, _ in traj.generator_refresh_sizes),
                "generator_refresh_generated_adapters": sum(a for _, a in traj.generator_refresh_sizes),
                "generator_ppo_input_contexts": ppo_input_contexts,
                "generator_ppo_generated_adapters": ppo_input_contexts * num_agents,
                "generator_nmd_input_contexts": nmd_input_contexts,
                "team_requery_batches": teams_per_batch,
                "agent_initial_queries": num_envs * num_agents,
                "agent_requeries": sum(len(idx) for _, idx in traj.requery_events),
                "first_terminal_fraction": float(np.asarray(traj.dones).any(axis=0).mean()),
            }
        logger.log({
            "episode": episode,
            **instrumentation,
            **gate_metrics,
            "avg_reward": (float(np.asarray(traj.rewards).sum() / (ppo_steps * num_agents))
                           if flat_steps is not None else float(np.mean(traj.episode_rewards))),
            "nmd": nmd,
            "alpha": alpha,
            "nmd_ema": nmd_ema,
            "target_nmd": target_nmd,
            **({"target_nmd_min": float(target_per_env.min()),
                "target_nmd_max": float(target_per_env.max())}
               if target_sampler is not None else {}),
            "team_size": current_num_agents,
            "requeries": sum(len(idx) for _, idx in traj.requery_events),
            "resets": traj.num_resets,
            "seconds": round(time.perf_counter() - started, 3),
            **{k: float(v) for k, v in loss_info.items()},
        })

        if trace is not None:
            trace.save(log_dir / "training_traces" / f"round_{rounds:04d}.npz")
            if gate_enabled:
                events.save(log_dir / "gate_training_traces" / f"round_{rounds:04d}.npz")
        if config.research_metrics:
            if scheduled_eval or final_round:
                research_evaluate(rounds, environment_steps)
        else:
            _maybe_evaluate(config, episode, env, logger, models=models, adapter=adapter,
                            num_agents=num_agents, obs_dim=obs_dim, nmd_ema=nmd_ema,
                            actions_fn=eval_actions_fn, tokens_fn=eval_tokens_fn,
                            after_step_fn=eval_after_step_fn)

        if config.logging.save_interval and episode and episode % config.logging.save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_{episode}.npz",
                            policy_params=models.policy_state.params,
                            critic_params=models.critic_state.params, episode=episode,
                            hn_params=models.hn_state.params, nmd_ema=nmd_ema,
                            gate_params=None if not gate_enabled else models.gate_state.params)

    save_checkpoint(checkpoint_dir / "final_checkpoint.npz",
                    policy_params=models.policy_state.params,
                    critic_params=models.critic_state.params, episode=config.ppo.num_episodes,
                    hn_params=models.hn_state.params,
                    gate_params=None if not gate_enabled else models.gate_state.params,
                    nmd_ema=nmd_ema if diversity.use_diversity_control else None)
    return models


def train_ps(config: Config, log_dir: Path | None = None, checkpoint_dir: Path | None = None):
    np_rng = seed_globals(config.seed)
    log_dir = Path(log_dir or config.logging.log_dir)
    checkpoint_dir = Path(checkpoint_dir or config.logging.checkpoint_dir)
    logger = RunLogger(log_dir)

    adapter = get_adapter(config.env.scenario)
    num_agents = team_size(config.env)
    env = make_env(config.env, initial_capabilities(config.env))
    obs = env.reset()
    obs_dim = obs[0].shape[-1]
    action_dim = _action_dim(config, env)

    models = build_ps_models(config, obs_dim, action_dim, num_agents)
    train_step = make_ps_train_step(models.policy, models.critic)
    _unused, get_actions, get_value = make_jitted(models.policy, None, models.critic)

    num_envs = config.env.num_envs
    batch_size = num_envs * num_agents
    input_dim = policy_obs_dim(config, obs_dim)
    adapters = zero_adapters(batch_size, input_dim, tuple(config.model.policy_hidden_dims),
                             action_dim)
    empty_tokens = HypernetworkInputs()
    eval_actions_fn, eval_tokens_fn = eval_actions.ps(models, config, adapter, num_agents, obs_dim,
                                                      input_dim, action_dim)

    curriculum_stage = None
    for episode in range(config.ppo.num_episodes):
        started = time.time()

        stage, stage_kwargs = _curriculum_kwargs(config, episode)
        if stage is not None and stage != curriculum_stage:
            env = make_env(config.env, initial_capabilities(config.env), curriculum=stage_kwargs)
            curriculum_stage = stage

        current_num_agents = sample_team_size(config.env, np_rng)
        capabilities = sample_capabilities(config.env, current_num_agents, np_rng)
        adapter.apply_capabilities(config, env, capabilities, current_num_agents)
        obs = env.reset()

        obs_suffix = None
        if adapter.env_context is not None and token_dims(config, obs_dim)["env_context"] > 0:
            context = adapter.env_context(config, env, num_envs, num_agents)
            obs_suffix = jnp.asarray(context.cpu().numpy()).reshape(batch_size, -1)

        result = collect_rollout(
            env=env, obs=obs, policy_params=models.policy_state.params,
            critic_params=models.critic_state.params, adapters=adapters, hn_apply=None,
            tokens=empty_tokens, events=NoEvents(num_envs, num_agents), get_actions=get_actions,
            get_value=get_value, config=config, num_agents=num_agents,
            current_num_agents=current_num_agents, episode=episode, alpha=1.0, buffer=[],
            obs_suffix=obs_suffix,
        )
        traj, obs = result.trajectory, result.obs

        advantages, returns, old_values = compute_gae(
            traj.rewards, [np.asarray(v) for v in traj.values], np.asarray(traj.dones),
            gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
            reward_scale=config.ppo.reward_scale, num_envs=num_envs,
            num_agents=num_agents, current_num_agents=current_num_agents,
            advantage_clip_range=config.ppo.advantage_clip_range,
            mixed_agent_training=config.env.mixed_agent_training,
        )

        obs_batch = jnp.concatenate(traj.obs, axis=0)
        global_state_batch = jnp.concatenate(traj.global_states, axis=0)
        if config.ppo.clip_global_state:
            global_state_batch = jnp.clip(global_state_batch, config.ppo.clip_global_state_min,
                                          config.ppo.clip_global_state_max)
        actions_batch = jnp.concatenate(traj.actions, axis=0)
        old_log_probs_batch = jnp.clip(
            jnp.nan_to_num(jnp.concatenate(traj.log_probs, axis=0), nan=-100.0, posinf=-100.0,
                           neginf=-100.0), -100.0, 100.0)
        advantages_batch = jnp.array(advantages)
        returns_batch = jnp.array(returns)
        if config.ppo.normalize_returns:
            returns_batch, _old_values = normalize_returns(returns_batch, old_values)

        entropy_coef = max(config.ppo.min_entropy_coef,
                           config.ppo.entropy_coef * (config.ppo.entropy_decay ** episode))

        loss_info = None
        for _epoch in range(config.ppo.ppo_epochs):
            shuffle_rng = None
            if config.ppo.use_minibatches and config.ppo.shuffle_minibatches:
                models.rng, shuffle_rng = jax.random.split(models.rng)
            rows = ps_minibatch_indices(obs_batch.shape[0], config.ppo.num_minibatches,
                                        config.ppo.shuffle_minibatches, shuffle_rng)

            accumulated = None
            for mb in range(config.ppo.num_minibatches):
                index = rows[mb]
                policy_state, critic_state, loss_info = train_step(
                    models.policy_state, models.critic_state,
                    obs_batch[index],
                    global_state_batch[index],
                    actions_batch[index], old_log_probs_batch[index],
                    advantages_batch[index], returns_batch[index], adapters,
                    config.ppo.ppo_clip_epsilon, entropy_coef, config.ppo.value_loss_coef,
                    num_agents, current_num_agents,
                )
                models.policy_state, models.critic_state = policy_state, critic_state
                accumulated = (loss_info if accumulated is None
                               else {k: accumulated[k] + v for k, v in loss_info.items()})
            loss_info = {k: v / config.ppo.num_minibatches for k, v in accumulated.items()}

        logger.log({
            "episode": episode,
            "avg_reward": float(np.mean(traj.episode_rewards)),
            "team_size": current_num_agents,
            "resets": traj.num_resets,
            "seconds": round(time.time() - started, 3),
            **{k: float(v) for k, v in loss_info.items()},
        })

        _maybe_evaluate(config, episode, env, logger, models=models, adapter=adapter,
                        num_agents=num_agents, obs_dim=obs_dim,
                        actions_fn=eval_actions_fn, tokens_fn=eval_tokens_fn)

        if config.logging.save_interval and episode and episode % config.logging.save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_{episode}.npz",
                            policy_params=models.policy_state.params,
                            critic_params=models.critic_state.params, episode=episode)

    save_checkpoint(checkpoint_dir / "final_checkpoint.npz",
                    policy_params=models.policy_state.params,
                    critic_params=models.critic_state.params,
                    episode=config.ppo.num_episodes)
    return models


def build_dico_models(config: Config, obs_dim: int, action_dim: int, num_agents: int) -> Models:
    model, opt = config.model, config.optimizer
    if not config.diversity.use_diversity_control:
        raise NotImplementedError(
            "DiCo requires diversity control")

    deep_sets = config.env.scenario == "football"
    input_dim = policy_obs_dim(config, obs_dim)

    if deep_sets:
        policy = DiCoDeepSetsPolicy(
            num_agents=num_agents, hidden_dims=tuple(model.policy_hidden_dims),
            action_dim=action_dim, gat_out_features=32, edge_radius=10.0)
        critic = CentralizedCriticDeepSets(
            hidden_dim=model.critic_hidden_dim, num_layers=model.critic_num_layers,
            num_agents=num_agents, gat_out_features=32,
            use_shared_value=model.use_shared_critic)
    else:
        policy = DiCoPolicy(num_agents=num_agents, hidden_dims=tuple(model.policy_hidden_dims),
                            action_dim=action_dim)
        critic = CentralizedCritic(
            hidden_dim=model.critic_hidden_dim, num_layers=model.critic_num_layers,
            num_agents=num_agents, use_shared_value=model.use_shared_critic)

    rng = jax.random.PRNGKey(0)
    rng, policy_rng, _hn_rng, critic_rng = jax.random.split(rng, 4)
    if deep_sets:
        dummy_obs = jnp.ones((num_agents, input_dim))
        dummy_agent_ids = jnp.arange(num_agents)
    else:
        dummy_obs = jnp.ones((1, input_dim))
        dummy_agent_ids = jnp.array([0])
    policy_params = policy.init(policy_rng, dummy_obs, dummy_agent_ids)["params"]
    critic_params = critic.init(critic_rng, jnp.ones((1, obs_dim * num_agents)))["params"]

    return Models(
        policy=policy, hypernetwork=None, critic=critic,
        policy_state=_train_state(policy, policy_params, opt.learning_rate,
                                  config.ppo.max_grad_norm, opt),
        hn_state=None,
        critic_state=_train_state(critic, critic_params, opt.critic_learning_rate,
                                  config.ppo.critic_grad_norm, opt),
        rng=rng, obs_dim=obs_dim, action_dim=action_dim,
    )


def _dico_capability_suffix(config: Config, capabilities, num_envs: int, num_agents: int):
    if not (config.env.scenario == "dispersion" and config.model.use_capability_context):
        return None
    speeds = np.tile(np.array(capabilities["max_speed"], dtype=np.float32), num_envs)
    return jnp.asarray(speeds[:, np.newaxis])


def train_dico(config: Config, log_dir: Path | None = None, checkpoint_dir: Path | None = None):
    np_rng = seed_globals(config.seed)
    log_dir = Path(log_dir or config.logging.log_dir)
    checkpoint_dir = Path(checkpoint_dir or config.logging.checkpoint_dir)
    logger = RunLogger(log_dir)

    adapter = get_adapter(config.env.scenario)
    num_agents = team_size(config.env)
    env = make_env(config.env, initial_capabilities(config.env))
    obs = env.reset()
    obs_dim = obs[0].shape[-1]
    action_dim = _action_dim(config, env)

    models = build_dico_models(config, obs_dim, action_dim, num_agents)
    ppo_update = make_dico_ppo_update(models.policy, models.critic)
    get_dico_actions, get_value = make_dico_jitted(models.policy, models.critic)

    diversity = config.diversity
    if diversity.num_snd_samples is None:
        raise ValueError("DiCo needs `diversity.num_snd_samples`")
    snd_ema = diversity.target_snd
    num_envs = config.env.num_envs
    empty_tokens = HypernetworkInputs()

    eval_actions_fn, eval_tokens_fn = eval_actions.dico(models, config, num_agents)

    curriculum_stage = None
    for episode in range(config.ppo.num_episodes):
        started = time.time()

        stage, stage_kwargs = _curriculum_kwargs(config, episode)
        if stage is not None and stage != curriculum_stage:
            env = make_env(config.env, initial_capabilities(config.env), curriculum=stage_kwargs)
            curriculum_stage = stage

        current_num_agents = sample_team_size(config.env, np_rng)
        capabilities = sample_capabilities(config.env, current_num_agents, np_rng)
        adapter.apply_capabilities(config, env, capabilities, current_num_agents)
        obs = env.reset()

        obs_suffix = _dico_capability_suffix(config, capabilities, num_envs, num_agents)

        sample_size = min(num_envs, diversity.num_snd_samples)
        common_obs = jnp.asarray(obs[0][:sample_size].detach().cpu().numpy())
        sample_obs = jnp.repeat(common_obs, num_agents, axis=0)
        if obs_suffix is not None:
            speeds = np.tile(np.array(capabilities["max_speed"], dtype=np.float32), sample_size)
            sample_obs = jnp.concatenate([sample_obs, jnp.asarray(speeds[:, np.newaxis])],
                                         axis=-1)
        snd = calculate_snd_dico(models.policy_state.params, sample_obs, models.policy,
                                 num_agents, sample_size, diversity_scaling=1.0)
        if not (jnp.isnan(snd) or jnp.isinf(snd)):
            snd_ema = update_ema(snd_ema, snd, diversity.snd_moving_average_coef)
        else:
            snd = snd_ema
        scaling = dico_scaling(diversity.target_snd, snd_ema,
                               diversity.max_diversity_scaling, diversity.min_snd_floor)

        agent_ids = jnp.tile(jnp.arange(num_agents), num_envs)
        get_actions = lambda params, obs_batch, _adapters, rng_key: get_dico_actions(
            params, obs_batch, agent_ids, scaling, rng_key)

        result = collect_rollout(
            env=env, obs=obs, policy_params=models.policy_state.params,
            critic_params=models.critic_state.params, adapters=None, hn_apply=None,
            tokens=empty_tokens, events=NoEvents(num_envs, num_agents), get_actions=get_actions,
            get_value=get_value, config=config, num_agents=num_agents,
            current_num_agents=current_num_agents, episode=episode, alpha=1.0, buffer=[],
            obs_suffix=obs_suffix,
        )
        traj, obs = result.trajectory, result.obs

        advantages, returns, old_values = compute_gae(
            traj.rewards, [np.asarray(v) for v in traj.values], np.asarray(traj.dones),
            gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
            reward_scale=config.ppo.reward_scale, num_envs=num_envs, num_agents=num_agents,
            current_num_agents=current_num_agents,
            advantage_clip_range=config.ppo.advantage_clip_range,
            mixed_agent_training=config.env.mixed_agent_training,
        )

        obs_batch = jnp.concatenate(traj.obs, axis=0)
        global_state_batch = jnp.concatenate(traj.global_states, axis=0)
        if config.ppo.clip_global_state:
            global_state_batch = jnp.clip(global_state_batch, config.ppo.clip_global_state_min,
                                          config.ppo.clip_global_state_max)
        actions_batch = jnp.concatenate(traj.actions, axis=0)
        old_log_probs_batch = jnp.clip(
            jnp.nan_to_num(jnp.concatenate(traj.log_probs, axis=0), nan=-100.0, posinf=-100.0,
                           neginf=-100.0), -100.0, 100.0)

        advantages_batch = jnp.array(advantages)
        returns_batch = jnp.array(returns)
        if config.ppo.normalize_returns:
            returns_batch, _old_values = normalize_returns(returns_batch, old_values)

        entropy_coef = max(config.ppo.min_entropy_coef,
                           config.ppo.entropy_coef * (config.ppo.entropy_decay ** episode))
        num_timesteps = obs_batch.shape[0] // (num_envs * num_agents)
        agent_ids_batch = jnp.tile(jnp.arange(num_agents), num_timesteps * num_envs)

        policy_state, critic_state, loss_info = ppo_update(
            models.policy_state, models.critic_state, obs_batch, global_state_batch,
            actions_batch, old_log_probs_batch, advantages_batch, returns_batch,
            agent_ids_batch, scaling, config.ppo.ppo_clip_epsilon, entropy_coef,
            config.ppo.value_loss_coef,
            num_minibatches=config.ppo.num_minibatches, ppo_epochs=config.ppo.ppo_epochs,
            num_agents=num_agents, current_num_agents=current_num_agents)
        models.policy_state, models.critic_state = policy_state, critic_state

        logger.log({
            "episode": episode,
            "avg_reward": float(np.mean(traj.episode_rewards)),
            "snd": float(snd),
            "lambda": float(scaling),
            "snd_ema": float(snd_ema),
            "team_size": current_num_agents,
            "resets": traj.num_resets,
            "seconds": round(time.time() - started, 3),
            **{k: float(v) for k, v in loss_info.items()},
        })

        _maybe_evaluate(config, episode, env, logger, models=models, adapter=adapter,
                        num_agents=num_agents, obs_dim=obs_dim,
                        actions_fn=eval_actions_fn, tokens_fn=eval_tokens_fn)

        if config.logging.save_interval and episode and episode % config.logging.save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_{episode}.npz",
                            policy_params=models.policy_state.params,
                            critic_params=models.critic_state.params, episode=episode,
                            snd_ema=snd_ema)

    save_checkpoint(checkpoint_dir / "final_checkpoint.npz",
                    policy_params=models.policy_state.params,
                    critic_params=models.critic_state.params,
                    episode=config.ppo.num_episodes, snd_ema=snd_ema)
    return models


def build_cash_models(config: Config, obs_dim: int, action_dim: int, num_agents: int,
                      capability_dim: int, policy_input_dim: int) -> Models:
    model, opt = config.model, config.optimizer
    missing = [name for name in ("gru_hidden_dim", "fc_dim_size", "cash_hyper_hidden_dim",
                                 "cash_hyper_layers", "cash_decoder_hidden_dim",
                                 "cash_use_two_layer_decoder")
               if getattr(model, name) is None]
    if missing:
        raise ValueError(f"CASH needs model.{{{', '.join(missing)}}}")

    policy = CASHPolicy(
        num_agents=num_agents,
        capability_dim=capability_dim,
        gru_hidden_dim=model.gru_hidden_dim,
        fc_dim_size=model.fc_dim_size,
        action_dim=action_dim,
        hyper_hidden_dim=model.cash_hyper_hidden_dim,
        hyper_num_layers=model.cash_hyper_layers,
        decoder_hidden_dim=model.cash_decoder_hidden_dim,
        use_two_layer_decoder=model.cash_use_two_layer_decoder,
        log_std_max=model.log_std_max,
        min_std=model.min_std,
    )
    critic = CentralizedCriticRNN(
        gru_hidden_dim=model.gru_hidden_dim,
        fc_dim_size=model.gru_hidden_dim,
        num_agents=num_agents,
        use_shared_value=model.use_shared_critic,
    )

    rng = jax.random.PRNGKey(0)
    rng, policy_rng, _hn_rng, critic_rng = jax.random.split(rng, 4)
    dummy_hidden = jnp.zeros((num_agents, model.gru_hidden_dim))
    dummy_x = (jnp.ones((1, num_agents, policy_input_dim)),
               jnp.zeros((1, num_agents), dtype=bool),
               jnp.ones((1, num_agents, capability_dim)))
    policy_params = policy.init(policy_rng, dummy_hidden, dummy_x)["params"]
    critic_params = critic.init(
        critic_rng, jnp.zeros((1, model.gru_hidden_dim)),
        (jnp.ones((1, 1, obs_dim * num_agents)), jnp.zeros((1, 1), dtype=bool)))["params"]

    return Models(
        policy=policy, hypernetwork=None, critic=critic,
        policy_state=_train_state(policy, policy_params, opt.learning_rate,
                                  config.ppo.max_grad_norm, opt),
        hn_state=None,
        critic_state=_train_state(critic, critic_params, opt.critic_learning_rate,
                                  config.ppo.critic_grad_norm, opt),
        rng=rng, obs_dim=obs_dim, action_dim=action_dim,
    )


def train_cash(config: Config, log_dir: Path | None = None, checkpoint_dir: Path | None = None):
    np_rng = seed_globals(config.seed)
    log_dir = Path(log_dir or config.logging.log_dir)
    checkpoint_dir = Path(checkpoint_dir or config.logging.checkpoint_dir)
    logger = RunLogger(log_dir)

    adapter = get_adapter(config.env.scenario)
    num_agents = team_size(config.env)
    env = make_env(config.env, initial_capabilities(config.env))
    obs = env.reset()
    obs_dim = obs[0].shape[-1]
    action_dim = _action_dim(config, env)
    capability_dim = token_dims(config, obs_dim)["context"]

    models = build_cash_models(config, obs_dim, action_dim, num_agents, capability_dim,
                               policy_obs_dim(config, obs_dim))
    train_step = make_cash_train_step(models.policy, models.critic)

    @jax.jit
    def get_actions(policy_params, obs_batch, capabilities, rng_key, hidden, dones):
        action, log_prob, new_hidden, mean, std = models.policy.apply(
            {"params": policy_params}, hidden,
            (obs_batch[None, ...], dones[None, ...], capabilities[None, ...]), rng_key,
            method=models.policy.get_action_and_log_prob)
        return (jnp.squeeze(action, axis=0), jnp.squeeze(log_prob, axis=0), new_hidden,
                jnp.squeeze(mean, axis=0), jnp.squeeze(std, axis=0))

    @jax.jit
    def get_value(critic_params, global_state, critic_hidden, dones):
        new_hidden, value = models.critic.apply(
            {"params": critic_params}, critic_hidden,
            (jnp.expand_dims(global_state, 0), jnp.expand_dims(dones, 0)))
        return new_hidden, jnp.squeeze(value, axis=0)

    num_envs = config.env.num_envs
    batch_size = num_envs * num_agents
    eval_actions_fn, eval_tokens_fn = eval_actions.cash(
        models, config, adapter, num_agents, obs_dim, policy_obs_dim(config, obs_dim))

    for episode in range(config.ppo.num_episodes):
        started = time.time()

        current_num_agents = sample_team_size(config.env, np_rng)
        capabilities = sample_capabilities(config.env, current_num_agents, np_rng)
        adapter.apply_capabilities(config, env, capabilities, current_num_agents)
        obs = env.reset()

        context = capability_context(config, env, num_envs, num_agents)
        jax_capabilities = jnp.asarray(context.cpu().numpy()).reshape(batch_size, -1)

        obs_suffix = None
        if adapter.env_context is not None and token_dims(config, obs_dim)["env_context"] > 0:
            env_ctx = adapter.env_context(config, env, num_envs, num_agents)
            obs_suffix = jnp.asarray(env_ctx.cpu().numpy()).reshape(batch_size, -1)

        result = collect_rollout_recurrent(
            env=env, obs=obs, policy_params=models.policy_state.params,
            critic_params=models.critic_state.params, capabilities=jax_capabilities,
            config=config, num_agents=num_agents, current_num_agents=current_num_agents,
            episode=episode, get_actions=get_actions, get_value=get_value,
            hidden_dim=config.model.gru_hidden_dim, critic_hidden_dim=config.model.gru_hidden_dim,
            obs_suffix=obs_suffix,
        )
        traj, obs = result.trajectory, result.obs

        advantages, returns, old_values = compute_gae(
            traj.rewards, [np.asarray(v) for v in traj.values], np.asarray(traj.dones),
            gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda,
            reward_scale=config.ppo.reward_scale, num_envs=num_envs, num_agents=num_agents,
            current_num_agents=current_num_agents,
            advantage_clip_range=config.ppo.advantage_clip_range,
            mixed_agent_training=config.env.mixed_agent_training,
        )

        num_steps = config.ppo.rollout_steps
        obs_seq = jnp.concatenate(traj.obs, axis=0).reshape(num_steps, batch_size, -1)
        global_state_seq = jnp.concatenate(traj.global_states, axis=0).reshape(
            num_steps, num_envs, -1)
        actions_seq = jnp.concatenate(traj.actions, axis=0).reshape(num_steps, batch_size, -1)
        old_log_probs_seq = jnp.clip(
            jnp.nan_to_num(jnp.concatenate(traj.log_probs, axis=0), nan=-100.0, posinf=-100.0,
                           neginf=-100.0), -100.0, 100.0).reshape(num_steps, batch_size)
        advantages_seq = jnp.array(advantages).reshape(num_steps, batch_size)
        returns_seq = jnp.array(returns).reshape(num_steps, batch_size)
        capability_seq = jnp.tile(jax_capabilities[None, :, :], (num_steps, 1, 1))
        dones_seq = jnp.any(
            jnp.concatenate([d[None, :] for d in traj.dones], axis=0).reshape(
                num_steps, num_envs, num_agents), axis=-1)

        entropy_coef = max(config.ppo.min_entropy_coef,
                           config.ppo.entropy_coef * (config.ppo.entropy_decay ** episode))

        loss_info = None
        for _epoch in range(config.ppo.ppo_epochs):
            if config.ppo.use_minibatches and config.ppo.shuffle_minibatches:
                models.rng, _shuffle_rng = jax.random.split(models.rng)

            accumulated = None
            for env_rows, agent_rows in sequence_minibatches(
                    num_envs, config.ppo.num_minibatches, num_agents):
                policy_state, critic_state, loss_info = train_step(
                    models.policy_state, models.critic_state,
                    obs_seq[:, agent_rows], capability_seq[:, agent_rows],
                    global_state_seq[:, env_rows], actions_seq[:, agent_rows],
                    old_log_probs_seq[:, agent_rows], advantages_seq[:, agent_rows],
                    returns_seq[:, agent_rows],
                    traj.init_hidden[agent_rows], traj.init_critic_hidden[env_rows],
                    dones_seq[:, env_rows],
                    config.ppo.ppo_clip_epsilon, entropy_coef, config.ppo.value_loss_coef,
                    num_agents, current_num_agents)
                models.policy_state, models.critic_state = policy_state, critic_state
                accumulated = (loss_info if accumulated is None
                               else {k: accumulated[k] + v for k, v in loss_info.items()})
            loss_info = {k: v / config.ppo.num_minibatches for k, v in accumulated.items()}

        logger.log({
            "episode": episode,
            "avg_reward": float(np.mean(traj.episode_rewards)),
            "team_size": current_num_agents,
            "resets": traj.num_resets,
            "seconds": round(time.time() - started, 3),
            **{k: float(v) for k, v in loss_info.items()},
        })

        _maybe_evaluate(config, episode, env, logger, models=models, adapter=adapter,
                        num_agents=num_agents, obs_dim=obs_dim,
                        actions_fn=eval_actions_fn, tokens_fn=eval_tokens_fn)

        if config.logging.save_interval and episode and episode % config.logging.save_interval == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_{episode}.npz",
                            policy_params=models.policy_state.params,
                            critic_params=models.critic_state.params, episode=episode)

    save_checkpoint(checkpoint_dir / "final_checkpoint.npz",
                    policy_params=models.policy_state.params,
                    critic_params=models.critic_state.params,
                    episode=config.ppo.num_episodes)
    return models

