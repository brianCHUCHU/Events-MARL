from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import torch


@dataclass
class Trajectory:
    obs: list = field(default_factory=list)
    global_states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    active_envs: list = field(default_factory=list)
    token_rows: list = field(default_factory=list)
    adapter_scales: list = field(default_factory=list)
    episode_rewards: list = field(default_factory=list)
    requery_events: list = field(default_factory=list)
    generator_refresh_sizes: list = field(default_factory=list)
    num_resets: int = 0


@dataclass
class RolloutResult:
    trajectory: Trajectory
    obs: list
    adapters: dict
    tokens: object
    alpha: float


def _apply_requery(requery, adapters, tokens, hn_apply, alpha, buffer, traj, step):
    if requery is None:
        return adapters, tokens, alpha
    tokens = requery.tokens if requery.tokens is not None else tokens
    if requery.alpha is not None:
        alpha = requery.alpha
    if len(requery.indices) == 0:
        return adapters, tokens, alpha
    scaling = alpha if requery.scaling is None else requery.scaling
    new_adapters = hn_apply(requery.inputs, scaling)
    if requery.generated_indices is not None:
        new_adapters = {key: matrix[requery.generated_indices]
                        for key, matrix in new_adapters.items()}
    adapters = {k: adapters[k].at[requery.indices].set(new_adapters[k]) for k in adapters}
    if requery.tokens is None:
        tokens = tokens.with_rows(requery.indices, requery.inputs)
    buffer.extend(requery.entries)
    traj.requery_events.append((step, requery.indices.tolist()))
    contexts, slots = requery.inputs.generation_shape()
    traj.generator_refresh_sizes.append((contexts, contexts * slots))
    return adapters, tokens, alpha


def _global_state(obs_transposed: torch.Tensor, num_envs: int, num_agents: int,
                  current_num_agents: int, mixed: bool) -> torch.Tensor:
    if mixed and current_num_agents < num_agents:
        active = obs_transposed[:, :current_num_agents, :]
        padding = torch.zeros(num_envs, num_agents - current_num_agents, obs_transposed.shape[-1],
                              dtype=obs_transposed.dtype, device=obs_transposed.device)
        active = torch.cat([active, padding], dim=1)
        return active.reshape(num_envs, -1)
    return obs_transposed.reshape(num_envs, -1)


def collect_rollout(*, env, obs, policy_params, critic_params, adapters, hn_apply, tokens, events,
                    config, num_agents, current_num_agents, episode: int, alpha: float,
                    buffer: list, get_actions, get_value, obs_suffix=None, trace=None) -> RolloutResult:
    env_config, ppo = config.env, config.ppo
    num_envs = env_config.num_envs
    batch_size = num_envs * num_agents
    mixed = env_config.mixed_agent_training
    traj = Trajectory()
    # Keep the original vectorized rollout shape. Only Ours' Pressure Plate
    # trainer consumes an active mask for its GAE and PPO updates.
    active_envs = (torch.ones(num_envs, dtype=torch.bool, device=obs[0].device)
                   if config.method == "ours" and (env_config.scenario == "pressure_plate"
                       or (config.research_metrics and env_config.scenario == "navigation"))
                   else None)

    jax_rng = jax.random.PRNGKey(episode)
    if config.research_metrics and config.seed:
        jax_rng = jax.random.fold_in(jax_rng, config.seed)
    if trace is not None:
        trace.initial_adapters(adapters)

    for step in range(ppo.rollout_steps):
        if active_envs is None:
            refresh = events.before_step(env, obs, tokens, step)
        else:
            refresh = events.before_step(env, obs, tokens, step, active_envs=active_envs)
            traj.active_envs.append(active_envs.cpu().numpy().copy())
        adapters, tokens, alpha = _apply_requery(refresh, adapters, tokens, hn_apply,
                                                 alpha, buffer, traj, step)
        if config.research_metrics:
            traj.adapter_scales.append(alpha)

        obs_transposed = torch.stack(obs, dim=0).transpose(0, 1)
        jax_obs = jnp.asarray(obs_transposed.reshape(batch_size, -1).cpu().numpy())
        if obs_suffix is not None:
            jax_obs = jnp.concatenate([jax_obs, obs_suffix], axis=-1)

        global_state = _global_state(obs_transposed, num_envs, num_agents, current_num_agents,
                                     mixed)
        jax_global_state = jnp.asarray(global_state.cpu().numpy())
        if ppo.clip_global_state:
            jax_global_state = jnp.clip(jax_global_state, ppo.clip_global_state_min,
                                        ppo.clip_global_state_max)
        values = get_value(critic_params, jax_global_state)

        jax_rng, action_rng = jax.random.split(jax_rng)
        action, log_prob, _hidden, mean, std = get_actions(
            policy_params, jax_obs, adapters, action_rng)

        actions_flat = np.asarray(action)
        if config.research_metrics:
            valid_agents = (np.ones(batch_size, dtype=bool) if active_envs is None else
                            np.repeat(active_envs.cpu().numpy(), num_agents))
            for name, data in (("observation", jax_obs), ("action", action),
                               ("log_prob", log_prob), ("policy_mean", mean),
                               ("policy_std", std), ("value", values)):
                if not np.isfinite(np.asarray(data).reshape(batch_size, -1)[valid_agents]).all():
                    raise FloatingPointError(f"Non-finite rollout {name} at round {episode + 1}, step {step}")
        if np.isnan(actions_flat).any():
            actions_flat = np.nan_to_num(actions_flat, nan=0.0, posinf=0.0, neginf=0.0)
        actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
        if mixed:
            agent_mask = np.arange(num_agents) < current_num_agents
            actions_reshaped = actions_reshaped * agent_mask[np.newaxis, :, np.newaxis]
        if active_envs is not None:
            actions_reshaped = np.where(active_envs.cpu().numpy()[:, None, None],
                                        actions_reshaped, 0.0)

        torch_actions = [
            torch.tensor(actions_reshaped[:, i, :], dtype=torch.float32) for i in range(num_agents)
        ]
        next_obs, rewards, dones, _info = env.step(torch_actions)
        if trace is not None:
            trace.step(env, obs, torch_actions, rewards, dones, tokens, refresh, adapters,
                       mean, std, active=active_envs, adapter_scale=alpha,
                       log_probs=log_prob, values=values)

        traj.obs.append(jax_obs)
        traj.global_states.append(jax_global_state)
        traj.actions.append(action)
        traj.token_rows.append(tokens.flatten_rows(batch_size))

        if mixed:
            for agent_idx in range(current_num_agents, num_agents):
                rewards[agent_idx] = torch.zeros_like(rewards[agent_idx])

        adapters, tokens, alpha = _apply_requery(events.after_step(env, next_obs, tokens, step),
                                                 adapters, tokens, hn_apply, alpha, buffer, traj,
                                                 step)
        events.commit()

        traj.log_probs.append(jnp.clip(
            jnp.nan_to_num(log_prob, nan=-100.0, posinf=-100.0, neginf=-100.0), -100.0, 100.0))
        traj.values.append(jnp.clip(
            jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0))

        rewards_np = torch.stack(rewards, dim=0).transpose(0, 1).cpu().numpy()
        if config.research_metrics:
            valid_teams = (np.ones(num_envs, dtype=bool) if active_envs is None else
                           traj.active_envs[-1])
            if not np.isfinite(rewards_np[valid_teams]).all():
                raise FloatingPointError(f"Non-finite rollout reward at round {episode + 1}, step {step}")
        rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=0.0, neginf=0.0)
        if mixed:
            rewards_np = rewards_np * (np.arange(num_agents) < current_num_agents)[np.newaxis, :]
        if active_envs is not None:
            active_rows = traj.active_envs[-1]
            rewards_np[~active_rows] = 0
            reward_mean = float(rewards_np.sum() / max(active_rows.sum() * num_agents, 1))
            dones = dones.bool() & active_envs
            active_envs = active_envs & ~dones
        else:
            reward_mean = float(rewards_np.mean())
        traj.rewards.append(np.asarray(rewards_np).flatten())
        traj.episode_rewards.append(0.0 if not np.isfinite(reward_mean) else reward_mean)

        traj.dones.append(jnp.array(np.repeat(dones.cpu().numpy(), num_agents), dtype=bool))

        if active_envs is None and not config.research_metrics and bool(dones.all().item()):
            traj.num_resets += 1
            next_obs = env.reset()

        obs = next_obs

    return RolloutResult(trajectory=traj, obs=obs, adapters=adapters, tokens=tokens,
                         alpha=alpha)


@dataclass
class RecurrentTrajectory(Trajectory):
    init_hidden: object = None
    init_critic_hidden: object = None


def collect_rollout_recurrent(*, env, obs, policy_params, critic_params, capabilities, config,
                              num_agents, current_num_agents, episode: int, get_actions,
                              get_value, hidden_dim: int, critic_hidden_dim: int,
                              obs_suffix=None) -> RolloutResult:
    env_config, ppo = config.env, config.ppo
    num_envs = env_config.num_envs
    batch_size = num_envs * num_agents
    mixed = env_config.mixed_agent_training
    traj = RecurrentTrajectory()

    hidden = jnp.zeros((batch_size, hidden_dim))
    critic_hidden = jnp.zeros((num_envs, critic_hidden_dim))
    traj.init_hidden, traj.init_critic_hidden = hidden, critic_hidden

    jax_rng = jax.random.PRNGKey(episode)
    dones = None

    for step in range(ppo.rollout_steps):
        prev_dones = (jnp.zeros(batch_size, dtype=bool) if step == 0
                      else jnp.asarray(traj.dones[-1]))

        obs_transposed = torch.stack(obs, dim=0).transpose(0, 1)
        jax_obs = jnp.asarray(obs_transposed.reshape(batch_size, -1).cpu().numpy())
        if obs_suffix is not None:
            jax_obs = jnp.concatenate([jax_obs, obs_suffix], axis=-1)

        global_state = _global_state(obs_transposed, num_envs, num_agents, current_num_agents,
                                     mixed)
        jax_global_state = jnp.asarray(global_state.cpu().numpy())
        if ppo.clip_global_state:
            jax_global_state = jnp.clip(jax_global_state, ppo.clip_global_state_min,
                                        ppo.clip_global_state_max)

        env_dones = torch.zeros(num_envs, dtype=torch.bool)
        if step > 0:
            for agent_idx in range(len(dones)):
                env_dones = env_dones | dones[agent_idx]
        critic_hidden, values = get_value(critic_params, jax_global_state, critic_hidden,
                                          jnp.asarray(env_dones.cpu().numpy()))

        jax_rng, action_rng = jax.random.split(jax_rng)
        action, log_prob, new_hidden, _mean, _std = get_actions(
            policy_params, jax_obs, capabilities, action_rng, hidden, prev_dones)
        hidden = new_hidden

        actions_flat = np.asarray(action)
        if np.isnan(actions_flat).any():
            actions_flat = np.nan_to_num(actions_flat, nan=0.0, posinf=0.0, neginf=0.0)
        actions_reshaped = actions_flat.reshape(num_envs, num_agents, -1)
        if mixed:
            agent_mask = np.arange(num_agents) < current_num_agents
            actions_reshaped = actions_reshaped * agent_mask[np.newaxis, :, np.newaxis]

        torch_actions = [
            torch.tensor(actions_reshaped[:, i, :], dtype=torch.float32) for i in range(num_agents)
        ]
        next_obs, rewards, dones, _info = env.step(torch_actions)

        traj.obs.append(jax_obs)
        traj.global_states.append(jax_global_state)
        traj.actions.append(action)

        if mixed:
            for agent_idx in range(current_num_agents, num_agents):
                rewards[agent_idx] = torch.zeros_like(rewards[agent_idx])

        traj.log_probs.append(jnp.clip(
            jnp.nan_to_num(log_prob, nan=-100.0, posinf=-100.0, neginf=-100.0), -100.0, 100.0))
        traj.values.append(jnp.clip(
            jnp.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0))

        rewards_np = torch.stack(rewards, dim=0).transpose(0, 1).cpu().numpy()
        rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=0.0, neginf=0.0)
        if mixed:
            rewards_np = rewards_np * (np.arange(num_agents) < current_num_agents)[np.newaxis, :]
        traj.rewards.append(np.asarray(rewards_np).flatten())
        reward_mean = float(rewards_np.mean())
        traj.episode_rewards.append(0.0 if not np.isfinite(reward_mean) else reward_mean)

        traj.dones.append(jnp.array(np.repeat(dones.cpu().numpy(), num_agents), dtype=bool))

        if bool(dones.all().item()):
            traj.num_resets += 1
            next_obs = env.reset()

        obs = next_obs

    return RolloutResult(trajectory=traj, obs=obs, adapters=None, tokens=None, alpha=1.0)

