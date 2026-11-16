from __future__ import annotations

from functools import partial

import distrax
import jax
import jax.numpy as jnp


def make_train_step(shared_policy, hypernetwork, critic, adapter_regularization_coef):
    @partial(jax.jit, static_argnums=(19, 20), static_argnames=("value_clip_range",))
    def train_step(
        policy_state,
        hn_state,
        critic_state,
        obs_batch,
        global_state_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
        task_batch,
        context_batch,
        lidar_batch,
        agent_position_batch,
        target_nmd_batch,
        env_context_batch,
        mask,
        clip_epsilon,
        entropy_coef,
        value_loss_coef,
        num_agents,
        current_num_agents,
        diversity_scaling=1.0,
        old_values_batch=None,
        value_clip_range=None,
        valid_steps=None,
        generation_lidar=None,
    ):
        def loss_fn(policy_params, hn_params, critic_params):
            adapters_dict = hypernetwork.apply(
                {"params": hn_params},
                task_batch,
                context_batch,
                lidar_batch if generation_lidar is None else generation_lidar,
                None,
                agent_position_batch,
                target_nmd_batch,
                env_context_batch,
                mask,
                diversity_scaling if generation_lidar is None else
                jnp.repeat(jnp.broadcast_to(jnp.asarray(diversity_scaling).reshape(-1),
                                           (obs_batch.shape[0] // num_agents,)), num_agents),
            )
            if generation_lidar is not None:
                # Each deployed agent retains the entire team context from its
                # own most recent refresh, including its attention partners.
                selected = jnp.arange(obs_batch.shape[0]) * num_agents + jnp.tile(
                    jnp.arange(num_agents), obs_batch.shape[0] // num_agents)
                adapters_dict = {key: matrix[selected] for key, matrix in adapters_dict.items()}

            mean, log_std = shared_policy.apply(
                {"params": policy_params}, obs_batch, adapters_dict
            )
            std = jnp.exp(log_std)
            std = jnp.maximum(std, shared_policy.min_std)

            base_dist = distrax.Normal(mean, std)
            dist = distrax.Transformed(base_dist, distrax.Tanh())
            new_log_probs = dist.log_prob(actions_batch).sum(axis=-1)
            entropy_per_agent = base_dist.entropy().sum(axis=-1)

            new_log_probs = jnp.nan_to_num(new_log_probs, nan=-100.0, posinf=-100.0, neginf=-100.0)
            new_log_probs = jnp.clip(new_log_probs, -100.0, 100.0)

            log_prob_diff = new_log_probs - old_log_probs_batch
            log_prob_diff = jnp.clip(log_prob_diff, -3.0, 3.0)
            ratio = jnp.exp(log_prob_diff)
            clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

            surr1 = ratio * advantages_batch
            surr2 = clipped_ratio * advantages_batch
            policy_loss_per_agent = -jnp.minimum(surr1, surr2)

            batch_size_total = obs_batch.shape[0]
            agent_indices = jnp.arange(num_agents, dtype=jnp.int32)
            agent_mask_template = agent_indices < current_num_agents
            valid_agent_mask = jnp.tile(agent_mask_template, batch_size_total // num_agents)

            if valid_steps is not None:
                valid_agent_mask &= jnp.repeat(valid_steps, num_agents)

            def mean_valid(array):
                rows = array.reshape(batch_size_total, -1)
                return jnp.mean(rows, where=valid_agent_mask[:, None])

            def variance_valid(array):
                return mean_valid(jnp.square(array - mean_valid(array)))

            mean_stat = jnp.mean if valid_steps is None else mean_valid
            variance_stat = jnp.var if valid_steps is None else variance_valid
            stat_mask = None if valid_steps is None else valid_agent_mask[:, None]

            policy_loss = mean_valid(policy_loss_per_agent)
            entropy = mean_valid(entropy_per_agent)
            policy_loss += 0.001 * mean_stat(jnp.square(mean))

            values = critic.apply({"params": critic_params}, global_state_batch)
            values_flat = values.reshape(-1)

            if value_clip_range is not None and old_values_batch is not None:
                old_values_flat = old_values_batch.reshape(-1)
                values_clipped = old_values_flat + jnp.clip(
                    values_flat - old_values_flat, -value_clip_range, value_clip_range
                )
                value_loss_per_agent = jnp.maximum(
                    (values_flat - returns_batch) ** 2, (values_clipped - returns_batch) ** 2
                )
            else:
                value_loss_per_agent = (values_flat - returns_batch) ** 2

            value_loss = mean_valid(value_loss_per_agent)

            adapter_norm = 0.0
            adapter_count = 0
            for v in adapters_dict.values():
                adapter_norm += mean_stat(jnp.square(v))
                adapter_count += 1
            adapter_mean_norm = (
                jnp.sqrt(adapter_norm / max(adapter_count, 1)) if adapter_count > 0 else 0.0
            )
            adapter_regularization_loss = adapter_regularization_coef * adapter_norm

            total_loss = (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy
                + adapter_regularization_loss
            )

            info_dict = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": mean_stat((new_log_probs - old_log_probs_batch) ** 2),
                "mean_ratio": mean_stat(ratio),
                "clip_fraction": mean_stat(jnp.abs(ratio - 1.0) > clip_epsilon),
                "value_explained_variance": 1 - variance_stat(returns_batch - values_flat)
                / jnp.maximum(variance_stat(returns_batch), 1e-8),
                "advantages_mean": mean_stat(advantages_batch),
                "advantages_std": jnp.sqrt(variance_stat(advantages_batch)),
                "advantages_max": jnp.max(jnp.abs(advantages_batch),
                                          where=None if stat_mask is None else valid_agent_mask,
                                          initial=0),
                "adapter_norm": adapter_mean_norm,
                "mean_std": mean_stat(std),
                "min_std": jnp.min(std, where=stat_mask, initial=jnp.inf),
                "max_std": jnp.max(std, where=stat_mask, initial=-jnp.inf),
            }
            return total_loss, info_dict

        (loss, info), grads = jax.value_and_grad(
            lambda p, h, c: loss_fn(p, h, c), argnums=(0, 1, 2), has_aux=True
        )(policy_state.params, hn_state.params, critic_state.params)
        policy_grads, hn_grads, critic_grads = grads

        info["policy_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(policy_grads))
        )
        info["hn_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(hn_grads))
        )
        info["critic_grad_norm"] = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads))
        )

        nan_to_zero = lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g)
        policy_grads = jax.tree_util.tree_map(nan_to_zero, policy_grads)
        hn_grads = jax.tree_util.tree_map(nan_to_zero, hn_grads)
        critic_grads = jax.tree_util.tree_map(nan_to_zero, critic_grads)

        return (
            policy_state.apply_gradients(grads=policy_grads),
            hn_state.apply_gradients(grads=hn_grads),
            critic_state.apply_gradients(grads=critic_grads),
            info,
        )

    return train_step

