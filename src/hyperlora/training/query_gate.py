"""Opt-in team query policy for the original simultaneous-goal Pressure Plate.

Coverage is a Bernoulli policy mixture, not a learned coverage estimator.
Actor/generator and gate use the same query-penalized learning reward; logs
retain the original task reward. Initial generation is mandatory and uncharged.
"""
from dataclasses import replace
from pathlib import Path

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from ..envs.events.pressure_plate import PressurePlateEvents
from .gae import compute_gae


class QueryGate(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, inputs):
        x = nn.tanh(nn.Dense(self.hidden_dim)(inputs))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        logits = nn.Dense(1, kernel_init=nn.initializers.zeros, name="query")(x)[..., 0]
        value = nn.Dense(1, name="value")(x)[..., 0]
        return logits, value


def gate_distribution(logits, epsilon):
    learned = jax.nn.sigmoid(logits)
    behavior = (1.0 - epsilon) * learned + 0.5 * epsilon
    return learned, jnp.clip(behavior, 1e-6, 1.0 - 1e-6)


def bernoulli_log_prob(probability, action):
    return jnp.where(action, jnp.log(probability), jnp.log1p(-probability))


@jax.jit
def gate_inference(state, inputs, key, epsilon):
    logits, values = state.apply_fn({"params": state.params}, inputs)
    learned, behavior = gate_distribution(logits, epsilon)
    action = jax.random.bernoulli(key, behavior)
    return action, bernoulli_log_prob(behavior, action), values, learned, behavior


class LearnedPressurePlateEvents(PressurePlateEvents):
    def __init__(self, config, num_envs, num_agents, state, round_index, evaluation=False,
                 eval_batch=0):
        super().__init__(config, num_envs, num_agents)
        self.state = state
        self.epsilon = (config.query_gate.coverage_epsilon
                        if config.query_gate.mode == "coverage" and not evaluation else 0.0)
        # Independent actor, gate and evaluation streams. Evaluation gates use
        # common random numbers across checkpoints and both gate variants.
        key = jax.random.PRNGKey(config.evaluation.seed if evaluation else config.seed)
        self.key = jax.random.fold_in(key, 9101 if evaluation else 9100)
        self.key = jax.random.fold_in(self.key, eval_batch if evaluation else round_index)
        self.last_query = np.zeros(num_envs, dtype=np.int32)
        self.records = []

    def before_step(self, env, obs, tokens, step, active_envs=None):
        active = (np.ones(self.num_envs, dtype=bool) if active_envs is None
                  else active_envs.cpu().numpy().copy())
        age = (step - self.last_query) / self.config.ppo.rollout_steps
        joint = torch.stack(obs, dim=1).reshape(self.num_envs, -1).cpu().numpy()
        inputs = np.concatenate([joint, age[:, None]], axis=-1).astype(np.float32)
        self.key, key = jax.random.split(self.key)
        q, logp, value, learned, behavior = gate_inference(
            self.state, jnp.asarray(inputs), key, self.epsilon)
        q = np.asarray(q) & active & (step > 0)
        record = dict(inputs=inputs, query=q, log_prob=np.asarray(logp),
                      value=np.asarray(value), learned_probability=np.asarray(learned),
                      behavior_probability=np.asarray(behavior), age=age.astype(np.float32),
                      valid=active & (step > 0), active=active)
        if not all(np.isfinite(v).all() for v in record.values()):
            raise FloatingPointError("Non-finite query gate rollout")
        self.records.append(record)
        self.last_query[q] = step
        if not q.any():
            return None
        refresh = self._build(env, tokens, torch.from_numpy(q))
        # Bound compilation shapes without generating adapters for every team.
        # Pad duplicate queried contexts; generated/deployed counts stay separate.
        count = int(q.sum())
        bucket = self.config.query_gate.generation_bucket
        padded = min(self.num_envs, ((count + bucket - 1) // bucket) * bucket)
        context = refresh.inputs.env_context
        if padded > count:
            context = jnp.concatenate([context, jnp.repeat(context[-1:], padded-count, axis=0)])
        return replace(refresh, inputs=replace(refresh.inputs, env_context=context),
                       generated_indices=np.arange(count * self.num_agents))

    def arrays(self):
        return {name: np.stack([row[name] for row in self.records]) for name in self.records[0]}

    def save(self, path):
        arrays = self.arrays()
        arrays["sample_inputs"] = arrays.pop("inputs")[:, :4]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays, coverage_epsilon=self.epsilon)


@jax.jit
def gate_update(state, inputs, actions, old_logp, old_values, advantages, returns,
                valid, active, epsilon, clip, entropy_coef, value_coef):
    def loss_fn(params):
        logits, values = state.apply_fn({"params": params}, inputs)
        _learned, probability = gate_distribution(logits, epsilon)
        logp = bernoulli_log_prob(probability, actions)
        ratio = jnp.exp(jnp.clip(logp - old_logp, -20, 20))
        mean = lambda x, mask: (x * mask).sum() / jnp.maximum(mask.sum(), 1)
        policy_loss = -mean(jnp.minimum(ratio * advantages,
                                       jnp.clip(ratio, 1-clip, 1+clip) * advantages), valid)
        clipped_value = old_values + jnp.clip(values - old_values, -clip, clip)
        value_loss = 0.5 * mean(jnp.maximum((values-returns)**2,
                                            (clipped_value-returns)**2), active)
        entropy = -probability*jnp.log(probability)-(1-probability)*jnp.log1p(-probability)
        loss = policy_loss + value_coef*value_loss - entropy_coef*mean(entropy, valid)
        return loss, dict(gate_policy_loss=policy_loss, gate_value_loss=value_loss,
                          gate_entropy=mean(entropy, valid),
                          gate_log_prob_mse=mean((logp-old_logp)**2, valid),
                          gate_clip_fraction=mean((jnp.abs(ratio-1)>clip).astype(float), valid))
    (loss, info), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    info = dict(info, gate_loss=loss, gate_grad_norm=optax.global_norm(grad))
    return state.apply_gradients(grads=grad), info


def update_gate(config, state, events, learning_rewards, dones, round_index):
    data = events.arrays()
    count, n = data["query"].shape
    advantages, returns, _ = compute_gae(
        learning_rewards, data["value"][..., None], np.asarray(dones).reshape(count, n, -1).any(-1),
        gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda, reward_scale=config.ppo.reward_scale,
        num_envs=n, num_agents=1, current_num_agents=1,
        advantage_clip_range=config.ppo.advantage_clip_range, mixed_agent_training=False,
        valid_steps=data["active"])
    # Normalize policy advantages only over actual learned decisions (not reset).
    decision = data["valid"].reshape(-1)
    a = advantages[decision]
    if not a.size:
        raise ValueError("Gate training needs at least one post-reset decision")
    advantages = np.clip((advantages-a.mean())/(a.std()+1e-8),
                         -config.ppo.advantage_clip_range, config.ppo.advantage_clip_range)
    flat = {k: jnp.asarray(v.reshape((-1, v.shape[-1])) if k == "inputs" else v.reshape(-1))
            for k,v in data.items()}
    minibatches = config.ppo.num_minibatches if config.ppo.use_minibatches else 1
    rng = jax.random.fold_in(jax.random.PRNGKey(config.seed), 9200+round_index)
    metrics = []
    first_mse = None
    for epoch in range(config.ppo.ppo_epochs):
        rng, shuffle_key = jax.random.split(rng)
        order = (jax.random.permutation(shuffle_key, count*n) if config.ppo.shuffle_minibatches
                 else jnp.arange(count*n))
        for ids in order.reshape(minibatches, -1):
            state, info = gate_update(
                state, flat["inputs"][ids], flat["query"][ids], flat["log_prob"][ids],
                flat["value"][ids], jnp.asarray(advantages)[ids], jnp.asarray(returns)[ids],
                flat["valid"][ids], flat["active"][ids], events.epsilon,
                config.ppo.ppo_clip_epsilon, config.query_gate.entropy_coef,
                config.ppo.value_loss_coef)
            info = {k: float(v) for k,v in info.items()}
            if not all(np.isfinite(v) for v in info.values()):
                raise FloatingPointError("Non-finite gate PPO update")
            if first_mse is None:
                first_mse = info["gate_log_prob_mse"]
                if first_mse > 1e-6:
                    raise RuntimeError(f"Gate rollout/PPO probability mismatch: {first_mse}")
            metrics.append(info)
    result = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    result.update(gate_pre_update_log_prob_mse=first_mse,
                  gate_query_rate=float(data["query"][data["valid"]].mean()),
                  gate_mean_learned_probability=float(data["learned_probability"][data["valid"]].mean()),
                  gate_mean_behavior_probability=float(data["behavior_probability"][data["valid"]].mean()),
                  gate_coverage_epsilon=events.epsilon)
    return state, result
