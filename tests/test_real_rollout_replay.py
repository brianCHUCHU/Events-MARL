"""Check every transition, including forced asynchronous refreshes, with real models."""

from pathlib import Path
import unittest

import distrax
import jax.numpy as jnp
import numpy as np
import torch

from hyperlora.config import load_config
from hyperlora.envs.adapters import get_adapter
from hyperlora.envs.capabilities import initial_capabilities
from hyperlora.envs.factory import make_env
from hyperlora.envs.tokens import HypernetworkInputs
from hyperlora.seeding import seed_globals
from hyperlora.training.jitted import make_jitted
from hyperlora.training.rollout import collect_rollout
from hyperlora.training.trainer import build_models

ROOT = Path(__file__).resolve().parents[1]


class RealRolloutReplayTest(unittest.TestCase):
    def check_replay(self, scenario):
        config = load_config(ROOT / f"configs/main/ours/{scenario}.yaml", {
            "research_metrics": True, "env.num_envs": 2, "ppo.rollout_steps": 6,
            "ppo.use_minibatches": False,
        })
        seed_globals(7)
        env = make_env(config.env, initial_capabilities(config.env))
        obs = env.reset()
        agents = config.env.num_agents
        if scenario == "navigation":
            for food in env.world.landmarks:
                food.set_pos(torch.tensor([10., 10.]), None)
            obs = [env.scenario.observation(a) for a in env.agents]
        models = build_models(config, obs[0].shape[-1], 2, agents)
        generate, act, value = make_jitted(models.policy, models.hypernetwork, models.critic)

        def hn_apply(inputs, scaling):
            return generate(models.hn_state.params, *inputs.as_args(), scaling)

        # Move known entities to force a refresh, then another refresh later.
        real_step = env.step
        step = 0

        def forced_step(actions):
            nonlocal step
            if scenario == "navigation":
                if step in (0, 2):
                    i = 0 if step == 0 else 2
                    env.world.landmarks[i].set_pos(env.agents[i].state.pos + torch.tensor([0.25, 0.]), None)
            elif step == 0:
                env.agents[0].set_pos(env.scenario.plate_left.state.pos, None)
            elif step == 2:
                env.agents[1].set_pos(env.scenario.plate_right.state.pos, None)
            step += 1
            return real_step(actions)

        env.step = forced_step
        adapter = get_adapter(scenario)
        tokens = adapter.build_tokens(config, env, obs, models.obs_dim, 2, agents, agents, 0.007)
        events = adapter.make_events(config, 2, agents, models.obs_dim, 0, 0.007)
        events.prime(obs)
        alpha = 1.7 if scenario == "navigation" else 1.
        result = collect_rollout(
            env=env, obs=obs, policy_params=models.policy_state.params,
            critic_params=models.critic_state.params, adapters=hn_apply(tokens, alpha),
            hn_apply=hn_apply, tokens=tokens, events=events, config=config, num_agents=agents,
            current_num_agents=agents, episode=0, alpha=alpha, buffer=[], get_actions=act,
            get_value=value,
        )
        trajectory = result.trajectory
        self.assertGreaterEqual(len(trajectory.requery_events), 2)
        rows = 6 * 2 * agents
        if scenario == "navigation":
            contexts = jnp.concatenate([t["generation_lidar"] for t in trajectory.token_rows])
            regenerated = hn_apply(HypernetworkInputs(lidar=contexts),
                                   jnp.repeat(jnp.asarray(trajectory.adapter_scales), 2 * agents))
            selected = jnp.arange(rows) * agents + jnp.tile(jnp.arange(agents), rows // agents)
            regenerated = {k: v[selected] for k, v in regenerated.items()}
        else:
            contexts = jnp.concatenate([t["env_context"].reshape(2, agents, -1)
                                        for t in trajectory.token_rows])
            regenerated = hn_apply(HypernetworkInputs(env_context=contexts), 1.)
        mean, log_std = models.policy.apply({"params": models.policy_state.params},
                                            jnp.concatenate(trajectory.obs), regenerated)
        dist = distrax.Transformed(distrax.Normal(mean, jnp.maximum(jnp.exp(log_std), models.policy.min_std)),
                                   distrax.Tanh())
        replay_logs = dist.log_prob(jnp.concatenate(trajectory.actions)).sum(axis=-1)
        np.testing.assert_allclose(replay_logs, jnp.concatenate(trajectory.log_probs), atol=2e-6, rtol=2e-6)

    def test_navigation_asynchronous_refresh_replays_all_actions(self):
        self.check_replay("navigation")

    def test_pressure_plate_refresh_replays_all_actions(self):
        self.check_replay("pressure_plate")
