import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hyperlora.config import load_config
from hyperlora.envs.capabilities import initial_capabilities
from hyperlora.envs.events.base import NoEvents
from hyperlora.envs.factory import make_env
from hyperlora.envs.tokens import HypernetworkInputs
from hyperlora.seeding import seed_before_vmas_import
from hyperlora.training.gae import compute_gae
from hyperlora.training.rollout import collect_rollout

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/main/ours/pressure_plate.yaml"


class PressurePlateTerminationTest(unittest.TestCase):
    def test_first_visit_bonus_is_shared_once_and_reset_per_environment(self):
        seed_before_vmas_import(0)
        config = load_config(CONFIG_PATH, {"env.num_envs": 2})
        env = make_env(config.env, initial_capabilities(config.env))
        env.scenario.distance_shaping_coef = 0
        env.reset()

        def rewards():
            env.scenario.post_step()
            return torch.stack([env.scenario.reward(agent) for agent in env.agents]).numpy()

        goal = env.scenario.goal.state.pos[0]
        agent = env.agents[1]
        agent.set_pos(goal, None)
        np.testing.assert_array_equal(rewards(), np.full((3, 2), 100.0))
        np.testing.assert_array_equal(rewards(), np.zeros((3, 2)))
        agent.set_pos(torch.tensor([-1.0, 0.0]), None)
        np.testing.assert_array_equal(rewards(), np.zeros((3, 2)))
        agent.set_pos(goal, None)
        np.testing.assert_array_equal(rewards(), np.zeros((3, 2)))
        env.reset_at(0)
        agent.set_pos(goal, None)
        np.testing.assert_array_equal(rewards(), np.tile([100.0, 0.0], (3, 1)))

    def test_finished_team_remains_inactive_and_stops_querying(self):
        num_envs, num_agents = 2, 3
        config = load_config(CONFIG_PATH, {
            "env.num_envs": num_envs, "ppo.rollout_steps": 3, "research_metrics": True,
            "ppo.use_minibatches": False,
        })
        obs = [torch.ones((num_envs, 2)) for _ in range(num_agents)]

        class Environment:
            t = 0

            def step(self, actions):
                if self.t > 0:
                    np.testing.assert_array_equal([a[0].item() for a in actions], [0, 0, 0])
                self.t += 1
                done = torch.tensor([self.t == 1, False])
                return obs, [torch.ones(num_envs) for _ in range(num_agents)], done, {}

        class Events(NoEvents):
            def __init__(self):
                super().__init__(num_envs, num_agents)
                self.active = []

            def before_step(self, env, obs, tokens, step, active_envs=None):
                self.active.append(active_envs.tolist())
                return None

        def get_actions(params, inputs, adapters, rng):
            import jax.numpy as jnp

            ones = jnp.ones((num_envs * num_agents, 1))
            return ones, ones[:, 0], None, ones, ones

        events = Events()
        result = collect_rollout(
            env=Environment(), obs=obs, policy_params=None, critic_params=None, adapters=None,
            hn_apply=None, tokens=HypernetworkInputs(), events=events, config=config,
            num_agents=num_agents, current_num_agents=num_agents, episode=0, alpha=1.0, buffer=[],
            get_actions=get_actions, get_value=lambda params, state: np.zeros((num_envs, num_agents)),
        )
        expected = [[True, True], [False, True], [False, True]]
        self.assertEqual(events.active, expected)
        np.testing.assert_array_equal(result.trajectory.active_envs, expected)
        np.testing.assert_array_equal(np.asarray(result.trajectory.rewards)[:, :3],
                                      [[1, 1, 1], [0, 0, 0], [0, 0, 0]])

    def test_gae_ignores_rewards_and_values_after_terminal(self):
        dimensions = (3, 2, 1)
        rewards = np.ones(dimensions, dtype=np.float32)
        values = np.zeros(dimensions, dtype=np.float32)
        dones = np.zeros(dimensions, dtype=bool)
        dones[0, 0] = True
        valid_steps = np.array([[True, True], [False, True], [False, True]])

        def advantages_and_returns():
            return compute_gae(
                rewards, values, dones, gamma=1.0, gae_lambda=1.0, reward_scale=1.0,
                num_envs=2, num_agents=1, current_num_agents=1, advantage_clip_range=10.0,
                mixed_agent_training=False, valid_steps=valid_steps,
            )[:2]

        initial_advantages, initial_returns = advantages_and_returns()
        rewards[1:, 0] = 500.0
        values[1:, 0] = -50.0
        actual_advantages, actual_returns = advantages_and_returns()
        np.testing.assert_array_equal(actual_advantages, initial_advantages)
        np.testing.assert_array_equal(actual_returns, initial_returns)
        np.testing.assert_array_equal(actual_returns.reshape(3, 2), [[1, 3], [0, 2], [0, 1]])

    def test_ppo_update_is_invariant_to_inactive_rows(self):
        import jax.numpy as jnp
        import optax
        from flax.training.train_state import TrainState

        from hyperlora.algorithms.ours import make_train_step

        policy = SimpleNamespace(min_std=0.3, apply=lambda variables, obs, adapters: (
            variables["params"]["scale"] * obs + adapters["offset"], jnp.zeros_like(obs)))
        generator = SimpleNamespace(apply=lambda variables, *args: {
            "offset": (variables["params"]["scale"] * args[6]).reshape(-1, 1),
        })
        critic = SimpleNamespace(apply=lambda variables, state: variables["params"]["scale"] * state)

        def state(model):
            return TrainState.create(apply_fn=model.apply, params={"scale": jnp.array(0.1)},
                                     tx=optax.sgd(0.01))

        update = make_train_step(policy, generator, critic, 0.2)

        def apply(team_rows):
            rows = jnp.repeat(jnp.asarray(team_rows, dtype=jnp.float32), 3)
            return update(
                state(policy), state(generator), state(critic), rows[:, None], rows.reshape(-1, 3),
                jnp.zeros_like(rows[:, None]), rows, rows, rows,
                None, None, None, None, None, rows.reshape(-1, 3, 1), None,
                0.2, 0.01, 0.5, 3, 3, valid_steps=jnp.arange(len(team_rows)) == 0,
            )

        expected = apply([1.0])
        padded = apply([1.0, 50.0])
        for left, right in zip(expected[:3], padded[:3]):
            np.testing.assert_allclose(left.params["scale"], right.params["scale"], atol=1e-6)
        for key, metric in expected[-1].items():
            np.testing.assert_allclose(metric, padded[-1][key], atol=1e-6)
