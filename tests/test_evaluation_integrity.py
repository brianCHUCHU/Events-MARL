import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import torch
from vmas.simulator.environment import Environment

from hyperlora.config import load_config
from hyperlora.envs.adapters import get_adapter
from hyperlora.envs.capabilities import initial_capabilities
from hyperlora.envs.factory import make_env
from hyperlora.evaluation import actions, research

ROOT = Path(__file__).resolve().parents[1]


class EvaluationIntegrityTest(unittest.TestCase):
    def config(self, scenario="navigation"):
        return load_config(ROOT / f"configs/main/ours/{scenario}.yaml", {
            "research_metrics": True, "env.num_envs": 2,
            "evaluation.num_eval_episodes": 2, "evaluation.max_eval_steps": 3,
        })

    def evaluate(self, config, directory):
        count = config.env.num_envs * config.env.num_agents
        mean = jnp.tile(jnp.array([[0.5, -1.5]]), (count, 1))
        models = SimpleNamespace(obs_dim=20, policy_state=SimpleNamespace(params=None))
        return research.evaluate(
            config, get_adapter("navigation"), models,
            lambda inputs, alpha: {"probe": jnp.zeros((count, 1))},
            lambda params, obs, adapters: (mean, jnp.ones_like(mean) * 0.3),
            0.007, 0, Path(directory), 0,
        )

    def test_research_actions_use_policy_squashing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.evaluate(self.config(), directory)
            with np.load(Path(directory) / "round_0000/batch_00.npz", allow_pickle=False) as z:
                expected = np.broadcast_to(np.tanh([0.5, -1.5]), z["sample_actions"].shape)
                np.testing.assert_allclose(z["sample_actions"], expected, atol=1e-6)

    def test_evaluation_preserves_training_and_host_rng_even_on_error(self):
        env = make_env(self.config().env, initial_capabilities(self.config().env))
        env.reset(seed=7)
        for fail in (False, True):
            state = copy.deepcopy(Environment.vmas_random_state)
            torch_state = torch.random.get_rng_state().clone()
            numpy_state = copy.deepcopy(np.random.get_state())
            python_state = random.getstate()
            with tempfile.TemporaryDirectory() as directory:
                if fail:
                    from unittest.mock import patch
                    with patch.object(research.RolloutTrace, "step", side_effect=RuntimeError("probe")):
                        with self.assertRaisesRegex(RuntimeError, "probe"):
                            self.evaluate(self.config(), directory)
                else:
                    self.evaluate(self.config(), directory)
            self.assertTrue(torch.equal(state[0], Environment.vmas_random_state[0]))
            np.testing.assert_array_equal(state[1][1], Environment.vmas_random_state[1][1])
            self.assertEqual(state[1][2:], Environment.vmas_random_state[1][2:])
            self.assertEqual(state[2], Environment.vmas_random_state[2])
            self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
            np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
            self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
            self.assertEqual(python_state, random.getstate())

    def test_evaluation_population_is_independent_of_training_resets(self):
        config = self.config()
        env = make_env(config.env, initial_capabilities(config.env))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.evaluate(config, first)
            for _ in range(3):
                env.reset()
            self.evaluate(config, second)
            for batch in range(2):
                relative = f"round_0000/batch_{batch:02d}.npz"
                with np.load(Path(first) / relative) as a, np.load(Path(second) / relative) as b:
                    np.testing.assert_array_equal(a["sample_observations"][0],
                                                  b["sample_observations"][0])

    def test_shared_action_builder_uses_policy_squashing(self):
        config = self.config()
        mean = jnp.tile(jnp.array([[0.5, -1.5]]), (8, 1))
        policy = SimpleNamespace(apply=lambda variables, obs, adapters: (mean, mean * 0))
        models = SimpleNamespace(policy=policy, policy_state=SimpleNamespace(params=None))
        action_fn, _, _ = actions.ours(models, config, None, 4, 20, None)
        actual, _ = action_fn(jnp.zeros((8, 20)), {"adapters": {}})
        np.testing.assert_allclose(actual, np.tanh(mean), atol=1e-6)

    def test_standard_evaluator_stops_reward_accounting_at_first_terminal(self):
        from dataclasses import replace
        from hyperlora.evaluation.quantitative import run_quantitative_evaluation

        config = self.config()
        config = replace(config, diversity=replace(config.diversity, use_diversity_control=False))
        obs = [torch.zeros((2, 20)) for _ in range(4)]

        class FakeEnvironment:
            scenario = None
            t = 0

            def reset(self):
                return obs

            def step(self, action):
                self.t += 1
                rewards = torch.tensor([1., 2.]) if self.t == 1 else torch.tensor([1000., 3.])
                return obs, [rewards] * 4, torch.tensor([True, self.t >= 2]), {}

        metrics = run_quantitative_evaluation(
            config=config, env=FakeEnvironment(), models=None, adapter=None,
            num_agents=4, obs_dim=20, num_eval_episodes=1,
            actions_fn=lambda obs, state: (jnp.zeros((8, 2)), state),
        )
        self.assertEqual(metrics["avg_reward"], 3.)
        self.assertEqual(metrics["completed_episodes"], 2)

    def test_standard_evaluation_uses_training_ema_convention(self):
        from hyperlora.diversity.control import update_ema
        from hyperlora.evaluation.quantitative import _update_eval_ema

        config = self.config()
        self.assertAlmostEqual(_update_eval_ema(config, 1., 10.),
                               update_ema(1., 10., config.diversity.nmd_moving_average_coef))


if __name__ == "__main__":
    unittest.main()
