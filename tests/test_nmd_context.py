import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from hyperlora.diversity.nmd import calculate_nmd, compute_nmd_from_action_means


class NmdContextTest(unittest.TestCase):
    def test_nmd_regenerates_adapters_with_the_recorded_teammates(self):
        contexts = jnp.array([[[1.], [3.]], [[2.], [8.]]])
        entries = [{"task": None, "context": None, "lidar": contexts[i, i],
                    "team_lidar": contexts[i], "agent_idx": i} for i in range(2)]

        def generate(variables, task, context, lidar, *args, **kwargs):
            return {"offset": (lidar + lidar.sum(axis=1, keepdims=True)).reshape(-1, 1)}

        generator = SimpleNamespace(
            max_agents=2, agent_position_dim=0, food_position_dim=0,
            target_nmd_dim=0, env_context_dim=0, apply=generate,
        )
        policy = SimpleNamespace(apply=lambda variables, obs, adapters: (
            obs + adapters["offset"], jnp.zeros_like(obs)))
        actual = calculate_nmd(entries, 2, None, None, generator, policy,
                               jax.random.PRNGKey(0), [jnp.array([[0.], [2.]])])
        # Full-team adapters are 1+(1+3)=5 and 8+(2+8)=18.
        expected = compute_nmd_from_action_means(jnp.array([[[5.], [7.]], [[18.], [20.]]]))
        self.assertAlmostEqual(actual, expected, places=6)

    def test_missing_team_lidar_is_rejected_instead_of_repeated(self):
        generator = SimpleNamespace(max_agents=2, agent_position_dim=0,
                                    food_position_dim=0, target_nmd_dim=0, env_context_dim=0)
        entries = [{"task": None, "context": None, "lidar": jnp.ones(1)}] * 2
        with self.assertRaisesRegex(ValueError, "team_lidar"):
            calculate_nmd(entries, 2, None, None, generator, None,
                          jax.random.PRNGKey(0), [jnp.zeros((2, 1))])
