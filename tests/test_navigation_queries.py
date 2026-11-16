import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import torch

from hyperlora.envs.events.navigation import NavigationEvents
from hyperlora.envs.tokens import HypernetworkInputs


class NavigationQueriesTest(unittest.TestCase):
    def make_inputs(self):
        obs = [torch.zeros((2, 20)) for _ in range(4)]
        lidar = jnp.arange(2 * 4 * 12, dtype=jnp.float32).reshape(2, 4, 12)
        for i in range(4):
            obs[i][:, 8:] = torch.tensor(np.asarray(lidar[:, i]).copy())
        tokens = HypernetworkInputs(
            lidar=lidar, generation_lidar=jnp.repeat(lidar[:, None], 4, axis=1))
        return obs, tokens

    def test_first_detection_updates_only_that_agent_with_its_whole_team_context(self):
        obs, tokens = self.make_inputs()
        events = NavigationEvents(2, 4, True, 20, research=True)
        events.prime(obs)
        self.assertIsNone(events.before_step(None, obs, tokens, 0))
        obs[2][1, 7] = 1
        events.after_step(None, obs, tokens, 0)
        events.commit()
        change = events.before_step(None, obs, tokens, 1,
                                    active_envs=torch.tensor([True, True]))
        np.testing.assert_array_equal(change.indices, [6])
        np.testing.assert_array_equal(change.inputs.lidar, tokens.lidar[1:2])
        np.testing.assert_array_equal(change.generated_indices, [2])
        np.testing.assert_array_equal(change.tokens.generation_lidar[1, 2], tokens.lidar[1])
        events.after_step(None, obs, change.tokens, 1)
        events.commit()
        self.assertIsNone(events.before_step(None, obs, change.tokens, 2))

    def test_fixed_schedule_respects_episode_boundaries(self):
        obs, tokens = self.make_inputs()
        events = NavigationEvents(2, 4, True, 20, research=True, interval=2)
        events.prime(obs)
        self.assertIsNone(events.before_step(None, obs, tokens, 0))
        self.assertIsNone(events.before_step(None, obs, tokens, 1))
        change = events.before_step(None, obs, tokens, 2, active_envs=torch.tensor([False, True]))
        np.testing.assert_array_equal(change.indices, [4, 5, 6, 7])
        np.testing.assert_array_equal(change.generated_indices, [0, 5, 10, 15])
        disabled = NavigationEvents(2, 4, False, 20, research=True)
        self.assertIsNone(disabled.before_step(None, obs, tokens, 2))

    def test_trace_counts_generated_work_separately_from_deployed_adapters(self):
        from hyperlora.logging.diagnostics import RolloutTrace

        obs, tokens = self.make_inputs()
        events = NavigationEvents(2, 4, True, 20, research=True, interval=2)
        events.prime(obs)
        active = torch.tensor([False, True])
        change = events.before_step(None, obs, tokens, 2, active_envs=active)
        env = SimpleNamespace(scenario=None, world=SimpleNamespace(landmarks=[
            SimpleNamespace(eaten=torch.zeros(2, dtype=torch.bool)) for _ in range(4)]))
        trace = RolloutTrace("navigation", 2, 4, 3, 0)
        trace.step_index = 2
        trace.step(env, obs, [torch.zeros((2, 2)) for _ in range(4)],
                   [torch.zeros(2) for _ in range(4)], torch.zeros(2, dtype=torch.bool),
                   tokens, change, {"probe": jnp.zeros((8, 1))}, jnp.zeros((8, 2)),
                   jnp.ones((8, 2)), active, 1.)
        self.assertEqual(int(trace.queries[2].sum()), 1)
        self.assertEqual(int(trace.query_batch_sizes[2]), 4)
        self.assertEqual(int(trace.generated_adapters[2]), 16)
        self.assertEqual(int(trace.deployed_adapters[2]), 4)
