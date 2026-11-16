import unittest
from types import SimpleNamespace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from hyperlora.algorithms.ours import make_train_step
from hyperlora.training.batching import make_indices
from hyperlora.training.gae import normalize_returns


class TrainingIntegrityTest(unittest.TestCase):
    def test_custom_min_std_matches_rollout_distribution_before_update(self):
        policy = SimpleNamespace(min_std=0.1, apply=lambda variables, obs, adapters: (
            variables["params"]["scale"] * obs + adapters["offset"], jnp.full_like(obs, -4.)))
        generator = SimpleNamespace(apply=lambda variables, *args: {
            "offset": (variables["params"]["scale"] * args[6]).reshape(-1, 1),
        })
        critic = SimpleNamespace(apply=lambda variables, state: variables["params"]["scale"] * state)

        def state(model):
            return TrainState.create(apply_fn=model.apply, params={"scale": jnp.array(0.)},
                                     tx=optax.sgd(0.))

        obs = jnp.ones((2, 1))
        action = jnp.ones((2, 1)) * 0.05
        old_logs = distrax.Transformed(distrax.Normal(jnp.zeros_like(obs), 0.1),
                                      distrax.Tanh()).log_prob(action).sum(axis=-1)
        _, _, _, info = make_train_step(policy, generator, critic, 0.)(
            state(policy), state(generator), state(critic), obs, obs, action,
            old_logs, jnp.ones(2), jnp.zeros(2), None, None, None, None, None,
            jnp.ones((2, 1, 1)), None, 0.2, 0., 0.5, 1, 1,
        )
        self.assertLess(float(info["approx_kl"]), 1e-12)
        self.assertAlmostEqual(float(info["mean_ratio"]), 1., places=6)

    def test_minibatches_keep_complete_teams_and_cover_all_data(self):
        indices = make_indices(24, 8, 3, 4, True, jax.random.PRNGKey(7))
        agent_rows, team_rows = [], []
        for mb in range(4):
            rows, teams = indices.slice(mb)
            np.testing.assert_array_equal(rows.reshape(-1, 3),
                                          np.asarray(teams)[:, None] * 3 + np.arange(3))
            agent_rows.extend(rows.tolist())
            team_rows.extend(teams.tolist())
        self.assertEqual(sorted(agent_rows), list(range(24)))
        self.assertEqual(sorted(team_rows), list(range(8)))

    def test_invalid_minibatches_are_rejected(self):
        for arguments in [(15, 5, 3, 2, False, None), (15, 6, 3, 2, False, None),
                          (12, 4, 3, 2, True, None)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                make_indices(*arguments)

    def test_return_normalization_ignores_terminal_padding(self):
        valid = np.array([True, True, False, False])
        expected = normalize_returns(np.array([1., 3.]), np.array([0., 1.]))
        actual = normalize_returns(np.array([1., 3., 10000., -20000.]),
                                   np.array([0., 1., 50., -80.]), valid)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b[:2], atol=1e-6)
            np.testing.assert_array_equal(b[2:], [0., 0.])

    def test_nmd_reference_observations_exclude_inactive_team_rows(self):
        from hyperlora.training.trainer import _nmd_observations

        trajectory = SimpleNamespace(
            obs=[jnp.arange(4)[:, None], jnp.arange(4, 8)[:, None]],
            active_envs=[np.array([True, True]), np.array([False, True])],
        )
        observations = _nmd_observations(trajectory, 2)
        np.testing.assert_array_equal(jnp.concatenate(observations)[:, 0], [0, 1, 2, 3, 6, 7])
