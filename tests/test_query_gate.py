from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import train_state

from hyperlora.config import load_config
from hyperlora.training.query_gate import (QueryGate, LearnedPressurePlateEvents,
    bernoulli_log_prob, gate_distribution, gate_update)

ROOT = Path(__file__).resolve().parents[1]


class QueryGateTest(unittest.TestCase):
    def config(self, mode="coverage", **overrides):
        return load_config(ROOT / "configs/main/ours/pressure_plate.yaml", {
            "research_metrics": True, "env.num_envs": 2, "ppo.rollout_steps": 8,
            "ppo.use_minibatches": False, "query_gate.mode": mode, **overrides})

    def state(self, bias=0):
        gate = QueryGate(8)
        p = gate.init(jax.random.PRNGKey(0), jnp.zeros((2, 7)))["params"]
        p["query"]["bias"] = jnp.array([bias], dtype=jnp.float32)
        return train_state.TrainState.create(apply_fn=gate.apply, params=p, tx=optax.adam(.001))

    def test_coverage_uses_mixture_likelihood_and_differentiates_through_it(self):
        logits = jnp.array([-3., 2.])
        action = jnp.array([True, False])
        learned, p = gate_distribution(logits, .2)
        np.testing.assert_allclose(p, .8*jax.nn.sigmoid(logits)+.1)
        actual = jax.grad(lambda l: bernoulli_log_prob(gate_distribution(l, .2)[1], action).sum())(logits)
        expected = (np.asarray(action)-p)/(p*(1-p)) * .8*learned*(1-learned)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertTrue(np.all(np.abs(actual) > 0))

    def test_initial_generation_is_not_queried_twice_and_terminal_teams_never_query(self):
        config = self.config()
        # Large positive logits make any remaining unmasked decision query.
        events = LearnedPressurePlateEvents(config, 2, 3, self.state(30), 0, evaluation=True)
        obs = [torch.zeros((2,2)) for _ in range(3)]
        self.assertIsNone(events.before_step(None, obs, None, 0, torch.ones(2,dtype=torch.bool)))
        self.assertIsNone(events.before_step(None, obs, None, 1, torch.zeros(2,dtype=torch.bool)))
        self.assertFalse(events.arrays()["query"].any())
        self.assertFalse(events.arrays()["valid"].any())
        self.assertEqual(events.epsilon, 0)

    def test_evaluation_removes_exploration_and_uses_repeatable_gate_stream(self):
        config = self.config()
        a = LearnedPressurePlateEvents(config, 2, 3, self.state(), 10, evaluation=True)
        b = LearnedPressurePlateEvents(config, 2, 3, self.state(), 99, evaluation=True)
        c = LearnedPressurePlateEvents(config, 2, 3, self.state(), 10)
        np.testing.assert_array_equal(a.key, b.key)
        self.assertEqual(a.epsilon, 0)
        self.assertEqual(c.epsilon, .2)
        self.assertFalse(np.array_equal(a.key, c.key))

    def test_invalid_and_unsupported_gate_configs_raise(self):
        for overrides in ({"query_gate.coverage_epsilon": 1}, {"query_gate.query_cost": -1},
                          {"query_gate.learning_rate": 0}, {"query_gate.hidden_dim": 0},
                          {"model.query_interval": 38}, {"research_metrics": False}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)

    def test_ppo_is_invariant_to_post_terminal_padding(self):
        state = self.state()
        x = jnp.arange(28,dtype=float).reshape(4,7)/28
        q = jnp.array([True,False,True,False])
        logits,v = state.apply_fn({"params":state.params},x)
        logp = bernoulli_log_prob(gate_distribution(logits,.2)[1],q)
        adv,ret = jnp.array([1.,-1.,.4,-.4]),v+.2
        def update(pad):
            rows = lambda a: jnp.concatenate([a,jnp.full((pad,)+a.shape[1:], 999,dtype=a.dtype)])
            valid = jnp.concatenate([jnp.ones(4,dtype=bool),jnp.zeros(pad,dtype=bool)])
            return gate_update(state,rows(x),rows(q),rows(logp),rows(v),rows(adv),rows(ret),
                               valid,valid,.2,.2,.01,.25)[0]
        a,b = update(0),update(3)
        for aa,bb in zip(jax.tree.leaves(a.params),jax.tree.leaves(b.params)):
            np.testing.assert_allclose(aa,bb,atol=1e-7)


if __name__ == "__main__":
    unittest.main()
