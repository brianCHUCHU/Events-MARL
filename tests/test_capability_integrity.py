from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from hyperlora.config import load_config
from hyperlora.envs.capabilities import initial_capabilities
from hyperlora.envs.factory import make_env
from hyperlora.logging.diagnostics import environment_manifest
from hyperlora.seeding import seed_globals

ROOT = Path(__file__).resolve().parents[1]


class CapabilityIntegrityTest(unittest.TestCase):
    def test_pressure_plate_config_matches_actual_homogeneous_physics(self):
        for method in ("ours", "ps", "dico", "cash"):
            with self.subTest(method=method):
                config = load_config(ROOT / f"configs/main/{method}/pressure_plate.yaml",
                                     {"env.num_envs": 2})
                capabilities = initial_capabilities(config.env)
                env = make_env(config.env, capabilities)
                manifest = environment_manifest(config, env)
                actual = [a["max_speed"] for a in manifest["actual_agents_at_initialization"]]
                self.assertEqual(actual, capabilities["speed"])
                self.assertEqual(actual, [1., 1., 1.])

    def test_ignored_pressure_plate_speed_setting_is_rejected(self):
        config = load_config(ROOT / "configs/main/ours/pressure_plate.yaml", {"env.num_envs": 2})
        changed = replace(config.env, fixed_capabilities={"speed": [0.8, 1., 1.2]})
        with self.assertRaisesRegex(ValueError, "max_speed"):
            initial_capabilities(changed)
        with self.assertRaisesRegex(ValueError, "max_speed"):
            make_env(config.env, {"speed": [0.8, 1., 1.2]})

    def test_navigation_uses_configured_prefix_without_substitution(self):
        config = load_config(ROOT / "configs/main/ours/navigation.yaml", {"env.num_envs": 2})
        fixed = {"speed": [.6, .7, .8, .9, 1.], "lidar_range": [.2, .3, .4, .5, .6]}
        changed = replace(config.env, fixed_capabilities=fixed)
        actual = initial_capabilities(changed)
        self.assertEqual(actual, {k: v[:4] for k, v in fixed.items()})
        with self.assertRaisesRegex(ValueError, "all 4 agents"):
            initial_capabilities(replace(changed, fixed_capabilities={"speed": [1.]}))

    def test_training_seed_resets_vmas_even_when_already_imported(self):
        config = load_config(ROOT / "configs/main/ours/navigation.yaml", {"env.num_envs": 2})
        env = make_env(config.env, initial_capabilities(config.env))
        seed_globals(42)
        first = np.array(env.reset()[0])
        env.reset()
        seed_globals(42)
        second = np.array(env.reset()[0])
        np.testing.assert_array_equal(first, second)
