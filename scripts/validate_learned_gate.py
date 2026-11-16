"""Short real-environment checks; this does not test task convergence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []
    for mode in ("learned", "coverage"):
        finals = []
        for interval in (1, 2):
            path = args.out / f"{mode}_eval{interval}"
            command = [sys.executable, "-m", "hyperlora.cli.train", "--config",
                       "configs/main/ours/pressure_plate.yaml", "--out", str(path.resolve()),
                       "--seed", "42"]
            for override in ("research_metrics=True", f"query_gate.mode={mode}",
                             "env.num_envs=4", "ppo.num_episodes=2", "ppo.rollout_steps=8",
                             "ppo.num_minibatches=2", "evaluation.num_eval_episodes=1",
                             "evaluation.max_eval_steps=8", f"evaluation.eval_interval={interval}"):
                command += ["--set", override]
            path.mkdir()
            with (path / "process.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=os.environ | {"PYTHONPATH": str(ROOT / "src"),
                                                 "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "4"})
            metrics = [json.loads(line) for line in (path / "logs/metrics.jsonl").read_text().splitlines()]
            train = [r for r in metrics if not r.get("eval")]
            assert len(train) == 2
            assert max(r["pre_update_log_prob_mse"] for r in train) <= 1e-6
            assert max(r["gate_pre_update_log_prob_mse"] for r in train) <= 1e-6
            expected_epsilon = .2 if mode == "coverage" else 0
            assert all(r["gate_coverage_epsilon"] == expected_epsilon for r in train)
            assert all(r["evaluation_coverage_epsilon"] == 0 for r in metrics if r.get("eval"))
            for file in (path / "logs/gate_evaluation_traces").rglob("*.npz"):
                with np.load(file, allow_pickle=False) as data:
                    assert data["coverage_epsilon"] == 0
                    assert not data["query"][0].any()
                    assert not data["query"][~data["valid"]].any()
                    np.testing.assert_allclose(data["behavior_probability"], data["learned_probability"], atol=1e-6)
                paired = path / "logs/evaluation_traces" / file.parent.name / file.name
                with np.load(paired, allow_pickle=False) as task, np.load(file, allow_pickle=False) as gate:
                    np.testing.assert_array_equal(task["extra_query_at_step"][:len(gate["query"])],gate["query"])
                    np.testing.assert_array_equal(task["team_queries"],1+gate["query"].sum(0))
            # Only load locally generated trusted parameter checkpoints.
            with np.load(path / "checkpoints/final_checkpoint.npz", allow_pickle=True) as checkpoint:
                finals.append({k: checkpoint[k].item() for k in
                               ("policy_params", "hn_params", "critic_params", "gate_params")})
            with np.load(path / "checkpoints/evaluated_round_0000.npz", allow_pickle=True) as checkpoint:
                initial_gate = checkpoint["gate_params"].item()
            assert any(not np.array_equal(a,b) for a,b in zip(
                jax.tree.leaves(initial_gate),jax.tree.leaves(finals[-1]["gate_params"])))
            checks.append(dict(mode=mode,eval_interval=interval,rounds=2,
                max_actor_replay_mse=max(r["pre_update_log_prob_mse"] for r in train),
                max_gate_replay_mse=max(r["gate_pre_update_log_prob_mse"] for r in train)))
            print(f"Passed {mode}, evaluation interval {interval}",flush=True)
        for a,b in zip(jax.tree.leaves(finals[0]),jax.tree.leaves(finals[1])):
            np.testing.assert_array_equal(a,b)
    (args.out / "checks.json").write_text(json.dumps(dict(
        passed=True, evaluation_frequency_invariance="exact_all_four_parameter_trees",cases=checks),indent=2)+"\n")


if __name__ == "__main__":
    main()
