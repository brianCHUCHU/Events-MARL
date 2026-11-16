"""Short, isolated checks of the two research pipelines; not full experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def equal_tree(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    return np.array_equal(left, right)


def checkpoint(path):
    # These are trusted checkpoints just produced by this local validation.
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k].item() if data[k].dtype == object else data[k] for k in data.files}


def inspect_run(path, scenario, schedule, interval, epochs, minibatches):
    rows = [json.loads(line) for line in (path / "logs/metrics.jsonl").read_text().splitlines()]
    train, evaluations = [r for r in rows if not r.get("eval")], [r for r in rows if r.get("eval")]
    assert len(train) == 2 and evaluations[-1]["training_round"] == 2
    assert (path / "checkpoints/final_checkpoint.npz").is_file()
    manifest = json.loads((path / "logs/environment.json").read_text())
    if scenario == "pressure_plate":
        assert [a["max_speed"] for a in manifest["actual_agents_at_initialization"]] == [1., 1., 1.]
    for row in train:
        assert row["pre_update_log_prob_mse"] < 1e-6
        assert row["generator_ppo_batches"] == epochs * minibatches
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), (path, key, value)
    for row in evaluations:
        assert row["evaluation_action_transform"] == "tanh_mean"
        assert row["evaluation_rng_isolated"]
    if scenario == "navigation":
        assert train[0]["nmd"] is None and train[0]["alpha"] == 1.
        assert train[0]["generator_nmd_input_contexts"] == 0
        assert train[1]["nmd"] is not None and np.isfinite(train[1]["nmd"])
    trace_count = 0
    for file in (path / "logs/evaluation_traces").glob("round_*/batch_*.npz"):
        with np.load(file, allow_pickle=False) as z:
            assert int(z["trace_schema_version"]) == 2
            active = z["active_at_step"]
            assert not np.any(z["extra_query_at_step"] & ~active)
            if scenario == "navigation":
                queries = z["agent_queries"]
                assert np.array_equal(z["completed"], (z["agent_first_collected_step"] >= 0).all(axis=1))
                assert not np.any(z["agent_query_at_step"] & ~active[:, :, None])
            else:
                queries = z["team_queries"][:, None]
                assert np.array_equal(z["completed"], z["first_all_at_goal_step"] >= 0)
            if schedule == "single":
                assert np.all(queries == 1)
            elif interval is not None:
                assert np.all(queries == (1 + (z["length"] - 1) // interval)[:, None])
            elif scenario == "navigation":
                assert np.all((queries >= 1) & (queries <= 2))
            assert np.all(z["generated_agent_adapters_at_step"] >= z["deployed_agent_adapters_at_step"])
            trace_count += 1
    return {"training_rounds": len(train), "evaluation_trace_files": trace_count,
            "max_pre_update_log_prob_mse": max(r["pre_update_log_prob_mse"] for r in train),
            "final_success": evaluations[-1]["completion_rate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    output = args.out.resolve()
    output.mkdir(mode=0o700)
    source = output / "source"
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "configs/main/ours", source / "configs/main/ours")
    shutil.copyfile(Path(__file__), source / "validate_query_pipeline.py")
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*") if p.is_file()}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ | {"PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                        "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}
    plan = []
    for scenario, intervals in (("pressure_plate", (38, 76, 152)), ("navigation", (25, 50, 100))):
        for schedule, interval in [("event", None), *[(f"fixed_{k}", k) for k in intervals], ("single", None)]:
            plan.append((f"{scenario}_{schedule}", scenario, schedule, interval, 0, 1, True))
    plan.extend([
        ("navigation_event_seed42", "navigation", "event", None, 42, 1, True),
        ("navigation_single_fullbatch", "navigation", "single", None, 0, 1, False),
        ("navigation_event_no_mid_eval", "navigation", "event", None, 0, 0, True),
        ("pressure_plate_event_no_mid_eval", "pressure_plate", "event", None, 0, 0, True),
    ])
    results = {}
    for name, scenario, schedule, interval, seed, eval_interval, use_minibatches in plan:
        path = output / name
        path.mkdir()
        command = [sys.executable, "-m", "hyperlora.cli.train", "--config",
                   str(source / f"configs/main/ours/{scenario}.yaml"), "--out", str(path), "--seed", str(seed)]
        epochs = 3 if scenario == "pressure_plate" else 4
        settings = ["research_metrics=True", "env.num_envs=2", "ppo.num_episodes=2",
                    f"ppo.ppo_epochs={epochs}", "ppo.num_minibatches=2", "evaluation.num_eval_episodes=1",
                    "diversity.adapter_nmd_sample_size=4", f"evaluation.eval_interval={eval_interval}",
                    f"ppo.use_minibatches={use_minibatches}"]
        if interval is not None:
            settings.append(f"model.query_interval={interval}")
        if schedule == "single":
            settings.append("model.adaptive_hypernetwork=False")
        for setting in settings:
            command.extend(("--set", setting))
        print(f"Validating {name}", flush=True)
        with (path / "process.log").open("w") as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[name] = inspect_run(path, scenario, schedule, interval, epochs, 2 if use_minibatches else 1)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for scenario in ("pressure_plate", "navigation"):
        left = checkpoint(output / f"{scenario}_event/checkpoints/final_checkpoint.npz")
        right = checkpoint(output / f"{scenario}_event_no_mid_eval/checkpoints/final_checkpoint.npz")
        assert equal_tree(left, right), f"Evaluation frequency changed {scenario} trained weights"
        results[f"{scenario}_evaluation_frequency_invariance"] = True
    initial0 = checkpoint(output / "navigation_event/checkpoints/evaluated_round_0000.npz")
    initial42 = checkpoint(output / "navigation_event_seed42/checkpoints/evaluated_round_0000.npz")
    assert not equal_tree(initial0["policy_params"], initial42["policy_params"])
    results["different_seeds_initialize_different_policies"] = True
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("All short validations passed; evaluation frequency preserved final model parameters exactly.", flush=True)


if __name__ == "__main__":
    main()
