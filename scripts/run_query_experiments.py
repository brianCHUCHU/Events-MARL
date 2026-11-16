"""Run the two-seed query comparison in independent, one-shot processes."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 42)
NAVIGATION = {
    "event": (),
    "fixed_25": ("model.query_interval=25",),
    "fixed_50": ("model.query_interval=50",),
    "fixed_100": ("model.query_interval=100",),
    "single": ("model.adaptive_hypernetwork=False",),
}
PRESSURE_PLATE = {
    "event": (),
    "fixed_38": ("model.query_interval=38",),
    "fixed_76": ("model.query_interval=76",),
    "fixed_152": ("model.query_interval=152",),
    "single": ("model.adaptive_hypernetwork=False",),
}


def compute_run_summary(path, scenario, schedule, seed, wall_seconds):
    log_dir = path / "logs"
    rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
    train = [row for row in rows if not row.get("eval")]
    evaluation = [row for row in rows if row.get("eval")]
    final = evaluation[-1]
    resources = json.loads((log_dir / "resources.json").read_text())
    if (final["training_round"] != len(train)
            or not (path / "checkpoints" / "final_checkpoint.npz").is_file()):
        raise RuntimeError(f"Incomplete metrics or checkpoint: {path}")
    x = [row["training_environment_steps"] for row in evaluation]
    total = x[-1]
    result = {
        "scenario": scenario,
        "schedule": schedule,
        "seed": seed,
        "training_environment_steps": total,
        "simulator_training_environment_steps": final.get("simulator_training_environment_steps"),
        "masked_post_terminal_env_steps": sum(row.get("masked_post_terminal_env_steps", 0)
                                              for row in train)
        if "masked_post_terminal_env_steps" in train[0] else None,
        "process_wall_seconds": round(wall_seconds, 3) if wall_seconds is not None else None,
        "training_wall_seconds": sum(row["seconds"] for row in train),
        "evaluation_wall_seconds": sum(row["evaluation_wall_seconds"] for row in evaluation),
        "cpu_seconds": resources["train_and_evaluate_cpu_seconds"],
        "peak_rss_bytes": resources["max_rss_bytes"],
        "training_team_queries": sum(row["team_initial_queries"] + row["team_requeries"] for row in train),
        "training_generator_deployment_batches": sum(row["generator_rollout_batches"] for row in train),
        "training_generator_ppo_batches": sum(row["generator_ppo_batches"] for row in train),
        "final_mean_team_queries": final["mean_team_queries"],
        "final_completion_rate": final["completion_rate"],
        "final_team_return": final["avg_reward"],
        "final_goal_completion_percent": final.get("goal_completion_percent"),
        "final_mean_agent_queries": final.get("mean_agent_queries"),
        "training_agent_queries": sum(row.get("agent_initial_queries", 0)
                                      + row.get("agent_requeries", 0) for row in train),
        "final_shielding": final.get("mean_small_agent_shielding"),
        "final_separation": final.get("mean_separation"),
    }
    result["metrics_schema_version"] = final.get("metrics_schema_version")
    for name in ("generator_setup_input_contexts", "generator_refresh_input_contexts",
                 "generator_refresh_generated_adapters", "generator_ppo_input_contexts",
                 "generator_ppo_generated_adapters", "generator_nmd_input_contexts"):
        result[f"training_{name}"] = (sum(row[name] for row in train)
                                      if final.get("metrics_schema_version") == 2 else None)
    for name in ("generator_input_contexts", "generator_generated_adapters", "generator_deployed_adapters"):
        result[f"final_{name}"] = final.get(name)
    scored = [row for row in evaluation if row["completion_rate"] is not None]
    peak = max(scored, key=lambda row: row["completion_rate"]) if scored else None
    result.update({
        "peak_completion_rate": peak["completion_rate"] if peak else None,
        "peak_training_round": peak["training_round"] if peak else None,
        "peak_training_environment_steps": peak["training_environment_steps"] if peak else None,
        "peak_simulator_training_environment_steps": peak["simulator_training_environment_steps"] if peak else None,
        "peak_goal_completion_percent": peak.get("goal_completion_percent") if peak else None,
        "cumulative_train_eval_seconds_at_peak": (
            sum(row["seconds"] for row in train if row["training_round"] <= peak["training_round"])
            + sum(row["evaluation_wall_seconds"] for row in evaluation
                  if row["training_round"] <= peak["training_round"]) if peak else None),
    })
    peak_checkpoint = (path / "checkpoints" / f"evaluated_round_{peak['training_round']:04d}.npz"
                       if peak else None)
    result["peak_checkpoint"] = (str(peak_checkpoint) if peak_checkpoint is not None
                                 and peak_checkpoint.is_file() else None)
    for key in ("completion_rate", "avg_reward", "goal_completion_percent"):
        y = [row.get(key) for row in evaluation]
        result[f"normalized_auc_{key}"] = (
            sum((a + b) / 2 * (hi - lo)
                for a, b, lo, hi in zip(y, y[1:], x, x[1:])) / total
            if y[0] is not None else None
        )
    hits = [row for row in evaluation if (row["completion_rate"] or 0) >= 50]
    result["first_observed_50_percent_steps"] = hits[0]["training_environment_steps"] if hits else None
    if hits:
        steps = result["first_observed_50_percent_steps"]
        result["wall_seconds_to_first_observed_50_percent"] = (
            sum(row["seconds"] for row in train if row["training_environment_steps"] <= steps)
            + sum(row["evaluation_wall_seconds"] for row in evaluation
                  if row["training_environment_steps"] <= steps)
        )
    else:
        result["wall_seconds_to_first_observed_50_percent"] = None

    late_completion = [r["completion_rate"] for r in evaluation[-3:]]
    result["last_three_completion_mean"] = (
        sum(late_completion) / len(late_completion) if late_completion[0] is not None else None
    )
    result["largest_completion_drop_pp"] = (
        max(0.0, max(a - b for a, b in zip(late_completion, late_completion[1:])))
        if len(late_completion) > 1 and late_completion[0] is not None else None
    )
    return result


def write_seed_summary(path, results):
    groups = {}
    for row in results:
        groups.setdefault((row["scenario"], row["schedule"]), []).append(row)
    fields = (
        "peak_completion_rate", "peak_training_environment_steps",
        "peak_simulator_training_environment_steps", "cumulative_train_eval_seconds_at_peak",
        "final_completion_rate", "normalized_auc_completion_rate", "final_team_return",
        "normalized_auc_avg_reward", "last_three_completion_mean", "largest_completion_drop_pp",
        "process_wall_seconds", "training_wall_seconds", "evaluation_wall_seconds",
        "cpu_seconds", "peak_rss_bytes", "final_mean_team_queries",
        "final_shielding", "final_separation", "final_goal_completion_percent",
        "final_mean_agent_queries", "normalized_auc_goal_completion_percent",
    )
    summaries = []
    for (scenario, schedule), runs in groups.items():
        summary = {
            "scenario": scenario,
            "schedule": schedule,
            "completed_seeds": len(runs),
            "seeds_reached_50_percent": (
                sum(row["first_observed_50_percent_steps"] is not None for row in runs)
                if scenario in ("pressure_plate", "navigation") else None
            ),
        }
        for key in fields:
            values = [row[key] for row in runs if row[key] is not None]
            summary[f"mean_{key}"] = mean(values) if values else None
            summary[f"sample_sd_{key}"] = stdev(values) if len(values) >= 2 else None
        summaries.append(summary)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)


def execute_run(entry, output, source, env, resume):
    scenario, schedule, seed, overrides = entry
    run_name = f"{scenario}_{schedule}_seed{seed}"
    path = output / run_name
    try:
        if resume and path.exists():
            result = compute_run_summary(path, scenario, schedule, seed, None)
            print(f"Reused {run_name}", flush=True)
            return result, None
        path.mkdir()
        print(f"Starting {run_name}", flush=True)
        command = [
            sys.executable, "-u", "-m", "hyperlora.cli.train",
            "--config", f"configs/main/ours/{scenario}.yaml",
            "--out", str(path), "--seed", str(seed), "--set", "research_metrics=True",
        ]
        for setting in overrides:
            command.extend(("--set", setting))
        started = perf_counter()
        with (path / "process.log").open("w") as log:
            completed = subprocess.run(command, cwd=source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        elapsed = perf_counter() - started
        if completed.returncode:
            raise RuntimeError(f"Training exited with code {completed.returncode}")
        result = compute_run_summary(path, scenario, schedule, seed, elapsed)
        print(f"Finished {run_name} in {elapsed:.1f} s", flush=True)
        return result, None
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Failed {run_name}: {error}", flush=True)
        return None, {"run": run_name, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scenario", choices=("pressure_plate", "wind_flocking", "navigation",
                                               "pressure_plate_navigation"))
    parser.add_argument("--resume", action="store_true", help="Reuse complete runs in this batch")
    parser.add_argument("--seeds", type=int, nargs="+", help="Training seeds for a new batch")
    parser.add_argument("--schedules", nargs="+", help="Schedule names for one selected environment")
    parser.add_argument("--eval-interval", type=int, help="Periodic evaluation interval for a new batch")
    parser.add_argument("--workers", type=int, default=1, help="Maximum simultaneous training processes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.resume and any(value is not None for value in (args.seeds, args.schedules, args.eval_interval)):
        parser.error("Resume uses its frozen plan; seed/schedule/evaluation overrides are not allowed")
    seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("Training seeds must be distinct unsigned 32-bit integers")
    if args.eval_interval is not None and args.eval_interval < 1:
        parser.error("--eval-interval must be positive")
    if args.schedules is not None:
        supported = {"pressure_plate": PRESSURE_PLATE, "navigation": NAVIGATION}
        if args.scenario not in supported:
            parser.error("--schedules requires --scenario pressure_plate or navigation")
        unknown = set(args.schedules) - supported[args.scenario].keys()
        if unknown or len(set(args.schedules)) != len(args.schedules):
            parser.error(f"Unknown or repeated schedules: {args.schedules}")
    output = args.out.resolve()
    os.umask(0o077)
    source = output / "source"
    if args.resume:
        if not args.scenario:
            parser.error("--resume requires --scenario")
        prior_plan = json.loads((output / "plan.json").read_text())
        scenarios = ({"pressure_plate", "navigation"} if args.scenario == "pressure_plate_navigation"
                     else {args.scenario})
        plan = [entry for entry in prior_plan if entry[0] in scenarios]
        if not plan or not (source / "src").is_dir():
            parser.error("The existing batch does not contain the requested scenario and source")
        original_plan = output / "initial_plan.json"
        if not original_plan.exists():
            shutil.copyfile(output / "plan.json", original_plan)
    else:
        output.mkdir(mode=0o700)
        shutil.copytree(ROOT / "src", source / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ROOT / "configs" / "main" / "ours", source / "configs" / "main" / "ours")
        shutil.copyfile(Path(__file__), source / "run_query_experiments.py")
        plan = [
            ("pressure_plate", "event", seed, PRESSURE_PLATE["event"]) for seed in seeds
        ] + [
            ("wind_flocking", "target_cycle", seed, ()) for seed in seeds
        ] + [
            ("pressure_plate", name, seed, overrides)
            for name, overrides in PRESSURE_PLATE.items() if name != "event" for seed in seeds
        ]
        if args.scenario == "navigation":
            plan = [("navigation", name, seed, (*overrides, "evaluation.eval_interval=25"))
                    for name, overrides in NAVIGATION.items() for seed in seeds]
        elif args.scenario == "pressure_plate_navigation":
            # Prioritize useful status-update comparisons without changing budgets.
            pp = [("pressure_plate", name, seed, (*PRESSURE_PLATE[name], "evaluation.eval_interval=10"))
                  for name in ("fixed_152", "event", "single", "fixed_76", "fixed_38") for seed in seeds]
            nav = [("navigation", name, seed, (*NAVIGATION[name], "evaluation.eval_interval=25"))
                   for name in ("single", "event", "fixed_100", "fixed_50", "fixed_25") for seed in seeds]
            plan = [entry for pair in zip(pp, nav) for entry in pair]
        elif args.scenario:
            plan = [entry for entry in plan if entry[0] == args.scenario]
        if args.schedules is not None:
            plan = [entry for entry in plan if entry[1] in args.schedules]
        if args.eval_interval is not None:
            plan = [(scenario, schedule, seed,
                     tuple(x for x in overrides if not x.startswith("evaluation.eval_interval="))
                     + (f"evaluation.eval_interval={args.eval_interval}",))
                    for scenario, schedule, seed, overrides in plan]
        with (output / "git_diff.patch").open("w") as diff:
            subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, stdout=diff, check=True)
        with (output / "upstream_commit.txt").open("w") as commit:
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, stdout=commit, check=True)
        with (output / "packages.txt").open("w") as packages:
            from importlib.metadata import distributions

            packages.write("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))
                           + "\n")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ | {
        "PYTHONPATH": str(source / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "4",
        "WANDB_MODE": "disabled",
    }

    results_by_index, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        remaining = list(enumerate(plan))
        limit_per_environment = args.scenario == "pressure_plate_navigation"

        def write_status():
            running = [f"{plan[i][0]}_{plan[i][1]}_seed{plan[i][2]}"
                       for future, i in pending.items() if future.running()]
            (output / "status.json").write_text(json.dumps({
                "planned": len(plan), "completed": len(results_by_index), "failed": failed,
                "parallel_workers": args.workers, "running": running,
                "per_environment_worker_limit": 1 if limit_per_environment else None,
            }, indent=2) + "\n")

        while remaining or pending:
            for index, entry in list(remaining):
                if len(pending) >= args.workers:
                    break
                if limit_per_environment and any(plan[i][0] == entry[0] for i in pending.values()):
                    continue
                pending[pool.submit(execute_run, entry, output, source, env, args.resume)] = index
                remaining.remove((index, entry))
            write_status()
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                result, failure = future.result()
                if failure is not None:
                    failed.append(failure)
                else:
                    results_by_index[index] = result
                results = [results_by_index[i] for i in sorted(results_by_index)]
                if results:
                    with (output / "summary.csv").open("w", newline="") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=results[0])
                        writer.writeheader()
                        writer.writerows(results)
                    write_seed_summary(output / "seed_summary.csv", results)
                write_status()
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
