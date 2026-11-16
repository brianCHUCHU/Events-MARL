from pathlib import Path
import csv
import json
import argparse
from collections import Counter

import numpy as np

OUT = Path(__file__).resolve().parent


def load_metrics(run):
    return [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]


def summarize_trace(run, round_number):
    arrays = {}
    samples = []
    for file in sorted((run / "logs/evaluation_traces" / f"round_{round_number:04d}").glob("batch_*.npz")):
        with np.load(file, allow_pickle=False) as z:
            for key in ["completed", "length", "team_return", "team_queries"]:
                arrays.setdefault(key, []).append(z[key])
            for key in z.files:
                if key.startswith("first_") or key in ("agent_queries", "agent_first_collected_step", "agent_first_detected_step"):
                    arrays.setdefault(key, []).append(z[key])
            idx = z["sample_environment_indices"]
            length = z["length"][idx]
            last = length - 1
            active = z["sample_active"].astype(bool)
            assert np.array_equal(active.sum(axis=0), length)
            sample = {"batch": file.stem, "ids": idx.tolist(),
                      "completed": z["completed"][idx].tolist(),
                      "initial_observations": z["sample_observations"][0].tolist()}
            if run.name.startswith("pressure_plate"):
                at_goal = z["sample_agent_at_goal"].astype(bool)
                visited = z["sample_agent_goal_visited"][last, np.arange(len(idx))]
                bonus = (z["sample_first_visit_team_bonus"] * active).sum(axis=0)
                np.testing.assert_allclose(bonus, visited.sum(axis=-1) * 100.)
                exit_events = at_goal[:-1] & ~at_goal[1:] & active[1:, :, None]
                stage = z["sample_task_stage"][last, np.arange(len(idx))]
                sample.update(
                    visited=visited.tolist(),
                    final_at_goal=at_goal[last, np.arange(len(idx))].tolist(),
                    stage=stage.tolist(),
                    final_goal_distances=z["sample_distance_to_goal"][last, np.arange(len(idx))].tolist(),
                    exit_count=exit_events.sum(axis=(0, 2)).tolist(),
                    first_arrival_steps=np.where(at_goal.any(axis=0), at_goal.argmax(axis=0) + 1, -1).tolist(),
                )
                roles = Counter()
                for t, s, agent in np.argwhere(exit_events):
                    left = z["sample_assigned_left_agent"][t + 1, s]
                    right = z["sample_assigned_right_agent"][t + 1, s]
                    roles["left" if agent == left else "right" if agent == right else "other"] += 1
                sample["exit_roles"] = dict(roles)
            samples.append(sample)
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    assert len(arrays["completed"]) == 1280
    summary = {
        "episodes": len(arrays["completed"]),
        "success_percent": float(arrays["completed"].mean() * 100),
        "mean_team_return": float(arrays["team_return"].mean()),
        "mean_length": float(arrays["length"].mean()),
        "mean_team_queries": float(arrays["team_queries"].mean()),
        "milestone_percent": {k: float((v >= 0).mean() * 100) for k, v in arrays.items() if k.startswith("first_")},
    }
    if run.name.startswith("pressure_plate"):
        visited = np.concatenate([s["visited"] for s in samples])
        at_goal = np.concatenate([s["final_at_goal"] for s in samples])
        completed = np.concatenate([s["completed"] for s in samples]).astype(bool)
        exits = np.concatenate([s["exit_count"] for s in samples])
        stages = np.concatenate([s["stage"] for s in samples])
        role_counts = Counter()
        for s in samples:
            role_counts.update(s["exit_roles"])
        summary["sample"] = {
            "episodes": len(completed),
            "visited_all_percent": float(visited.all(axis=1).mean() * 100),
            "visited_count_histogram": dict(Counter(visited.sum(axis=1).tolist())),
            "at_goal_count_histogram": dict(Counter(at_goal.sum(axis=1).tolist())),
            "all_visited_but_not_completed_percent": float((visited.all(axis=1) & ~completed).mean() * 100),
            "episodes_with_goal_exit_percent": float((exits > 0).mean() * 100),
            "total_goal_exits": int(exits.sum()), "goal_exit_roles": dict(role_counts),
            "stage_histogram": dict(Counter(stages.tolist())),
        }
    else:
        collected = arrays["agent_first_collected_step"] >= 0
        detected = arrays["agent_first_detected_step"] >= 0
        summary.update(
            goal_completion_percent=float(collected.mean() * 100),
            goals_per_episode_histogram=dict(Counter(collected.sum(axis=1).tolist())),
            per_agent_collected_percent=(collected.mean(axis=0) * 100).tolist(),
            per_agent_detected_percent=(detected.mean(axis=0) * 100).tolist(),
            detected_but_not_collected_percent=((detected & ~collected).mean(axis=0) * 100).tolist(),
            mean_agent_queries=float(arrays["agent_queries"].mean()),
        )
    return summary, arrays, samples


def optimize_window(rows):
    keys = ["policy_loss", "value_loss", "value_explained_variance", "entropy", "mean_std",
            "mean_ratio", "clip_fraction", "approx_kl", "policy_grad_norm", "hn_grad_norm",
            "critic_grad_norm", "nmd", "nmd_ema", "alpha"]
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in keys if any(r.get(k) is not None for r in rows)}


def main():
    parser = argparse.ArgumentParser(description="Reanalyze the retained raw local pilot batch")
    parser.add_argument("--batch", required=True, type=Path)
    args = parser.parse_args()
    BATCH = args.batch.resolve()
    summaries = {r["scenario"] + "_" + r["schedule"] + "_seed" + r["seed"]: r
                 for r in csv.DictReader((BATCH / "summary.csv").open())}
    all_results = {}
    flat_rows = []
    for run in sorted(BATCH.glob("*_seed42")):
        metrics = load_metrics(run)
        train = [r for r in metrics if not r.get("eval")]
        ev = [r for r in metrics if r.get("eval")]
        best = max(ev, key=lambda r: r["completion_rate"])
        last = ev[-1]
        peak_round = best["training_round"]
        peak_stats, peak_arrays, peak_samples = summarize_trace(run, peak_round)
        final_stats, final_arrays, final_samples = summarize_trace(run, last["training_round"])
        for a, b in zip(peak_samples, final_samples):
            np.testing.assert_array_equal(a["initial_observations"], b["initial_observations"])
        paired = {
            "success_to_failure": int((peak_arrays["completed"] & ~final_arrays["completed"]).sum()),
            "failure_to_success": int((~peak_arrays["completed"] & final_arrays["completed"]).sum()),
            "both_success": int((peak_arrays["completed"] & final_arrays["completed"]).sum()),
            "both_failure": int((~peak_arrays["completed"] & ~final_arrays["completed"]).sum()),
        }
        late = [r for r in ev if r["training_round"] > peak_round]
        largest_drop = max([0.] + [a["completion_rate"] - b["completion_rate"] for a, b in zip(ev, ev[1:])])
        result = {
            "peak_round": peak_round, "final_round": last["training_round"],
            "peak_success": best["completion_rate"], "final_success": last["completion_rate"],
            "peak_to_final_drop_pp": best["completion_rate"] - last["completion_rate"],
            "largest_adjacent_drop_pp": largest_drop,
            "post_peak_successes": [[r["training_round"], r["completion_rate"]] for r in late],
            "peak": peak_stats, "final": final_stats, "paired": paired,
            "optimizer_near_peak": optimize_window([r for r in train if peak_round - 10 < r["training_round"] <= peak_round]),
            "optimizer_last_10": optimize_window(train[-10:]),
            "max_pre_update_log_prob_mse": max(r["pre_update_log_prob_mse"] for r in train),
            "curve": [{k: r.get(k) for k in ["training_round", "training_environment_steps",
                       "simulator_training_environment_steps", "completion_rate", "goal_completion_percent", "avg_reward"]} for r in ev],
        }
        all_results[run.name] = result
        (OUT / f"{run.name}_sampled_peak_final.json").write_text(json.dumps({"peak": peak_samples, "final": final_samples}, indent=2) + "\n")
        s = summaries[run.name]
        flat_rows.append({"run": run.name, "peak_success_percent": best["completion_rate"],
                         "peak_goal_completion_percent": peak_stats.get("goal_completion_percent"),
                         "final_goal_completion_percent": final_stats.get("goal_completion_percent"),
                         "peak_round": peak_round if best["completion_rate"] > 0 else None,
                         "peak_valid_environment_steps": best["training_environment_steps"] if best["completion_rate"] > 0 else None,
                         "peak_simulator_environment_steps": best["simulator_training_environment_steps"] if best["completion_rate"] > 0 else None,
                         "final_success_percent": last["completion_rate"],
                         "peak_to_final_drop_pp": result["peak_to_final_drop_pp"],
                         "process_minutes": float(s["process_wall_seconds"]) / 60,
                         "train_eval_minutes_at_peak": float(s["cumulative_train_eval_seconds_at_peak"]) / 60 if best["completion_rate"] > 0 else None,
                         "peak_mean_queries": peak_stats.get("mean_agent_queries", peak_stats["mean_team_queries"]),
                         "peak_checkpoint": s["peak_checkpoint"] if best["completion_rate"] > 0 else None})
    (OUT / "analysis.json").write_text(json.dumps(all_results, indent=2) + "\n")
    with (OUT / "status_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_rows[0])
        writer.writeheader(); writer.writerows(flat_rows)
    log = BATCH.parent / "batch.log"
    wall = log.stat().st_mtime - log.stat().st_birthtime
    timing = {"batch_wall_seconds_from_log_lifetime": wall,
              "pressure_plate_sum_process_seconds": sum(float(r["process_wall_seconds"]) for r in summaries.values() if r["scenario"] == "pressure_plate"),
              "navigation_sum_process_seconds": sum(float(r["process_wall_seconds"]) for r in summaries.values() if r["scenario"] == "navigation"),
              "max_pre_update_log_prob_mse": max(r["max_pre_update_log_prob_mse"] for r in all_results.values())}
    (OUT / "timing_and_checks.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps({"timing": timing, "runs": len(all_results)}, indent=2))
    for name, r in all_results.items():
        print(name, json.dumps({"peak": r["peak_success"], "final": r["final_success"],
                               "peak_stats": r["peak"].get("sample", r["peak"].get("per_agent_collected_percent")),
                               "final_stats": r["final"].get("sample", r["final"].get("per_agent_collected_percent")),
                               "paired": r["paired"]}))


if __name__ == "__main__":
    main()
