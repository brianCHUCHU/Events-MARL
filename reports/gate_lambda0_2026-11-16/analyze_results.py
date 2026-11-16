"""Summarize completed gate pilots without modifying their frozen artifacts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_traces(run, evaluations):
    checked = 0
    for row in evaluations:
        round_number = row["training_round"]
        paths = sorted((run / "logs/evaluation_traces" / f"round_{round_number:04d}").glob("*.npz"))
        assert len(paths) == 10
        query_count = completed = episodes = 0
        team_return = 0.0
        for path in paths:
            gate_path = run / "logs/gate_evaluation_traces" / path.parent.name / path.name
            with np.load(path, allow_pickle=False) as z, np.load(gate_path, allow_pickle=False) as g:
                q = g["query"]
                assert float(g["coverage_epsilon"]) == 0
                assert not q[0].any() and not q[~g["valid"]].any()
                np.testing.assert_array_equal(q, z["extra_query_at_step"][:len(q)])
                np.testing.assert_array_equal(1 + q.sum(0), z["team_queries"])
                np.testing.assert_allclose(g["learned_probability"],g["behavior_probability"],atol=1e-6)
                assert not (z["extra_query_at_step"] & ~z["active_at_step"]).any()
                np.testing.assert_array_equal(z["completed"],z["first_all_at_goal_step"] >= 0)
                query_count += int(z["team_queries"].sum())
                completed += int(z["completed"].sum())
                episodes += len(z["completed"])
                team_return += float(z["team_return"].sum())
                checked += 1
        assert episodes == row["total_episodes"] == 1280
        assert completed == row["completed_episodes"]
        np.testing.assert_allclose(query_count/episodes,row["mean_team_queries"],atol=1e-10)
        np.testing.assert_allclose(team_return/episodes,row["avg_reward"],atol=1e-8)
    return checked


def final_samples(run):
    visitors, stages, occupancy = [], [], []
    exits = 0
    for p in sorted((run / "logs/evaluation_traces/round_0200").glob("*.npz")):
        with np.load(p,allow_pickle=False) as z:
            visitors.extend(z["sample_agent_goal_visited"][-1].sum(-1).tolist())
            stages.extend(z["sample_task_stage"][-1].tolist())
            g=z["sample_agent_at_goal"]
            occupancy.extend(g[-1].sum(-1).tolist())
            exits += int((g[:-1] & ~g[1:]).sum())
    histogram=lambda x: {str(k):v for k,v in sorted(Counter(x).items())}
    return dict(sample_episodes=len(visitors),ever_goal_visitors=histogram(visitors),
                final_stage=histogram(stages),final_agents_at_goal=histogram(occupancy),
                goal_exit_events=exits)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",type=Path,default=ROOT/"runs/gate_lambda0_2026-11-16/batch")
    parser.add_argument("--baseline-batch",type=Path,default=ROOT/"runs/status_update_2026-11-16/batch")
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    summaries={r["schedule"]:r for r in csv.DictReader((args.batch/"summary.csv").open())}
    all_results, comparison = {}, []
    conditions=[("event","Handcrafted event"),("fixed_38","Fixed K=38"),
                ("fixed_76","Fixed K=76"),("fixed_152","Fixed K=152"),("single","Single query"),
                ("learned_gate_lambda0","Learned gate, lambda=0"),
                ("coverage_gate_lambda0","Coverage gate, lambda=0")]
    for schedule,title in conditions:
        gated=schedule.endswith("lambda0")
        run=(args.batch if gated else args.baseline_batch)/f"pressure_plate_{schedule}_seed42"
        rows=load_rows(run/"logs/metrics.jsonl")
        train=[r for r in rows if not r.get("eval")]
        ev=[r for r in rows if r.get("eval")]
        assert len(train)==200 and len(ev)==21 and ev[-1]["training_round"]==200
        assert (run/"checkpoints/final_checkpoint.npz").is_file()
        peak=max(ev,key=lambda r:r["completion_rate"])
        result=dict(schedule=schedule,title=title,peak_success_percent=peak["completion_rate"],
            peak_round=peak["training_round"] if peak["completion_rate"]>0 else None,
            final_success_percent=ev[-1]["completion_rate"],
            final_queries_per_episode=ev[-1]["mean_team_queries"],final_team_return=ev[-1]["avg_reward"])
        comparison.append(result)
        detail=dict(**result,curve=ev)
        if gated:
            assert all(r["gate_query_cost_weight"]==0 and r["gate_total_query_cost"]==0 for r in train)
            assert all(r["evaluation_coverage_epsilon"]==0 for r in ev)
            assert ev[-1]["training_environment_steps"]==ev[-1]["simulator_training_environment_steps"]==5120000
            detail.update(final_samples=final_samples(run),verified_evaluation_trace_pairs=check_traces(run,ev),
                max_actor_pre_update_replay_mse=max(r["pre_update_log_prob_mse"] for r in train),
                max_gate_pre_update_replay_mse=max(r["gate_pre_update_log_prob_mse"] for r in train),
                successful_training_episodes=sum(round(r["first_terminal_fraction"]*128) for r in train),
                process_wall_seconds=float(summaries[schedule]["process_wall_seconds"]),
                training_team_queries=int(summaries[schedule]["training_team_queries"]),
                final_training_gate_query_rate=train[-1]["gate_query_rate"],
                final_training_gate_entropy=train[-1]["gate_entropy"])
        all_results[schedule]=detail
    (HERE/"results.json").write_text(json.dumps(all_results,indent=2,allow_nan=False)+"\n")
    with (HERE/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=comparison[0]);writer.writeheader();writer.writerows(comparison)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14.5,4.3),layout="constrained")
        colors={"event":"#2456A6","fixed_76":"#1D8A67","learned_gate_lambda0":"#A64B28","coverage_gate_lambda0":"#8655A6"}
        for schedule,color in colors.items():
            d=all_results[schedule];curve=d["curve"];x=[r["simulator_training_environment_steps"]/1e6 for r in curve]
            for ax,key in zip(axes,("completion_rate","avg_reward","mean_team_queries")):
                ax.plot(x,[r[key] for r in curve],label=d["title"],color=color,
                        linestyle="--" if schedule.startswith("coverage") else "-",linewidth=2)
        for ax,ylabel in zip(axes,("Team success (%)","Mean episodic team return","Team queries / episode")):
            ax.set(xlabel="Simulator environment steps (millions)",ylabel=ylabel)
            ax.grid(alpha=.2)
        axes[0].set_ylim(-3,103);axes[2].set_ylim(bottom=0)
        axes[1].legend(frameon=False,fontsize=8,loc="upper left")
        fig.suptitle("Pressure Plate | seed 42 | 1,280 evaluation episodes / checkpoint | gate query penalty = 0")
        for suffix in ("png","pdf"):fig.savefig(HERE/f"comparison.{suffix}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({k:{a:b for a,b in v.items() if a!="curve"} for k,v in all_results.items()},indent=2))


if __name__=="__main__":
    main()
