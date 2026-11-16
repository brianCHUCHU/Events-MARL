"""Measure full-shape gate pilots; short runs are throughput checks only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=(1,2))
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    def run(mode):
        path = args.out / mode
        path.mkdir()
        cmd = [sys.executable,"-u","-m","hyperlora.cli.train","--config",
               "configs/main/ours/pressure_plate.yaml","--seed","42","--out",str(path.resolve())]
        for setting in ("research_metrics=True",f"query_gate.mode={mode}",
                        f"ppo.num_episodes={args.rounds}","evaluation.num_eval_episodes=1",
                        "evaluation.eval_interval=2"):
            cmd += ["--set",setting]
        start = time.perf_counter()
        with (path/"process.log").open("w") as log:
            subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,
                           env=os.environ|{"PYTHONPATH":str(ROOT/"src"),"JAX_PLATFORMS":"cpu",
                                           "OMP_NUM_THREADS":"4"})
        metrics=[json.loads(line) for line in (path/"logs/metrics.jsonl").read_text().splitlines()]
        train=[r for r in metrics if not r.get("eval")]
        return dict(mode=mode,wall_seconds=time.perf_counter()-start,
                    training_round_seconds=[r["seconds"] for r in train],
                    training_round_cpu_seconds=[r["training_cpu_seconds"] for r in train],
                    evaluation_seconds=[r["evaluation_wall_seconds"] for r in metrics if r.get("eval")])
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run,("learned","coverage")))
    data=dict(workers=args.workers,rounds=args.rounds,elapsed_seconds=time.perf_counter()-start,runs=results)
    (args.out/"benchmark.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()
