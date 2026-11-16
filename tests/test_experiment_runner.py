import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("query_runner", ROOT / "scripts/run_query_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentRunnerTest(unittest.TestCase):
    def test_gate_selection_runs_only_two_requested_pressure_plate_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "src").mkdir()
            (project / "configs/main/ours").mkdir(parents=True)
            seen = []
            def execute(entry, *args):
                seen.append(entry)
                return {"scenario":entry[0],"schedule":entry[1],"seed":entry[2]},None
            argv=["runner","--out",str(project/"batch"),"--scenario","pressure_plate",
                  "--seeds","42","--schedules","learned_gate","coverage_gate","--workers","2"]
            with patch.object(runner,"ROOT",project),patch.object(runner,"execute_run",execute), \
                 patch.object(runner,"write_seed_summary"), \
                 patch.object(runner.subprocess,"run",return_value=SimpleNamespace(returncode=0)), \
                 patch("sys.argv",argv):
                self.assertEqual(runner.main(),0)
            self.assertEqual({entry[:3] for entry in seen},{("pressure_plate","learned_gate",42),
                                                          ("pressure_plate","coverage_gate",42)})
            self.assertTrue(all("evaluation.eval_interval=10" in entry[3] for entry in seen))

    def test_peak_records_the_best_evaluation_not_the_final_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "logs").mkdir()
            (path / "checkpoints").mkdir()
            for name in ("final_checkpoint.npz", "evaluated_round_0001.npz"):
                (path / "checkpoints" / name).touch()
            train = [{"training_round": r, "seconds": 2., "training_environment_steps": r * 100,
                      "team_initial_queries": 1, "team_requeries": 0,
                      "generator_rollout_batches": 1, "generator_ppo_batches": 3}
                     for r in (1, 2)]
            evaluation = [{"eval": True, "training_round": r, "training_environment_steps": r * 100,
                           "simulator_training_environment_steps": r * 200,
                           "evaluation_wall_seconds": 1., "mean_team_queries": 1.,
                           "completion_rate": score, "avg_reward": score / 100}
                          for r, score in [(0, 0.), (1, 80.), (2, 20.)]]
            rows = [evaluation[0], train[0], evaluation[1], train[1], evaluation[2]]
            (path / "logs/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (path / "logs/resources.json").write_text(json.dumps({
                "train_and_evaluate_cpu_seconds": 10., "max_rss_bytes": 1024}))
            result = runner.compute_run_summary(path, "pressure_plate", "event", 42, 6.)
            self.assertEqual(result["peak_completion_rate"], 80.)
            self.assertEqual(result["final_completion_rate"], 20.)
            self.assertEqual(result["peak_training_round"], 1)
            self.assertEqual(result["peak_training_environment_steps"], 100)
            self.assertEqual(result["peak_simulator_training_environment_steps"], 200)
            self.assertEqual(result["cumulative_train_eval_seconds_at_peak"], 4.)
            self.assertTrue(result["peak_checkpoint"].endswith("evaluated_round_0001.npz"))

    def test_two_workers_run_exactly_the_requested_environments_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "src").mkdir()
            (project / "configs/main/ours").mkdir(parents=True)
            output = project / "batch"
            lock, barrier = threading.Lock(), threading.Barrier(2)
            active = maximum = 0
            seen = []
            active_scenarios = set()

            def execute(entry, *args):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    seen.append(entry)
                    self.assertNotIn(entry[0], active_scenarios)
                    active_scenarios.add(entry[0])
                barrier.wait(timeout=5)
                with lock:
                    active -= 1
                    active_scenarios.remove(entry[0])
                return {"scenario": entry[0], "schedule": entry[1], "seed": entry[2]}, None

            argv = ["runner", "--out", str(output), "--scenario", "pressure_plate_navigation",
                    "--seeds", "42", "--workers", "2"]
            with patch.object(runner, "ROOT", project), patch.object(runner, "execute_run", execute), \
                    patch.object(runner, "write_seed_summary"), \
                    patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                    patch("sys.argv", argv):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(maximum, 2)
            self.assertEqual(len(seen), 10)
            self.assertEqual({entry[0] for entry in seen}, {"pressure_plate", "navigation"})
            self.assertEqual({entry[2] for entry in seen}, {42})
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["completed"], 10)
            self.assertEqual(status["failed"], [])
            self.assertEqual(status["running"], [])
