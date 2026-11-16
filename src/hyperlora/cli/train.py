from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import time
from pathlib import Path


def _parse_override(text: str):
    key, _, raw = text.partition("=")
    if not raw:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", type=_parse_override,
                        default=[], metavar="section.key=value")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="run directory; logs go to <out>/logs and checkpoints to "
                             "<out>/checkpoints (overrides --log-dir / --checkpoint-dir)")
    args = parser.parse_args()

    from ..config import dump_config, load_config

    overrides = dict(args.overrides)
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = load_config(args.config, overrides)

    from ..seeding import seed_before_vmas_import

    seed_before_vmas_import(config.seed)

    from ..training.trainer import train

    log_dir = Path(args.out / "logs" if args.out else (args.log_dir or config.logging.log_dir))
    checkpoint_dir = (args.out / "checkpoints" if args.out else args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, log_dir / "resolved_config.yaml")
    started, cpu_started = time.perf_counter(), time.process_time()
    models = train(config, log_dir=log_dir, checkpoint_dir=checkpoint_dir)
    if config.research_metrics:
        import resource

        import jax

        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "train_and_evaluate_wall_seconds": time.perf_counter() - started,
            "train_and_evaluate_cpu_seconds": time.process_time() - cpu_started,
            "max_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "parameter_counts": {
                name: sum(arr.size for arr in jax.tree.leaves(state.params))
                for name, state in (("actor", models.policy_state), ("generator", models.hn_state),
                                    ("critic", models.critic_state))
            },
        }
        (log_dir / "resources.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {log_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()

