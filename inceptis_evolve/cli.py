"""Command line interface for TIS sequence evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .jsonc import load_jsonc
from .optimizer import SequenceOptimizer, create_experiment_dir, seed_everything


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run config-driven TIS sequence evolution")
    parser.add_argument("--config", required=True, type=Path, help="JSON or JSONC optimizer config")
    args = parser.parse_args(argv)
    cfg = load_jsonc(args.config)
    out, cfg = create_experiment_dir(cfg, args.config)
    print(f"Experiment directory: {out}", flush=True)
    seed_everything(int(cfg.get("seed", 0)))
    optimizer = SequenceOptimizer(cfg)
    result = optimizer.run()
    summary = optimizer.write_outputs(result)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
