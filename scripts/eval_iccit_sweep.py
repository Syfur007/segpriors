#!/usr/bin/env python3
"""scripts/eval_iccit_sweep.py — companion to scripts/run_iccit_sweep.py.

Runs eval.py once per pre-registered seed against the checkpoint that
run_iccit_sweep.py produced for that seed (logging.experiment_name suffixed
_s<seed>). Each invocation self-mints its own test-eval token via
--allow-test-eval, exactly like a manual eval.py call — this study reads
each dataset's own held-out test split repeatedly across configs, which is
what that flag is for; it is the guarded *external* ColonDB cohort (A9 in
the master plan, run separately and only once) that must not be re-spent
this way.

Kept as a subprocess loop over the existing eval.py CLI rather than a
reusable Python import: eval.py's main() has no return value or importable
entry point separate from its own argparse/CLI body.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # see run_iccit_sweep.py's comment on this

from utils.config import load_config

DEFAULT_SEEDS = [1337, 2024, 7]
EVAL_PY = REPO_ROOT / "eval.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ICCIT config's checkpoints across the pre-registered seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    base_exp_name = load_config(args.config)["logging"]["experiment_name"]

    exit_code = 0
    for seed in args.seeds:
        experiment_name = f"{base_exp_name}_s{seed}"
        cmd = [
            sys.executable, str(EVAL_PY),
            "--config", args.config,
            "--experiment-name", experiment_name,
            "--allow-test-eval",
        ]
        if args.no_vis:
            cmd.append("--no-vis")

        print(f"=== eval {experiment_name} ===", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
