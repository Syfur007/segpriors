#!/usr/bin/env python3
"""scripts/run_iccit_sweep.py — CLI wrapper so a one-script-per-item runner
(XDash's dashboard: always `{python} {train_script} --config PATH`, no
--seed flag) can still launch a full run through
orchestration.runner.run_sweep instead of a bare train.py invocation.

Why this exists: train.py's own CLI never calls run_sweep, so a plain
`python train.py --config X` writes no manifest.json and no ledger row —
ICCIT2026_MASTER_PLAN.md section 6 requires every run to go through
run_sweep for exactly that bookkeeping. train.py also has no --seed flag,
and ANALYSIS_PLAN.md fixes 3 seeds identical for every config cell. This
script loops those seeds itself, one call to run_sweep(seeds=[seed]) per
seed — the same per-seed loop scripts/reproduce.sh already uses, generalised
into a standalone entry point.

Each seed gets its own logging.experiment_name (suffixed _s<seed>): a bare
run_sweep(config, seeds=[...]) called once with all seeds would still have
every seed share one checkpoint dir (checkpoints/<experiment_name>/ is keyed
by experiment_name only, not run_id/seed — see train.py's checkpoint_dir),
so seed 2 would silently overwrite seed 1's best.pth before seed 3 even
starts. Looping seeds here, one run_sweep([seed]) call per iteration with a
freshly suffixed experiment_name, is what keeps them on disk simultaneously.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

# Unlike train.py (repo root), this script lives under scripts/, so Python
# only puts scripts/ itself on sys.path[0] — orchestration/utils wouldn't
# be importable without this, the same fix gen_iccit_configs.py already
# applies for the same reason.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.runner import run_sweep
from utils.config import load_config

DEFAULT_SEEDS = [1337, 2024, 7]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ICCIT config across the pre-registered seeds via run_sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs (pre-flight smoke runs)")
    parser.add_argument("--force", action="store_true", help="Re-run even if a seed's manifest already says done")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--ledger-dir", default="artifacts/ledger")
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_exp_name = base_config["logging"]["experiment_name"]

    exit_code = 0
    for seed in args.seeds:
        config = copy.deepcopy(base_config)
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        config["logging"]["experiment_name"] = f"{base_exp_name}_s{seed}"

        print(f"=== {config['logging']['experiment_name']} (seed={seed}) ===", flush=True)
        results = run_sweep(
            config,
            seeds=[seed],
            artifacts_dir=args.artifacts_dir,
            ledger_dir=args.ledger_dir,
            force=args.force,
        )
        for r in results:
            print(f"  run_id={r['run_id']} status={r['status']} best_metric={r['best_metric']} error={r['error']}", flush=True)
            if r["status"] not in ("done", "skipped-done"):
                exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
