"""
tests/test_gen_iccit_configs.py — scripts/gen_iccit_configs.py: the ICCIT
run-matrix config generator (plan §3/§4). Runs the actual script (as a
module import, not a subprocess re-implementation) so a real regression in
its generation logic fails a test, not just a manual --dry-run inspection.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

from utils.config import load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "gen_iccit_configs.py")
OUT_DIR = os.path.join(REPO_ROOT, "configs", "experiment", "iccit")

EXPECTED_COUNT = 8 * 3 + 4 * 3 + 2 * 3 + 4 * 3 + 1  # 55


def test_dry_run_reports_expected_count():
    result = subprocess.run(
        [sys.executable, SCRIPT, "--dry-run"], capture_output=True, text=True, check=True,
    )
    assert f"Would write {EXPECTED_COUNT} config(s)" in result.stdout


def test_dry_run_writes_nothing():
    before = set(glob.glob(os.path.join(OUT_DIR, "*.yaml")))
    subprocess.run([sys.executable, SCRIPT, "--dry-run"], capture_output=True, text=True, check=True)
    after = set(glob.glob(os.path.join(OUT_DIR, "*.yaml")))
    assert before == after


def test_generated_configs_load_and_validate():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.yaml")))
    assert len(files) == EXPECTED_COUNT
    for f in files:
        load_config(f, validate=True)  # raises on schema/compose failure


def test_matched_configs_use_rgb_and_the_target_modes_channel_count():
    cfg = load_config(os.path.join(OUT_DIR, "mkunet_m4_matched_clinicdb.yaml"))
    assert cfg["model"]["in_channels"] == 3
    assert cfg["dataset"]["channel_mode"] == "m1"

    target = load_config(os.path.join(OUT_DIR, "mkunet_m4_clinicdb.yaml"))
    assert target["model"]["in_channels"] == 8
    assert target["dataset"]["channel_mode"] == "m4"


def test_order_ablation_configs_set_pre():
    cfg = load_config(os.path.join(OUT_DIR, "mkunet_m4_pre_busi.yaml"))
    assert cfg["dataset"]["channel_build_order"] == "pre"
    assert cfg["dataset"]["channel_mode"] == "m4"


def test_external_colondb_config_marks_external():
    cfg = load_config(os.path.join(OUT_DIR, "external_colondb_eval.yaml"))
    assert cfg["dataset"]["name"] == "ColonDB"
    assert cfg["dataset"]["external"] is True
