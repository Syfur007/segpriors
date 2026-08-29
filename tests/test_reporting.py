"""
tests/test_reporting.py — Phase 14: reporting layer
(reporting/tables.py, figures.py, inventory.py).

The test the plan names for this phase: test_reporting_blocks.
"""
from __future__ import annotations

import os

import pytest

from orchestration.ledger import LedgerWriter
from reporting.figures import (
    _pareto_front_indices,
    render_centre_bias_scatter,
    render_critical_difference_figure,
    render_degradation_curve_figure,
    render_occlusion_figure,
    render_pareto_frontier_figure,
    render_shortcut_figure,
)
from reporting.inventory import ARTEFACT_INVENTORY, ArtefactEntry, audit_artefact_inventory
from reporting.tables import (
    BlockingRuleError,
    check_minimum_seeds,
    check_no_dirty_tree_runs,
    check_saliency_sanitized,
    check_stats_entries_present,
    read_runs_ledger,
    read_stats_ledger,
    render_capacity_control_table,
    render_channel_mode_table,
    render_main_comparison_table,
    render_order_ablation_table,
)


def _seed_clean_ledger(ledger_dir, models=("gmk_unet", "unet"), seeds=(1, 2, 3), dirty=False):
    ledger = LedgerWriter(ledger_dir)
    for model in models:
        for seed in seeds:
            ledger.append_run_row(
                run_id=f"{model}-s{seed}", config_hash="abc", experiment_name="exp",
                model_name=model, dataset_name="clinicdb", seed=seed, fold="",
                status="done", start_time="t0", end_time="t1", gpu_hours=0.1,
                best_metric=0.8, monitor_metric="val_dice", git_commit="deadbeef",
                git_dirty=dirty, manifest_path="m.json",
            )
    return ledger_dir


def _results(models=("gmk_unet", "unet"), seeds=(1, 2, 3)):
    out = []
    for model, base in zip(models, [0.78, 0.70]):
        for seed in seeds:
            out.append({"model": model, "dataset": "clinicdb", "seed": seed, "dice": base + seed * 0.001, "miou": base - 0.1, "hd95": 30.0})
    return out


# ---------------------------------------------------------------------------
# test_reporting_blocks (spec's named test) — every blocking rule genuinely
# refuses to emit a table for the corresponding bad input.
# ---------------------------------------------------------------------------

def test_reporting_blocks(tmp_path):
    ledger_dir = str(tmp_path / "ledger")

    # 1. dirty-tree run
    _seed_clean_ledger(ledger_dir, dirty=True)
    runs = read_runs_ledger(ledger_dir)
    with pytest.raises(BlockingRuleError):
        check_no_dirty_tree_runs(runs)

    # 2. under-seeded config
    ledger_dir2 = str(tmp_path / "ledger2")
    _seed_clean_ledger(ledger_dir2, seeds=(1, 2), dirty=False)
    runs2 = read_runs_ledger(ledger_dir2)
    with pytest.raises(BlockingRuleError):
        check_minimum_seeds(runs2)

    # 3. missing stats entry
    with pytest.raises(BlockingRuleError):
        check_stats_entries_present(["a_vs_b", "a_vs_c"], [{"comparison": "a_vs_b"}])

    # 4. unsanitised saliency
    with pytest.raises(BlockingRuleError):
        check_saliency_sanitized(["fig1", "fig2"], {"fig1": True})

    # render_main_comparison_table must propagate a block, not silently render
    with pytest.raises(BlockingRuleError):
        render_main_comparison_table(_results(), ledger_dir, snapshot_id="snap")


def test_reporting_does_not_block_clean_inputs(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, dirty=False)
    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)  # must not raise
    check_minimum_seeds(runs)  # must not raise
    check_stats_entries_present(["a_vs_b"], [{"comparison": "a_vs_b"}])  # must not raise
    check_saliency_sanitized(["fig1"], {"fig1": True})  # must not raise

    out = render_main_comparison_table(_results(), ledger_dir, snapshot_id="snap")
    assert "gmk_unet" in out["csv"]
    assert "\\begin{tabular}" in out["latex"]


def test_check_minimum_seeds_only_counts_done_runs(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    ledger = LedgerWriter(ledger_dir)
    for seed, status in [(1, "done"), (2, "done"), (3, "failed")]:
        ledger.append_run_row(
            run_id=f"r{seed}", config_hash="h", experiment_name="e", model_name="m", dataset_name="d",
            seed=seed, fold="", status=status, start_time="t0", end_time="t1", gpu_hours=0.1,
            best_metric=0.8, monitor_metric="val_dice", git_commit="c", git_dirty=False, manifest_path="m.json",
        )
    runs = read_runs_ledger(ledger_dir)
    with pytest.raises(BlockingRuleError):
        check_minimum_seeds(runs)  # only 2 "done" seeds


def test_read_runs_ledger_missing_file_returns_empty(tmp_path):
    assert read_runs_ledger(str(tmp_path / "nonexistent")) == []


def test_read_stats_ledger_missing_file_returns_empty(tmp_path):
    assert read_stats_ledger(str(tmp_path / "nonexistent")) == []


def test_render_main_comparison_table_rejects_empty_results(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir)
    with pytest.raises(ValueError):
        render_main_comparison_table([], ledger_dir, snapshot_id="snap")


# ---------------------------------------------------------------------------
# T9: channel-mode / capacity-control / order-ablation tables
# ---------------------------------------------------------------------------

def _mode_dice(base, n=12, seed=0):
    import numpy as np
    return (base + np.random.default_rng(seed).normal(0, 0.01, n)).tolist()


def _channel_mode_stats(ledger_dir=None):
    from stats import run_family_comparison
    proposed = _mode_dice(0.80, seed=1)
    return run_family_comparison(
        family="channel_modes", proposed_name="m1",
        proposed_per_image=proposed,
        comparators={"m2": _mode_dice(0.81, seed=2), "m3": _mode_dice(0.79, seed=3)},
        out_dir=None, ledger_dir=ledger_dir,
    )


def _channel_mode_results():
    out = []
    for mode, base in [("m1", 0.80), ("m2", 0.81), ("m3", 0.79)]:
        for seed in (1, 2, 3):
            out.append({"mode": mode, "dataset": "clinicdb", "seed": seed, "dice": base + seed * 0.001})
    return out


def _channel_mode_profiling():
    return [
        {"mode": "m1", "effective_channels": 3, "params": 27384, "gflops": 0.1},
        {"mode": "m2", "effective_channels": 5, "params": 27612, "gflops": 0.11},
        {"mode": "m3", "effective_channels": 6, "params": 27732, "gflops": 0.12},
    ]


def test_render_channel_mode_table_blocks_on_under_seeded_ledger(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2))  # only 2 seeds
    stats = _channel_mode_stats()
    with pytest.raises(BlockingRuleError):
        render_channel_mode_table(_channel_mode_results(), stats, _channel_mode_profiling(), ledger_dir, "snap")


def test_render_channel_mode_table_blocks_on_missing_stats_entry(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    stats = _channel_mode_stats(ledger_dir=None)  # never written to the ledger
    with pytest.raises(BlockingRuleError):
        render_channel_mode_table(_channel_mode_results(), stats, _channel_mode_profiling(), ledger_dir, "snap")


def test_render_channel_mode_table_renders_on_clean_input(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    stats = _channel_mode_stats(ledger_dir=ledger_dir)
    out = render_channel_mode_table(_channel_mode_results(), stats, _channel_mode_profiling(), ledger_dir, "snap")
    assert "m2" in out["csv"]
    assert "\\begin{tabular}" in out["latex"]


def test_render_channel_mode_table_rejects_empty_results(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    with pytest.raises(ValueError):
        render_channel_mode_table([], _channel_mode_stats(), [], ledger_dir, "snap")


def test_render_capacity_control_table_blocks_on_under_seeded_ledger(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2))
    stats = _channel_mode_stats()
    results = _channel_mode_results() + [
        {"mode": "m2_matched", "dataset": "clinicdb", "seed": s, "dice": 0.80} for s in (1, 2, 3)
    ]
    with pytest.raises(BlockingRuleError):
        render_capacity_control_table(results, stats, ledger_dir, "snap")


def test_render_capacity_control_table_renders_on_clean_input(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    stats = _channel_mode_stats(ledger_dir=ledger_dir)
    results = _channel_mode_results() + [
        {"mode": f"{m}_matched", "dataset": "clinicdb", "seed": s, "dice": 0.80}
        for m in ("m2", "m3") for s in (1, 2, 3)
    ]
    out = render_capacity_control_table(results, stats, ledger_dir, "snap")
    assert "m2_matched" in out["csv"]


def test_render_order_ablation_table_blocks_on_missing_stats_entry(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    from stats import run_family_comparison
    stats = run_family_comparison(
        family="order_ablation", proposed_name="m4",
        proposed_per_image=_mode_dice(0.80, seed=4),
        comparators={"m4-pre": _mode_dice(0.75, seed=5)},
        out_dir=None, ledger_dir=None,
    )
    results = [
        {"mode": "m4", "order": order, "dataset": "clinicdb", "seed": s, "dice": 0.80}
        for order in ("post", "pre") for s in (1, 2, 3)
    ]
    with pytest.raises(BlockingRuleError):
        render_order_ablation_table(results, stats, ledger_dir, "snap")


def test_render_order_ablation_table_renders_on_clean_input(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    from stats import run_family_comparison
    stats = run_family_comparison(
        family="order_ablation", proposed_name="m4",
        proposed_per_image=_mode_dice(0.80, seed=4),
        comparators={"m4-pre": _mode_dice(0.75, seed=5)},
        out_dir=None, ledger_dir=ledger_dir,
    )
    results = [
        {"mode": "m4", "order": order, "dataset": "clinicdb", "seed": s, "dice": 0.80 if order == "post" else 0.75}
        for order in ("post", "pre") for s in (1, 2, 3)
    ]
    out = render_order_ablation_table(results, stats, ledger_dir, "snap")
    assert "clinicdb" in out["csv"]


# ---------------------------------------------------------------------------
# T9: shortcut / centre-bias / occlusion figures
# ---------------------------------------------------------------------------

def test_render_shortcut_figure_blocks_on_under_seeded_ledger(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2))
    shortcut_json = [{"dataset": "clinicdb", "coordonly_dice": 0.2, "constant_floor_dice": 0.25, "threshold": 0.3}]
    translation_json = {"clinicdb": {"m1": [{"severity": 0, "mean_dice": 0.8, "n": 5}],
                                      "m4": [{"severity": 0, "mean_dice": 0.82, "n": 5}]}}
    with pytest.raises(BlockingRuleError):
        render_shortcut_figure(shortcut_json, translation_json, ledger_dir, "snap")


def test_render_shortcut_figure_smoke(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    shortcut_json = [{"dataset": "clinicdb", "coordonly_dice": 0.2, "constant_floor_dice": 0.25, "threshold": 0.3}]
    translation_json = {"clinicdb": {"m1": [{"severity": 0, "mean_dice": 0.8, "n": 5},
                                             {"severity": 1, "mean_dice": 0.75, "n": 5}],
                                      "m4": [{"severity": 0, "mean_dice": 0.82, "n": 5},
                                             {"severity": 1, "mean_dice": 0.70, "n": 5}]}}
    fig = render_shortcut_figure(shortcut_json, translation_json, ledger_dir, "snap")
    assert len(fig.axes) == 2


def test_render_shortcut_figure_rejects_empty():
    with pytest.raises(ValueError):
        render_shortcut_figure([], {}, ledger_dir="/nonexistent", snapshot_id="snap")


def test_render_centre_bias_scatter_blocks_on_under_seeded_ledger(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2))
    centre_bias_json = {"clinicdb": {"constant_floor_dice": 0.4}}
    results = [{"mode": mode, "dataset": "clinicdb", "seed": 1, "dice": 0.8} for mode in ("m1", "m4")]
    with pytest.raises(BlockingRuleError):
        render_centre_bias_scatter(centre_bias_json, results, ledger_dir, "snap")


def test_render_centre_bias_scatter_smoke(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    centre_bias_json = {"clinicdb": {"constant_floor_dice": 0.4}, "isic18": {"constant_floor_dice": 0.2}}
    results = [
        {"mode": mode, "dataset": ds, "seed": 1, "dice": 0.8}
        for mode in ("m1", "m4") for ds in ("clinicdb", "isic18")
    ]
    fig = render_centre_bias_scatter(centre_bias_json, results, ledger_dir, "snap")
    assert len(fig.axes) == 1


def test_render_occlusion_figure_blocks_on_under_seeded_ledger(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2))
    occlusion_json = {"clinicdb": {"rgb": 0.1, "xy": 0.02}}
    shapley_json = {"clinicdb": {"rgb": 0.08, "xy": 0.01}}
    with pytest.raises(BlockingRuleError):
        render_occlusion_figure(occlusion_json, shapley_json, ledger_dir, "snap")


def test_render_occlusion_figure_smoke(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, models=("m",), seeds=(1, 2, 3))
    occlusion_json = {"clinicdb": {"rgb": 0.1, "xy": 0.02}}
    shapley_json = {"clinicdb": {"rgb": 0.08, "xy": 0.01}}
    fig = render_occlusion_figure(occlusion_json, shapley_json, ledger_dir, "snap")
    assert len(fig.axes) == 1


def test_render_occlusion_figure_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        render_occlusion_figure({}, {}, ledger_dir=str(tmp_path), snapshot_id="snap")


# ---------------------------------------------------------------------------
# figures.py
# ---------------------------------------------------------------------------

def test_pareto_front_indices_hand_computed():
    rows = [
        {"model": "A", "dice": 0.9, "cost": 100},
        {"model": "B", "dice": 0.85, "cost": 50},
        {"model": "C", "dice": 0.7, "cost": 200},
    ]
    front = _pareto_front_indices(rows, "dice", "cost", higher_metric_is_better=True)
    assert front == {0, 1}  # C is dominated by A on both axes


def test_render_pareto_frontier_figure_smoke():
    rows = [
        {"model": "A", "dice": 0.9, "cost": 100},
        {"model": "B", "dice": 0.85, "cost": 50},
    ]
    fig = render_pareto_frontier_figure(rows, "dice", "cost", snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_pareto_frontier_figure_rejects_empty():
    with pytest.raises(ValueError):
        render_pareto_frontier_figure([], "dice", "cost", snapshot_id="snap")


def test_render_critical_difference_figure_uses_real_nemenyi_output():
    from stats.ranking import nemenyi_posthoc

    scores = {"a": [0.9, 0.8, 0.85], "b": [0.7, 0.6, 0.65], "c": [0.8, 0.75, 0.78]}
    nem = nemenyi_posthoc(scores)
    fig = render_critical_difference_figure(nem, snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_degradation_curve_figure_smoke():
    curve = [{"severity": 0, "mean_dice": 1.0, "n": 5}, {"severity": 1, "mean_dice": 0.9, "n": 5}]
    fig = render_degradation_curve_figure(curve, title="test", snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_degradation_curve_figure_rejects_empty():
    with pytest.raises(ValueError):
        render_degradation_curve_figure([], title="test", snapshot_id="snap")


# ---------------------------------------------------------------------------
# inventory.py
# ---------------------------------------------------------------------------

def test_inventory_covers_conference_artefacts():
    # Every T9 renderer this branch actually ships has a corresponding
    # ARTEFACT_INVENTORY row — the same "codify the mapping so a rename
    # can't silently drift out of sync" guarantee inventory.py's own
    # docstring describes.
    items = {e.manuscript_item for e in ARTEFACT_INVENTORY}
    assert len(ARTEFACT_INVENTORY) == 6
    assert any("channel-mode" in i for i in items)
    assert any("capacity control" in i for i in items)
    assert any("order ablation" in i for i in items)
    assert any("shortcut" in i for i in items)
    assert any("centre-bias" in i for i in items)
    assert any("occlusion" in i for i in items)
    # produced_by/source_artefact are non-empty, real (non-manual, non-glob)
    # entries — audit_artefact_inventory can actually check them.
    for entry in ARTEFACT_INVENTORY:
        assert entry.produced_by != "manual"
        assert entry.source_artefact and "*" not in entry.source_artefact


def test_audit_artefact_inventory_reports_missing_when_nothing_exists(tmp_path, monkeypatch):
    synthetic = [
        ArtefactEntry("Fig. 1 Architecture", "manual", ""),
        ArtefactEntry("Table 1 Example", "S1", "tables/example.csv"),
    ]
    monkeypatch.setattr("reporting.inventory.ARTEFACT_INVENTORY", synthetic)
    result = audit_artefact_inventory(str(tmp_path))
    assert result["total"] == len(synthetic)
    assert "Fig. 1 Architecture" in result["skipped"]  # manual, always skipped
    assert len(result["missing"]) > 0
    assert result["present"] == []


def test_audit_artefact_inventory_detects_present_artefact(tmp_path, monkeypatch):
    synthetic = [ArtefactEntry("Table 1 Example", "S1", "tables/example.csv")]
    monkeypatch.setattr("reporting.inventory.ARTEFACT_INVENTORY", synthetic)
    reports_root = tmp_path / "reports"
    tables_dir = reports_root / "tables"
    os.makedirs(tables_dir)
    (tables_dir / "example.csv").write_text("a,b\n1,2\n")
    result = audit_artefact_inventory(str(reports_root))
    assert "Table 1 Example" in result["present"]
