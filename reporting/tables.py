"""
reporting/tables.py — spec §16's REPORTING LAYER: renders manuscript
tables from already-computed artefacts only (the orchestration ledger's
CSV rows, JSON stats/attribution/etc. outputs, Parquet result files) —
never a checkpoint, never a recomputed metric. Every number in a rendered
table is copied from an artefact, never derived from raw predictions here.

Four blocking rules (spec §16's own wording, verbatim) are enforced as
hard raises before any table is emitted — none of them a soft warning a
caller can silently proceed past.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from stats.tests import bootstrap_ci

REQUIRED_SEEDS = 3  # spec §18: "Seeds | 3 minimum, identical across all models"
_TRUE_STRINGS = {"true", "1", "yes"}


class BlockingRuleError(RuntimeError):
    """Raised when a table would otherwise include an untrustworthy row —
    spec §16's four blocking rules. Every ``render_*`` function below calls
    the relevant check(s) before building its output; none of these are
    recoverable by retrying the same inputs."""


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in _TRUE_STRINGS


def read_runs_ledger(ledger_dir: str) -> List[Dict[str, str]]:
    """Rows of ``<ledger_dir>/runs.csv`` (orchestration.ledger.LedgerWriter's
    Runs table) — ``[]`` if the file doesn't exist yet (a fresh project,
    not an error)."""
    path = os.path.join(ledger_dir, "runs.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_stats_ledger(ledger_dir: str) -> List[Dict[str, str]]:
    """Rows of ``<ledger_dir>/stats.csv`` (the Stats table
    ``stats.run_family_comparison`` — Phase 9 — appends to)."""
    path = os.path.join(ledger_dir, "stats.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Blocking rules
# ---------------------------------------------------------------------------

def check_no_dirty_tree_runs(runs: Sequence[Dict[str, str]]) -> None:
    """Blocking rule 1: a run whose manifest recorded an uncommitted-changes
    (``git.dirty == True``) working tree can't be attributed to a specific,
    citable commit — refuses the table outright rather than footnoting it.
    """
    dirty = [r.get("run_id") for r in runs if _is_true(r.get("git_dirty"))]
    if dirty:
        raise BlockingRuleError(f"dirty-tree run(s) present, refusing to emit table: {sorted(dirty)}")


def check_minimum_seeds(
    runs: Sequence[Dict[str, str]],
    group_keys: Sequence[str] = ("model_name", "dataset_name"),
    required: int = REQUIRED_SEEDS,
) -> None:
    """Blocking rule 2: every group (by default, each model/dataset pair)
    needs at least *required* distinct completed (``status == "done"``)
    seeds — spec §18's seed-count floor.
    """
    groups: Dict[tuple, set] = defaultdict(set)
    for r in runs:
        if r.get("status") != "done":
            continue
        key = tuple(r.get(k, "") for k in group_keys)
        groups[key].add(r.get("seed"))
    under_seeded = {k: len(v) for k, v in groups.items() if len(v) < required}
    if under_seeded:
        raise BlockingRuleError(
            f"config(s) with fewer than {required} completed seeds, refusing to emit table: {under_seeded}"
        )


def check_stats_entries_present(comparisons: Sequence[str], stats_rows: Sequence[Dict[str, str]]) -> None:
    """Blocking rule 3: every comparison a table lists (a "comparison"
    string, e.g. ``"gmkunet_t_vs_unet"`` — matching
    ``stats.run_family_comparison``'s own naming) must have a
    corresponding row in the Stats ledger.
    """
    present = {r.get("comparison") for r in stats_rows}
    missing = [c for c in comparisons if c not in present]
    if missing:
        raise BlockingRuleError(f"comparison(s) with no stats entry, refusing to emit table: {missing}")


def check_saliency_sanitized(saliency_figure_ids: Sequence[str], sanity_results: Dict[str, bool]) -> None:
    """Blocking rule 4: a saliency figure (attribution.segcam output) needs
    a corresponding *passing* sanity-check result
    (attribution.sanity.parameter_randomization_sanity_check's
    ``overall_pass`` — Phase 11) keyed by the same figure id; missing
    entirely is treated the same as failed.
    """
    bad = [fid for fid in saliency_figure_ids if not sanity_results.get(fid, False)]
    if bad:
        raise BlockingRuleError(f"saliency figure(s) without a passing sanity check, refusing to emit: {bad}")


# ---------------------------------------------------------------------------
# Provenance footer (spec §16: "Every table and figure carries a footer
# with the snapshot ID, git commit and generation date.")
# ---------------------------------------------------------------------------

def provenance_footer(snapshot_id: str, git_commit: Optional[str] = None) -> Dict[str, str]:
    if git_commit is None:
        from orchestration.manifest import git_commit as _git_commit
        git_commit = _git_commit() or "unknown"
    return {
        "snapshot_id": snapshot_id,
        "git_commit": git_commit,
        "generated": datetime.now(timezone.utc).isoformat(),
    }


def _footer_line_latex(footer: Dict[str, str]) -> str:
    return f"%% snapshot={footer['snapshot_id']} commit={footer['git_commit']} generated={footer['generated']}"


def _footer_line_csv(footer: Dict[str, str]) -> str:
    return f"# snapshot={footer['snapshot_id']} commit={footer['git_commit']} generated={footer['generated']}"


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

def render_main_comparison_table(
    results: List[Dict[str, Any]],
    ledger_dir: str,
    snapshot_id: str,
    metrics: Sequence[str] = ("dice", "miou", "hd95"),
) -> Dict[str, str]:
    """Spec's Table 2 (Main comparison, S6+S7): per-(model, dataset) mean
    +/- std over seeds for each of *metrics*, from *results* — a
    ``results.parquet``-shaped list of per-run dicts (``model``,
    ``dataset``, ``seed``, one column per metric) — cross-checked against
    the Runs ledger for the dirty-tree and minimum-seeds blocking rules,
    and against the Stats ledger for the missing-stats-entry rule (every
    (model, dataset) pair other than the row with the best mean Dice per
    dataset must have a stats comparison against that best row).

    Returns ``{"latex": str, "csv": str}``.
    """
    if not results:
        raise ValueError("render_main_comparison_table: no results given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    import pandas as pd

    df = pd.DataFrame(results)
    grouped = df.groupby(["model", "dataset"])[list(metrics)].agg(["mean", "std"])

    footer = provenance_footer(snapshot_id)

    csv_lines = [_footer_line_csv(footer), ",".join(["model", "dataset"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")])]
    latex_rows = []
    for (model, dataset), row in grouped.iterrows():
        csv_fields = [model, dataset]
        latex_fields = [model, dataset]
        for m in metrics:
            mean_v, std_v = row[(m, "mean")], row[(m, "std")]
            csv_fields += [f"{mean_v:.4f}", f"{std_v:.4f}" if pd.notna(std_v) else ""]
            latex_fields.append(f"{mean_v:.4f} $\\pm$ {std_v:.4f}" if pd.notna(std_v) else f"{mean_v:.4f}")
        csv_lines.append(",".join(str(f) for f in csv_fields))
        latex_rows.append(" & ".join(str(f) for f in latex_fields) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & Dataset & " + " & ".join(metrics) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}


def render_channel_mode_table(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    profiling: List[Dict[str, Any]],
    ledger_dir: str,
    snapshot_id: str,
) -> Dict[str, str]:
    """F1's channel-mode grid (plan §2 T9) — one row per channel mode:
    mode | effective channels | params(M) | GFLOPs | Dice mean+/-std
    (seeds) | 95% CI | vs m1 (p_corr) | verdict.

    Args:
        results: one dict per (mode, dataset, seed) run, at minimum
            ``{"mode", "dataset", "seed", "dice"}`` — pooled across
            datasets/seeds for the mean/std/CI columns (ANALYSIS_PLAN.md's
            primary endpoint: "pooled across the three training
            datasets").
        stats: ``stats.run_family_comparison()``'s return value for the F1
            ``channel_modes`` family (``proposed`` == "m1", comparators
            keyed by mode name) — supplies each non-m1 mode's
            corrected p-value and verdict.
        profiling: one dict per mode, ``{"mode", "effective_channels",
            "params", "gflops"}``.
    """
    if not results:
        raise ValueError("render_channel_mode_table: no results given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    proposed_name = stats.get("proposed", "m1")
    comparisons = [f"{proposed_name}_vs_{c['comparator']}" for c in stats.get("comparisons", [])]
    check_stats_entries_present(comparisons, read_stats_ledger(ledger_dir))

    by_comparator = {c["comparator"]: c for c in stats.get("comparisons", [])}
    profiling_by_mode = {p["mode"]: p for p in profiling}

    import pandas as pd

    df = pd.DataFrame(results)
    footer = provenance_footer(snapshot_id)

    columns = ["mode", "effective_channels", "params_M", "gflops", "dice_mean", "dice_std",
               "ci_low", "ci_high", "vs_m1_p_corr", "verdict"]
    csv_lines = [_footer_line_csv(footer), ",".join(columns)]
    latex_rows = []
    for mode in sorted(df["mode"].unique()):
        dice_vals = df.loc[df["mode"] == mode, "dice"].to_numpy()
        ci = bootstrap_ci(dice_vals)
        prof = profiling_by_mode.get(mode, {})
        comp = by_comparator.get(mode)
        p_corr = f"{comp['corrected_p_value']:.4f}" if comp else "-"
        verdict = comp.get("verdict", comp.get("meaningfulness_verdict", "-")) if comp else "-"

        row = [
            mode,
            prof.get("effective_channels", ""),
            f"{prof['params'] / 1e6:.2f}" if "params" in prof else "",
            f"{prof['gflops']:.2f}" if "gflops" in prof else "",
            f"{dice_vals.mean():.4f}",
            f"{dice_vals.std():.4f}",
            f"{ci['ci_low']:.4f}",
            f"{ci['ci_high']:.4f}",
            p_corr,
            verdict,
        ]
        csv_lines.append(",".join(str(v) for v in row))
        latex_rows.append(" & ".join(str(v) for v in row) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}


def render_capacity_control_table(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    ledger_dir: str,
    snapshot_id: str,
) -> Dict[str, str]:
    """F2's capacity-control comparison (plan §2 T9) — each non-RGB channel
    mode paired with its width-matched RGB control: mode | matched control |
    mode Dice mean+/-std | control Dice mean+/-std | delta | TOST verdict.

    Args:
        results: one dict per (mode, dataset, seed) run — must include rows
            for both each mode in *stats*'s comparators and their
            ``"<mode>_matched"`` control counterpart.
        stats: ``stats.run_family_comparison()``'s return value for the F2
            ``capacity_controls`` family, built with ``equivalence_bound``
            set (so each comparison carries an ``"equivalence"`` TOST
            result — see stats.tests.tost_equivalence).
    """
    if not results:
        raise ValueError("render_capacity_control_table: no results given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    proposed_name = stats.get("proposed", "")
    comparisons = [f"{proposed_name}_vs_{c['comparator']}" for c in stats.get("comparisons", [])]
    check_stats_entries_present(comparisons, read_stats_ledger(ledger_dir))

    import pandas as pd

    df = pd.DataFrame(results)
    footer = provenance_footer(snapshot_id)

    columns = ["mode", "control", "mode_dice_mean", "control_dice_mean", "delta", "tost_verdict"]
    csv_lines = [_footer_line_csv(footer), ",".join(columns)]
    latex_rows = []
    for comp in stats.get("comparisons", []):
        mode = comp["comparator"]
        control = f"{mode}_matched"
        mode_dice = df.loc[df["mode"] == mode, "dice"].mean()
        control_dice = df.loc[df["mode"] == control, "dice"].mean()
        equivalence = comp.get("equivalence", {})
        row = [
            mode, control,
            f"{mode_dice:.4f}", f"{control_dice:.4f}",
            f"{mode_dice - control_dice:.4f}",
            equivalence.get("verdict", "-"),
        ]
        csv_lines.append(",".join(str(v) for v in row))
        latex_rows.append(" & ".join(str(v) for v in row) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}


def render_order_ablation_table(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    ledger_dir: str,
    snapshot_id: str,
) -> Dict[str, str]:
    """F3's order ablation (plan §2 T9) — m4/m5 channel_build_order=post vs.
    pre, per dataset: mode | dataset | post Dice mean | pre Dice mean |
    delta | p_corr | verdict.

    Args:
        results: one dict per (mode, order, dataset, seed) run, at minimum
            ``{"mode", "order", "dataset", "seed", "dice"}`` — *order* is
            ``"post"``/``"pre"``.
        stats: ``stats.run_family_comparison()``'s return value for the F3
            ``order_ablation`` family, comparators named e.g.
            ``"m4-pre"``/``"m5-pre"`` (matching the plan's own
            ``m4-post vs m4-pre`` naming).
    """
    if not results:
        raise ValueError("render_order_ablation_table: no results given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    proposed_name = stats.get("proposed", "")
    comparisons = [f"{proposed_name}_vs_{c['comparator']}" for c in stats.get("comparisons", [])]
    check_stats_entries_present(comparisons, read_stats_ledger(ledger_dir))

    by_comparator = {c["comparator"]: c for c in stats.get("comparisons", [])}

    import pandas as pd

    df = pd.DataFrame(results)
    footer = provenance_footer(snapshot_id)

    columns = ["mode", "dataset", "post_dice_mean", "pre_dice_mean", "delta", "p_corr", "verdict"]
    csv_lines = [_footer_line_csv(footer), ",".join(columns)]
    latex_rows = []
    for mode in sorted(df["mode"].unique()):
        for dataset in sorted(df.loc[df["mode"] == mode, "dataset"].unique()):
            subset = df[(df["mode"] == mode) & (df["dataset"] == dataset)]
            post_dice = subset.loc[subset["order"] == "post", "dice"].mean()
            pre_dice = subset.loc[subset["order"] == "pre", "dice"].mean()
            comp = by_comparator.get(f"{mode}-pre")
            p_corr = f"{comp['corrected_p_value']:.4f}" if comp else "-"
            verdict = comp.get("verdict", comp.get("meaningfulness_verdict", "-")) if comp else "-"
            row = [mode, dataset, f"{post_dice:.4f}", f"{pre_dice:.4f}",
                   f"{post_dice - pre_dice:.4f}", p_corr, verdict]
            csv_lines.append(",".join(str(v) for v in row))
            latex_rows.append(" & ".join(str(v) for v in row) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}


def render_efficiency_table(profiling_rows: List[Dict[str, Any]], snapshot_id: str) -> Dict[str, str]:
    """Spec's Table 7 (Efficiency, S16) — one row per model, columns taken
    directly from *profiling_rows* (``profiling.flops.check_flops_agreement``
    + ``profiling.latency.measure_latency`` + ``profiling.memory``-shaped
    dicts, one merged dict per model, at minimum carrying ``model``,
    ``params``, ``reported_total`` (FLOPs), and ``median_ms``/
    ``throughput_ips``). No blocking rule applies here (efficiency numbers
    aren't seed-dependent statistical claims) beyond the artefact simply
    needing to exist, which the caller assembling *profiling_rows*
    already guarantees.
    """
    if not profiling_rows:
        raise ValueError("render_efficiency_table: no profiling rows given")

    footer = provenance_footer(snapshot_id)
    columns = ["model", "params", "reported_total", "median_ms", "throughput_ips"]
    csv_lines = [_footer_line_csv(footer), ",".join(columns)]
    latex_rows = []
    for row in profiling_rows:
        values = [row.get(c, "") for c in columns]
        csv_lines.append(",".join(str(v) for v in values))
        latex_rows.append(" & ".join(str(v) for v in values) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}
