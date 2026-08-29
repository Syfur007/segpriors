"""
reporting/figures.py — spec §16's REPORTING LAYER, figure side: renders
matplotlib figures directly from already-computed artefact shapes (this
module's inputs are exactly the return values of stats.ranking.nemenyi_posthoc,
profiling.flops.check_flops_agreement, robustness.common.degradation_curve,
etc. — no re-derivation of any number here, only plotting of what those
modules already computed). Every figure carries spec's provenance footer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .tables import (
    check_minimum_seeds,
    check_no_dirty_tree_runs,
    provenance_footer,
    read_runs_ledger,
)


def _add_provenance_footer(fig, snapshot_id: str, git_commit: Optional[str] = None) -> None:
    footer = provenance_footer(snapshot_id, git_commit)
    fig.text(
        0.01, 0.01,
        f"snapshot={footer['snapshot_id']}  commit={footer['git_commit'][:12]}  generated={footer['generated']}",
        fontsize=6, color="gray", ha="left", va="bottom",
    )


def render_degradation_curve_figure(curve: List[Dict[str, Any]], title: str, snapshot_id: str):
    """spec's Fig. 6 (Shortcut/translation curves, S11+S14) and the general
    "Degradation curve: metric vs severity" figure every corruption family
    produces — *curve* is exactly ``robustness.common.degradation_curve``'s
    (or ``robustness.geometric.geometric_degradation_curve``'s) return
    value: a list of ``{"severity": int, "mean_dice": float, "n": int}``
    dicts, severity 0 = clean baseline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curve:
        raise ValueError("render_degradation_curve_figure: empty curve")

    severities = [row["severity"] for row in curve]
    dices = [row["mean_dice"] for row in curve]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(severities, dices, marker="o")
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.set_xticks(severities)
    ax.grid(True, alpha=0.3)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def render_pareto_frontier_figure(
    profiling_rows: List[Dict[str, Any]],
    metric_key: str,
    cost_key: str,
    snapshot_id: str,
    higher_metric_is_better: bool = True,
):
    """spec's Fig. 2 (Pareto frontier, S6+S16) — *profiling_rows*: one dict
    per model, at minimum carrying ``"model"``, *metric_key* (e.g.
    ``"dice"``) and *cost_key* (e.g. ``profiling.flops.check_flops_agreement``'s
    ``"reported_total"``, or a latency figure). Points on the Pareto front
    (no other model is both better on *metric_key* and cheaper on
    *cost_key*) are highlighted and connected by a step line; dominated
    points are plotted but not connected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profiling_rows:
        raise ValueError("render_pareto_frontier_figure: no profiling rows given")

    pareto_indices = _pareto_front_indices(profiling_rows, metric_key, cost_key, higher_metric_is_better)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, row in enumerate(profiling_rows):
        on_front = i in pareto_indices
        ax.scatter(
            row[cost_key], row[metric_key],
            color="tab:blue" if on_front else "lightgray",
            zorder=3 if on_front else 2,
        )
        ax.annotate(row.get("model", str(i)), (row[cost_key], row[metric_key]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    front_sorted = sorted((profiling_rows[i] for i in pareto_indices), key=lambda r: r[cost_key])
    if front_sorted:
        ax.plot([r[cost_key] for r in front_sorted], [r[metric_key] for r in front_sorted], color="tab:blue", linestyle="--", zorder=1)

    ax.set_xlabel(cost_key)
    ax.set_ylabel(metric_key)
    ax.set_title("Pareto frontier")
    ax.grid(True, alpha=0.3)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def _pareto_front_indices(rows: List[Dict[str, Any]], metric_key: str, cost_key: str, higher_metric_is_better: bool) -> set:
    """Index set of the non-dominated rows: row i is on the front unless
    some other row j has metric >= i's metric (or <=, if lower-is-better)
    *and* cost <= i's cost, with at least one strict inequality (i.e. j
    Pareto-dominates i).
    """
    front = set()
    n = len(rows)
    for i in range(n):
        mi, ci = rows[i][metric_key], rows[i][cost_key]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            mj, cj = rows[j][metric_key], rows[j][cost_key]
            metric_at_least_as_good = (mj >= mi) if higher_metric_is_better else (mj <= mi)
            cost_at_least_as_good = cj <= ci
            strictly_better = (mj > mi if higher_metric_is_better else mj < mi) or (cj < ci)
            if metric_at_least_as_good and cost_at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def render_shortcut_figure(
    shortcut_json: List[Dict[str, Any]],
    translation_json: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ledger_dir: str,
    snapshot_id: str,
):
    """F4's shortcut audit (plan §2 T9) — two panels:
      A: per-dataset coord-only-trained Dice vs. the constant-mask floor,
         with ANALYSIS_PLAN.md's pre-registered shortcut threshold (0.300)
         drawn as a reference line.
      B: per-dataset degradation curves (Dice vs. translation-shift
         magnitude), m1 vs m4.

    Args:
        shortcut_json: one dict per dataset: {"dataset", "coordonly_dice"
            (robustness.geometric.shortcut_audit's output),
            "constant_floor_dice" (analysis.centre_bias.
            constant_mask_floor's test_dice), "threshold"}.
        translation_json: ``{dataset: {"m1": curve, "m4": curve}}``, each
            curve a ``robustness.common.degradation_curve``-shaped list of
            ``{"severity", "mean_dice", "n"}`` dicts.
    """
    if not shortcut_json:
        raise ValueError("render_shortcut_figure: no shortcut data given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    datasets = [row["dataset"] for row in shortcut_json]
    x = list(range(len(datasets)))
    ax_a.bar([i - 0.2 for i in x], [row["coordonly_dice"] for row in shortcut_json], width=0.4, label="coord-only (m8)")
    ax_a.bar([i + 0.2 for i in x], [row["constant_floor_dice"] for row in shortcut_json], width=0.4, label="constant-mask floor")
    threshold = shortcut_json[0].get("threshold")
    if threshold is not None:
        ax_a.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(datasets, rotation=30, ha="right")
    ax_a.set_ylabel("Dice")
    ax_a.set_title("Coord-only vs. constant-mask floor")
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.3)

    for dataset, curves in translation_json.items():
        for mode, curve in curves.items():
            severities = [row["severity"] for row in curve]
            dices = [row["mean_dice"] for row in curve]
            ax_b.plot(severities, dices, marker="o", label=f"{dataset}/{mode}")
    ax_b.set_xlabel("Translation magnitude (severity)")
    ax_b.set_ylabel("Dice")
    ax_b.set_title("m1 vs m4 under translation shift")
    ax_b.legend(fontsize=6)
    ax_b.grid(True, alpha=0.3)

    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def render_centre_bias_scatter(
    centre_bias_json: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
    ledger_dir: str,
    snapshot_id: str,
):
    """C4/dataset-selection framing — one point per dataset: x =
    analysis.centre_bias.centre_bias_index's constant_floor_dice, y = mean
    Dice gain of m4 over m1 for that dataset (does the channel gain
    correlate with positional predictability).

    Args:
        centre_bias_json: ``{dataset: analysis.centre_bias.
            centre_bias_index()output}``.
        results: one dict per (mode, dataset, seed) run, at minimum
            ``{"mode", "dataset", "seed", "dice"}`` — must include both
            "m1" and "m4" rows for every dataset in *centre_bias_json*.
    """
    if not centre_bias_json:
        raise ValueError("render_centre_bias_scatter: no centre-bias data given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(5, 4))
    for dataset, cb in centre_bias_json.items():
        m1_dice = df.loc[(df["dataset"] == dataset) & (df["mode"] == "m1"), "dice"].mean()
        m4_dice = df.loc[(df["dataset"] == dataset) & (df["mode"] == "m4"), "dice"].mean()
        gain = m4_dice - m1_dice
        ax.scatter(cb["constant_floor_dice"], gain, color="tab:blue")
        ax.annotate(dataset, (cb["constant_floor_dice"], gain), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Centre-bias index (constant-mask floor Dice)")
    ax.set_ylabel("m4 - m1 Dice gain")
    ax.set_title("Positional predictability vs. channel gain")
    ax.grid(True, alpha=0.3)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def render_occlusion_figure(
    occlusion_json: Dict[str, Dict[str, float]],
    shapley_json: Dict[str, Dict[str, float]],
    ledger_dir: str,
    snapshot_id: str,
):
    """Grouped bars: per-channel-group Dice drop (occlusion) and Shapley
    mass, per dataset.

    Args:
        occlusion_json: ``{dataset: {group_name: dice_drop}}`` —
            attribution.occlusion.run_channel_group_occlusion-shaped.
        shapley_json: ``{dataset: {group_name: shapley_value}}`` —
            attribution.shapley-shaped, same group keys per dataset.
    """
    if not occlusion_json:
        raise ValueError("render_occlusion_figure: no occlusion data given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = sorted(occlusion_json)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[0][i]
        groups = sorted(occlusion_json[dataset])
        occ_vals = [occlusion_json[dataset][g] for g in groups]
        shap_vals = [shapley_json.get(dataset, {}).get(g, 0.0) for g in groups]
        x = np.arange(len(groups))
        ax.bar(x - 0.2, occ_vals, width=0.4, label="occlusion Dice drop")
        ax.bar(x + 0.2, shap_vals, width=0.4, label="Shapley mass")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_title(dataset)
        if i == 0:
            ax.legend(fontsize=7)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def render_critical_difference_figure(nemenyi_result: Dict[str, Any], snapshot_id: str):
    """spec's "Critical-difference diagram" (S7) — *nemenyi_result* is
    exactly ``stats.ranking.nemenyi_posthoc``'s return value: average rank
    per method plotted on a single axis, with the critical-difference span
    drawn as a reference bar, and a horizontal bracket over any pair whose
    ranks are *not* significantly different (rank_diff <= critical_difference).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = nemenyi_result["methods"]
    avg_ranks = nemenyi_result["avg_ranks"]
    cd = nemenyi_result["critical_difference"]

    order = sorted(methods, key=lambda m: avg_ranks[m])
    ranks_sorted = [avg_ranks[m] for m in order]

    fig, ax = plt.subplots(figsize=(6, 2 + 0.3 * len(methods)))
    y_positions = list(range(len(order)))
    ax.scatter(ranks_sorted, y_positions, color="tab:blue", zorder=3)
    for y, m, r in zip(y_positions, order, ranks_sorted):
        ax.text(r, y, f"  {m} ({r:.2f})", va="center", fontsize=8)

    # CD reference bar, anchored at the best (lowest-rank) method.
    best_rank = ranks_sorted[0]
    ax.plot([best_rank, best_rank + cd], [-1, -1], color="black", linewidth=2, clip_on=False)
    ax.text(best_rank + cd / 2, -1.3, f"CD = {cd:.2f}", ha="center", fontsize=8, clip_on=False)

    # Brackets over statistically-indistinguishable pairs (adjacent in rank).
    bracket_y = len(order)
    for pair in nemenyi_result["pairwise"]:
        if not pair["significant"]:
            ia, ib = order.index(pair["method_a"]), order.index(pair["method_b"])
            lo, hi = sorted((ia, ib))
            ax.plot([ranks_sorted[lo], ranks_sorted[hi]], [bracket_y, bracket_y], color="tab:red", linewidth=1.5)
            bracket_y += 0.5

    ax.set_yticks([])
    ax.set_xlabel("Average rank")
    ax.set_title("Critical-difference diagram")
    ax.set_ylim(-2, bracket_y + 0.5)
    ax.invert_yaxis()
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig
