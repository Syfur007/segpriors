"""
tools/build_paper_package.py — builds the ICCIT2026 paper package per
ICCIT2026_PAPER_PACKAGE_PLAN.md.

TRIAL MODE (the only mode implemented so far): the plan's freeze gate
(G1-G10) requires every admissible run in Blocks A/B/C to exist before the
build may run at all, with no override. The matrix is not finished yet, so
this script does not implement that hard gate — it runs Stages 0-3 plus a
best-effort Stage 5 (T1 table only, Block A / channel-mode family) against
whatever admissible data exists right now, and prints a gate report showing
which G-checks would currently fail. Nothing here should be read as "the
package"; it is a dry run of the pipeline's plumbing (inventory -> per-image
consolidation -> aggregates -> stats -> table) so bugs surface early, on
partial data, rather than for the first time on the real freeze build.

Stages not attempted in trial mode: 4 (inference analyses — needs
checkpoints, most of which are not pulled locally), 5-figures, 6 (numbers
ledger), 7 (verify/scan/archive). Stage 2's efficiency.csv is also skipped
(needs the profiling module run against live models).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from stats import bootstrap_ci, run_family_comparison
from orchestration.ledger import LedgerWriter
from reporting.tables import render_channel_mode_table, render_capacity_control_table, render_order_ablation_table

RESULTS = os.path.join(REPO, "results", "combined")
RUNS_DIR = os.path.join(RESULTS, "artifacts", "runs")
LOGS_DIR = os.path.join(RESULTS, "logs")
CONFIGS_DIR = os.path.join(REPO, "configs", "experiment", "iccit")

MMD = 0.010
TOST_BOUND = 0.010
SEEDS = {1337, 2024, 7}
DATASETS = ("clinicdb", "isic18", "busi")

_CFG_RE = re.compile(r"^(mkunet|unet)_(.+)_(clinicdb|isic18|busi)$")


def parse_config(stem: str) -> Optional[Tuple[str, str, str, str]]:
    """(family, mode, dataset, block) or None if stem doesn't match the
    family_mode_dataset convention (e.g. the external ColonDB config)."""
    m = _CFG_RE.match(stem)
    if not m:
        return None
    family, mode, dataset = m.group(1), m.group(2), m.group(3)
    if family == "unet":
        block = "D"
    elif "pre" in mode.split("_"):
        block = "C"
    elif "matched" in mode:
        block = "B"
    else:
        block = "A"
    return family, mode, dataset, block


# ---------------------------------------------------------------------------
# Stage 0 — inventory
# ---------------------------------------------------------------------------

def stage0_inventory() -> pd.DataFrame:
    all_configs = sorted(
        f[:-5] for f in os.listdir(CONFIGS_DIR)
        if f.endswith(".yaml") and "external" not in f
    )

    # exp_base -> seed -> list of manifest dicts (run_id, gpu_hours, dirty, commit, path)
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for run_id in sorted(os.listdir(RUNS_DIR)):
        mpath = os.path.join(RUNS_DIR, run_id, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        m = json.load(open(mpath))
        if m.get("status") != "done":
            continue
        exp_base = re.sub(r"_s\d+$", "", m["resolved_config"]["logging"]["experiment_name"])
        seed = m["seed"]
        by_key[(exp_base, seed)].append({
            "run_id": run_id,
            "gpu_hours": m.get("gpu_hours") or 0.0,
            "dirty": bool(m.get("git", {}).get("dirty")),
            "commit": (m.get("git", {}).get("commit") or "")[:9],
            "config_hash": m.get("config_hash", ""),
        })

    rows = []
    for cfgstem in all_configs:
        parsed = parse_config(cfgstem)
        if parsed is None:
            continue
        family, mode, dataset, block = parsed
        exp_base = f"iccit_{cfgstem}"

        for seed in sorted(SEEDS):
            candidates = by_key.get((exp_base, seed), [])
            if not candidates:
                continue
            candidates_sorted = sorted(candidates, key=lambda c: c["gpu_hours"], reverse=True)
            keep = candidates_sorted[0]
            dupes = candidates_sorted[1:]
            total_gpu_hours = sum(c["gpu_hours"] for c in candidates)

            pq_path = os.path.join(LOGS_DIR, f"{exp_base}_s{seed}", "per_image.parquet")
            admissible = os.path.isfile(pq_path) and not keep["dirty"]
            reason = "" if admissible else ("dirty tree" if keep["dirty"] else "per_image.parquet missing")

            rows.append({
                "config": cfgstem, "exp_base": exp_base, "family": family, "mode": mode,
                "dataset": dataset, "block": block, "seed": seed,
                "run_id": keep["run_id"], "n_manifests": len(candidates),
                "gpu_hours_kept": keep["gpu_hours"], "gpu_hours_total": total_gpu_hours,
                "git_commit": keep["commit"], "dirty": keep["dirty"],
                "config_hash": keep["config_hash"],
                "per_image_parquet": pq_path if os.path.isfile(pq_path) else "",
                "admissible": admissible, "exclusion_reason": reason,
                "duplicate_run_ids": ";".join(d["run_id"] for d in dupes),
            })
            for d in dupes:
                rows.append({
                    "config": cfgstem, "exp_base": exp_base, "family": family, "mode": mode,
                    "dataset": dataset, "block": block, "seed": seed,
                    "run_id": d["run_id"], "n_manifests": len(candidates),
                    "gpu_hours_kept": d["gpu_hours"], "gpu_hours_total": total_gpu_hours,
                    "git_commit": d["commit"], "dirty": d["dirty"],
                    "per_image_parquet": "",
                    "admissible": False,
                    "exclusion_reason": f"duplicate manifest for same (config,seed); superseded by {keep['run_id']} "
                                         f"(kept={keep['gpu_hours']:.3f} gpu_h vs this {d['gpu_hours']:.3f} gpu_h) "
                                         f"— almost certainly a spurious resume-and-reeval of an already-done checkpoint",
                    "duplicate_run_ids": "",
                })

    df = pd.DataFrame(rows)
    return df


def apply_block_d_cut(inv: pd.DataFrame) -> pd.DataFrame:
    """G8: Block D (U-Net generality / F5) is formally cut from this
    package by user decision — see decisions.md. Every Block D row is
    marked inadmissible regardless of whether its data actually exists, so
    the exclusion is a declared decision, not a silent gap (plan §2's own
    distinction: "What is not legitimate is an undeclared partial Block
    D")."""
    inv = inv.copy()
    is_d = inv["block"] == "D"
    inv.loc[is_d, "admissible"] = False
    inv.loc[is_d, "exclusion_reason"] = (
        "Block D (U-Net generality, serves F5 only) formally cut from this package — see decisions.md"
    )
    return inv


def full_gate_report(inv: pd.DataFrame, all_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """G1-G10 from the paper package plan's freeze gate, scoped to Blocks
    A/B/C only (D already cut by apply_block_d_cut). Returns a list of
    (gate_id, status, detail) — status one of PASS / FAIL / DEFERRED.
    Unlike the plan's real freeze gate this does not abort the build on a
    FAIL/DEFERRED row; the caller decides what that means for the archive
    it's about to produce."""
    rows: List[Tuple[str, str, str]] = []
    adm = inv[inv["admissible"]]
    abc = inv[inv["block"].isin(("A", "B", "C"))]

    g1_fails, g2_fails = [], []
    for cfgstem in sorted(abc["config"].unique()):
        n = (adm["config"] == cfgstem).sum()
        seeds_present = set(adm.loc[adm["config"] == cfgstem, "seed"])
        if n < 3:
            g1_fails.append(f"{cfgstem} ({n}/3)")
        elif seeds_present != SEEDS:
            g2_fails.append(f"{cfgstem} {sorted(seeds_present)}")
    rows.append(("G1", "PASS" if not g1_fails else "FAIL",
                 "every Block A/B/C config has 3 admissible seeds" if not g1_fails
                 else f"under-seeded: {g1_fails}"))
    rows.append(("G2", "PASS" if not g2_fails else "FAIL",
                 f"seeds == {sorted(SEEDS)} everywhere" if not g2_fails else f"seed mismatch: {g2_fails}"))

    dirty_kept = adm[adm["block"].isin(("A", "B", "C")) & adm["dirty"]]
    rows.append(("G3", "PASS" if len(dirty_kept) == 0 else "FAIL",
                 "no dirty-tree admissible runs" if len(dirty_kept) == 0
                 else f"{len(dirty_kept)} dirty-tree run(s): {dirty_kept['run_id'].tolist()}"))

    rows.append(("G4", "PASS",
                 f"width-matched configs are only generated when models.build.build_width_matched reports "
                 f"within_tolerance at TOL={_gen_tol():.0%} (scripts/gen_iccit_configs.py raises otherwise) — "
                 f"verified at generation time, not re-checked here"))

    hash_by_ds: Dict[str, set] = defaultdict(set)
    for mpath in _manifest_paths(adm):
        m = json.load(open(mpath))
        if "projection_matrix_hash" in m:
            ds = re.search(r"_(clinicdb|isic18|busi)_s\d+$", m["resolved_config"]["logging"]["experiment_name"])
            if ds:
                hash_by_ds[ds.group(1)].add(m["projection_matrix_hash"])
    g5_bad = {ds: h for ds, h in hash_by_ds.items() if len(h) > 1}
    rows.append(("G5", "PASS" if not g5_bad else "FAIL",
                 "projection_matrix_hash identical across seeds within each dataset "
                 f"(hashes: { {ds: list(h)[0][:8] for ds, h in hash_by_ds.items()} }; "
                 "BUSI has none by design — randproj_rgb degenerates under grayscale, see channels.py)"
                 if not g5_bad else f"hash mismatch within a dataset: {g5_bad}"))

    missing_order = [mpath for mpath in _manifest_paths(adm) if "channel_order" not in json.load(open(mpath))]
    rows.append(("G6", "PASS" if not missing_order else "FAIL",
                 "channel_order recorded on every admissible A/B/C manifest" if not missing_order
                 else f"{len(missing_order)} manifest(s) missing channel_order"))

    m3_len = _channel_len(adm, "m3", "busi")
    m7_len = _channel_len(adm, "m7", "busi")
    rows.append(("G7", "PASS" if m3_len is not None and m3_len == m7_len else "FAIL",
                 f"m3 effective channels ({m3_len}) == m7 ({m7_len}) on BUSI — both degenerate to plain RGB "
                 "under grayscale modality, per the ycbcr/randproj_rgb grayscale fix"))

    rows.append(("G8", "PASS", "Block D formally cut — see decisions.md"))

    rows.append(("G9", "DEFERRED",
                 "A9 (external ColonDB) not yet spent — by design, run last per master plan §7/§15, "
                 "after every other result is final. Not attempted in this package."))

    commit_ts = _analysis_plan_commit_ts()
    rows.append(("G10", "PASS" if commit_ts is not None else "FAIL",
                 f"ANALYSIS_PLAN.md committed {commit_ts} — precedes the run matrix" if commit_ts
                 else "could not resolve ANALYSIS_PLAN.md commit timestamp"))

    return rows


def _gen_tol() -> float:
    with open(os.path.join(REPO, "scripts", "gen_iccit_configs.py")) as f:
        m = re.search(r"^TOL\s*=\s*([0-9.]+)", f.read(), re.M)
    return float(m.group(1)) if m else float("nan")


def _manifest_paths(adm: pd.DataFrame) -> List[str]:
    return [os.path.join(RUNS_DIR, rid, "manifest.json") for rid in adm["run_id"] if rid]


def _channel_len(adm: pd.DataFrame, mode: str, dataset: str) -> Optional[int]:
    row = adm[(adm["mode"] == mode) & (adm["dataset"] == dataset)]
    if row.empty:
        return None
    mpath = os.path.join(RUNS_DIR, row.iloc[0]["run_id"], "manifest.json")
    m = json.load(open(mpath))
    return len(m.get("channel_order", [])) if "channel_order" in m else None


def _analysis_plan_commit_ts() -> Optional[str]:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%aI", "--", "ANALYSIS_PLAN.md"], cwd=REPO,
        ).decode().strip()
        return out or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 1 — consolidate per-image scores
# ---------------------------------------------------------------------------

def stage1_consolidate(inv: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    frames = []
    for _, r in inv[inv["admissible"]].iterrows():
        df = pd.read_parquet(r["per_image_parquet"])
        df = df.copy()
        df["run_id"] = r["run_id"]
        df["config"] = r["config"]
        df["model_family"] = r["family"]
        df["mode"] = r["mode"]
        df["dataset"] = r["dataset"]
        df["block"] = r["block"]
        df["seed"] = r["seed"]
        for col in ("hd95", "asd", "nsd"):
            df[f"{col}_defined"] = df[col].notna() if col in df.columns else False
        frames.append(df)

    if not frames:
        raise RuntimeError("stage1: no admissible runs to consolidate")

    all_df = pd.concat(frames, ignore_index=True)

    dup = all_df.duplicated(subset=["run_id", "image_id"]).sum()
    if dup:
        raise RuntimeError(f"stage1: {dup} duplicate (run_id, image_id) rows — consolidation bug")

    warnings = []
    for (cfg, ds), g in all_df.groupby(["config", "dataset"]):
        n_seeds = g["seed"].nunique()
        if n_seeds != 3:
            warnings.append(f"{cfg}: {n_seeds}/3 seeds in consolidated table (trial data is partial)")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_image_all.parquet")
    all_df.to_parquet(path, index=False)
    return all_df, warnings


# ---------------------------------------------------------------------------
# Stage 2 — aggregates
# ---------------------------------------------------------------------------

def stage2_aggregates(all_df: pd.DataFrame, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for (cfg, ds, seed), g in all_df.groupby(["config", "dataset", "seed"]):
        per_seed_rows.append({
            "config": cfg, "dataset": ds, "seed": seed, "n_images": len(g),
            "dice_mean": g["dice"].mean(), "iou_mean": g["iou"].mean(),
            "hd95_mean": g.loc[g["hd95_defined"], "hd95"].mean() if g["hd95_defined"].any() else np.nan,
            "asd_mean": g.loc[g["asd_defined"], "asd"].mean() if g["asd_defined"].any() else np.nan,
            "nsd_mean": g.loc[g["nsd_defined"], "nsd"].mean() if g["nsd_defined"].any() else np.nan,
            "dice_p5": np.percentile(g["dice"], 5), "dice_p25": np.percentile(g["dice"], 25),
        })
    per_seed = pd.DataFrame(per_seed_rows)

    per_config_rows = []
    for (cfg, ds), g in per_seed.groupby(["config", "dataset"]):
        seed_means = g["dice_mean"].to_numpy()
        ci = bootstrap_ci(seed_means) if len(seed_means) > 1 else {"ci_low": float("nan"), "ci_high": float("nan")}
        per_config_rows.append({
            "config": cfg, "dataset": ds, "n_seeds": len(g),
            "dice_mean": seed_means.mean(), "dice_std": seed_means.std(),
            "dice_ci_low": ci["ci_low"], "dice_ci_high": ci["ci_high"],
            "dice_p5_mean": g["dice_p5"].mean(), "dice_p25_mean": g["dice_p25"].mean(),
            "iou_mean": g["iou_mean"].mean(), "hd95_mean": g["hd95_mean"].mean(),
            "asd_mean": g["asd_mean"].mean(), "nsd_mean": g["nsd_mean"].mean(),
        })
    per_config = pd.DataFrame(per_config_rows)

    os.makedirs(out_dir, exist_ok=True)
    per_seed.to_csv(os.path.join(out_dir, "aggregates_per_seed.csv"), index=False)
    per_config.to_csv(os.path.join(out_dir, "aggregates_per_config.csv"), index=False)
    return per_seed, per_config


# ---------------------------------------------------------------------------
# Stage 3 — statistics (F1: channel modes; F3: order ablation — both per dataset)
# ---------------------------------------------------------------------------

def _ordered_scores(sub: pd.DataFrame, mode: str) -> Optional[Tuple[List[float], pd.DataFrame]]:
    g = sub[sub["mode"] == mode]
    if g["seed"].nunique() != 3:
        return None
    g = g.sort_values(["seed", "image_id"])
    return g["dice"].tolist(), g[["seed", "image_id"]]


def _paired_or_skip(sub: pd.DataFrame, proposed_mode: str, comparator_modes: List[str]) -> Tuple[
        Optional[List[float]], Dict[str, List[float]], List[str]]:
    proposed = _ordered_scores(sub, proposed_mode)
    if proposed is None:
        return None, {}, [f"{proposed_mode} (proposed): <3 seeds or not present"]
    p_scores, p_keys = proposed

    comparators, skipped = {}, []
    for mode in comparator_modes:
        got = _ordered_scores(sub, mode)
        if got is None:
            skipped.append(f"{mode}: <3 seeds or not present")
            continue
        scores, keys = got
        if len(scores) != len(p_scores) or not (keys.reset_index(drop=True) == p_keys.reset_index(drop=True)).all().all():
            skipped.append(f"{mode}: (seed, image_id) pairing does not match {proposed_mode} — cannot pair for Wilcoxon")
            continue
        comparators[mode] = scores
    return p_scores, comparators, skipped


def stage3_f1_stats(all_df: pd.DataFrame, stats_dir: str, ledger_dir: str) -> Dict[str, Any]:
    os.makedirs(stats_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    block_a = all_df[(all_df["block"] == "A") & (all_df["model_family"] == "mkunet")]

    for ds in DATASETS:
        sub = block_a[block_a["dataset"] == ds]
        modes_present = sorted(set(sub["mode"].unique()) - {"m1"})
        p_scores, comparators, skipped = _paired_or_skip(sub, "m1", modes_present)
        if p_scores is None:
            results[ds] = {"skipped": skipped[0]}
            continue
        if not comparators:
            results[ds] = {"skipped": "no comparator mode has 3 paired seeds yet", "notes": skipped}
            continue
        res = run_family_comparison(
            family=f"F1_{ds}", proposed_name="m1", proposed_per_image=p_scores,
            comparators=comparators, min_meaningful_diff=MMD, equivalence_bound=TOST_BOUND,
            out_dir=stats_dir, ledger_dir=ledger_dir,
        )
        res["notes"] = skipped
        results[ds] = res

    return results


def stage3_f2_stats(all_df: pd.DataFrame, stats_dir: str, ledger_dir: str) -> Dict[str, Any]:
    """C1: capacity controls — each channel mode vs. its own width-matched
    RGB-only control. Matches reporting.tables.render_capacity_control_table's
    expected shape: one run_family_comparison call per (dataset, base_mode),
    proposed=the matched control's per-image dice, comparator={base_mode:
    the channel mode's per-image dice} — so comp['comparator'] is literally
    the base mode name and the table renderer can reconstruct
    f"{mode}_matched" to look the control back up."""
    os.makedirs(stats_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    mk = all_df[all_df["model_family"] == "mkunet"]

    for ds in DATASETS:
        sub = mk[mk["dataset"] == ds]
        for base_mode in ("m2", "m4", "m5", "m7"):
            control_mode = f"{base_mode}_matched"
            p_scores, comparators, skipped = _paired_or_skip(sub, control_mode, [base_mode])
            key = f"{ds}.{base_mode}"
            if p_scores is None or not comparators:
                results[key] = {"skipped": skipped[0] if skipped else "unknown"}
                continue
            res = run_family_comparison(
                family=f"F2_{ds}_{base_mode}", proposed_name="matched_control", proposed_per_image=p_scores,
                comparators=comparators, min_meaningful_diff=MMD, equivalence_bound=TOST_BOUND,
                out_dir=stats_dir, ledger_dir=ledger_dir,
            )
            results[key] = res

    return results


def stage3_c2_direct(all_df: pd.DataFrame, stats_dir: str, ledger_dir: str) -> Dict[str, Any]:
    """C2's actual falsification test: m3 (YCbCr) vs m7 (width-matched
    random-projection control), directly — not each vs m1 (that's F1).
    Outside F1-F5, so written to exploratory.json per the package plan
    (Stage 3 note: "Any comparison outside F1-F5 ... labelled exploratory")."""
    os.makedirs(stats_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    mk = all_df[all_df["model_family"] == "mkunet"]

    for ds in DATASETS:
        sub = mk[mk["dataset"] == ds]
        p_scores, comparators, skipped = _paired_or_skip(sub, "m7", ["m3"])
        if p_scores is None or not comparators:
            results[ds] = {"skipped": skipped[0] if skipped else "unknown"}
            continue
        res = run_family_comparison(
            family=f"C2_direct_{ds}", proposed_name="m7", proposed_per_image=p_scores,
            comparators=comparators, min_meaningful_diff=MMD, equivalence_bound=TOST_BOUND,
            out_dir=stats_dir, ledger_dir=ledger_dir,
        )
        results[ds] = res

    exploratory_path = os.path.join(stats_dir, "exploratory.json")
    payload = {ds: {k: v for k, v in r.items() if k != "_written_to"}
               for ds, r in results.items() if "skipped" not in r}
    with open(exploratory_path, "w") as f:
        json.dump(payload, f, indent=2)
    return results


def stage3_f3_stats(all_df: pd.DataFrame, stats_dir: str, ledger_dir: str) -> Dict[str, Any]:
    """C3: order ablation — mode-pre (Block C) vs. mode-post (Block A), for
    m4 and m5, per dataset. Proposed = the post (regenerated-after-augment)
    variant, since C3's hypothesis is that pre degrades relative to post."""
    os.makedirs(stats_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    mk = all_df[all_df["model_family"] == "mkunet"]

    for ds in DATASETS:
        sub = mk[mk["dataset"] == ds]
        for base_mode in ("m4", "m5"):
            pre_mode = f"{base_mode}_pre"
            p_scores, comparators, skipped = _paired_or_skip(sub, base_mode, [pre_mode])
            key = f"{ds}.{base_mode}"
            if p_scores is None or not comparators:
                results[key] = {"skipped": skipped[0] if skipped else "unknown"}
                continue
            # renamed to the hyphenated "<mode>-pre" key here (not
            # underscore) so the comparator name matches what
            # reporting.tables.render_order_ablation_table looks up
            # (f"{mode}-pre") — the ledger's comparison string is derived
            # from this same key, so it has to be set before the call, not
            # patched afterward.
            comparators_hyphen = {f"{base_mode}-pre": comparators[pre_mode]}
            # proposed_name is fixed to "post" (not base_mode) across both
            # m4 and m5 calls: render_order_ablation_table's blocking-rule
            # check reconstructs ledger comparison strings as
            # f"{stats['proposed']}_vs_{comparator}" using ONE shared
            # proposed label for the whole merged table, so every call
            # feeding that table must have written its ledger row under
            # the same proposed_name for the lookup to match.
            res = run_family_comparison(
                family=f"F3_{ds}_{base_mode}", proposed_name="post", proposed_per_image=p_scores,
                comparators=comparators_hyphen, min_meaningful_diff=MMD, equivalence_bound=TOST_BOUND,
                out_dir=stats_dir, ledger_dir=ledger_dir,
            )
            results[key] = res

    return results


# ---------------------------------------------------------------------------
# Stage 5 (T1/T2) — channel-mode and capacity-control tables per dataset
# ---------------------------------------------------------------------------

def seed_runs_ledger(all_df: pd.DataFrame, ledger_dir: str) -> None:
    """Seeds ledger_dir/runs.csv from every mkunet (config, dataset, mode,
    seed) we have, so reporting.tables' blocking-rule checks (dirty tree /
    min seeds) have something to read regardless of which table is rendered."""
    ledger = LedgerWriter(ledger_dir)
    mk = all_df[all_df["model_family"] == "mkunet"]
    for (cfg, ds, mode, seed), g in mk.groupby(["config", "dataset", "mode", "seed"]):
        ledger.append_run_row(
            run_id=f"{cfg}_s{seed}", config_hash="", experiment_name=f"iccit_{cfg}",
            model_name=mode, dataset_name=ds, seed=seed, fold="", status="done",
            start_time="", end_time="", gpu_hours="", best_metric="", monitor_metric="",
            git_commit="", git_dirty="false", manifest_path="",
        )


def stage5_t1(all_df: pd.DataFrame, per_config: pd.DataFrame, stats_results: Dict[str, Any],
              ledger_dir: str, snapshot_id: str, tables_dir: str) -> Dict[str, Any]:
    os.makedirs(tables_dir, exist_ok=True)
    block_a = all_df[(all_df["block"] == "A") & (all_df["model_family"] == "mkunet")]

    written = {}
    for ds in DATASETS:
        stats_res = stats_results.get(ds)
        if not stats_res or "skipped" in stats_res:
            written[ds] = f"skipped — {stats_res.get('skipped') if stats_res else 'no stats result'}"
            continue
        sub = per_config[per_config["dataset"] == ds]
        cfg_to_mode = {r["config"]: r["config"] for _, r in sub.iterrows()}
        results = []
        for _, r in block_a[block_a["dataset"] == ds].iterrows():
            results.append({"mode": r["mode"], "dataset": ds, "seed": r["seed"], "dice": r["dice"]})
        try:
            table = render_channel_mode_table(results, stats_res, profiling=[], ledger_dir=ledger_dir,
                                               snapshot_id=snapshot_id)
        except Exception as exc:
            written[ds] = f"render failed: {exc}"
            continue
        with open(os.path.join(tables_dir, f"T1_{ds}.tex"), "w") as f:
            f.write(table["latex"])
        with open(os.path.join(tables_dir, f"T1_{ds}.csv"), "w") as f:
            f.write(table["csv"])
        written[ds] = "ok"
    return written


def stage5_t2(all_df: pd.DataFrame, f2_results: Dict[str, Any], ledger_dir: str, snapshot_id: str,
              tables_dir: str) -> Dict[str, Any]:
    os.makedirs(tables_dir, exist_ok=True)
    mk = all_df[all_df["model_family"] == "mkunet"]

    written = {}
    for ds in DATASETS:
        comparisons = []
        for base_mode in ("m2", "m4", "m5", "m7"):
            res = f2_results.get(f"{ds}.{base_mode}")
            if res and "skipped" not in res:
                comparisons.append(res["comparisons"][0])
        if not comparisons:
            written[ds] = "skipped — no F2 comparison has 3 paired seeds on both mode and control yet"
            continue
        merged_stats = {"proposed": "matched_control", "comparisons": comparisons}

        sub = mk[mk["dataset"] == ds]
        results = [{"mode": r["mode"], "dataset": ds, "seed": r["seed"], "dice": r["dice"]}
                   for _, r in sub.iterrows()
                   if r["mode"] in {"m2", "m4", "m5", "m7", "m2_matched", "m4_matched", "m5_matched", "m7_matched"}]
        try:
            table = render_capacity_control_table(results, merged_stats, ledger_dir=ledger_dir,
                                                    snapshot_id=snapshot_id)
        except Exception as exc:
            written[ds] = f"render failed: {exc}"
            continue
        with open(os.path.join(tables_dir, f"T2_{ds}.tex"), "w") as f:
            f.write(table["latex"])
        with open(os.path.join(tables_dir, f"T2_{ds}.csv"), "w") as f:
            f.write(table["csv"])
        written[ds] = "ok"
    return written


def stage5_t3(all_df: pd.DataFrame, f3_results: Dict[str, Any], ledger_dir: str, snapshot_id: str,
              tables_dir: str) -> str:
    os.makedirs(tables_dir, exist_ok=True)
    mk = all_df[all_df["model_family"] == "mkunet"]

    comparisons = []
    for ds in DATASETS:
        for base_mode in ("m4", "m5"):
            res = f3_results.get(f"{ds}.{base_mode}")
            if res and "skipped" not in res:
                comparisons.append(res["comparisons"][0])
    if not comparisons:
        return "skipped — no F3 comparison has 3 paired seeds on both post and pre yet"
    merged_stats = {"proposed": "post", "comparisons": comparisons}

    results = []
    for ds in DATASETS:
        for base_mode in ("m4", "m5"):
            post = mk[(mk["dataset"] == ds) & (mk["mode"] == base_mode)]
            for _, r in post.iterrows():
                results.append({"mode": base_mode, "order": "post", "dataset": ds, "seed": r["seed"], "dice": r["dice"]})
            pre = mk[(mk["dataset"] == ds) & (mk["mode"] == f"{base_mode}_pre")]
            for _, r in pre.iterrows():
                results.append({"mode": base_mode, "order": "pre", "dataset": ds, "seed": r["seed"], "dice": r["dice"]})

    try:
        table = render_order_ablation_table(results, merged_stats, ledger_dir=ledger_dir, snapshot_id=snapshot_id)
    except Exception as exc:
        return f"render failed: {exc}"
    with open(os.path.join(tables_dir, "T3.tex"), "w") as f:
        f.write(table["latex"])
    with open(os.path.join(tables_dir, "T3.csv"), "w") as f:
        f.write(table["csv"])
    return "ok"


# ---------------------------------------------------------------------------
# Stage 6 — numbers ledger; Stage 7 — verify, scan, checksum, archive
# ---------------------------------------------------------------------------

CLAIM_TEXT = {
    "C1": "Reported gains from geometric/colour input channels are confounded with capacity",
    "C2": "YCbCr adds no information over RGB (invertible affine map)",
    "C3": "Coordinate channels must be regenerated after geometric augmentation",
}
FALSIFIED_IF = {
    "C1": "Gains persist at equal parameters and FLOPs",
    "C2": "m3 significantly beats m7",
    "C3": "No significant difference between orders",
}


_MEANINGFUL_REAL = "statistically significant and practically meaningful"
_MEANINGFUL_UNDERPOWERED = "not statistically significant despite exceeding the meaningful-difference threshold (underpowered?)"


def _bucket_verdict(comps: List[Dict[str, Any]]) -> str:
    """supported / falsified / equivalent / inconclusive, from a list of
    run_family_comparison comparison dicts. Driven by stats.tests.
    meaningfulness_gate's own 4-way vocabulary (not a re-derivation from
    raw p-values/TOST) so this can't silently disagree with what each
    comparison's own verdict string already says in stats/*.json — per
    master plan §10, a plain p < alpha alone is not evidence of a
    *meaningful* effect, and meaningfulness_gate is the thing that already
    encodes that distinction."""
    verdicts = [c["meaningfulness_verdict"] for c in comps]
    if any(v == _MEANINGFUL_REAL for v in verdicts):
        return "falsified"
    if any(v == _MEANINGFUL_UNDERPOWERED for v in verdicts):
        return "inconclusive"
    # every comparator is now either "significant but below threshold" or
    # "neither significant nor meaningful" — no comparator shows a real
    # meaningful effect, so the null hypothesis (e.g. "gains disappear")
    # holds; "equivalent" only when TOST also formally confirms it
    # everywhere, else the weaker (but still non-falsified) "supported".
    if all(c.get("equivalence", {}).get("verdict") == "equivalent" for c in comps):
        return "equivalent"
    return "supported"


def stage6_numbers(per_config: pd.DataFrame, f1_results: Dict[str, Any], f2_results: Dict[str, Any],
                    c2_results: Dict[str, Any], f3_results: Dict[str, Any], snapshot_id: str,
                    out_dir: str) -> None:
    rows = []

    for _, r in per_config.iterrows():
        base = f"AGG.{r['config']}.{r['dataset']}"
        for metric in ("dice_mean", "dice_std", "dice_ci_low", "dice_ci_high"):
            rows.append({"token": f"{base}.{metric}", "value": r[metric], "unit": "dice",
                         "source_file": "data/aggregates_per_config.csv", "source_row": "", "claim": "",
                         "family": "", "verdict": ""})

    for ds, res in f1_results.items():
        if "skipped" in res:
            continue
        for c in res["comparisons"]:
            base = f"F1.{ds}.m1_vs_{c['comparator']}"
            rows.append({"token": f"{base}.median_diff", "value": c["paired_median_diff"]["median_diff"],
                         "unit": "dice", "source_file": f"stats/F1_{ds}.json", "source_row": "", "claim": "",
                         "family": "F1", "verdict": c.get("verdict", c["meaningfulness_verdict"])})
            rows.append({"token": f"{base}.corrected_p", "value": c["corrected_p_value"], "unit": "p",
                         "source_file": f"stats/F1_{ds}.json", "source_row": "", "claim": "",
                         "family": "F1", "verdict": ""})

    for key, res in f2_results.items():
        if "skipped" in res:
            continue
        ds, mode = key.split(".")
        c = res["comparisons"][0]
        base = f"F2.{ds}.{mode}_vs_matched"
        rows.append({"token": f"{base}.median_diff", "value": c["paired_median_diff"]["median_diff"],
                     "unit": "dice", "source_file": f"stats/F2_{ds}_{mode}.json", "source_row": "", "claim": "C1",
                     "family": "F2", "verdict": c.get("verdict", c["meaningfulness_verdict"])})
        rows.append({"token": f"{base}.corrected_p", "value": c["corrected_p_value"], "unit": "p",
                     "source_file": f"stats/F2_{ds}_{mode}.json", "source_row": "", "claim": "C1",
                     "family": "F2", "verdict": ""})

    for ds, res in c2_results.items():
        if "skipped" in res:
            continue
        c = res["comparisons"][0]
        base = f"C2direct.{ds}.m3_vs_m7"
        rows.append({"token": f"{base}.median_diff", "value": c["paired_median_diff"]["median_diff"],
                     "unit": "dice", "source_file": f"stats/C2_direct_{ds}.json", "source_row": "", "claim": "C2",
                     "family": "C2_direct", "verdict": c.get("verdict", c["meaningfulness_verdict"])})

    for key, res in f3_results.items():
        if "skipped" in res:
            continue
        ds, mode = key.split(".")
        c = res["comparisons"][0]
        base = f"F3.{ds}.{mode}.post_vs_pre"
        rows.append({"token": f"{base}.median_diff", "value": c["paired_median_diff"]["median_diff"],
                     "unit": "dice", "source_file": f"stats/F3_{ds}_{mode}.json", "source_row": "", "claim": "C3",
                     "family": "F3", "verdict": c.get("verdict", c["meaningfulness_verdict"])})

    numbers_df = pd.DataFrame(rows)
    numbers_df.to_csv(os.path.join(out_dir, "numbers.csv"), index=False)
    with open(os.path.join(out_dir, "numbers.json"), "w") as f:
        json.dump({r["token"]: r for r in rows}, f, indent=2, default=str)

    trace_rows = []
    c1_comps = [res["comparisons"][0] for res in f2_results.values() if "skipped" not in res]
    c2_comps = [res["comparisons"][0] for res in c2_results.values() if "skipped" not in res]
    c3_comps = [res["comparisons"][0] for res in f3_results.values() if "skipped" not in res]
    for claim, comps, families, tables in (
        ("C1", c1_comps, "F2", "T2"), ("C2", c2_comps, "C2_direct (exploratory) + F1", "T1"),
        ("C3", c3_comps, "F3", "T3"),
    ):
        status = _bucket_verdict(comps) if comps else "inconclusive (no data)"
        trace_rows.append({
            "claim": claim, "text": CLAIM_TEXT[claim], "families": families, "tables": tables,
            "figures": "", "analysis_files": "", "falsified_if": FALSIFIED_IF[claim],
            "resolved_status": status,
        })
    trace_rows.append({"claim": "C4", "text": "Surviving gains are largely positional shortcut",
                        "families": "F4", "tables": "", "figures": "F1", "analysis_files": "A4-A6",
                        "falsified_if": "Gains are stable under shift and externally",
                        "resolved_status": "deferred — not attempted in this package"})
    trace_rows.append({"claim": "C5", "text": "Effect size is moderated by dataset centre bias, not modality",
                        "families": "-", "tables": "", "figures": "F3", "analysis_files": "A8",
                        "falsified_if": "No correlation",
                        "resolved_status": "deferred — not attempted in this package"})
    pd.DataFrame(trace_rows).to_csv(os.path.join(out_dir, "claims_traceability.csv"), index=False)


def stage7_verify_and_archive(out_dir: str, snapshot_id: str, tables_dir: str,
                               hard_fails: List[Tuple[str, str, str]]) -> str:
    problems = []

    expected = ["inventory.csv", "decisions.md", "package_manifest.json", "numbers.csv", "numbers.json",
                "claims_traceability.csv", "data/per_image_all.parquet", "data/aggregates_per_seed.csv",
                "data/aggregates_per_config.csv"]
    for rel in expected:
        p = os.path.join(out_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            problems.append(f"completeness: missing or empty {rel}")

    for ds in DATASETS:
        t1 = os.path.join(tables_dir, f"T1_{ds}.csv")
        if not os.path.isfile(t1):
            continue
        all_df = pd.read_parquet(os.path.join(out_dir, "data", "per_image_all.parquet"))
        block_a = all_df[(all_df["dataset"] == ds) & (all_df["block"] == "A") & (all_df["model_family"] == "mkunet")]
        rows = [ln.split(",") for ln in open(t1).read().splitlines()[2:]]
        for row in rows:
            mode, dice_mean = row[0], float(row[4])
            recomputed = block_a.loc[block_a["mode"] == mode, "dice"].mean()
            # T1.csv rounds to 4 decimals at render time (reporting/tables.py's
            # f"{dice_vals.mean():.4f}"), so the round-trip comparison has to
            # match at the same precision, not full float precision.
            if abs(round(recomputed, 4) - dice_mean) > 1e-4:
                problems.append(f"consistency: T1_{ds} {mode} dice_mean {dice_mean} != recomputed {recomputed:.6f}")

    if abs(MMD - 0.010) > 1e-9 or abs(TOST_BOUND - 0.010) > 1e-9:
        problems.append(f"threshold conformance: MMD={MMD} TOST_BOUND={TOST_BOUND} != ANALYSIS_PLAN.md's 0.010")

    leak_pattern = re.compile(
        r"mamba|ss2d|vss|cbffm|fusion|routing|auxiliary|dual.?encoder|dissert|thesis|/home/[a-z0-9_]+|@\w",
        re.IGNORECASE,
    )
    text_exts = {".csv", ".json", ".md", ".txt", ".tex"}
    leaks = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if os.path.splitext(fn)[1] not in text_exts:
                continue
            fp = os.path.join(root, fn)
            try:
                content = open(fp, errors="ignore").read()
            except Exception:
                continue
            for m in leak_pattern.finditer(content):
                leaks.append(f"{os.path.relpath(fp, out_dir)}: {m.group(0)!r}")
    if leaks:
        problems.append(f"anonymity scan: {len(leaks)} hit(s): {leaks[:10]}")

    verify_path = os.path.join(out_dir, "VERIFY.txt")
    with open(verify_path, "w") as f:
        if problems:
            f.write("PROBLEMS FOUND:\n" + "\n".join(problems) + "\n")
        else:
            f.write("completeness: OK\nconsistency: OK (T1 recomputed from per_image_all.parquet matches)\n"
                    "threshold conformance: OK\nanonymity scan: no hits\n")
        if hard_fails:
            f.write("\nNOTE: this package was built with FAILing freeze-gate rows (see decisions.md); "
                    "it is a declared-scope checkpoint, not a plan-compliant final package.\n")
    for line in (open(verify_path).read().splitlines()):
        print(f"  {line}")

    sha_path = os.path.join(out_dir, "SHA256SUMS")
    with open(sha_path, "w") as sha_f:
        for root, _, files in os.walk(out_dir):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                if fp == sha_path:
                    continue
                digest = hashlib.sha256(open(fp, "rb").read()).hexdigest()
                sha_f.write(f"{digest}  {os.path.relpath(fp, out_dir)}\n")

    suffix = "_PARTIAL" if (problems or hard_fails) else ""
    zip_name = f"iccit2026_paper_package_C1-C3_{snapshot_id}{suffix}.zip"
    zip_path = os.path.join(REPO, "build", zip_name)
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.join(os.path.basename(out_dir), os.path.relpath(fp, out_dir)))
    return zip_path


# ---------------------------------------------------------------------------

def compute_snapshot_id(inv: pd.DataFrame) -> str:
    """Plan §3: sha256 of the sorted (run_id, config_hash, per_image.parquet
    sha256) list for every admissible run, truncated to 8 hex chars."""
    adm = inv[inv["admissible"]].sort_values("run_id")
    parts = []
    for _, r in adm.iterrows():
        with open(r["per_image_parquet"], "rb") as f:
            pq_sha = hashlib.sha256(f.read()).hexdigest()
        parts.append(f"{r['run_id']}:{r['config_hash']}:{pq_sha}")
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:8]
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date}-{digest}"


def write_decisions_md(out_dir: str, snapshot_id: str, gate_rows: List[Tuple[str, str, str]]) -> None:
    lines = [
        f"# Decisions — snapshot {snapshot_id}", "",
        "## Scope", "",
        "This package covers **C1, C2, C3 only** (Blocks A, B, C — MK-UNet channel modes, "
        "capacity controls, order ablation). **C4 and C5 are deferred**, and with them Block D "
        "(the U-Net generality check, F5) and inference analyses A1-A10.", "",
        "## G8 — Block D declared cut", "",
        "Block D (`unet_*`, serves F5/generality only) is formally excluded from this package by "
        "explicit decision, not because it failed — several of its configs (ClinicDB, BUSI) are "
        "in fact complete. It is cut uniformly (every Block D row, regardless of local completion) "
        "so the exclusion is a clean declared boundary rather than a silently partial block. "
        "Its data remains in `results/combined/` and can be folded into a future snapshot.", "",
        "## G9 — A9 (external ColonDB) deferred", "",
        "Not run. Per master plan §7/§15, A9 is a one-shot spent last, after every other result is "
        "final — it is out of order to run it before Block D/C4/C5 are even scoped in. Deferred, "
        "not skipped.", "",
        "## Freeze gate (scoped to Blocks A/B/C)", "",
    ]
    for gid, status, detail in gate_rows:
        lines.append(f"- **{gid}**: {status} — {detail}")
    with open(os.path.join(out_dir, "decisions.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default: build/package/<SNAPSHOT_ID>)")
    args = ap.parse_args()

    print("=== ICCIT2026 paper package build (scope: C1-C3, Blocks A/B/C; U-Net/C4/C5 deferred) ===\n")

    print("--- Stage 0: inventory ---")
    inv = stage0_inventory()
    inv = apply_block_d_cut(inv)
    n_adm = int(inv["admissible"].sum())
    print(f"{len(inv)} manifest rows scanned, {n_adm} admissible (Blocks A/B/C only; D cut), "
          f"{len(inv) - n_adm} excluded")
    dup_reasons = inv[~inv["admissible"] & inv["exclusion_reason"].str.startswith("duplicate")]
    if len(dup_reasons):
        print(f"  {len(dup_reasons)} excluded as duplicate/superseded manifests (see inventory.csv)")

    snapshot_id = compute_snapshot_id(inv)
    out_dir = args.out or os.path.join(REPO, "build", "package", snapshot_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"snapshot: {snapshot_id}\nout: {out_dir}")
    inv_display = inv.copy()
    inv_display["per_image_parquet"] = inv_display["per_image_parquet"].apply(
        lambda p: os.path.relpath(p, REPO) if p else ""
    )
    inv_display.to_csv(os.path.join(out_dir, "inventory.csv"), index=False)

    print("\n--- Stage 1: consolidate per-image scores ---")
    data_dir = os.path.join(out_dir, "data")
    all_df, warnings = stage1_consolidate(inv, data_dir)
    print(f"per_image_all.parquet: {len(all_df)} rows, {all_df['config'].nunique()} configs, "
          f"{all_df['run_id'].nunique()} runs")
    for w in warnings:
        print(f"  WARNING: {w}")

    print("\n--- Freeze gate (G1-G10, scoped to Blocks A/B/C) ---")
    gate_rows = full_gate_report(inv, all_df)
    for gid, status, detail in gate_rows:
        print(f"  {gid} {status}: {detail}")
    write_decisions_md(out_dir, snapshot_id, gate_rows)
    hard_fails = [g for g in gate_rows if g[1] == "FAIL"]

    print("\n--- Stage 2: aggregates ---")
    per_seed, per_config = stage2_aggregates(all_df, data_dir)
    print(f"aggregates_per_seed.csv: {len(per_seed)} rows; aggregates_per_config.csv: {len(per_config)} rows")

    print("\n--- Stage 3: F1 stats (channel modes, per dataset) ---")
    stats_dir = os.path.join(out_dir, "stats")
    ledger_dir = os.path.join(out_dir, "ledger")
    stats_results = stage3_f1_stats(all_df, stats_dir, ledger_dir)
    for ds, res in stats_results.items():
        if "skipped" in res:
            print(f"  {ds}: SKIPPED — {res['skipped']}")
        else:
            verdicts = {c["comparator"]: c.get("verdict", c["meaningfulness_verdict"]) for c in res["comparisons"]}
            print(f"  {ds}: {len(res['comparisons'])} comparisons vs m1 -> {verdicts}")
            if res.get("notes"):
                print(f"    (skipped comparators: {res['notes']})")

    print("\n--- Stage 3: F2 stats (capacity controls, C1) ---")
    f2_results = stage3_f2_stats(all_df, stats_dir, ledger_dir)
    for key, res in f2_results.items():
        if "skipped" in res:
            print(f"  {key}: SKIPPED — {res['skipped']}")
        else:
            comp = res["comparisons"][0]
            print(f"  {key}: mode vs matched-control -> {comp.get('verdict', comp['meaningfulness_verdict'])} "
                  f"(median diff={comp['paired_median_diff']['median_diff']:.4f}, "
                  f"corrected p={comp['corrected_p_value']:.4f})")

    print("\n--- Stage 3: C2 direct test (m3 vs m7) ---")
    c2_results = stage3_c2_direct(all_df, stats_dir, ledger_dir)
    for ds, res in c2_results.items():
        if "skipped" in res:
            print(f"  {ds}: SKIPPED — {res['skipped']}")
        else:
            comp = res["comparisons"][0]
            print(f"  {ds}: m3 vs m7 -> {comp.get('verdict', comp['meaningfulness_verdict'])} "
                  f"(median diff={comp['paired_median_diff']['median_diff']:.4f}, "
                  f"corrected p={comp['corrected_p_value']:.4f})")

    print("\n--- Stage 3: F3 stats (order ablation, C3) ---")
    f3_results = stage3_f3_stats(all_df, stats_dir, ledger_dir)
    for key, res in f3_results.items():
        if "skipped" in res:
            print(f"  {key}: SKIPPED — {res['skipped']}")
        else:
            comp = res["comparisons"][0]
            print(f"  {key}: post vs pre -> {comp.get('verdict', comp['meaningfulness_verdict'])} "
                  f"(median diff={comp['paired_median_diff']['median_diff']:.4f}, "
                  f"corrected p={comp['corrected_p_value']:.4f})")

    print("\n--- Stage 5 (T1, T2, T3 tables) ---")
    tables_dir = os.path.join(out_dir, "tables")
    seed_runs_ledger(all_df, ledger_dir)
    t1_status = stage5_t1(all_df, per_config, stats_results, ledger_dir, snapshot_id, tables_dir)
    for ds, status in t1_status.items():
        print(f"  T1_{ds}: {status}")
    t2_status = stage5_t2(all_df, f2_results, ledger_dir, snapshot_id, tables_dir)
    for ds, status in t2_status.items():
        print(f"  T2_{ds}: {status}")
    t3_status = stage5_t3(all_df, f3_results, ledger_dir, snapshot_id, tables_dir)
    print(f"  T3: {t3_status}")

    print("\n--- Stage 6: numbers ledger ---")
    stage6_numbers(per_config, stats_results, f2_results, c2_results, f3_results, snapshot_id, out_dir)

    manifest = {
        "snapshot_id": snapshot_id, "scope": "C1-C3 (Blocks A/B/C); U-Net/Block D, C4, C5, A1-A10, A9 deferred",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_manifest_rows": len(inv), "n_admissible": n_adm,
        "freeze_gate": {gid: status for gid, status, _ in gate_rows},
        "stages_run": ["0-inventory", "1-consolidate", "2-aggregates", "3-stats(F1,F2/C1,C2-direct,F3/C3)",
                       "5-tables(T1,T2,T3)", "6-numbers-ledger", "7-verify/scan/checksum/archive"],
        "stages_skipped": ["3-C4/F4(shortcut, deferred by request)", "4-inference-analyses(A1-A10)",
                           "5-figures(F1-F3, need A1-A9)", "7-pdf-metadata-scrub(no PDFs built yet)"],
    }
    with open(os.path.join(out_dir, "package_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n--- Stage 7: verify, scan, checksum, archive ---")
    zip_path = stage7_verify_and_archive(out_dir, snapshot_id, tables_dir, hard_fails)
    print(f"  wrote {zip_path}")
    print(f"\n=== done. wrote {out_dir} ===")


if __name__ == "__main__":
    main()
