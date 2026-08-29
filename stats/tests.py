"""
stats/tests.py — paired significance testing + bootstrap CI + the
meaningfulness gate, per spec §10. Operates on per-image score arrays —
Phase 2's write_per_image_parquet() is the intended source of those, read
by whatever assembles a comparison (e.g. a future S7 pipeline stage), not
by this module directly (kept decoupled from any specific file layout).
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_paired_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on per-image scores, proposed (a) vs one
    comparator (b), the *same* images (paired) — spec §10's "Paired test"
    row.

    Returns ``{"statistic", "p_value", "n"}``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"wilcoxon_paired_test: scores_a and scores_b must be paired "
            f"(same length), got {len(a)} vs {len(b)}"
        )
    if len(a) < 1:
        raise ValueError("wilcoxon_paired_test: no scores given")

    if np.all(a == b):
        # scipy.stats.wilcoxon raises on an all-zero difference vector
        # instead of returning a "no evidence of difference" result — that
        # IS the correct verdict here (every paired score identical), so
        # return it directly rather than letting the exception surface.
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(a))}

    result = wilcoxon(a, b)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": int(len(a))}


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
    statistic: Callable[..., np.ndarray] = np.mean,
) -> Dict[str, float]:
    """Percentile bootstrap CI over *values* — spec §10's "CI" row calls
    for this twice per comparison, "over images" (values = one method's
    per-image scores) and "over seeds" (values = one method's per-seed
    aggregate scores), reported separately; this function doesn't know
    which population *values* represents — that's the caller's job.

    *statistic* must accept an ``axis`` kwarg (as ``np.mean``/``np.median``
    do) so it can be applied to every resample row at once.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("bootstrap_ci: no values given")

    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    stat_dist = statistic(resampled, axis=1)
    alpha = 1.0 - ci
    lo, hi = np.percentile(stat_dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(statistic(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_resamples": n_resamples,
        "n": int(len(values)),
    }


def tost_equivalence(
    scores_a: Sequence[float], scores_b: Sequence[float], bound: float, paired: bool = True
) -> Dict[str, object]:
    """Two one-sided tests (TOST) for equivalence of paired samples —
    C1/C2's null-result predictions need this: "no significant difference"
    (a plain Wilcoxon p >= alpha) is not evidence of *equivalence*, only
    absence of evidence. Delegates the actual two-one-sided-t-test
    machinery to ``statsmodels.stats.weightstats.ttost_paired`` rather than
    hand-rolling it (statsmodels is already a pinned dependency); the mean
    difference and its CI reuse this module's own ``bootstrap_ci`` (over
    the paired per-image differences), for the same reason
    ``run_family_comparison`` already reuses it elsewhere.

    Args:
        bound: equivalence bound — the interval (-bound, +bound) the true
            mean difference must lie within to call the two methods
            equivalent.
        paired: only True is currently implemented (every comparison this
            module supports elsewhere is paired, same-image scores).

    Returns:
        ``{p_lower, p_upper, p_tost, mean_diff, ci_low, ci_high, verdict}``
        — verdict is one of:
          - "equivalent": p_tost < 0.05 AND the CI lies entirely within
            (-bound, +bound).
          - "not_equivalent": the CI lies entirely *outside* the bound on
            one side — evidence the true difference exceeds it.
          - "inconclusive": neither — the CI straddles a bound edge
            (typically a high-variance/small-n, underpowered comparison).
    """
    from statsmodels.stats.weightstats import ttost_paired

    if not paired:
        raise NotImplementedError("tost_equivalence: only paired equivalence testing is implemented")

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"tost_equivalence: scores_a and scores_b must be paired (same length), "
            f"got {len(a)} vs {len(b)}"
        )
    if len(a) < 2:
        raise ValueError("tost_equivalence: need at least 2 paired observations")

    p_tost, (_, p_lower, _), (_, p_upper, _) = ttost_paired(a, b, -bound, bound)

    ci = bootstrap_ci(a - b)
    mean_diff, ci_low, ci_high = ci["point_estimate"], ci["ci_low"], ci["ci_high"]

    if p_tost < 0.05 and ci_low > -bound and ci_high < bound:
        verdict = "equivalent"
    elif ci_low > bound or ci_high < -bound:
        verdict = "not_equivalent"
    else:
        verdict = "inconclusive"

    return {
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "mean_diff": float(mean_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "verdict": verdict,
    }


def meaningfulness_gate(
    observed_diff: float, min_meaningful_diff: float, p_value: float, alpha: float = 0.05
) -> str:
    """Compares the observed difference to a pre-registered minimum
    meaningful difference (spec §10's "Meaningfulness gate" row) and
    returns a verdict string meant to be used *verbatim* in the paper —
    kept as fixed constants here rather than built ad hoc at each call
    site, so every comparison in the manuscript states the same four
    possible verdicts in the same words.
    """
    significant = p_value < alpha
    meaningful = abs(observed_diff) >= min_meaningful_diff
    if significant and meaningful:
        return "statistically significant and practically meaningful"
    if significant and not meaningful:
        return "statistically significant but below the pre-registered meaningful-difference threshold"
    if not significant and meaningful:
        return "not statistically significant despite exceeding the meaningful-difference threshold (underpowered?)"
    return "neither statistically significant nor practically meaningful"
