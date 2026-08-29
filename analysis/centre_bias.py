"""
analysis/centre_bias.py — per-dataset positional-predictability index (C4
framing: quantify how much of a dataset's mask signal is recoverable from
position alone, independent of any trained model).

No training required — everything here operates directly on ground-truth
masks. Dice is always computed via metrics.region.dice, never
reimplemented.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from metrics.region import dice

Size = Tuple[int, int]


def _resize_mask(mask: np.ndarray, size: Size) -> np.ndarray:
    """Binary (H, W) mask -> binary *size* mask, nearest-neighbour (no
    intermediate grey values from resampling a {0, 1} array)."""
    h, w = size
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def mask_density_map(masks: Sequence[np.ndarray], size: Size = (256, 256)) -> np.ndarray:
    """Mean binary mask over *masks*, each resampled to *size* first —
    per-pixel foreground frequency, in [0, 1]."""
    if not masks:
        raise ValueError("mask_density_map: no masks given")
    resized = [_resize_mask(m, size) for m in masks]
    return np.mean(np.stack(resized, axis=0), axis=0)


def _best_threshold_dice(density: np.ndarray, masks: Sequence[np.ndarray], size: Size) -> Dict[str, float]:
    """Grid-search the density-map threshold maximising mean Dice against
    *masks* (each resized to *size*); returns {"threshold", "dice"}."""
    candidates = np.unique(np.concatenate([density.ravel(), [0.0, 1.0]]))
    resized_masks = [_resize_mask(m, size) for m in masks]

    best_threshold, best_dice = 0.0, -1.0
    for t in candidates:
        pred = (density >= t).astype(np.uint8)
        mean_dice = float(np.mean([dice(pred, m) for m in resized_masks]))
        if mean_dice > best_dice:
            best_threshold, best_dice = float(t), mean_dice
    return {"threshold": best_threshold, "dice": best_dice}


def constant_mask_floor(
    train_masks: Sequence[np.ndarray], test_masks: Sequence[np.ndarray], size: Size = (256, 256)
) -> Dict[str, float]:
    """Threshold the *training* density map at the value maximising
    training Dice, then evaluate that one fixed (constant) predicted mask
    against *test_masks* — the "a model that learned nothing but frame
    position" floor F4 compares a coord-only-trained model's Dice against.

    Returns {"threshold", "train_dice", "test_dice"}.
    """
    density = mask_density_map(train_masks, size)
    fit = _best_threshold_dice(density, train_masks, size)

    pred = (density >= fit["threshold"]).astype(np.uint8)
    resized_test = [_resize_mask(m, size) for m in test_masks]
    test_dice = float(np.mean([dice(pred, m) for m in resized_test]))

    return {"threshold": fit["threshold"], "train_dice": fit["dice"], "test_dice": test_dice}


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Normalised (row, col) centroid in [-1, 1] x [-1, 1] (frame centre =
    (0, 0)), or None if *mask* has no foreground pixels."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    row = (ys.mean() / (h - 1)) * 2.0 - 1.0
    col = (xs.mean() / (w - 1)) * 2.0 - 1.0
    return float(row), float(col)


def centre_bias_index(masks: Sequence[np.ndarray], size: Size = (256, 256)) -> Dict[str, float]:
    """Per-dataset positional-predictability summary — no train/test split
    (unlike constant_mask_floor): this describes *masks* as a whole, for
    reporting/dataset-selection purposes.

    Returns:
        constant_floor_dice: best-threshold Dice of the density map
            against the same masks it was built from (self-consistency —
            how much of this dataset's signal is "always roughly here").
        centroid_mean/centroid_std: mean and std of per-mask centroids
            (row, col), normalised to [-1, 1]; low std means masks cluster
            at the same position across images.
        mean_radial_distance: mean Euclidean distance of centroids from
            frame centre (0, 0).
        density_entropy: Shannon entropy (nats) of the density map treated
            as a probability distribution over pixel locations — low
            entropy means foreground mass concentrates in a few locations.
    """
    density = mask_density_map(masks, size)
    fit = _best_threshold_dice(density, masks, size)

    centroids = [_mask_centroid(_resize_mask(m, size)) for m in masks if m.any()]
    centroids_arr = np.array(centroids, dtype=np.float64) if centroids else np.zeros((0, 2))

    if len(centroids_arr) > 0:
        centroid_mean = centroids_arr.mean(axis=0)
        centroid_std = centroids_arr.std(axis=0)
        radial = np.linalg.norm(centroids_arr, axis=1)
        mean_radial_distance = float(radial.mean())
    else:
        centroid_mean = np.zeros(2)
        centroid_std = np.zeros(2)
        mean_radial_distance = 0.0

    total = density.sum()
    if total > 0:
        p = (density / total).ravel()
        p = p[p > 0]
        density_entropy = float(-(p * np.log(p)).sum())
    else:
        density_entropy = 0.0

    return {
        "constant_floor_dice": fit["dice"],
        "centroid_mean": centroid_mean.tolist(),
        "centroid_std": centroid_std.tolist(),
        "mean_radial_distance": mean_radial_distance,
        "density_entropy": density_entropy,
    }


def write_centre_bias_report(
    dataset_name: str,
    masks: Sequence[np.ndarray],
    size: Size = (256, 256),
    out_dir: str = "reports/json/centre_bias",
) -> Dict[str, float]:
    """centre_bias_index(masks, size), written to
    <out_dir>/<dataset_name>.json (atomic tmp-file + os.replace, matching
    stats/__init__.py's write pattern)."""
    result = centre_bias_index(masks, size)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_name}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, path)
    return result
