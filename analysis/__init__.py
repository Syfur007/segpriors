"""
analysis/ — mechanism analysis: gradient-based Effective Receptive Field
(erf.py), linear Centered Kernel Alignment between representations
(cka.py), per-image failure taxonomy + gallery indexing
(failure_taxonomy.py), and per-dataset positional-predictability
(centre_bias.py).
"""
from __future__ import annotations

from .centre_bias import (
    centre_bias_index,
    constant_mask_floor,
    mask_density_map,
    write_centre_bias_report,
)
from .cka import cka_matrix, flatten_spatial_features, linear_cka
from .erf import compute_erf, erf_radius
from .failure_taxonomy import FAILURE_CATEGORIES, classify_failure, failure_counts, gallery_indices

__all__ = [
    "compute_erf",
    "erf_radius",
    "linear_cka",
    "cka_matrix",
    "flatten_spatial_features",
    "FAILURE_CATEGORIES",
    "classify_failure",
    "failure_counts",
    "gallery_indices",
    "mask_density_map",
    "constant_mask_floor",
    "centre_bias_index",
    "write_centre_bias_report",
]
