"""
attribution/ — channel-group occlusion, exact Shapley, Integrated
Gradients (cross-checked against each other via agreement_score),
Seg-Grad-CAM/Seg-XRes-CAM qualitative panels, and the model-parameter/
data-label randomisation sanity checks every saliency output must pass
before a downstream claim may cite it.

Inference-only throughout: nothing in this package builds an optimizer,
calls ``.backward()`` on a loss against ground truth, or updates any model
parameter in place (segcam.py's gradient computation is w.r.t. the
*input*, for one image's heatmap, not a training step; sanity.py's
parameter randomisation works on a ``copy.deepcopy`` of the model, never
the original).
"""
from __future__ import annotations

from .common import compute_training_mean_image, occlude_groups, predict_hard, resolve_group_slices
from .integrated_grads import agreement_score, run_integrated_gradients
from .occlusion import run_channel_group_occlusion
from .sanity import (
    label_randomization_sanity_check,
    parameter_randomization_sanity_check,
    randomize_model_,
)
from .segcam import seg_grad_cam, seg_xres_cam
from .shapley import run_exact_shapley, shapley_values_from_characteristic_function

__all__ = [
    # common
    "resolve_group_slices",
    "compute_training_mean_image",
    "occlude_groups",
    "predict_hard",
    # occlusion / shapley / integrated gradients
    "run_channel_group_occlusion",
    "run_exact_shapley",
    "shapley_values_from_characteristic_function",
    "run_integrated_gradients",
    "agreement_score",
    # qualitative saliency
    "seg_grad_cam",
    "seg_xres_cam",
    # sanity checks
    "parameter_randomization_sanity_check",
    "label_randomization_sanity_check",
    "randomize_model_",
]
