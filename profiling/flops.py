"""
profiling/flops.py — FLOP counting: fvcore's conv/linear-style count,
cross-checked against a hand-written analytic Conv2d/Linear count. If the
two disagree by more than a configured tolerance, the profiler raises.

fvcore's ``FlopCountAnalysis`` walks registered op handlers (conv, matmul,
batch_norm, upsample, ...) but mixes unit conventions across operator
types (empirically confirmed below, not assumed): its ``conv``/``addmm``
entries are multiply-*accumulate* counts (1 unit per multiply+add pair —
the standard "MACs" convention: a lone ``nn.Conv2d(3, 8, 3, padding=1,
bias=False)`` on a 16x16 input reports exactly
``out_elements * in_channels * kh * kw``, half the true arithmetic-op
count), while its ``batch_norm`` entry is already a true op count (a lone
``nn.BatchNorm2d(8)`` on that same input reports exactly
``2 * n_elements`` — one multiply and one add per element, not halved).
``true_flops_total`` below normalises both conventions to "true FLOPs"
(every multiply and every add counted separately) by doubling only the
MAC-style operators.

The two counts this module compares for agreement are therefore not "same
model, two tools" but:

  - fvcore's conv/linear-style subtotal (the operators fvcore is designed
    to count precisely), doubled to true-FLOPs units.
  - this module's own hand-written Conv2d/Linear formulas (kept
    independent of fvcore's implementation).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# fvcore operator names whose counts are in MAC (multiply-accumulate,
# halved) convention — everything else fvcore reports is treated as
# already being in true-FLOPs units.
_MAC_STYLE_OPS = {"conv", "addmm", "linear", "matmul", "einsum", "bmm", "conv_transpose"}


class FlopsAgreementError(RuntimeError):
    """Raised by check_flops_agreement when the analytic conv/linear total
    and fvcore's (unit-normalised) conv/linear-style total disagree by
    more than the configured tolerance — that indicates a bug in this
    module's hand-written Conv2d/Linear formulas (or an unaccounted-for
    MAC-style layer type)."""


def analytic_flops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Dict[str, int]:
    """Hand-computed true-FLOPs for *model*'s Conv2d/Linear layers (the
    part fvcore's conv/addmm handlers also cover, used for the agreement
    check). Uses forward hooks to capture real output shapes rather than
    reimplementing conv/pooling output-shape arithmetic — robust to any
    stride/padding/dilation/groups combination a registered model uses.

    Args:
        input_shape: (channels, height, width) for one image; batch=1.

    Returns ``{"conv_linear_total", "total"}`` (true FLOPs).
    """
    captured: Dict[nn.Module, Dict[str, object]] = {}
    handles = []

    def _hook(mod, inp, out):
        captured[mod] = {
            "input": inp[0].shape if inp and torch.is_tensor(inp[0]) else None,
            "output": out.shape if torch.is_tensor(out) else None,
        }

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            model(dummy)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    conv_linear_total = 0
    for m, info in captured.items():
        if isinstance(m, nn.Conv2d):
            if info["output"] is None:
                continue
            b, c_out, h_out, w_out = info["output"]
            k_h, k_w = m.kernel_size
            conv_linear_total += int(2 * b * c_out * h_out * w_out * (m.in_channels // m.groups) * k_h * k_w)
        elif isinstance(m, nn.Linear):
            if info["output"] is None:
                continue
            # Every output element (batch * ... * out_features, i.e. the
            # *full* output shape, not excluding out_features) costs
            # in_features multiply-adds.
            n_output_elements = 1
            for d in info["output"]:
                n_output_elements *= d
            conv_linear_total += int(2 * n_output_elements * m.in_features)

    return {
        "conv_linear_total": conv_linear_total,
        "total": conv_linear_total,
    }


def fvcore_flops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Dict[str, object]:
    """fvcore's per-operator breakdown for *model*, normalised to true
    FLOPs (see module docstring).

    Returns ``{"by_operator", "mac_style_true_flops", "other_true_flops",
    "true_flops_total"}``.
    """
    from fvcore.nn import FlopCountAnalysis

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            analysis = FlopCountAnalysis(model, dummy)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            by_op = dict(analysis.by_operator())
    finally:
        model.train(was_training)

    mac_style_raw = sum(v for k, v in by_op.items() if k in _MAC_STYLE_OPS)
    other_true_flops = sum(v for k, v in by_op.items() if k not in _MAC_STYLE_OPS)
    mac_style_true_flops = 2 * mac_style_raw

    return {
        "by_operator": by_op,
        "mac_style_true_flops": mac_style_true_flops,
        "other_true_flops": other_true_flops,
        "true_flops_total": mac_style_true_flops + other_true_flops,
    }


def check_flops_agreement(
    model: nn.Module, input_shape: Tuple[int, int, int], tolerance: float = 0.05
) -> Dict[str, object]:
    """Compares fvcore's conv/linear-style subtotal (true-FLOPs-normalised)
    against this module's hand-written Conv2d/Linear count — the part of
    the model both sides can, in principle, count identically. Raises
    FlopsAgreementError if they disagree by more than *tolerance*.
    """
    analytic = analytic_flops(model, input_shape)
    fv = fvcore_flops(model, input_shape)

    denom = max(fv["mac_style_true_flops"], 1)
    rel_error = abs(analytic["conv_linear_total"] - fv["mac_style_true_flops"]) / denom
    agree = rel_error <= tolerance

    reported_total = analytic["total"] + fv["other_true_flops"]

    result = {
        "fvcore_conv_linear_true_flops": fv["mac_style_true_flops"],
        "fvcore_other_true_flops": fv["other_true_flops"],
        "analytic_conv_linear_total": analytic["conv_linear_total"],
        "reported_total": reported_total,
        "relative_error": rel_error,
        "tolerance": tolerance,
        "agree": agree,
    }
    if not agree:
        raise FlopsAgreementError(
            f"fvcore conv/linear-style total ({fv['mac_style_true_flops']:,}) and analytic "
            f"conv/linear total ({analytic['conv_linear_total']:,}) disagree by "
            f"{rel_error:.1%}, exceeding tolerance {tolerance:.0%}. Detail: {result}"
        )
    return result
