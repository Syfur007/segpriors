"""
datasets/augment.py — AugmentationPolicy: built once per dataset from
(modality, ds_cfg), wrapping datasets/transforms.py's build_transforms().

Default channel_build_order="post": geometric augmentation runs on
(image, mask) first via Albumentations, then build_channels() regenerates
geometry (XY/Rθ) and colour-derived (YCbCr/random-projection) channels from
*that already-augmented* frame — never the other way around. A
pre-augmentation XY channel would encode stale coordinates once the frame is
flipped/rotated/cropped; regenerating it post-augmentation is the only way
an XY/Rθ channel means what it claims to mean. channel_build_order="pre" is
the deliberate opposite, kept for the C3 ablation (see build_model_input()).

build_model_input() is the actual connection between a run's
channel_mode/channel_build_order config and the data a model sees:
datasets.datamodule wires it in as the Dataset `transform` callable.
transform()/build_input_channels() below remain separate, lower-level
primitives (a bare geometric+normalize Compose, and a bare channel-group
assembler) — build_model_input() is what composes them correctly.

Modality-conditioned op tables (spec §5.4): colour vs. grayscale-ultrasound
vs. grayscale-microscopy get different augmentation intensity defaults
(aggressive brightness/contrast jitter that's reasonable for natural colour
images can push ultrasound speckle or microscopy staining intensity outside
any range a real scanner/slide would ever produce). The *config* schema
(orchestration/schema.py) only exposes the coarse `dataset.modality:
colour|grayscale` distinction the spec fixes at dataset level — the finer
ultrasound-vs-microscopy split used for augmentation intensity is resolved
here from `dataset.name` via `_GRAYSCALE_SUBTYPE`, since it's a fixed fact
about a specific named dataset, not something a two-value config flag can
express. Unrecognised grayscale datasets default to the same conservative
scale as ultrasound rather than the full colour-intensity augmentation set.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np

from .channels import MODE_GROUPS, build_channels_from_groups, modality_effective_channels
from .transforms import _IMAGENET_MEAN, _IMAGENET_STD, build_transforms

MODALITIES = ("colour", "grayscale")

# dataset.name (lowercased) -> grayscale subtype, for augmentation-intensity
# purposes only (channel construction only ever needs the coarse
# colour/grayscale distinction — see modality_effective_channels()).
# No grayscale-microscopy dataset is registered in this repo yet; the
# category exists here so one can be added without widening the config
# schema's modality field.
_GRAYSCALE_SUBTYPE = {
    "busi": "ultrasound",
}

# subtype -> augmentation-intensity scale layered on top of ds_cfg's own
# augmentation.brightness_contrast values. 1.0 = no adjustment.
_BRIGHTNESS_CONTRAST_SCALE = {
    "colour": 1.0,
    "ultrasound": 0.5,
    "microscopy": 0.5,
    "generic-grayscale": 0.5,  # unrecognised grayscale dataset: be conservative
}


class AugmentationPolicy:
    """
    Usage (the one datasets.datamodule actually wires in as a Dataset's
    ``transform``)::

        policy = AugmentationPolicy(modality="colour", ds_cfg=config["dataset"])
        image_tensor, mask = policy.build_model_input(image, mask, train=True)

    ``transform()``/``build_input_channels()`` remain available as
    lower-level primitives (a bare geometric+normalize Compose, and a bare
    channel-group assembler) for callers that need to compose the pipeline
    differently — e.g. attribution/common.py's resolve_group_slices() only
    needs ``effective_groups``, never runs augmentation at all.
    """

    def __init__(self, modality: str, ds_cfg: dict):
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got '{modality}'")
        self.modality = modality

        self.channel_mode = ds_cfg.get("channel_mode", "m1")
        if self.channel_mode not in MODE_GROUPS:
            raise ValueError(f"Unknown channel_mode '{self.channel_mode}'. Known: {sorted(MODE_GROUPS)}")

        requested_order = ds_cfg.get("channel_order")
        self._effective_groups = modality_effective_channels(self.channel_mode, modality)
        if requested_order is not None:
            filtered = [g for g in requested_order if g in self._effective_groups]
            if set(filtered) != set(self._effective_groups):
                raise ValueError(
                    f"channel_order {requested_order} does not cover the effective "
                    f"channel groups {self._effective_groups} for channel_mode="
                    f"'{self.channel_mode}' modality='{modality}'"
                )
            self._effective_groups = filtered

        self._dataset_name = ds_cfg.get("name", "")
        self.channel_build_order = ds_cfg.get("channel_build_order", "post")
        if self.channel_build_order not in ("post", "pre"):
            raise ValueError(
                f"channel_build_order must be 'post' or 'pre', got '{self.channel_build_order}'"
            )

        self._subtype = self._resolve_subtype(modality, ds_cfg.get("name", ""))
        adjusted_cfg = self._apply_modality_overrides(ds_cfg)
        h, w = ds_cfg["img_height"], ds_cfg["img_width"]
        self._train_tf, self._val_tf = build_transforms(h, w, adjusted_cfg)

        self._mean = tuple(adjusted_cfg.get("norm_mean") or _IMAGENET_MEAN)
        self._std = tuple(adjusted_cfg.get("norm_std") or _IMAGENET_STD)
        aug = adjusted_cfg.get("augmentation", {})
        ssr = aug.get("shift_scale_rotate", {})
        bc = aug.get("brightness_contrast", {})
        # Geometric-only (spatial, channel-count-agnostic) ops, used both on
        # the raw RGB frame (channel_build_order="post") and on the full
        # multi-channel stack (channel_build_order="pre") — deliberately
        # excludes RandomBrightnessContrast, which is only ever meaningful
        # applied to RGB pixel values, never to XY/Rθ/YCbCr/random-projection
        # channels.
        self._geo_train_tf = A.Compose([
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=aug.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=aug.get("vertical_flip_p", 0.5)),
            A.RandomRotate90(p=aug.get("random_rotate90_p", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=ssr.get("shift_limit", 0.1),
                scale_limit=ssr.get("scale_limit", 0.1),
                rotate_limit=ssr.get("rotate_limit", 30),
                p=ssr.get("p", 0.5),
                border_mode=0,
            ),
        ])
        self._geo_val_tf = A.Compose([A.Resize(height=h, width=w)])
        self._brightness_tf = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=bc.get("brightness_limit", 0.2),
                contrast_limit=bc.get("contrast_limit", 0.2),
                p=bc.get("p", 0.5),
            ),
        ])

    @staticmethod
    def _resolve_subtype(modality: str, dataset_name: str) -> str:
        if modality == "colour":
            return "colour"
        return _GRAYSCALE_SUBTYPE.get(dataset_name.lower(), "generic-grayscale")

    def _apply_modality_overrides(self, ds_cfg: dict) -> dict:
        scale = _BRIGHTNESS_CONTRAST_SCALE[self._subtype]
        if scale == 1.0:
            return ds_cfg
        adjusted = copy.deepcopy(ds_cfg)
        aug = adjusted.setdefault("augmentation", {})
        bc = aug.setdefault("brightness_contrast", {})
        bc["brightness_limit"] = bc.get("brightness_limit", 0.2) * scale
        bc["contrast_limit"] = bc.get("contrast_limit", 0.2) * scale
        return adjusted

    def transform(self, train: bool):
        """The underlying Albumentations Compose — val/test always get the
        no-augmentation pipeline (build_transforms already enforces this;
        see tests/test_channels.py::test_val_test_no_augment)."""
        return self._train_tf if train else self._val_tf

    @property
    def effective_groups(self) -> List[str]:
        return list(self._effective_groups)

    def effective_channel_count(self) -> int:
        from .channels import CHANNEL_GROUP_SIZES
        return sum(CHANNEL_GROUP_SIZES[g] for g in self._effective_groups)

    def build_input_channels(self, image: np.ndarray) -> np.ndarray:
        """Call this on the frame *after* self.transform(train)(...) has
        already run — see module docstring's ORDER=post note. image is
        (H, W, 3) float in [0, 1]."""
        return build_channels_from_groups(image, self._effective_groups, self._dataset_name)

    def build_model_input(
        self, image: np.ndarray, mask: np.ndarray, train: bool
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """The full channel-mode-aware pipeline: RGB brightness/contrast
        (train only), channel construction ordered per
        channel_build_order, geometric augmentation, per-channel
        normalisation, tensor conversion. This is what
        datasets.datamodule wires in as the Dataset's `transform`.

        val/test (train=False) always takes the "post" path — with no
        random geometric transform (val/test is resize-only), there's
        nothing for pre/post to differ over.

        Args:
            image: (H, W, 3) array — uint8 [0, 255] or already float
                [0, 1], as cv2.imread produces.
            mask: (H, W) array.

        Returns:
            (image_tensor (C, H, W) float32, mask) — mask is returned
            as whatever Albumentations produced (MedicalSegmentationDataset
            does its own mask post-processing).
        """
        import torch

        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.5:  # heuristic: input looked like [0, 255]
            image = image / 255.0

        if train:
            image = self._brightness_tf(image=image)["image"]

        if train and self.channel_build_order == "pre":
            stacked = self.build_input_channels(image)
            augmented = self._geo_train_tf(image=stacked, mask=mask)
            stacked, mask = augmented["image"], augmented["mask"]
        else:
            geo_tf = self._geo_train_tf if train else self._geo_val_tf
            augmented = geo_tf(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            stacked = self.build_input_channels(image)

        stacked = np.asarray(stacked, dtype=np.float32).copy()
        if "rgb" in self._effective_groups:
            # Locate rgb's slice from the actual (possibly caller-reordered
            # via channel_order) group layout — don't assume it's first.
            # Every other channel group is already in a bounded, roughly
            # zero-centred range by construction and left as constructed.
            from .channels import CHANNEL_GROUP_SIZES

            offset = sum(
                CHANNEL_GROUP_SIZES[g] for g in self._effective_groups[: self._effective_groups.index("rgb")]
            )
            mean = np.asarray(self._mean, dtype=np.float32)
            std = np.asarray(self._std, dtype=np.float32)
            stacked[..., offset : offset + 3] = (stacked[..., offset : offset + 3] - mean) / std

        tensor = torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1)))
        return tensor, mask


class PolicyTransform:
    """Adapts AugmentationPolicy.build_model_input to the Albumentations
    Compose calling convention (``transform(image=.., mask=..) ->
    {"image":.., "mask":..}``) that MedicalSegmentationDataset(transform=...)
    expects — the piece datasets.datamodule uses to actually connect a
    run's channel_mode/channel_build_order to the data a model sees.
    """

    def __init__(self, policy: AugmentationPolicy, train: bool):
        self._policy = policy
        self._train = train

    def __call__(self, image, mask, **kwargs):
        image_tensor, mask = self._policy.build_model_input(image, mask, train=self._train)
        return {"image": image_tensor, "mask": mask}
