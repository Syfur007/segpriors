"""
Generic dataset for image-mask segmentation pairs.

Features
--------
- Accepts either ``(image_dir, mask_dir, filenames)`` or a pre-built
  list of ``(img_path, mask_path)`` pairs.
- Optional integrity validation at construction (``validate=True``):
  confirms every image/mask pair exists, is readable, and has matching
  spatial dimensions.  All broken pairs are reported in a single error
  rather than crashing mid-epoch.
- Optional in-RAM caching (``cache=True``): loads all raw images and
  masks on construction if the estimated footprint is under
  ``cache_size_limit_gb``.  Skipped with a warning when the dataset is
  too large.
- ``__getitem__`` returns ``(image, mask, meta)`` — see METADATA_KEYS
  below. This is a project-wide contract (Phase 3 of
  IMPLEMENTATION_PLAN.md): every dataset, including ClinicDB/ColonDB, not
  just new ones, carries the same ``meta`` shape, so a leakage guard that
  reads ``meta["subject_id"]`` works identically regardless of which
  dataset handler produced a sample.
"""
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

# Keys always present in the ``meta`` dict __getitem__ returns.
METADATA_KEYS = ("subject_id", "source_dataset", "spacing", "artefact_flags")


def _default_subject_id_fn(img_path: str) -> str:
    """Frame-level identity: the image filename stem. The honest default
    for any dataset without a real subject/patient grouping mapping — see
    each handler's ARTEFACT_FLAGS for whether that's a meaningful
    simplification (BUSI/ISIC18: effectively one image per case already) or
    a real limitation (ClinicDB/ColonDB: frame extracts from a handful of
    source videos, so this under-counts how few truly independent subjects
    exist — see datasets/polyp/{clinicdb,colondb}.py's ARTEFACT_FLAGS)."""
    return Path(img_path).stem


class DataIntegrityError(Exception):
    """Raised when one or more image/mask pairs fail the integrity check."""


class MedicalSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir=None,
        mask_dir=None,
        filenames=None,
        pairs=None,
        transform=None,
        validate: bool = False,
        cache: bool = False,
        cache_size_limit_gb: float = 4.0,
        source_dataset: str = "",
        subject_id_fn: Optional[Callable[[str], str]] = None,
        artefact_flags: Optional[dict] = None,
    ):
        self.transform = transform
        # None  → no caching; dict → caching enabled (filled in _prefetch)
        self._cache: dict | None = {} if cache else None

        self.source_dataset = source_dataset
        self._subject_id_fn = subject_id_fn or _default_subject_id_fn
        # Shared by every sample from this dataset instance (dataset-level,
        # not per-image) — e.g. {"frame_level_only_no_video_grouping": True}.
        # Immutable per instance: build a new MedicalSegmentationDataset
        # rather than mutating this dict on a shared instance.
        self.artefact_flags = dict(artefact_flags) if artefact_flags else {}

        if pairs is not None:
            self.pairs = list(pairs)
        else:
            if filenames is None:
                filenames = (
                    sorted(f for f in os.listdir(image_dir)
                           if os.path.isfile(os.path.join(image_dir, f)))
                    if os.path.exists(image_dir) else []
                )
            mask_names_map = {}
            if mask_dir and os.path.exists(mask_dir):
                mask_names_map = {
                    os.path.splitext(f)[0].lower(): f
                    for f in os.listdir(mask_dir)
                }
            self.pairs = []
            for fname in filenames:
                stem = os.path.splitext(fname)[0].lower()
                mask_file = (
                    mask_names_map.get(stem)
                    or mask_names_map.get(f"{stem}_mask")
                    or mask_names_map.get(f"{stem}_gt")
                    or fname
                )
                self.pairs.append((
                    os.path.join(image_dir, fname),
                    os.path.join(mask_dir, mask_file),
                ))

        if validate:
            self._validate()

        if cache and self._cache is not None:
            self._prefetch(cache_size_limit_gb)

    # ------------------------------------------------------------------
    # Integrity validation
    # ------------------------------------------------------------------

    def _validate(self):
        """Check every pair for existence, readability, and spatial alignment."""
        errors = []
        for img_path, mask_path in self.pairs:
            pair_errors = []
            img, mask = None, None

            if not os.path.isfile(img_path):
                pair_errors.append("image missing")
            else:
                img = cv2.imread(img_path)
                if img is None:
                    pair_errors.append("image unreadable")

            if not os.path.isfile(mask_path):
                pair_errors.append("mask missing")
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    pair_errors.append("mask unreadable")

            if img is not None and mask is not None:
                if img.shape[:2] != mask.shape[:2]:
                    pair_errors.append(
                        f"dimension mismatch: image {img.shape[:2]} vs mask {mask.shape[:2]}"
                    )

            if pair_errors:
                errors.append(f"  {img_path}: {', '.join(pair_errors)}")

        if errors:
            raise DataIntegrityError(
                f"Dataset integrity check failed "
                f"({len(errors)}/{len(self.pairs)} broken pairs):\n"
                + "\n".join(errors)
            )
        logger.info(f"Dataset integrity check passed ({len(self.pairs)} pairs).")

    # ------------------------------------------------------------------
    # In-RAM caching
    # ------------------------------------------------------------------

    def _prefetch(self, limit_gb: float, probe_samples: int = 20):
        """Load all raw images/masks into self._cache if within size limit.

        Two independent guards, because either one alone has already failed
        in practice on this project's own datasets (see
        configs/dataset/isic18.yaml's history): ISIC18's raw images vary
        wildly in resolution, so a single pairs[0] probe can badly
        undercount the true average.

        1. The upfront estimate now samples up to *probe_samples* pairs
           spread across the whole dataset (not just index 0) and uses the
           *largest* observed per-pair size, not the first.
        2. Even that can still be wrong (an outlier the sample missed, or —
           as happened on a Kaggle worker — the configured root turning out
           to hold un-resized originals instead of the expected pre-resized
           copy). So the actual load below tracks a running byte total and
           aborts caching the moment it would cross *limit_gb*, falling
           back to disk reads for the rest, instead of continuing until the
           host OOMs.
        """
        if not self.pairs:
            return

        limit_bytes = limit_gb * (1024 ** 3)

        step = max(1, len(self.pairs) // probe_samples)
        sample_indices = range(0, len(self.pairs), step)
        max_bytes_per_pair = 0
        for idx in sample_indices:
            img_path, mask_path = self.pairs[idx]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            pair_bytes = img.nbytes + (mask.nbytes if mask is not None else 0)
            max_bytes_per_pair = max(max_bytes_per_pair, pair_bytes)

        if max_bytes_per_pair == 0:
            logger.warning("Cache: could not read any probe image; caching disabled.")
            self._cache = None
            return

        estimated_gb = max_bytes_per_pair * len(self.pairs) / (1024 ** 3)

        if estimated_gb > limit_gb:
            logger.warning(
                f"Cache: estimated footprint {estimated_gb:.2f} GB "
                f"(worst-case pair across a {len(list(sample_indices))}-sample probe) "
                f"exceeds limit {limit_gb:.2f} GB; caching disabled."
            )
            self._cache = None
            return

        logger.info(
            f"Caching {len(self.pairs)} pairs into RAM "
            f"(~{estimated_gb:.2f} GB estimated)..."
        )
        broken = []
        running_bytes = 0
        for idx, (img_path, mask_path) in enumerate(self.pairs):
            img  = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                # Don't cache a broken pair — leave it out of self._cache so
                # __getitem__ falls through to its non-cached path, which
                # raises a clear FileNotFoundError naming the missing file
                # instead of silently handing None to the transform
                # pipeline (an opaque crash deep inside albumentations,
                # far from the actual cause).
                broken.append(img_path if img is None else mask_path)
                continue

            running_bytes += img.nbytes + mask.nbytes
            if running_bytes > limit_bytes:
                logger.warning(
                    f"Cache: actual footprint exceeded the {limit_gb:.2f} GB "
                    f"limit after {idx}/{len(self.pairs)} pairs — the probe "
                    "estimate undercounted this dataset. Aborting caching; "
                    "falling back to disk reads for every pair instead of "
                    "risking an out-of-memory kill."
                )
                self._cache = None
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._cache[idx] = (img, mask)
        if broken:
            logger.warning(
                f"Cache: {len(broken)} pair(s) could not be read and were "
                f"left uncached (will raise on access): {broken}"
            )
        logger.info("Caching complete.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        if self._cache is not None and idx in self._cache:
            image, mask = self._cache[idx]
        else:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        meta = {
            "subject_id": self._subject_id_fn(img_path),
            "source_dataset": self.source_dataset,
            # () rather than None: torch's default DataLoader collate_fn
            # (as of this repo's torch 1.13 pin) raises on a bare None
            # inside a dict value, but happily collates an empty sequence
            # into an empty list — use () here, not None, for "no spacing
            # metadata available" (e.g. every dataset in this repo so far:
            # plain PNG/JPG, not DICOM/NIfTI).
            "spacing": (),
            "artefact_flags": self.artefact_flags,
        }

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.max() > 1.0:
            # Binarize rather than dividing unthresholded: some source masks
            # (e.g. ClinicDB) aren't strictly {0, 255} — antialiased/
            # compressed edges leave intermediate grey values, which would
            # otherwise become fractional soft labels concentrated exactly
            # at object boundaries, where HD95/ASD are most sensitive.
            mask = (mask > 127).float()

        return image, mask.float(), meta
