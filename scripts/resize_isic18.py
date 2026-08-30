"""One-time preprocessing: resize ISIC18's raw (highly variable-resolution)
images/masks to the study's fixed 256x256 training resolution. Safe no-op
w.r.t. results: datasets/augment.py's very first transform is already
A.Resize(256, 256) before any augmentation, so pre-resizing on disk produces
bit-for-bit the same input the training pipeline would have produced anyway
-- it just removes the need to decode/hold full-resolution originals
(some multiple megapixels) at cache/load time.
"""
import os
import cv2

SRC_ROOT = "data/isic18"
DST_ROOT = "data/isic18_256"
SIZE = 256

DIRS = [
    "ISIC2018_Task1-2_Training_Input",
    "ISIC2018_Task1_Training_GroundTruth",
    "ISIC2018_Task1-2_Validation_Input",
    "ISIC2018_Task1_Validation_GroundTruth",
    "ISIC2018_Task1-2_Test_Input",
    "ISIC2018_Task1_Test_GroundTruth",
]

for d in DIRS:
    src_dir = os.path.join(SRC_ROOT, d)
    dst_dir = os.path.join(DST_ROOT, d)
    os.makedirs(dst_dir, exist_ok=True)
    is_mask = "GroundTruth" in d
    files = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith((".jpg", ".png"))]
    print(f"{d}: {len(files)} files -> resizing")
    for i, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if is_mask:
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            resized = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, resized)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"{d}: done")

print("ALL_RESIZE_DONE")
