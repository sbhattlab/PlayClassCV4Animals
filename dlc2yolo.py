#!/usr/bin/env python3
"""
Convert multi-animal DeepLabCut CollectedData_*.h5 (columns: [scorer, individuals, bodyparts, coords])
to Ultralytics YOLO Pose dataset with visibility flags (v in {0,1,2}).
- Writes labels: one line per individual present in an image.
- Computes per-individual bounding boxes from visible keypoints; falls back to full-image when needed.
- Creates data.yaml with kpt_shape=[K,3], keypoint names, and optional flip_idx.

Requirements: pandas, numpy, pillow, pyyaml, tqdm


python dlc2yolo.py \
  --h5 "/mnt/birds/dlc-training/multipath-campy-24-25-prince-2025-07-09/training-datasets/iteration-0/UnaugmentedDataSet_multipath-campy-24-25Jul9/CollectedData_prince.h5" \
  --images-root "/mnt/birds/dlc-training/multipath-campy-24-25-prince-2025-07-09/labeled-data" \
  --out ./yolo_pose_dataset \
  --train-ratio 0.8 \
  --class-name chicken \
  --copy-images


"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# ---------- IO helpers ----------


def load_dlc_h5(h5_path: Path) -> pd.DataFrame:
    df = pd.read_hdf(h5_path)
    # Expect MultiIndex columns: [scorer, individuals, bodyparts, coords]
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 4:
        raise ValueError(
            "Expected DLC columns with 4 levels: [scorer, individuals, bodyparts, coords]."
        )
    return df


def get_levels(df: pd.DataFrame):
    scorers = list(df.columns.get_level_values(0).unique())
    individuals = list(df.columns.get_level_values(1).unique())
    bodyparts = list(df.columns.get_level_values(2).unique())
    coords = list(df.columns.get_level_values(3).unique())
    if not set(["x", "y"]).issubset(set([str(c) for c in coords])):
        raise ValueError("Coord level must contain 'x' and 'y'. Found: %s" % coords)
    return scorers, individuals, bodyparts, coords


def resolve_image_path(img_idx, images_root: Path) -> Path:
    """
    Resolve various index styles:
    - tuple/list parts, e.g. ('labeled-data', 'videoA', 'frame.png')
    - absolute path string
    - relative path under images_root
    - bare filename under images_root
    """
    # 1) If it's a tuple/list, join parts into a Path
    if isinstance(img_idx, (tuple, list)):
        parts = [str(p) for p in img_idx]
        candidate_rel = Path(*parts)

        # Avoid duplicating the "labeled-data" segment if images_root already points there
        try:
            # Normalize both for comparison
            root_name = images_root.name
            if len(parts) > 0 and parts[0] == root_name:
                candidate_rel = Path(*parts[1:])
        except Exception:
            pass

        candidate = images_root / candidate_rel
        if candidate.exists():
            return candidate

        # Also try treating the tuple path as absolute/relative as-is
        if candidate_rel.exists():
            return candidate_rel

        # Fallback: search by basename
        basename = candidate_rel.name
        matches = list(images_root.rglob(basename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Could not resolve image path for tuple '{img_idx}' under '{images_root}'."
        )

    # 2) If it's a Path-like or string
    p = Path(img_idx)
    if p.exists():
        return p

    candidate = images_root / p
    if candidate.exists():
        return candidate

    # Fallback: search by basename
    basename = p.name
    matches = list(images_root.rglob(basename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve image path for '{img_idx}' under '{images_root}'."
    )


def image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # (W, H)


# ---------- DLC → keypoints extraction ----------


def first_non_nan(values: np.ndarray):
    """Pick the first non-NaN from an array (e.g., if multiple scorers exist)."""
    for v in values:
        if not (pd.isna(v) or (isinstance(v, float) and np.isnan(v))):
            return float(v)
    return np.nan


def extract_individual_keypoints(
    df: pd.DataFrame, img_idx, individual: str, bodyparts: list[str]
) -> list[tuple[float, float, int]]:
    """
    Returns a list of (x_norm, y_norm, v) for all bodyparts of the given individual.
    v = 2 if labeled (x,y are finite), else 0 (missing).
    """
    # For x and y, slice scorers freely (slice(None)) and choose the first non-NaN
    kpts = []
    # We do normalization later; here we just return raw x,y and v flags
    for bp in bodyparts:
        xs = df.loc[img_idx, (slice(None), individual, bp, "x")].values
        ys = df.loc[img_idx, (slice(None), individual, bp, "y")].values
        x = first_non_nan(xs)
        y = first_non_nan(ys)
        if np.isfinite(x) and np.isfinite(y):
            kpts.append((x, y, 2))  # labeled & visible
        else:
            kpts.append((0.0, 0.0, 0))  # not labeled in this frame
    return kpts


def any_keypoint_present(kpts: list[tuple[float, float, int]]) -> bool:
    return any(v == 2 for (_, _, v) in kpts)


def normalize_kpts(kpts: list[tuple[float, float, int]], W: int, H: int) -> np.ndarray:
    arr = []
    for x, y, v in kpts:
        x_norm = float(x) / float(W) if W > 0 else 0.0
        y_norm = float(y) / float(H) if H > 0 else 0.0
        arr.append((x_norm, y_norm, v))
    return np.array(arr, dtype=float)  # shape (K, 3)


def bbox_from_kpts(kpts_norm: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute normalized bbox (xc, yc, w, h) from visible keypoints (v==2).
    If none visible, return full-image bbox (0.5, 0.5, 1.0, 1.0).
    """
    visible = kpts_norm[kpts_norm[:, 2] == 2]
    if visible.shape[0] == 0:
        return 0.5, 0.5, 1.0, 1.0
    xs = visible[:, 0]
    ys = visible[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    xc = xmin + w / 2
    yc = ymin + h / 2
    return xc, yc, w, h


# ---------- Label writing ----------


def write_yolo_pose_line(
    label_path: Path,
    cls_id: int,
    bbox: tuple[float, float, float, float],
    kpts_norm: np.ndarray,
):
    xc, yc, w, h = bbox
    flat = []
    for i in range(kpts_norm.shape[0]):
        x, y, v = kpts_norm[i]
        flat.extend([f"{x:.6f}", f"{y:.6f}", str(int(v))])  # v must be integer (0/1/2)
    parts = [str(cls_id), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"] + flat
    with open(label_path, "a", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")


# ---------- YAML helpers ----------


def auto_flip_idx(bodyparts: list[str]) -> list[int] | None:
    """
    Try to infer left-right symmetry indices; return list or None if no pairs found.
    """
    n = len(bodyparts)
    idx_map = list(range(n))
    name_to_idx = {bp.lower(): i for i, bp in enumerate(bodyparts)}
    changed = False
    for i, bp in enumerate(bodyparts):
        name = bp.lower()
        if "left" in name:
            candidate = name.replace("left", "right")
            j = name_to_idx.get(candidate)
            if j is not None:
                idx_map[i] = j
                idx_map[j] = i
                changed = True
        elif "right" in name:
            candidate = name.replace("right", "left")
            j = name_to_idx.get(candidate)
            if j is not None:
                idx_map[i] = j
                idx_map[j] = i
                changed = True
    return idx_map if changed else None


def write_data_yaml(
    out_root: Path,
    K: int,
    class_names: list[str],
    bodyparts: list[str],
    flip_idx: list[int] | None,
):
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,  # one class: 'chicken' by default
        "kpt_shape": [K, 3],  # (x,y,visible) triplets
        "keypoints": bodyparts,  # optional, helpful reference
    }
    if flip_idx is not None:
        data["flip_idx"] = flip_idx
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


# ---------- Main conversion ----------


def convert(
    h5_path: Path,
    images_root: Path,
    out_root: Path,
    train_ratio: float = 0.8,
    copy_images: bool = False,
    class_names: list[str] | None = None,
):
    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"
    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    df = load_dlc_h5(h5_path)
    scorers, individuals, bodyparts, coords = get_levels(df)
    K = len(bodyparts)
    if class_names is None:
        class_names = ["chicken"]

    # train/val split by index order
    image_indices = df.index.tolist()
    n = len(image_indices)
    n_train = int(round(train_ratio * n))
    train_set = set(image_indices[:n_train])

    # Optional flip_idx
    flip_idx = auto_flip_idx(bodyparts)

    # Convert each image
    for img_idx in tqdm(image_indices, desc="Converting"):
        img_path = resolve_image_path(
            img_idx, images_root
        )  # pass the raw index (tuple or string)
        W, H = image_size(img_path)

        # Decide split targets
        if img_idx in train_set:
            img_out = out_images_train / img_path.name
            label_out = out_labels_train / (img_path.stem + ".txt")
        else:
            img_out = out_images_val / img_path.name
            label_out = out_labels_val / (img_path.stem + ".txt")

        # Copy or symlink image
        if copy_images:
            if not img_out.exists():
                shutil.copy2(img_path, img_out)
        else:
            if not img_out.exists():
                try:
                    os.symlink(img_path, img_out)
                except OSError:
                    shutil.copy2(img_path, img_out)  # fallback if symlink not allowed

        # For each individual, write a row if present
        wrote_any = False
        for ind in individuals:
            kpts = extract_individual_keypoints(df, img_idx, ind, bodyparts)
            if not any_keypoint_present(kpts):
                continue  # this individual not present in this frame
            kpts_norm = normalize_kpts(kpts, W, H)
            bbox = bbox_from_kpts(kpts_norm)
            write_yolo_pose_line(label_out, cls_id=0, bbox=bbox, kpts_norm=kpts_norm)
            wrote_any = True

        # If nothing was written, ensure no empty file lingers
        if not wrote_any and label_out.exists():
            label_out.unlink()

    # Write YAML
    write_data_yaml(
        out_root, K=K, class_names=class_names, bodyparts=bodyparts, flip_idx=flip_idx
    )
    print(
        f"\n✅ Done.\nData root: {out_root.resolve()}\nTrain labels: {out_labels_train}\nVal labels: {out_labels_val}\nYAML: {out_root / 'data.yaml'}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Convert DLC multi-animal CollectedData.h5 to Ultralytics YOLO Pose dataset."
    )
    ap.add_argument("--h5", type=str, required=True, help="Path to CollectedData_*.h5")
    ap.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="Root directory to find image files referenced in the h5 index.",
    )
    ap.add_argument("--out", type=str, required=True, help="Output dataset root.")
    ap.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train/val split ratio."
    )
    ap.add_argument(
        "--copy-images", action="store_true", help="Copy images instead of symlinking."
    )
    ap.add_argument(
        "--class-name", type=str, default="chicken", help="Single class name."
    )
    args = ap.parse_args()

    convert(
        h5_path=Path(args.h5),
        images_root=Path(args.images_root),
        out_root=Path(args.out),
        train_ratio=args.train_ratio,
        copy_images=args.copy_images,
        class_names=[args.class_name],
    )


if __name__ == "__main__":
    main()
