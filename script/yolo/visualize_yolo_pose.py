import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------
def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Normalize keys that might be given in different orders
    if "path" not in data:
        raise ValueError("YAML must define 'path' to the dataset root.")
    if "train" not in data or "val" not in data:
        raise ValueError("YAML must define 'train' and 'val' subpaths.")
    if "kpt_shape" not in data:
        raise ValueError("YAML must define 'kpt_shape': [num_keypoints, dims].")
    return data


def yolo_bbox_to_xyxy(bx, by, bw, bh, W, H):
    x1 = (bx - bw / 2.0) * W
    y1 = (by - bh / 2.0) * H
    x2 = (bx + bw / 2.0) * W
    y2 = (by + bh / 2.0) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def clip_point(x, y, W, H):
    return int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))


def parse_label_line(line, kpt_num):
    """
    YOLO pose: class cx cy w h (px1 py1 v1) ... (pxN pyN vN)
    Returns: cls, (cx,cy,w,h), keypoints list [(x,y,v), ...].
    """
    t = line.strip().split()
    if len(t) < 5 + 3 * kpt_num:
        raise ValueError(
            f"Label line has too few values: {len(t)} < expected {5 + 3 * kpt_num}"
        )
    cls = int(float(t[0]))
    cx, cy, w, h = map(float, t[1:5])
    k = list(map(float, t[5 : 5 + 3 * kpt_num]))
    kpts = []
    for i in range(kpt_num):
        x = k[3 * i + 0]
        y = k[3 * i + 1]
        v = int(round(k[3 * i + 2]))
        kpts.append((x, y, v))
    return cls, (cx, cy, w, h), kpts


def default_skeleton(kpt_num):
    """
    If no skeleton is provided in YAML, create a reasonable chain for 5 points:
      0(head) -> 1(centre_left) -> 3(saddle) -> 4(tail)
      and 0(head) -> 2(centre_right) -> 3(saddle)
    For other kpt counts, just connect sequentially.
    """
    if kpt_num == 5:
        return [(0, 1), (1, 3), (3, 4), (0, 2), (2, 3)]
    # fallback: chain
    return [(i, i + 1) for i in range(kpt_num - 1)]


def draw_example(
    img,
    labels,
    kpt_num,
    W,
    H,
    draw_box,
    kpt_names=None,
    skeleton=None,
    radius=4,
    thickness=2,
    text=False,
):
    """
    labels: list of tuples (cls, (cx,cy,w,h), [(x,y,v)*N])
    Draws bboxes, keypoints, and skeleton.
    """
    # Colors
    color_box = (36, 255, 12)  # green
    color_vis = (0, 170, 255)  # orange for v=1 (labeled but not visible)
    color_ok = (0, 255, 0)  # green for v=2 (visible)
    color_miss = (160, 160, 160)  # grey for v=0 (not labeled)
    color_line = (255, 128, 0)  # blue/orange line

    if skeleton is None:
        skeleton = default_skeleton(kpt_num)

    for cls, (cx, cy, w, h), kpts in labels:
        if draw_box:
            x1, y1, x2, y2 = yolo_bbox_to_xyxy(cx, cy, w, h, W, H)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)

        # draw skeleton first (uses current keypoint visibility)
        for a, b in skeleton:
            xa, ya, va = kpts[a]
            xb, yb, vb = kpts[b]
            if va > 0 and vb > 0:
                xa_img, ya_img = clip_point(xa * W, ya * H, W, H)
                xb_img, yb_img = clip_point(xb * W, yb * H, W, H)
                cv2.line(
                    img,
                    (xa_img, ya_img),
                    (xb_img, yb_img),
                    color_line,
                    max(1, thickness - 1),
                )

        # draw keypoints
        for i, (x, y, v) in enumerate(kpts):
            px, py = x * W, y * H
            # warn if OOB
            oob = not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
            px, py = clip_point(px, py, W, H)
            col = color_miss if v == 0 else (color_vis if v == 1 else color_ok)
            cv2.circle(img, (px, py), radius, col, -1 if v == 2 else 2)
            if oob:
                cv2.circle(
                    img, (px, py), radius + 5, (0, 0, 255), 1
                )  # red ring if out-of-bounds
            if text and kpt_names and i < len(kpt_names):
                cv2.putText(
                    img,
                    f"{i}:{kpt_names[i]}",
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return img


def collect_pairs(img_dir, lbl_dir):
    """
    Return list of (image_path, label_path or None)
    """
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths.extend(sorted(Path(img_dir).glob(ext)))
    pairs = []
    for ip in img_paths:
        lp = Path(lbl_dir) / (ip.stem + ".txt")
        pairs.append((ip, lp if lp.exists() else None))
    return pairs


# ----------------------------
# Main
# ----------------------------
def main(args):
    data = load_yaml(args.data)
    root = Path(data["path"])
    splits = args.splits if args.splits else ["train", "val"]
    kpt_shape = data["kpt_shape"]
    if isinstance(kpt_shape, (list, tuple)):
        kpt_num = int(kpt_shape[0])
    else:
        raise ValueError("kpt_shape must be a list like [N,3].")

    kpt_names = data.get("keypoints", None)
    skeleton = data.get("skeleton", None)  # optional in your YAML

    # Resolve split subpaths
    split_to_sub = {"train": data["train"], "val": data["val"]}
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_drawn = 0
    for split in splits:
        img_dir = root / split_to_sub[split]
        lbl_dir = root / split_to_sub[split].replace("images", "labels")
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        pairs = collect_pairs(img_dir, lbl_dir)
        if args.limit > 0:
            pairs = pairs[: args.limit]

        pbar = tqdm(pairs, desc=f"Rendering {split}")
        for img_path, lbl_path in pbar:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue
            H, W = img.shape[:2]

            labels = []
            if lbl_path is not None and lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = [ln for ln in f.readlines() if ln.strip()]
                for ln in lines:
                    try:
                        cls, bbox, kpts = parse_label_line(ln, kpt_num)
                        # Warn if kpts length mismatched
                        if len(kpts) != kpt_num:
                            print(
                                f"[WARN] {lbl_path.name}: keypoint count != kpt_shape[0]"
                            )
                        labels.append((cls, bbox, kpts))
                    except Exception as e:
                        print(f"[WARN] Parse error in {lbl_path.name}: {e}")

            canvas = img.copy()
            canvas = draw_example(
                canvas,
                labels,
                kpt_num,
                W,
                H,
                draw_box=args.draw_box,
                kpt_names=kpt_names,
                skeleton=skeleton,
                radius=args.radius,
                thickness=args.thickness,
                text=args.names,
            )

            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), canvas)
            total_drawn += 1

    print(f"Done. Wrote {total_drawn} visualizations to: {out_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize YOLO-Pose labels on images."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML (with path/train/val/kpt_shape).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val"],
        help="One or more splits to render, e.g., --splits train val",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="vis_images",
        help="Output directory for rendered images.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max images per split (0 = all)."
    )
    parser.add_argument("--draw-box", action="store_true", help="Draw bounding boxes.")
    parser.add_argument(
        "--names",
        action="store_true",
        help="Draw keypoint indices/names next to points.",
    )
    parser.add_argument("--radius", type=int, default=4, help="Keypoint circle radius.")
    parser.add_argument(
        "--thickness", type=int, default=2, help="Line thickness for skeleton/points."
    )
    args = parser.parse_args()
    main(args)
