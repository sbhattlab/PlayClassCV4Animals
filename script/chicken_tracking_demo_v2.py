import json
import os
import shutil
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.track_utils import sample_points_from_masks

# ---------------------------
# Args and small utilities
# ---------------------------


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    group_source = parser.add_mutually_exclusive_group(required=True)
    group_source.add_argument(
        "-i",
        "--video_dir",
        type=str,
        help="Directory of JPEG frames named <frame_index>.jpg (0-based or 1-based).",
    )
    group_source.add_argument(
        "--video-file", type=str, help="Path to a single video file to stream decode."
    )

    parser.add_argument(
        "--decoder",
        type=str,
        choices=("cv2", "torchcodec"),
        default="cv2",
        help="Frame decoder to use when --video-file is provided.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="text queries. Need to be lowercased + end with a dot",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mask_file",
        type=str,
        help="Path to .json file containing masks (optional)",
    )
    parser.add_argument(
        "--fps", type=int, default=25, help="frame rate of the output video"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=("large", "base_plus", "small", "tiny"),
        default="large",
        help="SAM2 model size to be used",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.25, help="Grounding DINO box threshold"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO text threshold",
    )
    parser.add_argument(
        "-o", "--save-dir", type=str, default="./tracking_results", help="Output dir"
    )
    parser.add_argument(
        "--save-masks", action="store_true", help="Save mask JSON per frame"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Additionally save annotated JPG frames",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Frames per chunk")
    parser.add_argument(
        "--chunk-overlap", type=int, default=1, help="Overlaps between chunks"
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        default=True,
        help="Offload frames to CPU inside SAM2 state to reduce VRAM",
    )
    parser.add_argument(
        "--seed-frame-index",
        type=int,
        default=0,
        help="Frame index (global) used for seeding detections on first chunk",
    )
    return parser.parse_args()


def single_mask_to_rle(mask: np.ndarray):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def list_frame_names(video_dir: str) -> List[str]:
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # Sort by numeric stem
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


# ---------------------------
# Decoding helpers (cv2 / torchcodec stub)
# ---------------------------


def write_frames_to_chunk_dirs_cv2(
    video_file: str, tmp_root: str, chunk_size: int, chunk_overlap: int
) -> Tuple[List[str], int, int]:
    """
    Decode video_file with cv2, write per-chunk frames into subdirectories under tmp_root.
    Returns (chunk_dirs, width, height).
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    chunk_dirs = []
    chunk_idx = 0
    frame_idx_global = 0
    frames_written_in_chunk = 0
    cur_chunk_dir = None

    def open_new_chunk_dir(idx: int):
        d = os.path.join(tmp_root, f"chunk_{idx:05d}")
        ensure_dir(d)
        return d

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cur_chunk_dir is None:
            cur_chunk_dir = open_new_chunk_dir(chunk_idx)
            chunk_dirs.append(cur_chunk_dir)
            frames_written_in_chunk = 0

        # Write frame as JPEG into chunk dir
        out_path = os.path.join(cur_chunk_dir, f"{frames_written_in_chunk:05d}.jpg")
        cv2.imwrite(out_path, frame)
        frames_written_in_chunk += 1
        frame_idx_global += 1

        # Roll to next chunk when we hit chunk_size, but copy overlap frames
        if frames_written_in_chunk == chunk_size:
            # Prepare overlap frames for next chunk (we re-read from cap, so we need to store overlap)
            # Simpler approach: for next chunk, we will *re-use* last `chunk_overlap` frames by copying them.
            # Read and write overlap frames now:
            overlap_frames = []
            for _ in range(chunk_overlap):
                ret2, f2 = cap.read()
                if not ret2:
                    break
                overlap_frames.append(f2)
            # If we read any overlap, create next chunk and write overlap frames at its beginning
            if overlap_frames:
                chunk_idx += 1
                cur_chunk_dir = open_new_chunk_dir(chunk_idx)
                chunk_dirs.append(cur_chunk_dir)
                frames_written_in_chunk = 0
                # Write overlap frames as first frames in new chunk
                for f in overlap_frames:
                    out_path = os.path.join(
                        cur_chunk_dir, f"{frames_written_in_chunk:05d}.jpg"
                    )
                    cv2.imwrite(out_path, f)
                    frames_written_in_chunk += 1
                    frame_idx_global += 1
            else:
                cur_chunk_dir = None  # no more frames

    cap.release()
    return chunk_dirs, width, height


# Placeholder for torchcodec. If you want to wire it up later, replace decoding logic above.
def write_frames_to_chunk_dirs_torchcodec(
    video_file: str, tmp_root: str, chunk_size: int, chunk_overlap: int
) -> Tuple[List[str], int, int]:
    """
    STUB: You can integrate torchcodec VideoDecoder here if desired.
    For now, we just delegate to cv2 implementation to keep things stable.
    """
    return write_frames_to_chunk_dirs_cv2(
        video_file, tmp_root, chunk_size, chunk_overlap
    )


# ---------------------------
# Core pipeline
# ---------------------------


@torch.no_grad()
def main():
    args = parse_args()

    # Fast, safe matmul on Ampere+
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Mixed precision
    torch.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
    ).__enter__()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init models (load once, reuse across chunks)
    sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{args.size}.pt"
    model_cfg_size = args.size[0].split("_")[0].lower()
    if "plus" in args.size:
        model_cfg_size += "+"
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_cfg_size}.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )

    # Make output dir
    ensure_dir(args.save_dir)

    # Prepare chunk sources
    tmp_root = None
    frame_chunks: List[str] = []
    width = height = None

    if args.video_file:
        # Decode the video into per-chunk directories of frames
        tmp_root = tempfile.mkdtemp(prefix="sam2_chunks_")
        if args.decoder == "cv2":
            frame_chunks, width, height = write_frames_to_chunk_dirs_cv2(
                args.video_file, tmp_root, args.chunk_size, args.chunk_overlap
            )
        else:
            frame_chunks, width, height = write_frames_to_chunk_dirs_torchcodec(
                args.video_file, tmp_root, args.chunk_size, args.chunk_overlap
            )
        if len(frame_chunks) == 0:
            raise RuntimeError("No frames decoded from video.")
        output_video_path = os.path.join(
            args.save_dir,
            f"{os.path.splitext(os.path.basename(args.video_file))[0]}_{args.size}.mp4",
        )
    else:
        # Use the provided directory of frames; virtually slice it into chunks
        frame_names = list_frame_names(args.video_dir)
        if len(frame_names) == 0:
            raise RuntimeError("No frames found in --video_dir.")
        # Get dimensions
        first_image_path = os.path.join(args.video_dir, frame_names[0])
        first_image = cv2.imread(first_image_path)
        height, width = first_image.shape[:2]

        # Create virtual chunk paths (no copies): we’ll compute per-chunk index slices
        # For simplicity, we will create temp dirs with symlinks to avoid duplication.
        tmp_root = tempfile.mkdtemp(prefix="sam2_chunks_")
        N = len(frame_names)
        start = 0
        chunk_idx = 0
        while start < N:
            end = min(N, start + args.chunk_size)
            chunk_dir = os.path.join(tmp_root, f"chunk_{chunk_idx:05d}")
            ensure_dir(chunk_dir)
            # Symlink (or copy if symlink not supported) frames into chunk_dir
            for local_i, gi in enumerate(range(start, end)):
                src = os.path.join(args.video_dir, frame_names[gi])
                dst = os.path.join(chunk_dir, f"{local_i:05d}.jpg")
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy(src, dst)
            frame_chunks.append(chunk_dir)

            # Overlap: add next chunk with last `chunk_overlap` frames at its beginning
            if end < N and args.chunk_overlap > 0:
                # Prepare next chunk dir and pre-fill overlap frames
                next_chunk_dir = os.path.join(tmp_root, f"chunk_{chunk_idx + 1:05d}")
                ensure_dir(next_chunk_dir)
                # Write overlap frames as first frames in next chunk
                overlap_count = min(args.chunk_overlap, N - end)
                for k in range(overlap_count):
                    src = os.path.join(args.video_dir, frame_names[end + k])
                    dst = os.path.join(next_chunk_dir, f"{k:05d}.jpg")
                    try:
                        os.symlink(src, dst)
                    except Exception:
                        shutil.copy(src, dst)
                # The loop will fill the rest when we reach the next iteration
            start = end
            chunk_idx += 1

        output_video_path = os.path.join(
            args.save_dir,
            f"{os.path.basename(os.path.normpath(args.video_dir))}_{args.size}.mp4",
        )

    # Create a single video writer for the whole output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, args.fps, (width, height))

    # Globals we maintain across chunks
    ID_TO_OBJECTS: Dict[int, str] = {}
    carryover_path = os.path.join(
        args.save_dir, "carryover.json"
    )  # holds last-frame masks/boxes to re-seed next chunk
    have_carryover = False

    # Work through chunks
    global_frame_counter = 0  # counts frames written to final video
    print(f"[INFO] Processing {len(frame_chunks)} chunks...")
    for chunk_i, chunk_dir in enumerate(frame_chunks):
        # Init per-chunk inference state with streaming-friendly settings
        inference_state = video_predictor.init_state(
            video_path=chunk_dir, offload_video_to_cpu=args.offload_video_to_cpu
        )

        # Gather this chunk’s frame list
        chunk_frames = list_frame_names(chunk_dir)
        # Which frame index (inside chunk) will we seed on?
        # If we have overlap, the first frame of this chunk equals last frame of previous chunk
        # => Perfect for mask re-seed.
        ann_frame_idx = 0

        # PROMPTS for this chunk
        PROMPT_TYPE_FOR_VIDEO = None
        OBJECTS = []

        # If we have carryover from previous chunk, re-add objects with same IDs
        if have_carryover:
            with open(carryover_path, "r", encoding="utf-8") as f:
                carry = json.load(f)
            saved = carry["objects"]  # list of {obj_id, label, bbox, rle}

            # Prefer exact mask prompt on the overlapped frame
            PROMPT_TYPE_FOR_VIDEO = "mask"
            OBJECTS = [str(it["label"]) for it in saved]
            for item in saved:
                obj_id = int(item["obj_id"])
                # Reconstruct mask from RLE
                mask = mask_util.decode(item["rle"]).astype(np.uint8)
                _, _, _ = video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
            # ID_TO_OBJECTS persists across chunks; nothing to change
        else:
            # First chunk: either use provided masks file OR GroundingDINO+SAM2ImagePredictor
            if args.mask_file:
                with open(args.mask_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)["annotations"]
                masks = [mask_util.decode(ann["segmentation"]) for ann in annotations]
                PROMPT_TYPE_FOR_VIDEO = "mask"
                OBJECTS = [str(i) for i in range(len(annotations))]
                for object_id, mask in enumerate(masks, start=1):
                    _ = video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        mask=mask,
                    )
                ID_TO_OBJECTS = dict(enumerate(OBJECTS, start=1))
            else:
                # Run Grounding DINO on seed frame
                seed_img_path = os.path.join(
                    chunk_dir, chunk_frames[args.seed_frame_index]
                )
                image = Image.open(seed_img_path)
                inputs = processor(
                    images=image, text=args.text, return_tensors="pt"
                ).to(device)
                outputs = grounding_model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    target_sizes=[image.size[::-1]],
                )

                input_boxes = results[0]["boxes"].detach().cpu().numpy()
                labels = results[0]["labels"]
                OBJECTS = [str(_) for _ in range(len(labels))]

                if input_boxes.shape[0] == 0:
                    print(
                        f"[WARN] No detections for text='{args.text}' on first chunk. Skipping."
                    )
                    input_boxes = np.zeros((0, 4), dtype=np.float32)
                    PROMPT_TYPE_FOR_VIDEO = "box"
                else:
                    # Refine boxes to masks with SAM2 image predictor
                    image_predictor.set_image(np.array(image.convert("RGB")))
                    all_masks = []
                    for k in range(input_boxes.shape[0]):
                        box_k = np.asarray(input_boxes[k], dtype=np.float32)[None, ...]
                        m_k, s_k, l_k = image_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box_k,
                            multimask_output=False,
                        )
                        # Normalize dims to (H,W)
                        if m_k.ndim == 4:
                            m_k = m_k.squeeze(0).squeeze(0)
                        elif m_k.ndim == 3:
                            m_k = m_k.squeeze(0)
                        all_masks.append(m_k.astype(np.uint8))
                    masks = (
                        np.stack(all_masks, axis=0)
                        if len(all_masks)
                        else np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
                    )
                    PROMPT_TYPE_FOR_VIDEO = "mask" if masks.shape[0] > 0 else "box"

                # Register prompts with same IDs starting at 1
                if PROMPT_TYPE_FOR_VIDEO == "mask" and (len(OBJECTS) > 0):
                    for object_id, mask in enumerate(masks, start=1):
                        _ = video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            mask=mask,
                        )
                else:
                    for object_id, box in enumerate(input_boxes, start=1):
                        _ = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            box=box,
                        )
                ID_TO_OBJECTS = dict(enumerate(OBJECTS, start=1))

        # Propagate and write outputs on-the-fly
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        # We'll save masks of the last UNIQUE frame in this chunk to carry over
        last_unique_frame_idx_local = len(chunk_frames) - 1
        # If this chunk is not the last one and there is overlap, the last frame will also
        # appear as first frame of next chunk. That's what we want for re-seeding.
        carryover_objects = []

        # If this is not the first chunk and we used overlap, avoid writing the first `chunk_overlap` frames
        # to the final video to prevent duplicates.
        frames_to_skip_at_start = (
            args.chunk_overlap if (chunk_i > 0 and args.chunk_overlap > 0) else 0
        )

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in video_predictor.propagate_in_video(inference_state):
            # out_frame_idx is local within chunk
            img_path = os.path.join(chunk_dir, chunk_frames[out_frame_idx])
            img = cv2.imread(img_path)

            # Convert logits to masks
            segments = {
                # int(out_obj_ids[i]): (out_mask_logits[i] > 0.0)
                int(out_obj_ids[i]): np.squeeze(
                    (out_mask_logits[i] > 0.0).cpu().numpy()
                ).astype(np.uint8)
                for i in range(len(out_obj_ids))
            }

            # Build detections for visualization
            object_ids = list(segments.keys())
            if len(object_ids) == 0:
                # Write original frame if no detections
                if out_frame_idx >= frames_to_skip_at_start:
                    video_writer.write(img)
                    if args.save_images:
                        cv2.imwrite(
                            os.path.join(
                                args.save_dir,
                                f"annotated_frame_{global_frame_counter:05d}.jpg",
                            ),
                            img,
                        )
                    global_frame_counter += 1
                continue

            masks_np = np.stack([segments[i] for i in object_ids], axis=0)  # (n, h, w)
            xyxy = sv.mask_to_xyxy(masks_np)
            detections = sv.Detections(
                xyxy=xyxy, mask=masks_np, class_id=np.array(object_ids, dtype=np.int32)
            )

            annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
            annotated = label_annotator.annotate(
                annotated,
                detections=detections,
                labels=[ID_TO_OBJECTS.get(i, str(i)) for i in object_ids],
            )
            annotated = mask_annotator.annotate(annotated, detections=detections)

            # Save per-frame JSON if requested
            if args.save_masks:
                mask_rles = [single_mask_to_rle(m) for m in masks_np]
                results = {
                    "image_path": img_path,
                    "annotations": [
                        {"segmentation": rle, "bbox": box}
                        for rle, box in zip(mask_rles, xyxy.tolist())
                    ],
                    "object_ids": object_ids,
                    "labels": [ID_TO_OBJECTS.get(i, str(i)) for i in object_ids],
                }
                with open(
                    os.path.join(args.save_dir, f"{global_frame_counter:05d}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(results, f, indent=2)

            # Write annotated frame unless it's part of the overlapping prefix for non-first chunks
            if out_frame_idx >= frames_to_skip_at_start:
                video_writer.write(annotated)
                if args.save_images:
                    cv2.imwrite(
                        os.path.join(
                            args.save_dir,
                            f"annotated_frame_{global_frame_counter:05d}.jpg",
                        ),
                        annotated,
                    )
                global_frame_counter += 1

            # If this is the last local frame of the chunk, capture carryover data
            if out_frame_idx == last_unique_frame_idx_local:
                # Save masks and boxes for re-seeding next chunk (exact on overlap)
                carryover_objects = []
                for k, obj_id in enumerate(object_ids):
                    carryover_objects.append(
                        {
                            "obj_id": int(obj_id),
                            "label": ID_TO_OBJECTS.get(int(obj_id), str(obj_id)),
                            "bbox": xyxy[k].tolist(),
                            "rle": single_mask_to_rle(masks_np[k]),
                        }
                    )

        # Persist carryover for next chunk (if any)
        if chunk_i < len(frame_chunks) - 1 and len(carryover_objects) > 0:
            with open(carryover_path, "w", encoding="utf-8") as f:
                json.dump({"objects": carryover_objects}, f, indent=2)
            have_carryover = True
        else:
            have_carryover = False
            if os.path.exists(carryover_path):
                os.remove(carryover_path)

        # Free per-chunk state
        del inference_state
        torch.cuda.empty_cache()

    # Finalize
    video_writer.release()
    print(f"[OK] Video saved at {output_video_path}")

    # Cleanup temp data
    if tmp_root and os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
