"""
Run transformers implementation of SAM3 with chunking (to keep VRAM usage low)
Frame iteration now uses TorchCodec VideoDecoder. Each chunk is annotated into RAM and written at once at the end of the chunk via OpenCV VideoWriter.
"""

import gc
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch
from accelerate import Accelerator
from loguru import logger
from simpler_timer import SimplerTimer

# NEW: TorchCodec imports
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor

try:
    from torchcodec.decoders import set_cuda_backend  # optional
except Exception:
    set_cuda_backend = None

# OpenCV used only for color conversion and writing output video
import cv2

# ---------------------------
# Utilities
# ---------------------------


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_xyxy_pixels(xyxy, frame_shape):
    """
    Ensures boxes are pixel-space xyxy. If they look <= 2.0, treat as normalized and scale.
    """
    if xyxy is None or len(xyxy) == 0:
        return xyxy
    xyxy = xyxy.astype(float)
    h, w = frame_shape[:2]
    if np.nanmax(xyxy) <= 2.0:
        xyxy[:, [0, 2]] *= w
        xyxy[:, [1, 3]] *= h
    return xyxy


def single_mask_to_rle(mask):
    # mask: HxW (uint8 or bool)
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    if "size" in rle and isinstance(rle["size"], (list, tuple)):
        rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]
    return rle


def setup_logger(filename: str, debug: bool = False):
    level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )
    logger.add(
        filename,
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return logger


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Run with flags: CUDA_VISIBLE_DEVICES=1 "
            "PYTORCH_ALLOC_CONF='garbage_collection_threshold:0.6,max_split_size_mb:128'"
        ),
    )
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("text", type=str, help="Prompt text for SAM3")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=5,
        help="Number of chunks to split video into (helps with memory usage)",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    # TorchCodec decode controls
    parser.add_argument(
        "--decode-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="TorchCodec VideoDecoder device (NVDEC on 'cuda' if available).",
    )
    parser.add_argument(
        "--cuda-backend",
        choices=["default", "beta"],
        default="default",
        help="TorchCodec CUDA backend (beta can be faster). Only used when decoding on CUDA.",
    )
    # Output encode
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC for OpenCV VideoWriter (e.g., mp4v, avc1, XVID)",
    )
    return parser.parse_args()


def build_decoder(
    video_path: str, decode_device: str, cuda_backend: str
) -> VideoDecoder:
    want_cuda = (decode_device == "cuda") or (
        decode_device == "auto" and torch.cuda.is_available()
    )

    if want_cuda:
        if cuda_backend == "beta" and set_cuda_backend is not None:
            with set_cuda_backend("beta"):
                try:
                    return VideoDecoder(video_path, device="cuda")
                except Exception as e:
                    logger.warning(
                        f"CUDA/NVDEC decoding failed with backend=beta ({e}); falling back to CPU."
                    )
        else:
            try:
                return VideoDecoder(video_path, device="cuda")
            except Exception as e:
                logger.warning(
                    f"CUDA/NVDEC decoding failed with default backend ({e}); falling back to CPU."
                )

    return VideoDecoder(video_path, device="cpu")


def write_video_chunk_cv2(
    frames_bgr: list[np.ndarray],
    path: Path,
    fps: float,
    size_wh: tuple[int, int],
    fourcc_str: str = "mp4v",
):
    if len(frames_bgr) == 0:
        return
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    w, h = size_wh
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        # Try a fallback
        alt = "avc1" if fourcc_str.lower() != "avc1" else "mp4v"
        logger.warning(f"VideoWriter failed for FOURCC='{fourcc_str}', trying '{alt}'.")
        fourcc = cv2.VideoWriter_fourcc(*alt)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter for output video.")
    for f in frames_bgr:
        writer.write(f)
    writer.release()


# ---------------------------
# Main
# ---------------------------


def main():
    args = parse_args()
    video_path = args.video_path
    num_chunks = args.num_chunks
    text = args.text
    decode_device = args.decode_device
    cuda_backend = args.cuda_backend
    fourcc_str = args.fourcc

    # Output dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(Path(video_path).stem, "tracking_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = (
        output_dir
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sam3_log_{Path(video_path).stem.replace(' ', '_')}.log"
    )
    _logger = setup_logger(log_filename)
    global logger
    logger = _logger

    logger.info(f"Input video path: {video_path}")
    logger.info(f"Number of chunks: {num_chunks}")
    logger.info(f"Query prompt: {text}")
    logger.info(f"Output dir: {output_dir}")

    # Device & dtype
    device = Accelerator().device
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    logger.info(f"Using dtype: {dtype}")

    # Models
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
    model.eval()
    torch.set_grad_enabled(False)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # TorchCodec decoder & metadata
    decoder = build_decoder(video_path, decode_device, cuda_backend)
    meta = decoder.metadata
    total_frames = int(getattr(meta, "num_frames", 0) or 0)
    fps = float(getattr(meta, "average_fps", 0.0) or 0.0)
    width = int(getattr(meta, "width", 0) or 0)
    height = int(getattr(meta, "height", 0) or 0)
    if (width == 0 or height == 0) and total_frames > 0:
        f0 = decoder[0]  # [C,H,W] uint8 RGB
        height, width = int(f0.shape[1]), int(f0.shape[2])
        del f0
    if total_frames <= 0:
        raise RuntimeError(
            "Could not determine total frame count from TorchCodec metadata."
        )
    if fps <= 0.0:
        logger.warning("FPS from metadata not found; defaulting to 30 fps.")
        fps = 30.0
    logger.info(f"Total frames: {total_frames} @ {fps:.2f} fps | {width}x{height}")

    # Chunk boundaries
    chunk_size = total_frames // num_chunks
    boundaries = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_frames
        boundaries.append((start, end))
    logger.info(f"Chunk boundaries: {boundaries}")

    # Supervision annotators (drawing only; not for iteration)
    palette = sv.ColorPalette.DEFAULT
    mask_annotator = sv.MaskAnnotator(color=palette)
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    default_label = text if isinstance(text, str) else "object"

    video_stem = Path(video_path).stem

    # Process chunks
    for chunk_idx, (start_f, end_f) in enumerate(boundaries, start=1):
        logger.info(
            f"\n=== Processing chunk {chunk_idx}/{num_chunks}: frames [{start_f}, {end_f}) ==="
        )

        # Fresh inference session per chunk
        inference_session = processor.init_video_session(
            inference_device=device,
            inference_state_device="cpu",
            processing_device="cpu",
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=dtype,
        )
        inference_session = processor.add_text_prompt(inference_session, text)

        annotated_out = (
            output_dir / f"{video_stem}_annotated_part{chunk_idx}of{num_chunks}.mp4"
        )
        outputs_per_frame: dict[int, dict] = {}

        timer = SimplerTimer()
        frames_annotated_bgr: list[np.ndarray] = []  # buffer frames for this chunk

        total_in_chunk = end_f - start_f
        pbar = tqdm(
            total=total_in_chunk,
            desc=f"Chunk {chunk_idx}/{num_chunks}",
            unit="frames",
            dynamic_ncols=True,
        )

        for current_idx in range(start_f, end_f):
            # Decode frame via TorchCodec: [C,H,W] uint8 RGB on CPU or CUDA
            frame_chw = decoder[current_idx]
            # Convert to HWC RGB on CPU for processor + BGR for drawing
            frame_rgb = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()  # HWC RGB
            frame_bgr = frame_rgb[..., ::-1]  # HWC BGR

            # Prepare per-frame inputs (processor usually on CPU), then move only the tensor to GPU
            inputs = processor(images=frame_rgb, device="cpu", return_tensors="pt")
            frame_tensor = inputs.pixel_values[0].to(
                device, dtype=dtype, non_blocking=True
            )

            with torch.inference_mode():
                model_outputs = model(
                    inference_session=inference_session,
                    frame=frame_tensor,
                    reverse=False,
                )
                po = processor.postprocess_outputs(
                    inference_session,
                    model_outputs,
                    original_sizes=inputs.original_sizes,
                )

            # Extract results
            obj_ids_np = to_numpy(po.get("object_ids"))
            scores_np = to_numpy(po.get("scores"))
            xyxy_np = to_numpy(po.get("boxes"))
            masks_np = to_numpy(po.get("masks"))

            # RLE encoding
            masks_rle = None
            if masks_np is not None:
                m = masks_np
                if m.dtype != np.bool_:
                    m = m > 0.5
                masks_rle = [single_mask_to_rle(mi.astype(np.uint8)) for mi in m]

            # JSON-friendly
            object_ids_py = (
                None if obj_ids_np is None else [int(x) for x in obj_ids_np.tolist()]
            )
            scores_py = (
                None if scores_np is None else [float(x) for x in scores_np.tolist()]
            )
            boxes_py = (
                None
                if xyxy_np is None
                else [[float(v) for v in row] for row in xyxy_np.tolist()]
            )

            prompt_map = po.get("prompt_to_obj_ids", None)
            prompt_to_obj_ids_py = (
                {str(k): [int(v) for v in vals] for k, vals in prompt_map.items()}
                if prompt_map is not None
                else None
            )

            outputs_per_frame[int(current_idx)] = {
                "object_ids": object_ids_py,
                "scores": scores_py,
                "boxes": boxes_py,
                "masks_rle": masks_rle,
                "prompt_to_obj_ids": prompt_to_obj_ids_py,
            }

            # Visualization
            xyxy = ensure_xyxy_pixels(xyxy_np, frame_bgr)
            if xyxy is not None and len(xyxy) > 0:
                n = xyxy.shape[0]
                ids_for_vis = (
                    np.arange(n, dtype=int)
                    if obj_ids_np is None or len(obj_ids_np) != n
                    else obj_ids_np.astype(int)
                )
                scores_for_vis = (
                    np.ones(n, dtype=float)
                    if scores_np is None or len(scores_np) != n
                    else scores_np.astype(np.float32)
                )

                detections = sv.Detections(
                    xyxy=xyxy.astype(np.float32),
                    mask=masks_np,
                    confidence=scores_for_vis,
                    class_id=ids_for_vis,
                    tracker_id=ids_for_vis,
                )
                labels = [
                    f"{default_label} • id={int(oid)} • {float(sc):.2f}"
                    for oid, sc in zip(ids_for_vis, scores_for_vis)
                ]

                annotated = frame_bgr.copy()
                if detections.mask is not None:
                    annotated = mask_annotator.annotate(annotated, detections)
                annotated = box_annotator.annotate(annotated, detections)
                annotated = label_annotator.annotate(
                    annotated, detections, labels=labels
                )
            else:
                annotated = frame_bgr

            # Buffer this annotated frame (BGR, HxWx3, uint8)
            frames_annotated_bgr.append(annotated)

            # Cleanup
            del frame_tensor, model_outputs, po, inputs, frame_chw

            if torch.cuda.is_available() and (current_idx - start_f) % 200 == 0:
                torch.cuda.empty_cache()

            pbar.update(1)

        pbar.close()

        # --- WRITE THE WHOLE CHUNK AT ONCE ---
        write_video_chunk_cv2(
            frames_bgr=frames_annotated_bgr,
            path=annotated_out,
            fps=fps,
            size_wh=(width, height),
            fourcc_str=fourcc_str,
        )

        timer.end()
        avg_sec = timer.recall() / max(1, len(frames_annotated_bgr))
        logger.info(
            f"Average time per frame (incl. decode+infer+annotate): {avg_sec:.3f} secs"
        )
        logger.info(
            f"✅ Chunk {chunk_idx} complete. Wrote {len(frames_annotated_bgr)} frames to: {annotated_out}"
        )

        # ---------------------------
        # Save per-chunk JSON + Parquet
        # ---------------------------
        output_json_p = (
            output_dir / f"{video_stem}_output_part{chunk_idx}of{num_chunks}.json"
        )
        with open(output_json_p, "w", encoding="utf-8") as f:
            json.dump(outputs_per_frame, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote JSON results: {output_json_p}")

        # Build results -> DataFrame -> Parquet
        all_results = {}
        default_class = text if isinstance(text, str) else None

        for frame_idx, frame_dict in outputs_per_frame.items():
            obj_ids = frame_dict.get("object_ids") or []
            scores = frame_dict.get("scores") or []
            boxes = frame_dict.get("boxes") or []
            masks_rle = frame_dict.get("masks_rle")
            prompt_to_obj_ids = frame_dict.get("prompt_to_obj_ids") or {}

            id_to_class = {}
            for prompt, id_list in (
                prompt_to_obj_ids.items() if prompt_to_obj_ids else []
            ):
                for oid in id_list:
                    id_to_class[int(oid)] = prompt

            per_obj = {}
            for i in range(len(obj_ids)):
                oid = int(obj_ids[i])
                score = float(scores[i]) if i < len(scores) else None
                bbox = boxes[i] if i < len(boxes) else None

                rle = None
                if masks_rle is not None and i < len(masks_rle):
                    r = masks_rle[i]
                    size = r.get("size", None)
                    counts = r.get("counts", None)
                    if isinstance(size, (list, tuple)):
                        rle = {
                            "size": [int(size[0]), int(size[1])]
                            if len(size) == 2
                            else list(map(int, size)),
                            "counts": str(counts) if counts is not None else None,
                        }
                    else:
                        rle = {
                            "size": None,
                            "counts": str(counts) if counts is not None else None,
                        }

                cls_name = id_to_class.get(oid, default_class)
                per_obj[oid] = {
                    "segmentation": rle,
                    "bbox": bbox,
                    "class_name": cls_name,
                    "score": score,
                }

            all_results[int(frame_idx)] = per_obj

        records, mi_tuples = [], []
        for frame_idx in sorted(all_results.keys()):
            obj_map = all_results[frame_idx]
            for obj_id in sorted(obj_map.keys()):
                det = obj_map[obj_id]
                rle = det.get("segmentation") or {}
                size = rle.get("size", None)
                counts = rle.get("counts", None)
                bbox = det.get("bbox", None)
                cls = det.get("class_name", None)
                score = det.get("score", None)
                if isinstance(cls, str) and cls.strip() == "":
                    cls = np.nan
                records.append([size, counts, bbox, cls, score])
                mi_tuples.append((int(frame_idx), int(obj_id)))

        if len(records) > 0:
            df = pd.DataFrame(
                records,
                index=pd.MultiIndex.from_tuples(
                    mi_tuples, names=["frame", "object_id"]
                ),
                columns=["size", "counts", "bbox", "class", "score"],
            )
        else:
            df = pd.DataFrame(
                columns=["size", "counts", "bbox", "class", "score"],
                index=pd.MultiIndex.from_tuples([], names=["frame", "object_id"]),
            )

        output_parquet_p = (
            output_dir / f"{video_stem}_output_part{chunk_idx}of{num_chunks}.parquet"
        )
        df.to_parquet(output_parquet_p, index=True, engine="pyarrow")
        logger.info(f"Wrote Parquet: {output_parquet_p}")

        # Teardown (free session + VRAM)
        del inference_session, outputs_per_frame, all_results, df, frames_annotated_bgr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                f"[After chunk {chunk_idx}] VRAM allocated={allocated_gb:.2f} GB | reserved={reserved_gb:.2f} GB"
            )

    logger.info("\n🎉 All chunks processed and saved.")
    logger.info(f"Log file found at: {log_filename}")


if __name__ == "__main__":
    main()
