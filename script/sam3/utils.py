"""
Utility functions for processing SAM3 tracking outputs.
"""

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch


def to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.array(x)


def process_tracking_outputs(outputs_per_frame):
    index_tuples = []
    bboxes = []
    counts_list = []
    sizes = []
    scores_list = []

    for frame_idx, proc in outputs_per_frame.items():
        object_ids = to_numpy(proc["object_ids"])
        boxes = to_numpy(proc["boxes"])
        masks = proc["masks"]  # keep lazy until conversion per-item
        scores = to_numpy(proc.get("scores", np.zeros(len(object_ids))))

        for i, oid in enumerate(object_ids):
            # bbox -> list (x1,y1,x2,y2)
            bbox = boxes[i].tolist()

            # mask -> RLE
            mask_item = masks[i]
            # if mask already RLE-like (dict with 'counts'/'size'), use it
            if (
                isinstance(mask_item, dict)
                and "counts" in mask_item
                and "size" in mask_item
            ):
                rle = mask_item
                counts = rle["counts"]
                try:
                    counts = counts.decode("utf-8")
                except Exception:
                    pass
                size = rle["size"]
            else:
                m = to_numpy(mask_item).astype(np.uint8)
                # squeeze singleton channel dimension if present
                if m.ndim == 3 and m.shape[0] in (1,):
                    m = m.squeeze(0)
                # ensure Fortran order required by pycocotools
                rle = mask_util.encode(np.asfortranarray(m))
                counts = rle["counts"]
                try:
                    counts = counts.decode("ascii")
                except Exception:
                    pass
                size = rle["size"]

            score = float(scores[i])
            index_tuples.append((int(frame_idx), int(oid)))
            bboxes.append(bbox)
            counts_list.append(counts)
            sizes.append(size)
            scores_list.append(score)

    mi = pd.MultiIndex.from_tuples(index_tuples, names=["frame_idx", "object_id"])
    df_results = pd.DataFrame(
        {"bbox": bboxes, "counts": counts_list, "size": sizes, "scores": scores_list},
        index=mi,
    )
    return df_results


def create_annotation_callback(outputs_per_frame: dict):
    """
    Creates a callback function for sv.process_video that annotates frames
    using pre-computed SAM3 tracking outputs.
    """
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        # Get outputs for this frame (may be missing for some frames)
        if frame_idx not in outputs_per_frame:
            return frame  # return original frame if no detections

        frame_out = outputs_per_frame[frame_idx]

        # Prepare masks: ensure shape (N, 1, H, W)
        masks_cpu = frame_out["masks"].detach().cpu()
        if masks_cpu.ndim == 3:  # (N, H, W)
            masks_cpu = masks_cpu.unsqueeze(1)  # -> (N, 1, H, W)
        masks_cpu = masks_cpu.to(torch.uint8)

        # Build transformers-style results
        transformers_res = {
            "boxes": frame_out["boxes"].detach().cpu(),
            "masks": masks_cpu,
            "labels": frame_out["object_ids"].detach().cpu(),
            "scores": frame_out["scores"].detach().cpu(),
        }

        # Create id2label mapping
        id2label = {
            int(i): f"id:{int(i)}" for i in transformers_res["labels"].cpu().numpy()
        }

        # Build detections
        detections = sv.Detections.from_transformers(
            transformers_results=transformers_res, id2label=id2label
        )

        # Create labels with ID and confidence
        labels = [
            f"#{int(obj_id)} {confidence:.2f}"
            for obj_id, confidence in zip(
                frame_out["object_ids"].cpu().numpy(), detections.confidence
            )
        ]

        # Apply annotations
        annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated

    return callback


def annotate_video_with_sam3_outputs(
    source_path: str, target_path: str, outputs_per_frame: dict
):
    """
    Process entire video with SAM3 tracking outputs and save annotated version.
    """
    callback = create_annotation_callback(outputs_per_frame)
    sv.process_video(
        source_path=source_path, target_path=target_path, callback=callback
    )
