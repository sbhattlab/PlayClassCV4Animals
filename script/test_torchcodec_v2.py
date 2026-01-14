import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

def torchcodec_frame_to_rgb_numpy(frame: torch.Tensor) -> np.ndarray:
    """
    Converts torchcodec frame tensor to RGB uint8 HWC numpy array.
    Handles CHW / HWC / grayscale safely.
    """
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    # If CHW → HWC
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)

    frame = frame.cpu().numpy()

    # Grayscale → RGB
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)

    return frame


# ============================================================
# Step 1: Environment settings and model initialization
# ============================================================

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- SAM 2 ----------------
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# ---------------- Grounding DINO ----------------
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = (
    AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    .to(device)
    .eval()
)

text_prompt = "car."  # must be lowercase + end with dot

# ============================================================
# Step 2: Video decoding (streaming)
# ============================================================

video_path = "notebooks/videos/car/out.mp4"
decoder = VideoDecoder(video_path)

num_frames = len(decoder)
print(f"Video loaded: {num_frames} frames")

# Initialize SAM-2 video state using video path directly
inference_state = video_predictor.init_state(video_path=video_path)

ann_frame_idx = 0  # frame used for prompting


# ============================================================
# Step 3: Run Grounding DINO + SAM image predictor on 1 frame
# ============================================================

with torch.no_grad():
    # frame = decoder[ann_frame_idx]  # torch.Tensor (H, W, 3), uint8, CPU
    # frame_np = frame.numpy()
    # image_pil = Image.fromarray(frame_np)
    frame = decoder[ann_frame_idx]
    frame_np = torchcodec_frame_to_rgb_numpy(frame)
    image_pil = Image.fromarray(frame_np)

    inputs = processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image_pil.size[::-1]],
    )

# ---------------- SAM image predictor ----------------
image_predictor.set_image(frame_np)

input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

if masks.ndim == 3:
    masks = masks[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)


# ============================================================
# Step 4: Register prompts with SAM-2 video predictor
# ============================================================

PROMPT_TYPE_FOR_VIDEO = "box"  # "point" | "box" | "mask"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"]

if PROMPT_TYPE_FOR_VIDEO == "point":
    all_sample_points = sample_points_from_masks(masks, num_points=10)

    for obj_id, points in enumerate(all_sample_points, start=1):
        labels = np.ones(len(points), dtype=np.int32)
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

elif PROMPT_TYPE_FOR_VIDEO == "box":
    for obj_id, box in enumerate(input_boxes, start=1):
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            box=box,
        )

elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for obj_id, mask in enumerate(masks, start=1):
        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            mask=mask,
        )


# ============================================================
# Step 5: Propagate through video (streaming, GPU-safe)
# ============================================================

video_segments = {}

with torch.no_grad():
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(
        inference_state
    ):
        video_segments[frame_idx] = {
            obj_id: (mask_logits[i] > 0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }


# ============================================================
# Step 6: Visualization (re-decode frames on demand)
# ============================================================

save_dir = Path("./tracking_results")
save_dir.mkdir(exist_ok=True)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    # frame = decoder[frame_idx].numpy()
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = decoder[frame_idx]
    frame_np = torchcodec_frame_to_rgb_numpy(frame)
    img = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    


    object_ids = list(segments.keys())
    masks = np.concatenate(list(segments.values()), axis=0)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        class_id=np.array(object_ids, dtype=np.int32),
    )

    annotated = sv.BoxAnnotator().annotate(img.copy(), detections)
    annotated = sv.LabelAnnotator().annotate(
        annotated,
        detections,
        labels=[ID_TO_OBJECTS[i] for i in object_ids],
    )
    annotated = sv.MaskAnnotator().annotate(annotated, detections)

    cv2.imwrite(
        str(save_dir / f"annotated_frame_{frame_idx:05d}.jpg"),
        annotated,
    )


# ============================================================
# Step 7: Convert frames to video
# ============================================================

output_video_path = "./children_tracking_demo_video.mp4"
create_video_from_images(str(save_dir), output_video_path)
print(f"Wrote results to {output_video_path}")

