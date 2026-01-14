import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# ============================================================
# Utilities
# ============================================================


def torchcodec_frame_to_rgb_numpy(frame: torch.Tensor) -> np.ndarray:
    # Handle CHW / HWC / grayscale
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    frame = frame.cpu().numpy()
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


# ============================================================
# Environment setup
# ============================================================

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Models
# ============================================================

# ---------- SAM-2 ----------
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# ---------- Grounding DINO ----------
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = (
    AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
)

TEXT_PROMPT = "chicken.bird."  # must be lowercase + end with dot


# ============================================================
# Video
# ============================================================

video_path = "/mnt/birds/rebecca2025/test/video_1_5min.mp4"
decoder = VideoDecoder(video_path)
num_frames = len(decoder)

print(f"Video loaded: {num_frames} frames")


# ============================================================
# Initialize SAM-2 inference state (STREAMING MODE)
# ============================================================

inference_state = video_predictor.init_state(
    video_path=video_path,
    offload_video_to_cpu=True,  # 🔑 prevents GPU OOM
    offload_state_to_cpu=True,  # 🔑 keeps state lightweight
    async_loading_frames=True,  # 🔑 frame streaming
)


# ============================================================
# Step 1 — Initial prompt frame
# ============================================================

ann_frame_idx = 0

frame0 = torchcodec_frame_to_rgb_numpy(decoder[ann_frame_idx])
image_pil = Image.fromarray(frame0)

# ---- Grounding DINO ----
with torch.no_grad():
    inputs = processor(
        images=image_pil,
        text=TEXT_PROMPT,
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

boxes = results[0]["boxes"].cpu().numpy()
labels = results[0]["labels"]

# ---- SAM image predictor ----
image_predictor.set_image(frame0)

masks, scores, logits = image_predictor.predict(
    box=boxes,
    multimask_output=False,
)

if masks.ndim == 4:
    masks = masks.squeeze(1)

# ---- Register objects (POINT-BASED, robust) ----
for obj_id, mask in enumerate(masks, start=1):
    points = sample_points_from_masks(mask[None], num_points=20)[0]
    video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=points,
        labels=np.ones(len(points), dtype=np.int32),
    )


# ============================================================
# Step 2 — Propagate through entire video (STREAMING)
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
# Step 3 — Visualization
# ============================================================

save_dir = Path("tracking_results")
save_dir.mkdir(exist_ok=True)

ID_TO_LABEL = {i + 1: lbl for i, lbl in enumerate(labels)}

for frame_idx, segments in video_segments.items():
    frame = torchcodec_frame_to_rgb_numpy(decoder[frame_idx])
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    obj_ids = list(segments.keys())
    masks = np.stack(list(segments.values()))

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        class_id=np.array(obj_ids, dtype=np.int32),
    )

    img = sv.BoxAnnotator().annotate(img, detections)
    img = sv.LabelAnnotator().annotate(
        img,
        detections,
        labels=[f"chicken {i}" for i in obj_ids],
    )
    img = sv.MaskAnnotator().annotate(img, detections)

    cv2.imwrite(
        str(save_dir / f"frame_{frame_idx:06d}.jpg"),
        img,
    )


# ============================================================
# Step 4 — Write video
# ============================================================

output_video_path = "chicken_tracking.mp4"
create_video_from_images(str(save_dir), output_video_path)

print(f"✅ Tracking complete. Output written to {output_video_path}")
