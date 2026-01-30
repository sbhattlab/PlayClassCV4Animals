import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

from src.utils import autoselect_torch_device

TEXT = "bird"


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        # Resize mask to match image dimensions if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = autoselect_torch_device()
logger.info(f"Using device: {device}")

CUDA_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES")
logger.info(f"CUDA_VISIBLE_DEVICES: {CUDA_GPU_ID}")
logger.info(f"Using GPU ID: {CUDA_GPU_ID}")

# Load video frames
logger.info("Loading test video...")
video_frames, _ = load_video("data/video/test_10_sec_560x560.mp4")
if video_frames is None or len(video_frames) == 0:
    raise ValueError("No frames in video.")
logger.info(f"Loaded video with {len(video_frames)} frames")

frame_height, frame_width = np.asarray(video_frames[0]).shape[:2]
image_size = max(frame_height, frame_width)
logger.info(
    "Detected frame dimensions %sx%s; configuring model with image_size=%s",
    frame_height,
    frame_width,
    image_size,
)

logger.info("Loading SAM3-HF Video and Processor Model...")
config = Sam3VideoConfig.from_pretrained("facebook/sam3", image_size=image_size)
model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
    device, dtype=torch.bfloat16
)
processor = Sam3VideoProcessor.from_pretrained(
    "facebook/sam3",
    size={"height": frame_height, "width": frame_width},
)


# Initialize video inference session
logger.info("Initializing video inference session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device=device,
    video_storage_device=device,
    dtype=torch.bfloat16,
)

# Add TEXT prompt to detect and track objects
logger.info(f"Adding text prompt to inference session: {TEXT}")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=TEXT,
)

# Process all frames in the video
logger.info("Processing video frames...")
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session,
    max_frame_num_to_track=len(video_frames),
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

# Access results for a specific frame
logger.info("Accessing results for frame 0")
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(
    f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}"
)
print(f"Masks shape: {frame_0_outputs['masks'].shape}")

# Convert numpy array to PIL Image
output_dir = Path("sandbox/test/sam3-hf-video/")
output_dir.mkdir(parents=True, exist_ok=True)
for i in range(0, len(video_frames), 25):
    frame_image = Image.fromarray(video_frames[i]).convert("RGB")
    frame_image_with_masks = overlay_masks(frame_image, outputs_per_frame[i]["masks"])
    output_img_path = output_dir / f"sam3-hf-video-frame-{i}-masks.png"
    frame_image_with_masks.save(output_img_path)
    logger.info(f"Saved frame {i} with masks overlay to {output_img_path}")
