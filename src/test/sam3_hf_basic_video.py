import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

# from transformers.video_utils import load_video
from src.utils import load_video_frames_range
from src.viz import overlay_masks

TEXT = "bird"
START_IDX = 0
START_IDX = 6879
N_GROUNDING_FRAMES = 125

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

device = Accelerator().device
logger.info(f"Using device: {device}")

CUDA_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES")
logger.info(f"CUDA_VISIBLE_DEVICES: {CUDA_GPU_ID}")
logger.info(f"Using GPU ID: {CUDA_GPU_ID}")

# Load video frames
logger.info(f"Loading {N_GROUNDING_FRAMES} frames from test video...")
video_frames = load_video_frames_range(
    "ext-data/raw/C5G2_Test_1_day_28_1_Camera_5_2025_02_04_11_35_00_2.mp4",
    START_IDX,
    START_IDX + N_GROUNDING_FRAMES,
)

if video_frames is None or len(video_frames) == 0:
    raise ValueError("No frames in video.")
logger.info(f"Loaded video with {len(video_frames)} frames")

logger.info("Loading SAM3-HF Video and Processor Model...")
config = Sam3VideoConfig.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
    device, dtype=torch.bfloat16
)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Initialize video inference session
logger.info("Initializing video inference session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
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

logger.info(f"Processed {len(outputs_per_frame)} frames")

# Access results for a specific frame
# logger.info("Accessing results for frame 0")
# frame_0_outputs = outputs_per_frame[0]
# logger.info(f"Detected {len(frame_0_outputs['object_ids'])} objects")
# logger.info(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
# logger.info(f"Scores: {frame_0_outputs['scores'].tolist()}")
# logger.info(
#     f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}"
# )
# logger.info(f"Masks shape: {frame_0_outputs['masks'].shape}")

# PIL_Image = Image.fromarray(video_frames[0]).convert("RGB")
# overlay_masks(PIL_Image, frame_0_outputs["masks"])

logger.info("Summary of detected objects per frame:")
for frame_idx, outputs in outputs_per_frame.items():
    logger.info(
        f"Frame {frame_idx}: Detected {len(outputs['object_ids'])} objects, Detection confidence scores: {outputs['scores'].tolist()}, Object IDs: {outputs['object_ids'].tolist()}"
    )

# Convert numpy array to PIL Image
output_dir = Path("sandbox/test/sam3-hf-video/")
output_dir.mkdir(parents=True, exist_ok=True)
for i in range(0, len(video_frames), 25):
    frame_image = Image.fromarray(video_frames[i]).convert("RGB")
    frame_image_with_masks = overlay_masks(frame_image, outputs_per_frame[i]["masks"])
    output_img_path = output_dir / f"sam3-hf-video-frame-{i}-masks.png"
    frame_image_with_masks.save(output_img_path)
    logger.info(f"Saved frame {i} with masks overlay to {output_img_path}")
