import logging
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

from script.sam3.utils import autoselect_torch_device


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

logger.info("Loading SAM3-HF Video and Processor Model...")
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
logger.info("Loading test video...")
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)
logger.info(f"Loaded video from {video_url}")

# Initialize video inference session
logger.info("Initializing video inference session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device=device,
    video_storage_device=device,
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "person"
logger.info(f"Adding text prompt to inference session: {text}")
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
logger.info("Processing video frames...")
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=25
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
output_img_path = "sandbox/sam3-hf-video-frame-0-masks.png"
frame_0_image = Image.fromarray(video_frames[0]).convert("RGB")
Path("sandbox").mkdir(exist_ok=True)
frame_0_image_with_masks = overlay_masks(frame_0_image, frame_0_outputs["masks"])
frame_0_image_with_masks.save(output_img_path)
logger.info(f"Saved frame 0 with masks overlay to {output_img_path}")
