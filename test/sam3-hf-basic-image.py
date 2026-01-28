import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from src.utils import autoselect_torch_device

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


TEXT = "bird"


device = autoselect_torch_device()
logger.info(f"Using device: {device}")

CUDA_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES")
logger.info(f"CUDA_VISIBLE_DEVICES: {CUDA_GPU_ID}")
logger.info(f"Using GPU ID: {CUDA_GPU_ID}")


logger.info("Loading SAM3-HF Image and Processor Model...")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

logger.info("Loading test image...")
image = Image.open("data/test-img/0120.jpg").convert("RGB")


logger.info(f"Prompting model with text: {TEXT})")
inputs = processor(images=image, text="bird", return_tensors="pt").to(device)

logger.info("Running model inference...")
with torch.no_grad():
    outputs = model(**inputs)

logger.info("Model inference completed.")
logger.info("Post-processing outputs for instance segmentation...")
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]

print(f"Found {len(results['masks'])} objects")

# Overlay masks on the image and save
logger.info("Overlaying masks on image...")
output_dir = Path("sandbox/test/sam3-hf-image/")
output_dir.mkdir(parents=True, exist_ok=True)
result_image = overlay_masks(image.copy(), results["masks"])
output_path = output_dir / "result.png"
result_image.save(output_path)
logger.info(f"Saved result image to {output_path}")

logger.info("Test completed successfully.")
