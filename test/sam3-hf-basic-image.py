import logging
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def autoselect_torch_device():
    """
    Automatically selects the best available PyTorch device:
    - Prioritizes CUDA if available.
    - Fallback to MPS if CUDA is not available.
    - Default to CPU if neither CUDA nor MPS are available.

           torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


TEXT = "bird"


device = autoselect_torch_device()
logger.info(f"Using device: {device}")

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
logger.info("Test completed successfully.")
