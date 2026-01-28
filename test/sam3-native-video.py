import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Parse GPU arguments or prompt user
# cuda_idx_str = sys.argv[1]
# gpu_to_use = [int(cuda_idx_str)]
# logger.info(f"Received GPU argument from command line: {sys.argv[1]}")

# Set CUDA_VISIBLE_DEVICES to filter available GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx_str
# logger.info(f"gpu_to_use before remapping: {gpu_to_use}")

# After setting CUDA_VISIBLE_DEVICES, remap GPU indices to [0, 1, 2, ...]
# (since CUDA_VISIBLE_DEVICES reindexes the visible GPUs)
# gpu_to_use = [0]  # Always use index 0 since we filtered with CUDA_VISIBLE_DEVICES

# logger.info(f"gpu_to_use after remapping: {gpu_to_use}")

import glob
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sam3
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


CUDA_GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES")
logger.info(f"CUDA_VISIBLE_DEVICES: {CUDA_GPU_ID}")
logger.info(f"Using GPU ID: {CUDA_GPU_ID}")

TEXT = "bird"

logger.info(
    "CUDA VISIBLE DEVICES filters the visible GPUs, so we can *always* use GPU 0."
)
predictor = build_sam3_video_predictor(gpus_to_use=[0])

video_path = Path("data/test_15_sec.mp4")

logger.info("Loading test video...")
video_path_str = str(video_path)
if video_path_str.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path_str)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        logger.info(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()

logger.info("Initialising video inference session...")
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path.resolve().as_posix(),
    )
)
session_id = response["session_id"]

logger.info("Running video promptable concept segmentation...")
frame_idx = 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=TEXT,
    )
)

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)
logger.info("Finished inference.")

outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
logger.info("Prepared masks for viz.")

# Visualize and save every 25th frame to sandbox
logger.info("Visualizing results...")
output_dir = Path("sandbox/test/sam3-native-video/")
output_dir.mkdir(parents=True, exist_ok=True)

vis_frame_stride = 25
plt.close("all")
for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    logger.info(f"Processing frame {frame_idx}...")
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[outputs_per_frame],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
    )

    # Save the current figure to sandbox
    out_path = output_dir / f"frame_{frame_idx:06d}.png"
    try:
        plt.savefig(out_path, bbox_inches="tight", dpi=100)
        logger.info(f"Saved frame {frame_idx} to {os.path.abspath(out_path)}")
    except Exception as e:
        logger.error(f"Error saving frame {frame_idx}: {e}")
    finally:
        plt.close("all")

logger.info(f"Visualization complete. Check the {output_dir} for output frames.")
