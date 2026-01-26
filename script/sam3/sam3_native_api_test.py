import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization


import sam3


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

VIDEO = "../data/test.mp4"

# Build predictor (single GPU). Multi-GPU: pass gpus_to_use=[0,1,...]
predictor = build_sam3_video_predictor(
    gpus_to_use=None,
    async_loading_frames=True,  # suggest enabling async frame loading
    video_loader_type="cv2",
)

# 1) start session
resp = predictor.handle_request(
    request=dict(type="start_session", resource_path=VIDEO)
)
session_id = resp["session_id"]

# 2) add separate text prompts (PCS)
predictor.handle_request(
    request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="chicken")
)
predictor.handle_request(
    request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="bird")
)

# 3) stream-collect all frames (your helper)
outputs_per_frame = propagate_in_video(predictor, session_id)
outputs_per_frame_transformed = prepare_masks_for_visualization(outputs_per_frame)

# 4) close + shutdown
predictor.handle_request(request=dict(type="close_session", session_id=session_id))
predictor.shutdown()

