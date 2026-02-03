import torch
from accelerate import Accelerator
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video

CUSTOM_RESOLUTION = 560
FRAMES_TO_TRACK = 250

# Load model and processor
device = Accelerator().device
print(f"Using device: {device}")

print(
    f"Loading model and processor (custom resolution: {CUSTOM_RESOLUTION}x{CUSTOM_RESOLUTION})..."
)
config = Sam3VideoConfig.from_pretrained("facebook/sam3")
config.image_size = CUSTOM_RESOLUTION
model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
    device, dtype=torch.bfloat16
)
processor = Sam3VideoProcessor.from_pretrained(
    "facebook/sam3", size={"height": CUSTOM_RESOLUTION, "width": CUSTOM_RESOLUTION}
)

# Load video frames
print("Loading video...")
video_path = "data/video/test_1_min_560x560.mp4"
video_frames, _ = load_video(video_path)

# Initialize video inference session
print("Initializing video inference session...")
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "bird"
print("Adding text prompt to inference session:", text)
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
outputs_per_frame = {}

print(f"Run propogation on {FRAMES_TO_TRACK} frames...")

# Pass show_progress_bar=True to display a tqdm progress bar.
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=FRAMES_TO_TRACK
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

print("Resetting inference session...")
inference_session.reset_inference_session()

# Access results for a specific frame
FRAME_IDX = 110
single_frame_outputs = outputs_per_frame[FRAME_IDX]
print(f"Detected {len(single_frame_outputs['object_ids'])} objects")
print(f"Object IDs: {single_frame_outputs['object_ids'].tolist()}")
print(f"Scores: {single_frame_outputs['scores'].tolist()}")
print(
    f"Boxes shape (XYXY format, absolute coordinates): {single_frame_outputs['boxes'].shape}"
)
print(f"Masks shape: {single_frame_outputs['masks'].shape}")

print("Per-frame post-processed outputs contain:")
for key in single_frame_outputs.keys():
    print(f"{key}, type: {type(single_frame_outputs.get(key))}")
print("Per-frame (raw) model outputs contain:")
for key, value in model_outputs.items():
    print(f"{key}, type: {type(value)}")

print("Demo complete.")
