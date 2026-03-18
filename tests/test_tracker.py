"""Smoke tests for SAM3 video tracking inference (GPU required).

Loads the SAM3 video model, runs text-prompted tracking on a short clip,
and verifies that per-frame outputs are produced with expected structure.
Requires CUDA and the test video at
``ext-data/raw/C5G2_Test_1_day_28_1_Camera_5_2025_02_04_11_35_00_2.mp4``.

Usage::

    CUDA_VISIBLE_DEVICES=1 pixi run -e sam3-hf pytest tests/test_tracker_video.py -v
"""

from pathlib import Path

import pytest
import torch

from src.io import load_video_frames_torchcodec as load_video_frames

TEXT = "bird"
START_IDX = 10
N_GROUNDING_FRAMES = 125
VIDEO_PATH = Path("data/video/test_10_sec.mp4")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not VIDEO_PATH.exists(), reason="Test video not found"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    from accelerate import Accelerator

    return Accelerator().device


@pytest.fixture(scope="module")
def video_frames():
    """Load a short clip of video frames."""

    frames = load_video_frames(
        str(VIDEO_PATH), START_IDX, START_IDX + N_GROUNDING_FRAMES
    )
    assert frames is not None and len(frames) > 0, "No frames loaded from video"
    return frames


@pytest.fixture(scope="module")
def model_and_processor(device):
    """Load SAM3 video model and processor (once per module)."""
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

    config = Sam3VideoConfig.from_pretrained("facebook/sam3")
    model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
        device, dtype=torch.bfloat16
    )
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return model, processor


@pytest.fixture(scope="module")
def outputs_per_frame(video_frames, model_and_processor, device):
    """Run text-prompted video tracking and collect per-frame outputs."""
    model, processor = model_and_processor

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=TEXT,
    )

    outputs = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=len(video_frames),
    ):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        outputs[model_outputs.frame_idx] = processed

    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVideoFrameLoading:
    def test_frame_count(self, video_frames):
        """Should load exactly N_GROUNDING_FRAMES frames."""
        assert len(video_frames) == N_GROUNDING_FRAMES

    def test_frame_shape(self, video_frames):
        """Each frame should be a 3-channel (H, W, 3) numpy array."""
        frame = video_frames[0]
        assert frame.ndim == 3
        assert frame.shape[2] == 3


class TestSam3VideoInference:
    def test_all_frames_processed(self, outputs_per_frame, video_frames):
        """Model should produce outputs for every input frame."""
        assert len(outputs_per_frame) == len(video_frames)

    def test_majority_frames_have_detections(self, outputs_per_frame):
        """At least 90% of frames should have at least one detected object."""
        n_with_objects = sum(
            1 for out in outputs_per_frame.values() if len(out["object_ids"]) > 0
        )
        ratio = n_with_objects / len(outputs_per_frame)
        assert ratio >= 0.9, f"Only {ratio:.0%} of frames have detections"

    def test_scores_in_range(self, outputs_per_frame):
        """Detection scores should be in [0, 1]."""
        for out in outputs_per_frame.values():
            for score in out["scores"]:
                assert 0.0 <= score.item() <= 1.0

    def test_consistent_object_ids(self, outputs_per_frame):
        """Object IDs in the first frame should appear in all subsequent frames."""
        sorted_frames = sorted(outputs_per_frame.keys())
        first_ids = set(outputs_per_frame[sorted_frames[0]]["object_ids"].tolist())
        for frame_idx in sorted_frames[1:]:
            frame_ids = set(outputs_per_frame[frame_idx]["object_ids"].tolist())
            assert first_ids.issubset(
                frame_ids
            ), f"Frame {frame_idx} missing IDs: {first_ids - frame_ids}"

    def test_output_keys(self, outputs_per_frame):
        """Each frame output should contain expected keys."""
        expected_keys = {"object_ids", "scores", "boxes", "masks"}
        for out in outputs_per_frame.values():
            assert expected_keys.issubset(set(out.keys()))


class TestSam3VideoOverlay:
    def test_overlay_masks(self, video_frames, outputs_per_frame):
        """overlay_masks should return an RGBA image matching frame dimensions."""
        from PIL import Image

        from src.viz import overlay_masks

        frame_image = Image.fromarray(video_frames[0]).convert("RGB")
        result = overlay_masks(frame_image, outputs_per_frame[0]["masks"])
        assert result.mode == "RGBA"
        assert result.size == frame_image.size
