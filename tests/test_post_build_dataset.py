"""Data integrity tests for built dataset outputs.

These tests require a built dataset at the default dataset directory.
Run after: pixi run -e sam3-hf build_dataset
"""

from pathlib import Path

import pandas as pd
import pytest

from src._config import DEFAULT_DATASET_DIR

TRACKS_PATH = Path(DEFAULT_DATASET_DIR) / "tracks.parquet"

pytestmark = pytest.mark.skipif(
    not TRACKS_PATH.exists(), reason="Built dataset not found"
)


@pytest.fixture
def tracks():
    return pd.read_parquet(TRACKS_PATH)


@pytest.fixture
def window_frame_counts(tracks):
    return tracks.groupby(["video_id", "bird_id", "window"]).size().rename("n_frames")


class TestWindowLengths:
    def test_no_oversized_windows(self, window_frame_counts):
        """No window should exceed 127 frames (125 + fps rounding tolerance)."""
        assert window_frame_counts.max() <= 127

    def test_min_coverage(self, window_frame_counts):
        """No window should have fewer than 50% of expected frames (63)."""
        assert window_frame_counts.min() >= 63

    def test_majority_at_expected_length(self, window_frame_counts):
        """At least 90% of windows should have 125 frames."""
        at_125 = (window_frame_counts == 125).sum()
        assert at_125 / len(window_frame_counts) >= 0.9

    def test_total_window_count(self, window_frame_counts):
        """Dataset should have ~3600 windows (7 videos × 3 birds × ~170 windows)."""
        assert len(window_frame_counts) > 3000
