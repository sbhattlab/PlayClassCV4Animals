"""Tests for LOVO val rotation: cage-aware, no duplicates."""

import pytest

from src.classification.datamodule import select_val_video

ALL_VIDEOS = [
    "C1G1", "C1G2", "C1G3",
    "C2G1", "C2G2", "C2G3",
    "C3G1", "C3G2", "C3G3",
    "C4G1", "C4G2", "C4G3",
    "C5G1", "C5G2", "C5G3",
]


@pytest.fixture
def rotation():
    """Build the full test→val mapping."""
    return {tv: select_val_video(tv, ALL_VIDEOS) for tv in ALL_VIDEOS}


class TestValRotation:
    def test_val_from_different_cage(self, rotation):
        """Val video must never share a cage with the test video."""
        for test_video, val_video in rotation.items():
            assert test_video[:2] != val_video[:2], (
                f"test={test_video} and val={val_video} are from the same cage"
            )

    def test_each_video_is_test_exactly_once(self, rotation):
        """Every video appears as test exactly once."""
        assert sorted(rotation.keys()) == sorted(ALL_VIDEOS)

    def test_each_video_is_val_at_most_once(self, rotation):
        """No video is used as val more than once."""
        val_counts = {}
        for val_video in rotation.values():
            val_counts[val_video] = val_counts.get(val_video, 0) + 1
        duplicates = {v: c for v, c in val_counts.items() if c > 1}
        assert not duplicates, f"Videos used as val more than once: {duplicates}"
