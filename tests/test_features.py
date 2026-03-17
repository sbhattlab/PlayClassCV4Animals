"""Tests for mask feature extraction and label utilities."""

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import pytest

from src.dataset.features import (
    extract_mask_features,
    extract_pairwise_features,
    extract_spatial_features,
    extract_temporal_features,
    summarize_features_by_window,
)
from src.dataset.labels import resolve_dual_groups
from src.dataset.tracking_postprocessing import filter_incomplete_windows

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 100, 100
RECT_SIZE = 10  # 10x10 rectangle masks


def _make_rle_mask(x0, y0, w, h, img_h=IMG_H, img_w=IMG_W):
    """Create a binary rectangle mask and return (counts, size) as RLE."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8, order="F")
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    rle = mask_util.encode(mask)
    return rle["counts"], list(rle["size"])


@pytest.fixture
def dummy_tracks_with_masks():
    """2 videos x 2 birds x 50 frames with synthetic 10x10 rectangle masks.

    Bird 1: mask at (10, 10+frame), shifting down by 1 pixel/frame.
    Bird 2: mask at (50, 50), stationary.
    Window column: 5 windows of 10 frames each.
    """
    rows = []
    for vid in ["v1", "v2"]:
        for frame in range(50):
            window = frame // 10

            # Bird 1: centroid shifts down by 1px/frame
            counts1, size1 = _make_rle_mask(10, 10 + frame, RECT_SIZE, RECT_SIZE)
            rows.append(
                {
                    "video_id": vid,
                    "bird_id": 1,
                    "frame_idx": frame,
                    "bbox": [10, 10 + frame, 10 + RECT_SIZE, 10 + frame + RECT_SIZE],
                    "counts": counts1,
                    "size": size1,
                    "window": window,
                }
            )

            # Bird 2: stationary at (50, 50)
            counts2, size2 = _make_rle_mask(50, 50, RECT_SIZE, RECT_SIZE)
            rows.append(
                {
                    "video_id": vid,
                    "bird_id": 2,
                    "frame_idx": frame,
                    "bbox": [50, 50, 50 + RECT_SIZE, 50 + RECT_SIZE],
                    "counts": counts2,
                    "size": size2,
                    "window": window,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------


class TestExtractSpatialFeatures:
    def test_output_columns(self, dummy_tracks_with_masks):
        result = extract_spatial_features(dummy_tracks_with_masks)
        expected = {
            "video_id",
            "bird_id",
            "frame_idx",
            "mask_area",
            "bbox_area",
            "aspect_ratio",
            "centroid_x",
            "centroid_y",
        }
        assert expected == set(result.columns)

    def test_row_count(self, dummy_tracks_with_masks):
        result = extract_spatial_features(dummy_tracks_with_masks)
        assert len(result) == len(dummy_tracks_with_masks)

    def test_mask_area(self, dummy_tracks_with_masks):
        result = extract_spatial_features(dummy_tracks_with_masks)
        # All masks are 10x10 rectangles = 100 pixels
        assert (result["mask_area"] == 100).all()


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


class TestExtractTemporalFeatures:
    def test_first_frame_nan(self, dummy_tracks_with_masks):
        spatial = extract_spatial_features(dummy_tracks_with_masks)
        temporal = extract_temporal_features(spatial)

        # First frame per (video_id, bird_id) should have NaN velocity
        for (vid, bid), group in temporal.groupby(["video_id", "bird_id"]):
            first = group.sort_values("frame_idx").iloc[0]
            assert np.isnan(first["velocity"])

    def test_velocity(self, dummy_tracks_with_masks):
        spatial = extract_spatial_features(dummy_tracks_with_masks)
        temporal = extract_temporal_features(spatial)

        # Bird 1 shifts down by 1px/frame → velocity ≈ 1.0
        bird1 = temporal.query("bird_id == 1 and video_id == 'v1'").sort_values(
            "frame_idx"
        )
        non_first = bird1.iloc[1:]
        np.testing.assert_allclose(non_first["velocity"].values, 1.0, atol=0.01)

        # Bird 2 is stationary → velocity ≈ 0.0
        bird2 = temporal.query("bird_id == 2 and video_id == 'v1'").sort_values(
            "frame_idx"
        )
        non_first2 = bird2.iloc[1:]
        np.testing.assert_allclose(non_first2["velocity"].values, 0.0, atol=0.01)

    def test_no_cross_video(self, dummy_tracks_with_masks):
        spatial = extract_spatial_features(dummy_tracks_with_masks)
        temporal = extract_temporal_features(spatial)

        # First frame of EACH video group should have NaN (no carry-over)
        for (vid, bid), group in temporal.groupby(["video_id", "bird_id"]):
            first = group.sort_values("frame_idx").iloc[0]
            assert np.isnan(first["velocity"]), (
                f"Expected NaN velocity at first frame for ({vid}, {bid})"
            )


# ---------------------------------------------------------------------------
# Pairwise features
# ---------------------------------------------------------------------------


class TestExtractPairwiseFeatures:
    def test_distances(self, dummy_tracks_with_masks):
        spatial = extract_spatial_features(dummy_tracks_with_masks)
        pairwise = extract_pairwise_features(spatial)

        # At frame 0, bird 1 centroid ≈ (15, 15), bird 2 centroid ≈ (55, 55)
        # Distance ≈ sqrt(40^2 + 40^2) ≈ 56.57
        f0_b1 = pairwise.query(
            "video_id == 'v1' and bird_id == 1 and frame_idx == 0"
        ).iloc[0]
        expected_dist = np.sqrt(40**2 + 40**2)
        assert abs(f0_b1["min_dist_to_other"] - expected_dist) < 1.0

    def test_no_cross_video(self, dummy_tracks_with_masks):
        spatial = extract_spatial_features(dummy_tracks_with_masks)
        pairwise = extract_pairwise_features(spatial)

        # nearest_neighbor_id should always be a bird from the same video
        # Since we only have 2 birds per video, nearest neighbor should be the other bird
        for _, row in pairwise.iterrows():
            if not np.isnan(row["nearest_neighbor_id"]):
                assert row["nearest_neighbor_id"] in [1, 2]
                assert row["nearest_neighbor_id"] != row["bird_id"]


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------


class TestExtractMaskFeatures:
    def test_all_columns(self, dummy_tracks_with_masks):
        result = extract_mask_features(dummy_tracks_with_masks)
        expected_cols = {
            "video_id",
            "bird_id",
            "frame_idx",
            # Spatial
            "mask_area",
            "bbox_area",
            "aspect_ratio",
            "centroid_x",
            "centroid_y",
            # Temporal
            "velocity",
            "area_change_rate",
            # Pairwise
            "min_dist_to_other",
            "mean_dist_to_other",
            "nearest_neighbor_id",
        }
        assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Summarize by window
# ---------------------------------------------------------------------------


class TestSummarizeFeaturesByWindow:
    def test_one_row_per_group(self, dummy_tracks_with_masks):
        features = extract_mask_features(dummy_tracks_with_masks)
        features = features.merge(
            dummy_tracks_with_masks[["video_id", "bird_id", "frame_idx", "window"]],
            on=["video_id", "bird_id", "frame_idx"],
        )
        result = summarize_features_by_window(features)

        # 2 videos × 2 birds × 5 windows = 20 rows
        assert len(result) == 20

    def test_n_frames(self, dummy_tracks_with_masks):
        features = extract_mask_features(dummy_tracks_with_masks)
        features = features.merge(
            dummy_tracks_with_masks[["video_id", "bird_id", "frame_idx", "window"]],
            on=["video_id", "bird_id", "frame_idx"],
        )
        result = summarize_features_by_window(features)

        # Each window has 10 frames
        assert (result["n_frames"] == 10).all()


# ---------------------------------------------------------------------------
# resolve_dual_groups
# ---------------------------------------------------------------------------


class TestResolveDualGroups:
    def test_single_group_unchanged(self):
        labels = pd.DataFrame({"behav_group": ["worm", "worm", "locomotor"]})
        result = resolve_dual_groups(labels)
        assert result.loc[0, "behav_label"] == "worm"
        assert result.loc[2, "behav_label"] == "locomotor"

    def test_dual_picks_rarest(self):
        # worm appears 2× total, locomotor appears 10× total
        behav_groups = ["locomotor"] * 10 + ["locomotor, worm"] * 2
        labels = pd.DataFrame({"behav_group": behav_groups})
        result = resolve_dual_groups(labels)

        # The dual entries should resolve to "worm" (rarer)
        dual_rows = result.iloc[10:]
        assert (dual_rows["behav_label"] == "worm").all()

    def test_none_unchanged(self):
        labels = pd.DataFrame({"behav_group": ["none", "locomotor"]})
        result = resolve_dual_groups(labels)
        assert result.loc[0, "behav_label"] == "none"


# ---------------------------------------------------------------------------
# filter_incomplete_windows
# ---------------------------------------------------------------------------

FPS = 25.0
WINDOW_DURATION = 5.0  # seconds
EXPECTED_FRAMES = int(WINDOW_DURATION * FPS)  # 125


def _make_window_data(windows, frames_per_window, vid="v1", bird=1):
    """Build tracks and labels for given window specs.

    Parameters
    ----------
    windows : list[int]
        Window indices.
    frames_per_window : list[int]
        Number of track frames for each window.
    """
    track_rows = []
    label_rows = []
    for win, n_frames in zip(windows, frames_per_window):
        t_start = win * WINDOW_DURATION
        for i in range(n_frames):
            track_rows.append(
                {
                    "video_id": vid,
                    "bird_id": bird,
                    "frame_idx": int(t_start * FPS) + i,
                    "window": win,
                }
            )
        label_rows.append(
            {
                "video_id": vid,
                "bird_id": bird,
                "time": t_start + WINDOW_DURATION,
                "window": win,
            }
        )
    return pd.DataFrame(track_rows), pd.DataFrame(label_rows)


class TestFilterIncompleteWindows:
    def test_drops_sparse_window(self):
        """Window with 10/125 frames (8%) gets dropped at 50% threshold."""
        tracks, labels = _make_window_data([0, 1], [10, EXPECTED_FRAMES])
        fps_lookup = {"v1": FPS}

        tracks_out, labels_out = filter_incomplete_windows(
            tracks, labels, fps_lookup, min_coverage=0.5
        )

        assert set(tracks_out["window"].unique()) == {1}
        assert set(labels_out["window"].unique()) == {1}

    def test_keeps_full_window(self):
        """Window with 125/125 frames survives."""
        tracks, labels = _make_window_data([0], [EXPECTED_FRAMES])
        fps_lookup = {"v1": FPS}

        tracks_out, labels_out = filter_incomplete_windows(
            tracks, labels, fps_lookup, min_coverage=0.5
        )

        assert len(tracks_out) == EXPECTED_FRAMES
        assert len(labels_out) == 1

    def test_keeps_partial_above_threshold(self):
        """Window with 80/125 frames (64%) survives at 50% threshold."""
        tracks, labels = _make_window_data([0], [80])
        fps_lookup = {"v1": FPS}

        tracks_out, labels_out = filter_incomplete_windows(
            tracks, labels, fps_lookup, min_coverage=0.5
        )

        assert len(tracks_out) == 80
        assert len(labels_out) == 1

    def test_labels_filtered_too(self):
        """Labels for dropped windows are removed."""
        tracks, labels = _make_window_data([0, 1, 2], [10, EXPECTED_FRAMES, 5])
        fps_lookup = {"v1": FPS}

        _, labels_out = filter_incomplete_windows(
            tracks, labels, fps_lookup, min_coverage=0.5
        )

        assert set(labels_out["window"].unique()) == {1}

    def test_custom_threshold(self):
        """min_coverage=0.8 drops a 64% window."""
        tracks, labels = _make_window_data([0, 1], [80, EXPECTED_FRAMES])
        fps_lookup = {"v1": FPS}

        tracks_out, _ = filter_incomplete_windows(
            tracks, labels, fps_lookup, min_coverage=0.8
        )

        assert set(tracks_out["window"].unique()) == {1}
