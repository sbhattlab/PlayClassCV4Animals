"""Tests for the tracking postprocessing pipeline."""

import pandas as pd
import pytest

from src.dataset.tracking_postprocessing import (
    align_labels,
    assign_windows,
    check_postprocessing,
    match_bird_ids,
    merge_id_on_switch,
    process_tracks,
    trim,
)

FPS = 25.0


@pytest.fixture
def dummy_tracks():
    """Flat DataFrame: 3 tracking IDs (0, 1, 2) across frames 0–99 at 25 fps."""
    rows = []
    for frame in range(100):
        for oid in range(3):
            rows.append({"frame_idx": frame, "tracking_id": oid, "tracker_score": 0.9})
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_labels():
    """Labels at 1-second intervals spanning the same time range."""
    return pd.DataFrame(
        {
            "video_id": ["v1"] * 4,
            "bird_id": [1, 1, 1, 1],
            "time": [0.0, 1.0, 2.0, 3.0],
            "behav": ["idle", "walk", "peck", "idle"],
            "behav_group": ["rest", "loco", "feed", "rest"],
        }
    )


# ── trim ─────────────────────────────────────────────────────────────


class TestTrim:
    def test_trim_id_scoped(self, dummy_tracks, dummy_labels):
        """
        Input: IDs {0,1,2} × frames [0,99]
        Transform: trim ID 1 [20,40]
        Expected output: ID 1 rows in [20,40] gone, labels untouched.
        """
        trims = [{"from": 20, "to": 40, "id": 1}]
        result, labels_out = trim(dummy_tracks, dummy_labels, trims, FPS)

        # Rows for ID 1 in [20, 40] are gone
        id1_in_range = result.query("tracking_id == 1 and 20 <= frame_idx <= 40")
        assert id1_in_range.empty

        # Other IDs in [20, 40] survive
        others_in_range = result.query("tracking_id != 1 and 20 <= frame_idx <= 40")
        assert len(others_in_range) == 2 * 21  # IDs 0 and 2, 21 frames each

        # Exact row count reduction
        trimmed_count = 21  # frames 20..40 inclusive, 1 ID
        assert len(result) == len(dummy_tracks) - trimmed_count

        # ID-scoped trim must NOT drop labels
        assert len(labels_out) == len(dummy_labels)

    def test_trim_low_score_single_id(self, dummy_tracks, dummy_labels):
        """
        Input: IDs {0,1,2} × frames [0,99]
        Transform: trim ID 2 [60,80]
        Expected output: ID 2 rows in [60,80] gone, others intact, labels untouched.
        """
        trims = [{"from": 60, "to": 80, "id": 2, "cause": "low_score"}]
        result, labels_out = trim(dummy_tracks, dummy_labels, trims, FPS)

        trimmed_frames = 21  # 60..80 inclusive
        assert len(result) == len(dummy_tracks) - trimmed_frames

        # ID 2 gone inside range
        assert result.query("tracking_id == 2 and 60 <= frame_idx <= 80").empty

        # ID 2 survives outside range
        id2_before = result.query("tracking_id == 2 and frame_idx < 60")
        id2_after = result.query("tracking_id == 2 and frame_idx > 80")
        assert len(id2_before) == 60  # frames 0..59
        assert len(id2_after) == 19  # frames 81..99

        # Other IDs untouched everywhere
        for oid in [0, 1]:
            assert len(result.query("tracking_id == @oid")) == 100

        # ID-scoped trim must NOT drop labels
        assert len(labels_out) == len(dummy_labels)

    def test_trim_global(self, dummy_tracks, dummy_labels):
        """
        Input: IDs {0,1,2} × frames [0,99]
        Transform: global trim [20,40]
        Expected output: all rows in [20,40] gone."""
        trims = [{"from": 20, "to": 40}]
        result, _ = trim(dummy_tracks, dummy_labels, trims, FPS)

        in_range = result.query("20 <= frame_idx <= 40")
        assert in_range.empty
        assert len(result) == len(dummy_tracks) - 3 * 21

    def test_trim_drops_labels(self, dummy_tracks, dummy_labels):
        """
        Input: Labels at {0,1,2,3}s
        Transform: global trim [25,75] (1.0–3.0s)
        Expected output: only label at 0.0s survives."""
        trims = [{"from": 25, "to": 75}]
        _, labels_out = trim(dummy_tracks, dummy_labels, trims, FPS)

        # Labels at 1.0, 2.0, 3.0 fall in [1.0, 3.0] → dropped
        assert set(labels_out["time"].tolist()) == {0.0}


# ── merge_id_on_switch ───────────────────────────────────────────────


class TestMergeIdOnSwitch:
    def test_merge_single_remap(self, dummy_tracks):
        """
        Input: IDs {0,1,2} × frames [0,99]
        Transform: remap 0→1 at frame 50
        Expected output: pre-50 ID 0 becomes ID 1, post-50 unchanged.
        """
        remaps = [{"frame": 50, "from": 0, "to": 1}]
        result = merge_id_on_switch(dummy_tracks.copy(), remaps)

        # Pre-frame-50 rows that were ID 0 are now ID 1
        pre = result.query("frame_idx < 50")
        assert (pre["tracking_id"] != 0).all()
        assert len(pre.query("tracking_id == 1")) == 50 + 50  # original ID 1 + remapped

        # ID 0 at/after frame 50 is unchanged
        post_id0 = result.query("frame_idx >= 50 and tracking_id == 0")
        assert len(post_id0) == 50  # frames 50..99

    def test_merge_same_frame_swap(self, dummy_tracks):
        """
        Input: IDs {0,1,2} × frames [0,99]
        Transform: simultaneous remaps 0→1 and 1→2 at frame 50
        Expected output: pre-50 IDs become {0:0, 1:50, 2:100}; orig 0 rows land on 1, not on 2.
        """
        remaps = [
            {"frame": 50, "from": 0, "to": 1},
            {"frame": 50, "from": 1, "to": 2},
        ]
        result = merge_id_on_switch(dummy_tracks.copy(), remaps)

        # Masks computed before any renames (simultaneous):
        #   mask 0→1: selects original ID 0, frame<50 → 50 rows
        #   mask 1→2: selects original ID 1, frame<50 → 50 rows
        # After applying both:
        #   pre-50 ID 0: 0 (all remapped to 1)
        #   pre-50 ID 1: 50 (from orig 0; orig 1 went to 2)
        #   pre-50 ID 2: 100 (orig 1→2 + orig 2 unchanged)
        pre = result.query("frame_idx < 50")
        assert len(pre.query("tracking_id == 0")) == 0
        assert len(pre.query("tracking_id == 1")) == 50
        assert len(pre.query("tracking_id == 2")) == 100

        # Post-50 rows unchanged: 50 each for IDs 0, 1, 2
        post = result.query("frame_idx >= 50")
        assert len(post.query("tracking_id == 0")) == 50
        assert len(post.query("tracking_id == 1")) == 50
        assert len(post.query("tracking_id == 2")) == 50


# ── match_bird_ids ───────────────────────────────────────────────────


class TestMatchBirdIds:
    def test_match_bird_ids(self, dummy_tracks):
        """
        Input: IDs {0,1,2} × 100 frames
        Transform: match tracking_id 1 → protocol_id 2664
        Expected output: all ID 1 rows become bird_id 2664."""
        matches = [{"tracking_id": 1, "protocol_id": 2664, "frame": 0}]
        result = match_bird_ids(dummy_tracks.copy(), matches)

        assert "bird_id" in result.columns
        assert "tracking_id" not in result.columns
        assert len(result.query("bird_id == 2664")) == 100
        assert result.query("bird_id == 1").empty

    def test_match_bird_ids_frame_verification(self, dummy_tracks):
        """
        Input: IDs {0,1,2} × 100 frames
        Transform: match tracking_id 99 at frame 0, but ID 99 does not exist at that frame
        Expected output: no rename applied, column still renamed to bird_id.
        """
        matches = [{"tracking_id": 99, "protocol_id": 2664, "frame": 0}]
        result = match_bird_ids(dummy_tracks.copy(), matches)

        # Column renamed but values unchanged
        assert "bird_id" in result.columns
        assert result.query("bird_id == 2664").empty
        assert len(result.query("bird_id == 1")) == 100


# ── check_postprocessing ────────────────────────────────────────────


class TestCheckPostprocessing:
    def test_valid(self):
        """Entries with all fields filled → no exception."""
        entries = [
            {"type": "trim", "from": 0, "to": 10, "id": 1, "cause": "overlap"},
            {"type": "id_switch", "frame": 50, "from": 0, "to": 1},
            {"type": "id_match", "protocol_id": 2664, "tracking_id": 1, "frame": 0},
        ]
        check_postprocessing(entries)  # should not raise

    def test_null_id_switch_to(self):
        """id_switch entry with to=null → ValueError raised."""
        entries = [{"type": "id_switch", "frame": 50, "from": 0, "to": None}]
        with pytest.raises(ValueError, match="unfilled"):
            check_postprocessing(entries)

    def test_null_id_match_tracking_id(self):
        """id_match entry with tracking_id=null → ValueError raised."""
        entries = [
            {"type": "id_match", "protocol_id": 2664, "tracking_id": None, "frame": 0}
        ]
        with pytest.raises(ValueError, match="unfilled"):
            check_postprocessing(entries)


# ── process_tracks (integration) ────────────────────────────────────


class TestProcessTracks:
    def test_integration(self, dummy_tracks, dummy_labels):
        """
        Input: IDs {0,1,2} × [0,99]
        Transform: trim ID 2 [80,99] + remap 0→1@50 + match 1→2664
        Output: {2664:150, 0:50, 2:80}, labels untouched.
        """
        postprocessing = [
            # Trim ID 2 in [80, 99]
            {"type": "trim", "from": 80, "to": 99, "id": 2, "cause": "low_score"},
            # ID switch at frame 50: merge old ID 0 into ID 1
            {"type": "id_switch", "frame": 50, "from": 0, "to": 1},
            # Match surviving ID 1 → protocol_id 2664
            {"type": "id_match", "tracking_id": 1, "protocol_id": 2664, "frame": 0},
        ]
        result, labels = process_tracks(
            dummy_tracks.copy(), dummy_labels.copy(), postprocessing, FPS
        )

        # Trim removed 20 rows (ID 2, frames 80..99)
        trimmed = 20
        assert len(result) == len(dummy_tracks) - trimmed

        # ID 2664 should have: pre-50 orig 0 (50) + all orig 1 (100) = 150 rows
        assert len(result.query("bird_id == 2664")) == 150

        # ID 0 survives only at/after frame 50 (50 rows)
        assert len(result.query("bird_id == 0")) == 50

        # ID 2 survives only in [0, 79] (80 rows, since [80, 99] trimmed)
        assert len(result.query("bird_id == 2")) == 80

        # ID-scoped trim → labels untouched
        assert len(labels) == len(dummy_labels)


# ── Fixtures for align_labels / assign_windows ─────────────────────


@pytest.fixture
def dataset_tracks():
    """Post-process_tracks + rename state: 2 birds across 250 frames at 25 fps.

    Bird 1: frames 125–249 (5.0–9.96 s)
    Bird 2: frames 0–249   (0.0–9.96 s)
    """
    rows = []
    for frame in range(125, 250):
        rows.append(
            {"video_id": "v1", "bird_id": 1, "frame_idx": frame, "tracker_score": 0.9}
        )
    for frame in range(250):
        rows.append(
            {"video_id": "v1", "bird_id": 2, "frame_idx": frame, "tracker_score": 0.9}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def dataset_labels():
    """Labels for 2 birds at 5-second intervals."""
    return pd.DataFrame(
        {
            "video_id": ["v1"] * 8,
            "bird_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0],
            "behav": ["idle"] * 8,
        }
    )


FPS_LOOKUP = {"v1": 25.0}


# ── align_labels ───────────────────────────────────────────────────


class TestAlignLabels:
    def test_align_drops_uncovered_labels(self, dataset_tracks, dataset_labels):
        """
        Input: bird 1 tracks frames 125–249 (5.0–9.96s), labels at [0, 5, 10, 15]s
        Transform: align_labels
        Expected output: bird 1 keeps only 5.0s (0s before coverage, 10s and
        15s after t_max=9.96s). Bird 2 tracks frames 0–249 (0.0–9.96s), keeps
        0s and 5s (10s and 15s after t_max=9.96s).
        """
        result = align_labels(dataset_tracks, dataset_labels, FPS_LOOKUP)

        bird1 = result.query("bird_id == 1")
        assert set(bird1["time"].tolist()) == {5.0}

        bird2 = result.query("bird_id == 2")
        assert set(bird2["time"].tolist()) == {0.0, 5.0}

    def test_align_drops_unmatched_birds(self, dataset_tracks, dataset_labels):
        """
        Input: labels for bird 99 (no tracks exist)
        Transform: align_labels
        Expected output: all bird 99 labels dropped.
        """
        extra = pd.DataFrame(
            {
                "video_id": ["v1", "v1"],
                "bird_id": [99, 99],
                "time": [0.0, 5.0],
                "behav": ["idle", "idle"],
            }
        )
        labels_with_extra = pd.concat([dataset_labels, extra], ignore_index=True)
        result = align_labels(dataset_tracks, labels_with_extra, FPS_LOOKUP)

        assert result.query("bird_id == 99").empty
        # Original birds still present
        assert not result.query("bird_id == 1").empty
        assert not result.query("bird_id == 2").empty


# ── assign_windows ─────────────────────────────────────────────────


class TestAssignWindows:
    def test_window_assignment_basic(self):
        """
        Input: bird 1 tracks frames 0–249 (0–9.96s at 25fps), labels at [5, 10]s
        Transform: assign_windows
        Expected output: frames 0–125 (≤5.0s) → window 0, frames 126–249 (>5.0s) → window 1.
        searchsorted(side="left") maps exact boundary (5.0s = frame 125) to window 0.
        """
        rows = [
            {"video_id": "v1", "bird_id": 1, "frame_idx": f, "tracker_score": 0.9}
            for f in range(250)
        ]
        tracks = pd.DataFrame(rows)
        labels = pd.DataFrame(
            {
                "video_id": ["v1", "v1"],
                "bird_id": [1, 1],
                "time": [5.0, 10.0],
                "behav": ["idle", "walk"],
            }
        )

        tracks_out, labels_out = assign_windows(tracks, labels, FPS_LOOKUP)

        # Frames 0–125 (time 0.0–5.0s) → window 0
        w0 = tracks_out.query("window == 0")
        assert w0["frame_idx"].min() == 0
        assert w0["frame_idx"].max() == 125

        # Frames 126–249 (time 5.04–9.96s) → window 1
        w1 = tracks_out.query("window == 1")
        assert w1["frame_idx"].min() == 126
        assert w1["frame_idx"].max() == 249

        assert len(tracks_out) == 250

    def test_window_labels_sequential(self):
        """
        Input: labels at [5, 10, 15]s for one bird
        Transform: assign_windows
        Expected output: label windows are [0, 1, 2].
        """
        rows = [
            {"video_id": "v1", "bird_id": 1, "frame_idx": f, "tracker_score": 0.9}
            for f in range(400)
        ]
        tracks = pd.DataFrame(rows)
        labels = pd.DataFrame(
            {
                "video_id": ["v1"] * 3,
                "bird_id": [1, 1, 1],
                "time": [5.0, 10.0, 15.0],
                "behav": ["idle", "walk", "peck"],
            }
        )

        _, labels_out = assign_windows(tracks, labels, FPS_LOOKUP)

        assert labels_out["window"].tolist() == [0, 1, 2]
