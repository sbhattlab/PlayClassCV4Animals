"""Tests for label parsing and bird_info correctness."""

from glob import glob

import pytest

from src._config import DEFAULT_LABEL_DIR, DEFAULT_N_BIRDS
from src.dataset.labels import process_labels
from src.dataset.utils import cage_id_from_video_id, extract_video_id


@pytest.mark.parametrize(
    "dirname, expected",
    [
        ("C1G3_Test_1_day_28_1_Camera_8_2025_02_04_10_59_56_3", "C1G3D28"),
        ("C2G1_Test_2_day_29_2_Camera_4_2025_02_05_09_57_55_1", "C2G1D29"),
        ("C5G2_Test_3_day_37_3_Camera_5_2025_02_11_10_00_00_2", "C5G2D37"),
        ("some_random_directory", None),
        ("C1G3_no_day_info", None),
    ],
)
def test_extract_video_id(dirname, expected):
    assert extract_video_id(dirname) == expected


@pytest.fixture(scope="module")
def parsed_labels():
    label_files = sorted(glob(f"{DEFAULT_LABEL_DIR}/*.xlsx"))
    assert label_files, f"No .xlsx files found in {DEFAULT_LABEL_DIR}"
    return process_labels(label_files)


def test_bird_info_n_birds(parsed_labels):
    """Each video_id must have exactly DEFAULT_N_BIRDS birds."""
    _, bird_info = parsed_labels
    assert bird_info, "bird_info is empty"
    for video_id, birds in bird_info.items():
        assert len(birds) == DEFAULT_N_BIRDS, (
            f"{video_id}: expected {DEFAULT_N_BIRDS} birds, got {len(birds)} — {birds}"
        )


def test_video_id_format(parsed_labels):
    """video_id must follow CxGyDz format (e.g. C1G3D28)."""
    _, bird_info = parsed_labels
    import re

    pattern = re.compile(r"^C\dG\dD\d+$")
    for video_id in bird_info:
        assert pattern.match(video_id), f"Invalid video_id format: {video_id!r}"


def test_no_duplicate_birds_across_groups_same_day(parsed_labels):
    """Within the same cage+day, no bird should appear in multiple groups."""
    _, bird_info = parsed_labels
    from collections import defaultdict

    # Group by (cage, day) — e.g. ("C1", "D28")
    cage_day_birds: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for video_id, birds in bird_info.items():
        cage = cage_id_from_video_id(video_id)
        day = video_id[4:]  # "D28"
        group = video_id[2:4]  # "G3"
        cage_day_birds[(cage, day)][group] = set(birds.keys())

    for (cage, day), groups in cage_day_birds.items():
        all_birds = []
        for group, birds in groups.items():
            all_birds.extend(birds)
        assert len(all_birds) == len(set(all_birds)), (
            f"{cage}{day}: duplicate bird IDs across groups"
        )


def test_birds_stay_within_cage(parsed_labels):
    """Each bird must appear in only one cage across all observation days."""
    _, bird_info = parsed_labels
    from collections import defaultdict

    bird_to_cages: dict[int, set[str]] = defaultdict(set)
    for video_id, birds in bird_info.items():
        cage = cage_id_from_video_id(video_id)
        for bird_id in birds:
            bird_to_cages[bird_id].add(cage)

    for bird_id, cages in bird_to_cages.items():
        assert len(cages) == 1, (
            f"Bird {bird_id} appears in multiple cages: {sorted(cages)}"
        )
