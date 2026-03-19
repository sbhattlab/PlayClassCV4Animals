"""Tests for LOVO and LOCO val rotation."""

import pytest

from src.classification.model_selection import LOCO, LOVO
from src.dataset.utils import cage_id_from_video_id

ALL_VIDEOS = [
    "C1G1D28", "C1G2D28", "C1G3D28",
    "C2G1D28", "C2G2D28", "C2G3D28",
    "C3G1D28", "C3G2D28", "C3G3D28",
    "C4G1D28", "C4G2D28", "C4G3D28",
    "C5G1D28", "C5G2D28", "C5G3D28",
]
ALL_CAGES = sorted({cage_id_from_video_id(v) for v in ALL_VIDEOS})


class _TestLOO:
    """Base tests for leave-one-out val rotation.

    Subclasses must define a ``folds`` fixture returning ``[(test, val), ...]``.
    """

    def test_val_never_equals_test(self, folds):
        for test, val in folds:
            assert test != val, f"test and val are the same: {test}"

    def test_each_id_is_test_at_most_once(self, folds):
        test_counts = {}
        for test, _ in folds:
            test_counts[test] = test_counts.get(test, 0) + 1
        duplicates = {k: c for k, c in test_counts.items() if c > 1}
        assert not duplicates, f"Used as test more than once: {duplicates}"

    def test_each_id_is_val_at_most_once(self, folds):
        val_counts = {}
        for _, val in folds:
            val_counts[val] = val_counts.get(val, 0) + 1
        duplicates = {k: c for k, c in val_counts.items() if c > 1}
        assert not duplicates, f"Used as val more than once: {duplicates}"

    def test_val_from_next_cage(self, folds):
        """Val comes from the next cage in sorted circular order."""
        for test, val in folds:
            test_cage = cage_id_from_video_id(test)
            val_cage = cage_id_from_video_id(val)
            expected_cage = ALL_CAGES[(ALL_CAGES.index(test_cage) + 1) % len(ALL_CAGES)]
            assert val_cage == expected_cage, (
                f"test={test} -> val={val}, expected cage {expected_cage}"
            )


class TestLOVOValRotation(_TestLOO):
    @pytest.fixture
    def folds(self):
        return list(LOVO().split(ALL_VIDEOS))

    def test_fold_count(self, folds):
        assert len(folds) == len(ALL_VIDEOS)


class TestLOCOValRotation(_TestLOO):
    @pytest.fixture
    def folds(self):
        return list(LOCO().split(ALL_VIDEOS))

    def test_fold_count(self, folds):
        assert len(folds) == len(ALL_CAGES)
