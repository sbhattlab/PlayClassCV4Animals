"""Sklearn-style cross-validation splitters for LOVO/LOCO."""

from abc import ABC, abstractmethod

from src.dataset.utils import cage_id_from_video_id


class LOO(ABC):
    """Abstract leave-one-out cross-validation splitter.

    Subclasses implement ``split(video_ids)`` which yields
    ``(test_id, val_id)`` pairs, one per fold.
    """

    @abstractmethod
    def split(self, video_ids, shuffle=False, random_state=None):
        """Yield (test_id, val_id) for each fold.

        Parameters
        ----------
        video_ids : list[str]
            All video IDs in the dataset (e.g. ``["C1G1D28", ...]``).
        shuffle : bool
            If True, randomize val selection instead of deterministic rotation.
        random_state : int | None
            Seed for reproducibility when ``shuffle=True``.
        """

    @abstractmethod
    def get_groups(self, video_ids):
        """Return per-sample group labels for splitting."""

    @abstractmethod
    def _select_val_circular(self, test_id, all_ids):
        """Pick val ID for a given test ID (deterministic rotation)."""


class LOCO(LOO):
    """Leave-One-Cage-Out cross-validation."""

    def _select_val_circular(self, test_cage, all_cages):
        """Pick val as the next cage in sorted circular order."""
        idx = all_cages.index(test_cage)
        return all_cages[(idx + 1) % len(all_cages)]

    def split(self, video_ids, shuffle=False, random_state=None):
        cages = sorted({cage_id_from_video_id(v) for v in video_ids})
        if len(cages) < 2:
            raise ValueError(f"LOCO requires at least 2 cages, got {len(cages)}")
        if shuffle:
            import random

            rng = random.Random(random_state)
            remaining = list(cages)
            rng.shuffle(remaining)
            for test_cage in remaining:
                val_cage = rng.choice([c for c in cages if c != test_cage])
                yield test_cage, val_cage
        else:
            for test_cage in cages:
                yield test_cage, self._select_val_circular(test_cage, cages)

    def get_groups(self, video_ids):
        return [cage_id_from_video_id(v) for v in video_ids]


class LOVO(LOO):
    """Leave-One-Video-Out cross-validation."""

    def _select_val_circular(self, test_video, all_videos, cage_to_videos, cages):
        """Pick val video from the next cage (cage-aware rotation, group-matched)."""
        test_cage = test_video[:2]
        next_cage = cages[(cages.index(test_cage) + 1) % len(cages)]
        test_group_idx = cage_to_videos[test_cage].index(test_video)
        next_videos = cage_to_videos[next_cage]
        return next_videos[test_group_idx % len(next_videos)]

    def split(self, video_ids, shuffle=False, random_state=None):
        videos = sorted(set(video_ids))
        # Precompute cage mapping once
        cage_to_videos: dict[str, list[str]] = {}
        for v in videos:
            cage_to_videos.setdefault(v[:2], []).append(v)
        cages = sorted(cage_to_videos)
        if len(cages) < 2:
            raise ValueError(f"LOVO requires at least 2 cages, got {len(cages)}")
        if shuffle:
            import random

            rng = random.Random(random_state)
            for test_video in videos:
                others = [
                    v
                    for v in videos
                    if cage_id_from_video_id(v) != cage_id_from_video_id(test_video)
                ]
                val_video = rng.choice(others)
                yield test_video, val_video
        else:
            for test_video in videos:
                yield test_video, self._select_val_circular(
                    test_video, videos, cage_to_videos, cages
                )

    def get_groups(self, video_ids):
        return list(video_ids)
