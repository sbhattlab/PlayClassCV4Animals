from src.dataset.tracking_issues import (
    merge_id_on_switch,
    remove_overlaps,
)


def process_tracks(tracks, issues, labels, id_remaps=None):
    tracks, labels = remove_overlaps(tracks, issues, labels)
    tracks = merge_id_on_switch(tracks, id_remaps or [])
    return tracks, labels
