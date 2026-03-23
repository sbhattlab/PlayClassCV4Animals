from typing import Final

DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/dataset"
DEFAULT_CHECKPOINT_DIR: Final = f"{DEFAULT_DATA_DIR}/eval"
DEFAULT_LABEL_DIR: Final = f"{DEFAULT_DATA_DIR}/labels"
DEFAULT_TRACKING_DIR: Final = f"{DEFAULT_DATA_DIR}/postprocessing"
DEFAULT_VIDEO_DIR: Final = f"{DEFAULT_DATA_DIR}/video"
# Discard windows where less than this fraction of frames are available after postprocessing
DEFAULT_MIN_WINDOW_COVERAGE: Final = 0.5
DEFAULT_FPS: Final = 25.0
DEFAULT_N_BIRDS: Final = 3
LABEL_ORDER: Final = ["none", "worm", "locomotor", "social"]
