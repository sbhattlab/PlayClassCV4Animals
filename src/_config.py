from typing import Final

DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/dataset"
DEFAULT_CHECKPOINT_DIR: Final = f"{DEFAULT_DATA_DIR}/eval"
DEFAULT_LABEL_DIR: Final = f"{DEFAULT_DATA_DIR}/labels"
DEFAULT_TRACKING_DIR: Final = f"{DEFAULT_DATA_DIR}/tracking"
# Discard windows where less than this fraction of frames are available after postprocessing
DEFAULT_MIN_WINDOW_COVERAGE: Final = 0.5
DEFAULT_FPS: Final = 25.0
LABEL_ORDER: Final = ["none", "worm", "locomotor", "social"]
