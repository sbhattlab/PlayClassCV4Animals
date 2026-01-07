from pathlib import Path

import deeplabcut

DLC_CONFIG = "data/config.yaml"
videos = [Path("sandbox/test.mp4").resolve()]
deeplabcut.analyze_videos(config=DLC_CONFIG, videos=videos, device="cuda")
deeplabcut.create_labeled_video(config=DLC_CONFIG, videos=videos)
