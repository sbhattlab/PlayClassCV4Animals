import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def parse_args():
    parser = ArgumentParser(
        description="Visualize chunk boundary frames from a config file or prescan run directory"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to YAML config with manual_chunk_frames")
    group.add_argument(
        "--run-dir", help="Path to a prescan run directory containing chunk_info.json"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory to save the output image"
    )
    return parser.parse_args()


def grab_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fmt_timestamp(frame_idx, fps):
    total_s = frame_idx / fps
    m, s = divmod(total_s, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main():
    args = parse_args()

    if args.output_dir is None:
        output_dir = Path("sandbox")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.config:
        cfg = OmegaConf.load(args.config)
        video_path = cfg.video_path
        chunks = list(cfg.manual_chunk_frames)  # list of [start, end] pairs
    else:
        run_dir = Path(args.run_dir)
        chunk_info_path = run_dir / "chunk_info.json"
        assert chunk_info_path.exists(), f"No chunk_info.json found in: {run_dir}"
        with open(chunk_info_path) as f:
            chunk_info = json.load(f)
        # Read video_path from the config copy saved in the run dir
        config_copies = list(run_dir.glob("*.yaml"))
        assert config_copies, f"No YAML config copy found in: {run_dir}"
        cfg = OmegaConf.load(config_copies[0])
        video_path = cfg.video_path
        chunks = [c["frame_range"] for c in chunk_info["chunks"]]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    n_chunks = len(chunks)
    fig, axes = plt.subplots(
        n_chunks,
        2,
        figsize=(14, n_chunks * 2.5),
        constrained_layout=True,
    )
    if n_chunks == 1:
        axes = [axes]

    for i, (start, end) in enumerate(chunks):
        for j, (frame_idx, label) in enumerate([(start, "start"), (end, "end")]):
            frame = grab_frame(cap, frame_idx)
            ax = axes[i][j]
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.set_facecolor("black")
                ax.text(
                    0.5,
                    0.5,
                    "read error",
                    color="white",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ts = fmt_timestamp(frame_idx, fps)
            ax.set_title(
                f"Chunk {i} — {label}\nframe {frame_idx}  ({ts})",
                fontsize=8,
            )
            ax.axis("off")

    cap.release()
    fig.suptitle("Chunk boundary frames", fontsize=12, fontweight="bold")
    plt.savefig(
        output_plot_path := output_dir
        / f"chunk_boundaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        dpi=300,
    )
    print(f"Saved chunk boundary visualization to: {output_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
