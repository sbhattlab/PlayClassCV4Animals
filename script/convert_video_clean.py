import re
import subprocess
import sys
from pathlib import Path


def normalize_filename(name: str) -> str:
    """Normalize a filename by replacing non-alphanumeric sequences with underscores."""
    p = Path(name)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", p.stem).strip("_")
    return stem + p.suffix


VIDEO_EXTENSIONS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


def clean_video(input_path: str):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Normalize filename (stem only)
    normalized_stem = re.sub(r"[^A-Za-z0-9]+", "_", input_path.stem).strip("_")

    # Force .mp4 extension
    output_path = input_path.parent / f"{normalized_stem}.mp4"

    if output_path.exists():
        print(f"Skipping (output exists): {output_path}")
        return

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-i",
        str(input_path),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print(f"Running:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"Saved cleaned file → {output_path}")


def clean_video_dir(dir_path: str):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    videos = sorted(
        f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No video files found in {dir_path}")
        return

    print(f"Found {len(videos)} video(s) in {dir_path}\n")

    failed = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        try:
            clean_video(str(video))
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}")
            failed.append(video.name)
        print()

    print(f"Done. {len(videos) - len(failed)}/{len(videos)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_video_clean.py <path/to/video_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        clean_video_dir(str(target))
    else:
        clean_video(str(target))
