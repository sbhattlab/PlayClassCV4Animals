import re
import subprocess
import sys
from pathlib import Path


def normalize_filename(name: str) -> str:
    """Normalize a filename by replacing non-alphanumeric sequences with underscores."""
    p = Path(name)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", p.stem).strip("_")
    return stem + p.suffix


def clean_video(input_path: str):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Normalize filename (stem only)
    normalized_stem = re.sub(r"[^A-Za-z0-9]+", "_", input_path.stem).strip("_")

    # Force .mp4 extension
    output_path = input_path.parent / f"{normalized_stem}.mp4"

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_video_clean.py <path/to/video>")
        sys.exit(1)
    clean_video(sys.argv[1])

## 🎯 Usage sh python convert_video_clean.py "C1G1 Test 1 day 28(1) Camera 4 2025-02-04 10_59_56 1.mpg"  Output example: Saved cleaned file → C1G1_Test_1_day_28_1_Camera_4_2025_02_04_10_59_56_1.mpg
