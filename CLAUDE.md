# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-object tracking and segmentation of chickens in video data using SAM3, Grounded-SAM-2, and YOLO. The system processes long videos by chunking them, running text-prompted segmentation on the first chunk, then propagating tracks via point prompts across subsequent chunks.

## Environment & Setup

**Package manager**: Pixi (not pip/conda directly). Python 3.11 only.

```sh
# Fetch git submodules (Grounded-SAM-2 fork)
git submodule update --init --recursive

# Install an environment
pixi install -e sam3-hf        # SAM3 HuggingFace (main pipeline)
pixi install -e sam3-native    # SAM3 native (Facebook Research)
pixi run -e gs2 setup-gs2      # Grounded-SAM-2 (downloads checkpoints + installs)

# Enter an environment shell
pixi shell -e sam3-hf
```

Environments: `sam3-hf`, `sam3-hf-macos`, `sam3-native`, `gs2`, `gs2-macos`. CUDA environments are Linux-only; macOS variants use CPU/MPS.

## Running Scripts

Scripts are run as Python modules from the project root:

```sh
python -m script.sam3.run_sam3_hf_chunking --config config/sam3_hf_config.yaml
python -m script.gs2.chicken_tracking_demo --gs2-repo-path Grounded-SAM-2-fork -i ext-data/imgs/imgs_1min --text ".chicken.bird."
```

Main scripts read `CUDA_VISIBLE_DEVICES` from their YAML config file (e.g. `config/sam3_hf_config.yaml`). For tests, set it as an env var:

## Running Tests

```sh
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-image     # Single image inference (sam3-hf env)
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-video     # Video chunking test (sam3-hf env)
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-native-video  # Native SAM3 video test (sam3-native env)
CUDA_VISIBLE_DEVICES=1 pixi run test-gs2                # Grounded-SAM-2 chicken tracking demo (gs2 env)
```

Tests are standalone scripts in `test/`, not pytest-based.

## Architecture

```
src/                          # Core library modules
  utils.py                    # Config (OmegaConf), device selection, mask utils, video I/O, rendering
  sam3_hf.py                  # SAM3 HuggingFace pipeline: chunking, text→segmentation, point→tracking
  tracking_metrics.py         # Per-frame/chunk metrics: IoU, centroids, occlusion, identity switches

script/                       # Executable scripts (run as python -m script.X.Y)
  sam3/                       # SAM3 pipelines
    run_sam3_hf_chunking.py   # Main chunked video processing pipeline
  gs2/                        # Grounded-SAM-2 pipelines
    chicken_tracking_demo.py  # GS2 image-sequence tracking
  yolo/                       # YOLO format conversion and visualization

config/                       # YAML configs (OmegaConf) for pipeline runs
test/                         # Test scripts (run via pixi tasks)
notebook/                     # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: Pipelines read YAML configs via OmegaConf. Configs live in `config/` and control GPU selection, chunk duration, tracking parameters, output paths.
- **Chunked processing**: Long videos are split into chunks (default 45s). First chunk uses text-prompted SAM3 segmentation; subsequent chunks use point prompts extracted from previous chunk's masks for tracking continuity.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation across chunks.
- **Metrics pipeline**: `tracking_metrics.py` computes IoU, centroid distances, occlusion detection, and identity switch heuristics per frame and chunk. Results export to Parquet/CSV.
- **Device auto-detection**: `src/utils.py` automatically selects CUDA > MPS > CPU.

## Data Layout

- `data/` — Small test data (images, short video clips, DLC annotations, ethogram parquets)
- `ext-data/` — Symlink to `/mnt/birds/rebecca2025/` (longer videos, output results, image sequences)
- `video-data/` — Symlink to `/mnt/birds/rebecca2025/raw` (raw video files)
- `Grounded-SAM-2-fork/` — Git submodule with SAM2/Grounding DINO code and checkpoints

## Key Dependencies

- PyTorch 2.9.1 (CUDA 12.6 on Linux)
- HuggingFace Transformers v5.0.0rc2 (installed from git)
- supervision (annotation/visualization), loguru (logging), OmegaConf (config)
