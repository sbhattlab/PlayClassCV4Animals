# Data directory
- Small and tabular data in `data` directory
- Larger video data in symlinked to `video-data-dir`
```sh
# ku-01
ln -s "/mnt/birds/rebecca2025/raw" video-data
ln -s "/mnt/birds/rebecca2025/" ext-data
```

## Overview of test data
- Small (<=1 min) in `data/img` and `data/video`
    - 10 sec, 15 sec, 30 sec, 1 min clips
- Medium (>=5 min) at `/mnt/birds/rebecca2025/`
    - 5 min

## Overview of `ext-data` directory
```
├── imgs    : video sequences converted to images (durations vary from 15 sec to 15 mins)
├── output  : results from runs
├── raw     : raw video files
└── test    : *longer* video files (i.e. > 1 min)
```

# Environment
- Fetch git submodules:
```sh
git submodule update --init --recursivegit
```

- Install environments (currently supported: `sam3-hf` and `gs2`)
```sh
# Install main (default) environment
pixi install

# Install SAM3 (sam3-hf) environment
pixi install -e sam3-hf

# install grounded-sam-2 (gs2) environment
pixi run -e gs2 setup-gs2
# Add location of gs2 fork to shell profile
export PYTHONPATH="/path/to/submodule/chicken-behaviour-classifier/Grounded-SAM-2-fork":$PYTHONPATH

# Launch shell in specific environment
pixi shell -e sam3-hf
```

## Supported platforms
In general, environments assume Linux. 

Currently the following environments are supported in addition to Linux, on macOS:
- SAM3
    - huggingface

# How to run scripts
> [!IMPORTANT] Please read base config file usually named (`config/<tool name>_config.yaml`), and modify appropriately (e.g. which CUDA device to run)
- In general, run scripts as Python modules, e.g.:
```sh
python -m script.sam3.run_sam3_hf_chunking --config config/sam3_hf_config.yaml
```

# Overview of currently implemented test scripts
> [!IMPORTANT]
> Set `CUDA_VISIBLE_DEVICES` explicitly before running the commands below, e.g.:
```sh
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-image
```

```sh
# Test torch/cuda in sam3 environments
pixi run -e sam3-hf python -c "import torch; print(f'PyTorch is installed: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
pixi run -e sam3-native python -c "import torch; print(f'PyTorch is installed: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# SAM3-hf
pixi run test-sam3-hf-image
pixi run test-sam3-hf-video

# SAM3-native 
pixi run test-sam3-native-video  

# grounded-sam-2
pixi run test-gs2
```

# Methods tested
- Object detection 
    - YOLO (yolo8n, yolo11x)
- Pose-estimation (w/ fine-tuning from manually-labelled data)
    - DeepLabCut
    - YOLO model (yolo11x-pose)
- Segmenter
    - OpenCV + SciPy (i.e. "pure" computer vision, virtually no pre-trained model-based prediction)
    - Grounded-SAM-2  
    - SAM3 (huggingface (hf) and native implementations)

# Future methods to implemented
- DINOv2/v3-derived features
    - Possibly adding trad. object detector or segmenter as preprocessing step to isolate subjects