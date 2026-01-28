# Data directory
- Small and tabular data in `data` directory
- Larger video data in symlinked to `video-data-dir`
```sh
# e.g. macOS
ln -s  "$HOME/Library/CloudStorage/OneDrive-UniversityofCopenhagen/IFSV/proj/chicken-behaviour-classifier-data/video-raw-data" video-data
# ku-01
ln -s "/mnt/birds/rebecca2025/raw" video-data
```

# Environment
```sh
# Install main (default) environment
pixi install

# All environments
pixi install --all
# Launch shell in specific environment
pixi shell -e sam3-hf
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
pixi run test-sam3-hf-image # Only test currently available on macOS
pixi run test-sam3-hf-video

# SAM3-native 
pixi run test-sam3-native-video  
```

# Future methods to implemented
- DINOv2/v3-derived features
    - Possibly adding trad. object detector or segmenter as preprocessing step to isolate subjects
