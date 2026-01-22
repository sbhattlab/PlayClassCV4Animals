# Data directory
- Small and tabular data in `data` directory
- Larger video data in symlinked to `video-data-dir`
```sh
# e.g. macOS
ln -s  "$HOME/Library/CloudStorage/OneDrive-UniversityofCopenhagen/IFSV/proj/chicken-behaviour-classifier-data/video-raw-data" video-data
# ku-01
ln -s "/mnt/birds/rebecca2025/raw" video-data
```

# Recreate main environemnt
```sh
mamba env create -p ENV/chicken-behav/ENV.yml && mamba activate -y
uv pip install ENV/chicken-behav/requirements_linux.txt 
```

# Methods tested
- Background‑subtraction–based multi‑object tracking (trad.)
    - OpenCV + SciPy (Segmentation‑based MOT with Hungarian assignment)
- Object detection (trad.)
    - YOLO (yolo11x)
- Pose-estimation w/ fine-tuning from manually-labelled data
    - Pre-trained DeepLabCut model 
    - YOLO model (yolo11x-pose) finetuned on dlc-derived points
- Segmenter
    - Grounded-SAM-2 (segmentation) – **best overall, so far**
    - SAM3 (segmentation)

# Future methods to test
- DINOv2/v3-derived features
    - Possibly adding trad. bbox object tracking or segmenter as preprocessing step to isolate subjects
