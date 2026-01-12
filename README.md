# Data directory
- Small and tabular data in `data` directory
- Larger video data in symlinked to `video-data-dir`
```sh
ln -s  "$HOME/Library/CloudStorage/OneDrive-UniversityofCopenhagen/IFSV/proj/chicken-behaviour-classifier-data/video-raw-data" video-data
```

# Methods tested
- Pre-trained dlc model
- YOLO pose estimation based model trained on dlc points
- Grounded-SAM-2 