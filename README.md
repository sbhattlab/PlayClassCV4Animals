# Data directory
- Small and tabular data in `data` directory
- Larger video data in symlinked to `video-data-dir`
```sh
ln -s  "$HOME/Library/CloudStorage/OneDrive-UniversityofCopenhagen/IFSV/proj/chicken-behaviour-classifier-data/video-raw-data" video-data
```
# Environment installations
- DLC - install dlc conda environment as per: https://github.com/DeepLabCut/DeepLabCut 
- YOLO - use general conda environment by installing ENV.yml and platform-specific requirements file
- Grounded-SAM-2 - uv venv install virtual environment as per: https://github.com/IDEA-Research/Grounded-SAM-2

# Methods tested
- Pre-trained dlc model
- YOLO pose estimation based model trained on dlc points
- Grounded-SAM-2 