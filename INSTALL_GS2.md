# Installation instructions for grounded-sam-2
Adapted from: https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#installation
```sh
# Enter Grounded-SAM-2-fork directory and download checkpoints
cd Grounded-SAM-2-fork
cd checkpoints
bash download_ckpts.sh
../
cd gdino_checkpoints
bash download_ckpts.sh
cd ../
# Install PyTorch with CUDA support
uv install torch torchvision torchaudio
export CUDA_HOME="/usr/lib/nvidia-cuda-toolkit/"
uv pip install -e .
uv pip install --no-build-isolation -e grounding_dino
```