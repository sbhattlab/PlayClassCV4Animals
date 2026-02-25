# Set up conda environment
```sh
mamba create -n deva python=3.10.10 -y
mamba activate deva
mamba install cuda -c nvidia/label/cuda-12.4 -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

# Install DEVA fork
```sh
mkdir -p repos && cd repos 
git clone https://github.com/Neclow/Tracking-Anything-with-DEVA/
mv Tracking-Anything-with-DEVA/ Tracking-Anything-with-DEVA-fork/ && cd Tracking-Anything-with-DEVA-fork/

pip install -e .

bash scripts/download_models.sh
```

# Install Grounded-SAM
```sh
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda

cd ..
git clone https://github.com/hkchengrex/Grounded-Segment-Anything && cd Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
cd ..
git submodule update --init --recursive
cd Tag2Text && pip install -r requirements.txt
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

# Test installation
```sh
python -c "from groundingdino.util.inference import Model as GroundingDINOModel"
```