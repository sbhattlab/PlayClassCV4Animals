# SAM3 installation instructions
- Get approval for sam3 weights on huggingface

# NEW METHOD
```sh
cd ENV/sam3
uv sync
```

# OLD METHOD 
- Clone / fetch git submodule for sam3, transformers
```
mamba create -n sam3 python=3.12 uv ffmpeg nvidia::cudatoolkit -y
mamba activate sam3

### if _no_ torchcodec
uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
### if torchcodec
uv pip install torch torchvision torchaudio
uv pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128

cd sam3
uv pip install -e .
uv pip install -e ".[notebooks]"
cd ../transformers
uv pip install '.[torch]'
uv pip install kernels # better mask quality
uv pip install pandas # to run sam3_video_predictor_example.ipynb
```
