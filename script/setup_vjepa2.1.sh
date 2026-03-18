#!/usr/bin/env bash
# Download V-JEPA 2.1 checkpoint and patch the torch.hub cache to avoid
# the src/ namespace collision with this project's own src/ package.
#
# Usage:
#   bash script/setup_vjepa21.sh [model]
#
# Models:
#   vjepa2_1_vit_large_384   (default, ViT-L distilled from ViT-G)
#   vjepa2_1_vit_base_384    (ViT-B distilled from ViT-G)
#   vjepa2_1_vit_giant_384   (ViT-G)
#   vjepa2_1_vit_gigantic_384 (ViT-G2)

set -euo pipefail

MODEL="${1:-vjepa2_1_vit_large_384}"

# Map model names to checkpoint filenames
declare -A CHECKPOINTS=(
    ["vjepa2_1_vit_base_384"]="vjepa2_1_vitb_dist_vitG_384"
    ["vjepa2_1_vit_large_384"]="vjepa2_1_vitl_dist_vitG_384"
    ["vjepa2_1_vit_giant_384"]="vjepa2_1_vitg_384"
    ["vjepa2_1_vit_gigantic_384"]="vjepa2_1_vitG_384"
)

if [[ ! -v "CHECKPOINTS[$MODEL]" ]]; then
    echo "Unknown model: $MODEL"
    echo "Available: ${!CHECKPOINTS[*]}"
    exit 1
fi

CKPT_NAME="${CHECKPOINTS[$MODEL]}"
CKPT_URL="https://dl.fbaipublicfiles.com/vjepa2/${CKPT_NAME}.pt"
CKPT_DIR="${HOME}/.cache/torch/hub/checkpoints"
CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}.pt"

HUB_DIR="${HOME}/.cache/torch/hub/facebookresearch_vjepa2_main"

# --- Step 1: Download checkpoint ---
mkdir -p "$CKPT_DIR"
if [[ -f "$CKPT_PATH" ]]; then
    echo "Checkpoint already exists: $CKPT_PATH"
else
    echo "Downloading ${CKPT_NAME}.pt ..."
    wget -q --show-progress -O "$CKPT_PATH" "$CKPT_URL"
    echo "Saved to $CKPT_PATH"
fi

# --- Step 2: Clone hub repo (if not cached) ---
if [[ -d "$HUB_DIR" ]]; then
    echo "Hub repo already cached: $HUB_DIR"
else
    echo "Fetching hub repo via torch.hub ..."
    python -c "import torch; torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')" 2>/dev/null || true
fi

# --- Step 3: Rename src/ -> vjepa2/ to avoid namespace collision ---
if [[ -d "$HUB_DIR/src" ]]; then
    echo "Renaming hub repo src/ -> vjepa2/ ..."
    mv "$HUB_DIR/src" "$HUB_DIR/vjepa2"
    find "$HUB_DIR" -name "*.py" -exec sed -i 's/from src\./from vjepa2./g; s/import src\./import vjepa2./g' {} +
    echo "Done."
elif [[ -d "$HUB_DIR/vjepa2" ]]; then
    echo "Hub repo already patched (vjepa2/ exists)."
else
    echo "Error: neither src/ nor vjepa2/ found in $HUB_DIR"
    exit 1
fi

echo ""
echo "Setup complete. Run extraction with:"
echo "  pixi run -e tracker python -m script.extract_embeddings_vjepa2 --video-dir data/video/batch data/video/batch2 --device cuda:0 --model-name $MODEL"
