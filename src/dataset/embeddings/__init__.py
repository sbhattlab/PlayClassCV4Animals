# Re-exports for backwards compatibility
from src.dataset.embeddings.dinov3 import (
    _split_mask_thirds,
    extract_bodypart_embeddings,
    extract_embeddings,
)

__all__ = ["extract_embeddings", "extract_bodypart_embeddings", "_split_mask_thirds"]
