# Architecture

```
script/
  run_tracker.py            # Config-driven tracking pipeline (main script)

src/
  utils.py                  # Config/logging/output dirs, chunking, parquet export, video annotation
  io.py                     # Frame loading (sequential + seek-based), video I/O
  masks.py                  # Mask/bbox/point extraction, tracking output normalization
  grounding.py              # Text-prompt grounding, best-frame selection, ID matching
  metrics.py                # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
  viz.py                    # Visualizations: ID timeline, dashboard, score plots, mask evolution, prompt points
  chunk_boundaries.py       # Per-frame metrics, occlusion detection, separation windows, adaptive chunking
  yolo_scan.py              # YOLO inference only (run_yolo_scan); re-exports src.chunk_boundaries for compat
  ethogram.py               # Behavior label parsing from Excel registration protocols
  dataset/                  # Dataset construction package
    __init__.py             # Package marker (no re-exports; import from submodules directly)
    utils.py                # Shared helpers: fmt_time, get_video_fps, resolve_video_path, assert_embedding_label_alignment, load_video_frames_sequential
    tracking_issues.py      # Detection: ID switches, mask overlaps, low-score periods
    tracking_postprocessing.py  # Remediation: prefill from issues, ID-scoped trims, ID remaps, process_tracks
    labels.py               # Behaviour label parsing from Excel registration protocols
    features.py             # Handcrafted mask features: spatial, temporal, pairwise, window summarization (vectorized)
    embeddings/             # Embedding extraction package
      __init__.py           # Re-exports for backwards compat
      dinov3.py             # DINOv3 CLS-token embedding extraction from bbox crops
      vjepa2.py             # V-JEPA 2/2.1 video embedding extraction
      videoprism.py         # VideoPrism video embedding extraction
    crops.py              # Shared crop modes: crop_frame, compute_union_origin, compute_union_bbox
  classification/           # Behaviour classification package
    models.py               # Backbones: SimpleLinear, SimpleMLP, TemporalMLP, TemporalCNNv2. MODEL_REGISTRY maps names to (cls, temporal_flag).
    datamodule.py           # BehaviourDataset + BehaviourDataModule (LOVO split, segment pooling)
    trainer.py              # BehaviourClassifier LightningModule (weighted CE, AdamW, MetricCollection)
    stats.py                # LOVO aggregation: scalar summary CSV, summed confusion matrices
  debug/                    # Interactive debugging utilities and standalone grounding test script

script/
  build_dataset.py                 # Labels, postprocessing, windowing -> tracks + labels parquets
  extract_features.py              # Mask feature extraction + window summarization (CPU-only)
  extract_embeddings_dinov3.py     # DINOv3 CLS-token embeddings (GPU required)
  extract_embeddings_vjepa2.py     # V-JEPA 2/2.1 video embeddings (GPU required)
  extract_embeddings_videoprism.py # VideoPrism video embeddings (JAX/GPU)
  save_cid_crops.py                # Save union384 crops to disk for CID pretraining
  cid_vjepa21.py                   # CID pretraining of V-JEPA 2.1
  train.py                         # Classification training CLI (PyTorch Lightning)
  train_xgboost.py                 # XGBoost baseline (LOCO/LOVO)
  compute_chunk_boundaries.py      # Recompute metrics + boundaries from yolo_tracking.parquet

config/                     # YAML configs (OmegaConf)
tests/                      # pytest tests (dataset features, labels, postprocessing)
src/test/                   # Test scripts for SAM3 inference (run via pixi tasks, not pytest)
notebook/                   # Jupyter notebooks for EDA and demos
```
