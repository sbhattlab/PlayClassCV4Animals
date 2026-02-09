# YOLO Prescan Quick Reference

## One-Liner Usage

```python
from src.yolo_prescan import compute_yolo_prescan_results
prescan = compute_yolo_prescan_results(yolo_df, fps=25.0)
```

## Key Outputs

```python
prescan['occlusion_periods']     # [(start, end), ...] high-occlusion periods
prescan['transition_frames']     # [frame1, frame2, ...] chunk boundaries
prescan['per_frame_metrics']     # List[Dict] with 16 features per frame
```

## Core Metrics

| Metric | Trigger | Meaning |
|--------|---------|---------|
| `max_pairwise_bbox_iou` | > 0.15 | Visual overlap detected |
| `clustering_coefficient` | > 0.5 | Spatial grouping detected |
| `min_centroid_distance` | < 0.15 | Very close proximity |

## Occlusion Detection Logic

```
Frame is HIGH-OCCLUSION if:
  max_bbox_iou > 0.15  OR  clustering_coefficient > 0.5

Period is HIGH-OCCLUSION if:
  >30% of frames in 1-second window are flagged
```

## Tuning Quick Guide

**More sensitive (detect more):**
- ⬇️ `occlusion_iou_threshold` → 0.10
- ⬇️ `clustering_distance_threshold` → 0.10  
- ⬇️ `high_occlusion_threshold` → 0.20

**More conservative (fewer false positives):**
- ⬆️ `occlusion_iou_threshold` → 0.25
- ⬆️ `clustering_distance_threshold` → 0.25
- ⬆️ `high_occlusion_threshold` → 0.50

## Integration Pattern

```python
# Create chunks from transitions
boundaries = sorted([0] + list(prescan['transition_frames']) + [total_frames])
chunks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

# Process with awareness
for start, end in chunks:
    is_occ = any(occ_start <= start <= occ_end 
                 for occ_start, occ_end in prescan['occlusion_periods'])
    
    if is_occ:
        process_high_occlusion_chunk(start, end)
    else:
        process_standard_chunk(start, end)
```

## Files Created

```
src/yolo_prescan.py                         # Core implementation
docs/YOLO_PRESCAN_README.md                 # Detailed docs
docs/YOLO_PRESCAN_SUMMARY.md                # Design summary
examples/yolo_prescan_demo.py               # Working example
sandbox/results/yolox/yolo_prescan_*.{parquet,json,png}  # Outputs
```

## Your Results (Chicken Video)

- **3 occlusion periods** detected
- **Frames**: 720-777, 778-830, 1199-1254  
- **Transitions**: [720, 777, 778, 830, 1199, 1254]
- **6.3%** of frames flagged as high-occlusion

## Comparison to KMeans

| | YOLO | KMeans |
|-|------|--------|
| **Speed** | ⚡⚡⚡ | ⚡ |
| **Semantics** | Object-level | Pixel-level |
| **Requires** | YOLO tracks | Video frames |
| **Accuracy** | Direct overlap | Color proxy |

## Common Use Cases

1. **Adaptive chunking**: Use transitions as chunk boundaries
2. **Resource allocation**: More time/memory for occlusion chunks  
3. **Model selection**: Temporal models for high-occlusion regions
4. **Feature extraction**: Focus on clear frames for embeddings
5. **Quality control**: Flag difficult periods for review

## See Also

- Full documentation: `docs/YOLO_PRESCAN_README.md`
- Design details: `docs/YOLO_PRESCAN_SUMMARY.md`  
- Example script: `examples/yolo_prescan_demo.py`
