# YOLO-Based Occlusion Detection: Design & Implementation Summary

## Executive Summary

I've designed and implemented a **YOLO-based occlusion detection pipeline** that analyzes pre-computed tracking data to identify high-occlusion periods in video. This approach offers a semantic alternative to pixel-based clustering (KMeans) by operating directly on object-level features.

## Design Principles

### 1. **Reuse Existing Data**
- Leverages YOLO tracking outputs already computed in your pinned notebook
- No need for frame extraction or re-processing
- Extremely efficient: metrics computed on 2,431 detections in <1 second

### 2. **Semantic Object Understanding**
- Operates on **bounding boxes** and **centroids**, not raw pixels
- Computes physically meaningful metrics:
  - **IoU (Intersection over Union)**: Visual overlap between subjects
  - **Centroid distances**: Spatial proximity
  - **Clustering coefficient**: Group tightness

### 3. **Inspired by Your Metrics Framework**
The pipeline mirrors the structure of your existing `metrics.py` module:
- Per-frame metric computation
- Temporal aggregation via sliding windows
- Transition point identification for adaptive chunking
- But uses **object-level features** instead of **pixel clusters**

## Implementation Overview

### Core Module: `src/yolo_prescan.py`

**Key Functions:**

1. **`compute_yolo_per_frame_metrics()`**
   - Input: YOLO tracking DataFrame
   - Output: Per-frame metrics (16 features)
   - Computes: IoU, distances, clustering, bbox stats, confidence

2. **`identify_occlusion_periods()`**
   - Input: Per-frame metrics + window parameters
   - Output: List of (start_frame, end_frame) tuples
   - Logic: Sliding window with configurable threshold

3. **`compute_yolo_prescan_results()`**
   - Main entry point
   - Returns comprehensive results dict with periods, transitions, and metadata

### Pipeline Flow

```
YOLO Tracks (parquet)
    ↓
Extract per-frame detections
    ↓
Compute pairwise metrics
  • Bbox IoU matrix (N×N)
  • Centroid distance matrix (N×N)
    ↓
Aggregate to frame-level
  • max/mean IoU
  • min/mean distance
  • clustering coefficient
  • bbox area statistics
    ↓
Flag high-occlusion frames
  • IoU > 0.15 OR
  • Clustering > 0.5
    ↓
Sliding window analysis
  • Window: 25 frames (1s @ 25fps)
  • Threshold: 30% frames flagged
    ↓
Extract occlusion periods
  • Contiguous high-occlusion regions
  • Transition frames for chunking
```

## Results on Your Chicken Video

### Detection Statistics
- **Total frames**: 1,494 (59.76 seconds @ 25fps)
- **Frames with detections**: 1,251 (83.7%)
- **Total detections**: 2,431
- **Unique track IDs**: 94
- **Mean detections/frame**: 1.94

### Occlusion Analysis
- **High-occlusion frames**: 79 / 1,251 (6.3%)
- **Detected occlusion periods**: 3

| Period | Frames | Time (s) | Duration | Avg Objects | Max IoU | Avg Clustering |
|--------|--------|----------|----------|-------------|---------|----------------|
| 1 | 720-777 | 28.8-31.1 | 2.28s | 2.4 | 0.358 | 0.149 |
| 2 | 778-830 | 31.1-33.2 | 2.08s | 2.1 | 0.243 | 0.174 |
| 3 | 1199-1254 | 48.0-50.2 | 2.20s | 1.9 | 0.192 | 0.229 |

### Transition Frames
`[720, 777, 778, 830, 1199, 1254]`

These can be used as **chunk boundaries** to avoid splitting occlusion events.

## Key Metrics Explained

### 1. **Max Pairwise Bbox IoU**
- **Range**: 0-1 (0 = no overlap, 1 = perfect overlap)
- **Interpretation**: Highest visual overlap between any two subjects
- **Threshold**: 0.15 (15% overlap triggers high-occlusion flag)
- **Use case**: Detect when subjects visually occlude each other

### 2. **Clustering Coefficient**
- **Range**: 0-1 (0 = dispersed, 1 = all pairs close)
- **Interpretation**: Fraction of object pairs within threshold distance
- **Threshold**: 0.5 (50% of pairs must be close)
- **Use case**: Detect spatial grouping even without visual overlap

### 3. **Min Centroid Distance**
- **Range**: 0-1+ (normalized coordinates)
- **Interpretation**: Closest pair of objects
- **Use case**: Identify moments of extreme proximity

### 4. **Mean Confidence**
- **Range**: 0-1
- **Interpretation**: Average YOLO detection confidence
- **Use case**: Quality check for detections

## Advantages Over KMeans Approach

| Aspect | YOLO-Based | KMeans-Based |
|--------|------------|--------------|
| **Speed** | ⚡ Very fast (reuses data) | 🐢 Slower (frame extraction) |
| **Semantics** | 🎯 Object-level | 🎨 Pixel-level |
| **Interpretability** | ✅ High (IoU, distance) | ⚠️ Moderate (silhouette) |
| **Identity** | ✅ Track IDs available | ❌ No identity |
| **Accuracy** | ✅ Semantic overlap | ⚠️ Color-based proxy |
| **Requirement** | YOLO tracks must exist | Any video |

## Integration with Adaptive Chunking

### 1. **Chunk Boundary Placement**
Use transition frames to define chunks that respect occlusion boundaries:

```python
transition_frames = prescan_results['transition_frames']
chunks = create_chunks_from_transitions(
    total_frames=total_frames,
    transitions=transition_frames,
    max_chunk_size=500
)
```

### 2. **Occlusion-Aware Processing**
Apply different strategies based on occlusion status:

```python
for start, end in chunks:
    is_high_occ = is_in_occlusion_period(start, end, occlusion_periods)
    
    if is_high_occ:
        # Use temporal models, increase context window
        features = extract_with_temporal_context(video, start, end)
    else:
        # Standard frame-level processing
        features = extract_standard(video, start, end)
```

### 3. **Resource Allocation**
Allocate more processing time/memory to high-occlusion chunks:

```python
for start, end in chunks:
    if is_high_occ:
        batch_size = 8   # Smaller batches
        context = 10      # More temporal context
    else:
        batch_size = 32
        context = 0
```

## Deliverables

### 1. **Core Implementation**
- ✅ `src/yolo_prescan.py` - Complete pipeline with comprehensive docstrings

### 2. **Documentation**
- ✅ `docs/YOLO_PRESCAN_README.md` - Detailed usage guide and API reference

### 3. **Example Script**
- ✅ `examples/yolo_prescan_demo.py` - Complete working example

### 4. **Results & Visualizations**
- ✅ `sandbox/results/yolox/yolo_prescan_metrics.parquet` - Per-frame metrics
- ✅ `sandbox/results/yolox/yolo_prescan_config.json` - Configuration & periods
- ✅ `sandbox/results/yolox/yolo_prescan_analysis.png` - Full timeline visualization
- ✅ `sandbox/results/yolox/yolo_prescan_detailed.png` - Detailed period analysis

## Usage Example

```python
import pandas as pd
from src.yolo_prescan import compute_yolo_prescan_results, yolo_prescan_to_df

# Load your YOLO tracking data
yolo_df = pd.read_parquet("sandbox/results/yolox/chicken_tracks_yolo.parquet")

# Run pre-scan analysis
prescan = compute_yolo_prescan_results(
    yolo_df,
    fps=25.0,
    window_seconds=1.0,
    high_occlusion_threshold=0.3,
)

# Get occlusion periods
print(f"Detected {len(prescan['occlusion_periods'])} occlusion periods:")
for start, end in prescan['occlusion_periods']:
    print(f"  Frames {start}-{end}")

# Get transition frames for chunking
print(f"Transition frames: {prescan['transition_frames']}")

# Convert to DataFrame for analysis
metrics_df = yolo_prescan_to_df(prescan['per_frame_metrics'])
```

## Tuning Guidelines

### Increasing Sensitivity (Detect More Occlusions)
- **Lower** `occlusion_iou_threshold` (e.g., 0.10)
- **Lower** `clustering_distance_threshold` (e.g., 0.10)
- **Lower** `high_occlusion_threshold` (e.g., 0.20)
- **Increase** `window_seconds` (e.g., 2.0)

### Reducing False Positives (More Conservative)
- **Raise** `occlusion_iou_threshold` (e.g., 0.25)
- **Raise** `clustering_distance_threshold` (e.g., 0.25)
- **Raise** `high_occlusion_threshold` (e.g., 0.50)
- **Decrease** `window_seconds` (e.g., 0.5)

## Future Enhancements

1. **Multi-scale Windows**: Analyze at multiple temporal scales simultaneously
2. **Confidence Weighting**: Downweight low-confidence detections
3. **Track Stability**: Incorporate track lifetime and consistency
4. **Velocity Features**: Add motion-based occlusion prediction
5. **Scene-level Metrics**: Global crowding scores per frame
6. **Adaptive Thresholds**: Learn optimal thresholds from data distribution

## Conclusion

This YOLO-based pipeline provides a **fast**, **interpretable**, and **semantic** approach to occlusion detection that:

✅ **Reuses existing YOLO tracking data** - no additional frame processing  
✅ **Computes meaningful metrics** - IoU and distances have clear physical interpretation  
✅ **Integrates with adaptive chunking** - provides transition frames as chunk boundaries  
✅ **Matches your metrics framework** - similar structure to existing `metrics.py`  
✅ **Handles identity instability** - works even when track IDs switch (uses proximity, not identity)  

The approach successfully identified 3 high-occlusion periods in your 60-second chicken video, providing actionable boundaries for adaptive processing strategies.
