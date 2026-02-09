# YOLO-Based Occlusion Detection Pipeline

## Overview

This pipeline uses pre-computed YOLO tracking outputs to identify periods of high occlusion in video data. Unlike pixel-based clustering approaches (like KMeans), this method operates on semantic object-level features (bounding boxes, centroids) to detect when subjects are in close proximity or overlapping.

## Key Features

- **Semantic Understanding**: Operates on detected objects rather than raw pixels
- **Efficient**: Reuses existing YOLO tracking data without re-processing frames
- **Interpretable**: Uses physically meaningful metrics (IoU, centroid distance)
- **Adaptive**: Configurable thresholds for different use cases
- **Identity-Aware**: Leverages track IDs for temporal context

## Architecture

```
YOLO Tracking Data (parquet)
    ↓
Per-Frame Metrics Computation
    • Bounding box IoU (overlap)
    • Centroid distances (proximity)
    • Clustering coefficient (spatial grouping)
    • Bbox area statistics
    ↓
High-Occlusion Frame Flagging
    • IoU > threshold OR
    • Clustering coefficient > threshold
    ↓
Sliding Window Analysis
    • Window size: N frames (e.g., 1 second)
    • Threshold: % of frames flagged
    ↓
Occlusion Period Identification
    • Contiguous high-occlusion regions
    • Transition frames for chunking
```

## Metrics Computed Per Frame

| Metric | Description | Use Case |
|--------|-------------|----------|
| `num_objects` | Detection count | Track subject presence |
| `max_pairwise_bbox_iou` | Maximum bbox overlap (0-1) | Detect visual occlusion |
| `mean_pairwise_bbox_iou` | Average bbox overlap | Overall crowding |
| `num_overlapping_pairs` | Pairs with IoU > threshold | Count overlapping subjects |
| `min_centroid_distance` | Closest object pair (normalized) | Detect proximity |
| `mean_centroid_distance` | Average inter-object distance | Spatial dispersion |
| `clustering_coefficient` | Fraction of pairs within threshold | Group tightness |
| `mean_bbox_area` | Average detection size | Scale/distance indicator |
| `bbox_area_variance` | Variance in detection sizes | Size heterogeneity |
| `mean_confidence` | Average YOLO confidence | Detection quality |
| `is_high_occlusion` | Boolean flag | Frame-level classification |

## Occlusion Detection Logic

### Frame-Level Classification

A frame is flagged as **high-occlusion** if **either** condition is met:

1. **High Overlap**: `max_pairwise_bbox_iou > 0.15`
   - At least one pair of bounding boxes overlaps significantly
   - Indicates visual occlusion

2. **Tight Clustering**: `clustering_coefficient > 0.5`
   - More than 50% of object pairs are within threshold distance
   - Indicates spatial grouping/proximity

### Period-Level Detection

Occlusion periods are identified using a **sliding window approach**:

1. Define window size (e.g., 25 frames = 1 second at 25fps)
2. For each window:
   - Calculate fraction of frames flagged as high-occlusion
   - If fraction > threshold (e.g., 30%) → mark as occlusion period
3. Extend period until occlusion drops below threshold
4. Extract transition frames (start/end of each period)

## Usage

### Basic Usage

```python
from src.yolo_prescan import compute_yolo_prescan_results, yolo_prescan_to_df
import pandas as pd

# Load YOLO tracking data
yolo_df = pd.read_parquet("path/to/yolo_tracks.parquet")

# Run pre-scan analysis
prescan_results = compute_yolo_prescan_results(
    yolo_df,
    fps=25.0,                            # Video frame rate
    window_seconds=1.0,                  # Sliding window duration
    high_occlusion_threshold=0.3,        # 30% of frames in window
    occlusion_iou_threshold=0.15,        # Bbox overlap threshold
    clustering_distance_threshold=0.15,  # Centroid distance threshold
)

# Extract metrics DataFrame
metrics_df = yolo_prescan_to_df(prescan_results['per_frame_metrics'])

# Get occlusion periods
occlusion_periods = prescan_results['occlusion_periods']
# [(start_frame1, end_frame1), (start_frame2, end_frame2), ...]
```

### Integration with Adaptive Chunking

```python
# Use transition frames as chunk boundaries
transition_frames = prescan_results['transition_frames']

# Create chunks that respect occlusion boundaries
chunks = []
boundaries = sorted([0] + list(transition_frames) + [total_frames])

for i in range(len(boundaries) - 1):
    chunks.append((boundaries[i], boundaries[i + 1]))

# Process chunks with occlusion-aware strategies
for start, end in chunks:
    is_high_occ = any(
        occ_start <= start <= occ_end or occ_start <= end <= occ_end
        for occ_start, occ_end in occlusion_periods
    )
    
    if is_high_occ:
        # Apply specialized processing for high-occlusion chunks
        process_with_temporal_model(video, start, end)
    else:
        # Standard processing
        process_standard(video, start, end)
```

## Configuration Parameters

### Required Parameters

- **`yolo_df`**: DataFrame with YOLO tracking data
  - Required columns: `frame`, `track_id`, `x1`, `y1`, `x2`, `y2`, `cx_norm`, `cy_norm`, `confidence`

- **`fps`**: Video frame rate (float)

### Tunable Thresholds

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `window_seconds` | 1.0 | Sliding window duration | Increase for smoother detection; decrease for finer temporal resolution |
| `high_occlusion_threshold` | 0.3 | Fraction of frames in window to trigger | Lower = more sensitive; higher = fewer false positives |
| `occlusion_iou_threshold` | 0.15 | Bbox IoU for overlap detection | Depends on subject size and expected overlap |
| `clustering_distance_threshold` | 0.15 | Normalized centroid distance | Lower = tighter clustering required; higher = more permissive |

### Threshold Tuning Guidelines

#### IoU Threshold (`occlusion_iou_threshold`)

- **0.05 - 0.10**: Very sensitive, detects any bbox overlap
- **0.10 - 0.20**: Moderate, good for typical multi-subject tracking
- **0.20 - 0.30**: Conservative, only significant overlaps

#### Clustering Threshold (`clustering_distance_threshold`)

For normalized coordinates (0-1 scale):

- **0.05 - 0.10**: Very tight clustering (subjects nearly touching)
- **0.10 - 0.20**: Moderate clustering (subjects in proximity)
- **0.20 - 0.30**: Loose clustering (subjects in same general area)

#### Window Threshold (`high_occlusion_threshold`)

- **0.20 - 0.30**: Sensitive, catches brief occlusion events
- **0.30 - 0.50**: Balanced, filters transient overlaps
- **0.50 - 0.70**: Conservative, only sustained occlusion

## Output Structure

### `prescan_results` Dictionary

```python
{
    'per_frame_metrics': List[Dict],      # Detailed metrics per frame
    'occlusion_periods': List[Tuple],     # [(start, end), ...]
    'transition_frames': np.ndarray,      # [frame1, frame2, ...]
    'total_frames': int,                  # Total frame count
    'video_duration_seconds': float,      # Video duration
    'fps': float,                         # Frame rate
    'window_frames': int,                 # Window size in frames
    'high_occlusion_threshold': float,    # Threshold used
}
```

### `metrics_df` DataFrame Columns

- `frame_idx`: Frame index
- `num_objects`: Object count
- `objects_present`: Comma-separated track IDs
- `min_centroid_distance`: Minimum pairwise distance
- `mean_centroid_distance`: Mean pairwise distance
- `clustering_coefficient`: Clustering metric
- `max_pairwise_bbox_iou`: Maximum IoU
- `mean_pairwise_bbox_iou`: Mean IoU
- `num_overlapping_pairs`: Count of overlapping pairs
- `mean_bbox_area`: Average bbox area
- `min_bbox_area`: Minimum bbox area
- `max_bbox_area`: Maximum bbox area
- `bbox_area_variance`: Bbox area variance
- `is_high_occlusion`: Boolean flag
- `is_object_count_change`: Boolean flag
- `mean_confidence`: Average YOLO confidence

## Example Results

### Chicken Tracking Video (59.76s, 1494 frames)

**Detected Occlusion Periods:**

| Period | Frames | Time | Duration | Avg Objects | Max IoU | Avg Clustering |
|--------|--------|------|----------|-------------|---------|----------------|
| 1 | 720-777 | 28.8s - 31.1s | 2.28s | 2.4 | 0.358 | 0.149 |
| 2 | 778-830 | 31.1s - 33.2s | 2.08s | 2.1 | 0.243 | 0.174 |
| 3 | 1199-1254 | 48.0s - 50.2s | 2.20s | 1.9 | 0.192 | 0.229 |

**Statistics:**
- High-occlusion frames: 79 / 1251 (6.3%)
- Transition frames: [720, 777, 778, 830, 1199, 1254]
- Mean detections per frame: 1.94
- Unique track IDs: 94

## Comparison with KMeans Approach

| Aspect | YOLO-Based | KMeans-Based |
|--------|------------|--------------|
| **Input** | Pre-computed YOLO tracks | Raw video frames |
| **Processing** | Lightweight metrics on bboxes | Frame extraction + pixel clustering |
| **Semantics** | Object-level (bboxes, centroids) | Pixel-level (color clusters) |
| **Efficiency** | Very fast (reuses existing data) | Slower (frame extraction + clustering) |
| **Interpretability** | High (IoU, distance) | Moderate (silhouette score) |
| **Identity awareness** | Yes (track IDs) | No |
| **Best for** | When YOLO tracking exists | When no detection data available |

## Visualization

The pipeline generates a multi-panel plot showing:

1. **# Objects over time** - Detection count per frame
2. **Max Bbox IoU** - Highest overlap per frame (with threshold line)
3. **Clustering Coefficient** - Spatial grouping metric (with threshold line)
4. **Min Centroid Distance** - Closest object pair
5. **High Occlusion Flag** - Binary indicator

Red shaded regions indicate detected occlusion periods.

## File Structure

```
src/
  yolo_prescan.py                    # Core pipeline implementation

examples/
  yolo_prescan_demo.py               # Complete usage example

sandbox/results/yolox/
  chicken_tracks_yolo.parquet        # Input: YOLO tracking data
  yolo_prescan_metrics.parquet       # Output: Per-frame metrics
  yolo_prescan_config.json           # Output: Prescan configuration
  yolo_prescan_analysis.png          # Output: Visualization

docs/
  YOLO_PRESCAN_README.md             # This file
```

## Future Enhancements

1. **Multi-scale Analysis**: Compute metrics at multiple window sizes
2. **Temporal Smoothing**: Apply moving average to reduce noise
3. **Confidence Weighting**: Weight metrics by YOLO detection confidence
4. **Track Stability**: Incorporate track lifetime/consistency
5. **Scene Context**: Add global scene metrics (e.g., frame-level crowding)
6. **Adaptive Thresholds**: Learn thresholds from data distribution

## References

- YOLO tracking implementation: See pinned notebook
- Adaptive chunking: `src/adaptive_chunking.py`
- Metrics computation: `src/metrics.py` (KMeans-based approach)
- Example usage: `examples/yolo_prescan_demo.py`

## License

Same as parent project.
