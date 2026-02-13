# Parameter Sensitivity Testing Guide

## Overview

After running a YOLO prescan (which takes ~5 minutes), you can test different occlusion detection parameters **without re-running YOLO inference**. This is much faster because it only recomputes the metrics and chunk boundaries from the existing detection data.

## Quick Start

### 1. Run a prescan-only mode (one time)

```bash
# Edit config to enable prescan_only
python -m script.sam3.run_sam3_hf
```

This saves `yolo_tracking.parquet` in the run directory (e.g., `20260213_142342_yolo_prescan/`).

### 2. Test different parameters (fast, repeatable)

```bash
python notebook/parameter_sensitivity_example.py ext-data/output/results/sam3-hf/20260213_142342_yolo_prescan
```

This will:
- Load the existing YOLO tracking data
- Test 3 parameter sets (strict, moderate, relaxed)
- Generate comparison visualizations
- Save results to `parameter_comparison/` subdirectory

## What Gets Tested

The script tests different thresholds for occlusion detection:

| Parameter Set | `occlusion_iou_threshold` | `high_occlusion_threshold` | Result |
|---------------|---------------------------|----------------------------|--------|
| **Strict**    | 0.08 | 0.15 | Flags many periods as occlusion (conservative) |
| **Moderate** (default) | 0.15 | 0.30 | Balanced detection |
| **Relaxed**   | 0.20 | 0.40 | Only severe occlusions flagged |

## Understanding the Output

### Generated Files

```
20260213_142342_yolo_prescan/
└── parameter_comparison/
    ├── yolo_prescan_overview_strict.png
    ├── yolo_prescan_overview_moderate_(default).png
    ├── yolo_prescan_overview_relaxed.png
    └── parameter_comparison_summary.csv
```

### Visualization Components

Each `yolo_prescan_overview_*.png` contains:
1. **Object count over time** (top panel)
2. **Max bbox IoU** - shows object overlap (2nd panel)
3. **Clustering coefficient** - shows spatial clustering (3rd panel)
4. **High occlusion flag** - binary indicator (bottom panel)

**Red shaded regions** = Detected occlusion periods
**Blue dashed lines** = Chunk boundaries (tracker re-initialization points)

### Summary CSV

The `parameter_comparison_summary.csv` shows:
- Number of occlusion periods detected
- Number of transition frames identified
- Number of chunks after adaptive adjustment
- Number of boundaries that were shifted

## Customizing Parameters

Edit `PARAMETER_SETS` in `notebook/parameter_sensitivity_example.py`:

```python
PARAMETER_SETS = [
    {
        "name": "custom_test",
        "occlusion_iou_threshold": 0.12,      # Adjust this
        "high_occlusion_threshold": 0.25,     # Adjust this
        "clustering_distance_threshold": 0.15,
    },
    # Add more sets...
]
```

## When to Use This

1. **Initial tuning**: Run prescan once, test multiple thresholds to find optimal values
2. **Debugging**: Understand why certain frames/periods are flagged as occlusion
3. **Validation**: Compare different parameter sets side-by-side before committing to a full SAM3 run

## Key Insight

The parameter sensitivity script **reuses YOLO detections** (saved in `yolo_tracking.parquet`). Only the metric computation and boundary placement logic runs again, which is very fast (<1 second per parameter set).

This is much more efficient than re-running the full prescan (~5 min) or full SAM3 pipeline (~75 min) for each parameter test.
