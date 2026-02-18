"""
Example usage of the YOLO-based occlusion detection pipeline.

This script demonstrates how to:
1. Load YOLO tracking data
2. Run the pre-scan analysis
3. Integrate results with adaptive chunking
4. Visualize and export results
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.yolo_prescan import compute_yolo_prescan_results, yolo_prescan_to_df


def main():
    """Run the YOLO pre-scan pipeline on chicken tracking data."""
    
    # ========================================================================
    # STEP 1: Load YOLO tracking data
    # ========================================================================
    print("Loading YOLO tracking data...")
    yolo_df = pd.read_parquet("sandbox/results/yolox/chicken_tracks_yolo.parquet")
    
    print(f"  • Loaded {len(yolo_df)} detections")
    print(f"  • Frame range: {yolo_df['frame'].min()} - {yolo_df['frame'].max()}")
    print(f"  • Unique tracks: {yolo_df['track_id'].nunique()}")
    
    # ========================================================================
    # STEP 2: Run pre-scan analysis
    # ========================================================================
    print("\nRunning YOLO pre-scan analysis...")
    
    prescan_results = compute_yolo_prescan_results(
        yolo_df,
        fps=25.0,                            # Video frame rate
        window_seconds=1.0,                  # Sliding window duration
        high_occlusion_threshold=0.3,        # 30% of frames in window
        occlusion_iou_threshold=0.15,        # Bbox overlap threshold
        clustering_distance_threshold=0.15,  # Centroid distance threshold
    )
    
    print(f"  • Analyzed {prescan_results['total_frames']} frames")
    print(f"  • Detected {len(prescan_results['occlusion_periods'])} occlusion periods")
    
    # ========================================================================
    # STEP 3: Extract metrics DataFrame
    # ========================================================================
    print("\nConverting per-frame metrics to DataFrame...")
    metrics_df = yolo_prescan_to_df(prescan_results['per_frame_metrics'])
    
    print(f"  • Metrics computed for {len(metrics_df)} frames")
    print(f"  • High-occlusion frames: {metrics_df['is_high_occlusion'].sum()}")
    
    # ========================================================================
    # STEP 4: Display occlusion periods
    # ========================================================================
    print("\n" + "=" * 70)
    print("DETECTED OCCLUSION PERIODS")
    print("=" * 70)
    
    for i, (start, end) in enumerate(prescan_results['occlusion_periods'], 1):
        duration = (end - start) / prescan_results['fps']
        time_start = start / prescan_results['fps']
        
        period_metrics = metrics_df[
            (metrics_df['frame_idx'] >= start) & 
            (metrics_df['frame_idx'] <= end)
        ]
        
        print(f"\nPeriod {i}:")
        print(f"  Frames:     {start} - {end}")
        print(f"  Time:       {time_start:.2f}s - {time_start + duration:.2f}s")
        print(f"  Duration:   {duration:.2f}s")
        print(f"  Avg objects: {period_metrics['num_objects'].mean():.1f}")
        print(f"  Max IoU:     {period_metrics['max_pairwise_bbox_iou'].max():.3f}")
        print(f"  Avg clustering: {period_metrics['clustering_coefficient'].mean():.3f}")
    
    # ========================================================================
    # STEP 5: Integration with adaptive chunking
    # ========================================================================
    print("\n" + "=" * 70)
    print("ADAPTIVE CHUNKING INTEGRATION")
    print("=" * 70)
    
    print("\nTransition frames (suggested chunk boundaries):")
    print(f"  {prescan_results['transition_frames'].tolist()}")
    
    print("\nExample: Create chunks respecting occlusion boundaries")
    chunks = create_adaptive_chunks(
        total_frames=prescan_results['total_frames'],
        transition_frames=prescan_results['transition_frames'],
        max_chunk_size=500,
    )
    
    print(f"\n  Generated {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks, 1):
        is_occlusion = any(
            occ_start <= start <= occ_end or occ_start <= end <= occ_end
            for occ_start, occ_end in prescan_results['occlusion_periods']
        )
        marker = "🚨" if is_occlusion else "  "
        print(f"    {marker} Chunk {i:2d}: frames {start:4d} - {end:4d} ({end-start:3d} frames)")
    
    # ========================================================================
    # STEP 6: Save outputs
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    output_dir = Path("sandbox/results/yolox")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics DataFrame
    metrics_path = output_dir / "yolo_prescan_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"  ✓ Saved metrics: {metrics_path}")
    
    # Save configuration
    config_path = output_dir / "yolo_prescan_config.json"
    config = {
        "occlusion_periods": prescan_results['occlusion_periods'],
        "transition_frames": prescan_results['transition_frames'].tolist(),
        "total_frames": prescan_results['total_frames'],
        "video_duration_seconds": prescan_results['video_duration_seconds'],
        "fps": prescan_results['fps'],
        "chunks": chunks,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config: {config_path}")
    
    # ========================================================================
    # STEP 7: Visualization
    # ========================================================================
    print("\nCreating visualization...")
    fig = create_prescan_visualization(metrics_df, prescan_results)
    
    viz_path = output_dir / "yolo_prescan_analysis.png"
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved visualization: {viz_path}")
    
    print("\n" + "=" * 70)
    print("✓ YOLO pre-scan pipeline completed successfully!")
    print("=" * 70)


def create_adaptive_chunks(
    total_frames: int,
    transition_frames: list,
    max_chunk_size: int = 500,
) -> list:
    """
    Create chunks respecting occlusion period boundaries.
    
    Args:
        total_frames: Total number of frames in video
        transition_frames: List of frames marking occlusion boundaries
        max_chunk_size: Maximum frames per chunk
    
    Returns:
        List of (start_frame, end_frame) tuples
    """
    chunks = []
    
    # Add video start and end to transition points
    boundaries = sorted([0] + list(transition_frames) + [total_frames])
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # If segment is too large, subdivide it
        while end - start > max_chunk_size:
            chunks.append((start, start + max_chunk_size))
            start += max_chunk_size
        
        if start < end:
            chunks.append((start, end))
    
    return chunks


def create_prescan_visualization(
    metrics_df: pd.DataFrame,
    prescan_results: dict,
) -> plt.Figure:
    """Create comprehensive visualization of pre-scan results."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Plot 1: Number of objects
    ax = axes[0]
    ax.plot(metrics_df['frame_idx'], metrics_df['num_objects'], 
            linewidth=0.8, alpha=0.7, color='blue')
    ax.set_ylabel('# Objects', fontsize=10)
    ax.set_title('YOLO-based Occlusion Detection', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight occlusion periods
    for start, end in prescan_results['occlusion_periods']:
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    # Plot 2: Max bbox IoU
    ax = axes[1]
    ax.plot(metrics_df['frame_idx'], metrics_df['max_pairwise_bbox_iou'], 
            linewidth=0.8, color='orange', alpha=0.7)
    ax.axhline(0.15, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Max Bbox IoU', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    for start, end in prescan_results['occlusion_periods']:
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    # Plot 3: Clustering coefficient
    ax = axes[2]
    ax.plot(metrics_df['frame_idx'], metrics_df['clustering_coefficient'], 
            linewidth=0.8, color='purple', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Clustering\nCoefficient', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    for start, end in prescan_results['occlusion_periods']:
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    # Plot 4: High occlusion flag
    ax = axes[3]
    ax.fill_between(metrics_df['frame_idx'], 0, 
                     metrics_df['is_high_occlusion'].astype(int), 
                     alpha=0.5, color='red')
    ax.set_ylabel('High\nOcclusion', fontsize=10)
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    
    for start, end in prescan_results['occlusion_periods']:
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
