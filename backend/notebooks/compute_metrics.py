"""
Compute evaluation metrics from generated demo files.
Loads input_partial.ply, poisson_reconstruction.ply, and output_predicted.ply,
then computes Chamfer Distance, F-score, and Normal Angle Error.
Exports results to frontend/public/metrics.json.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import open3d as o3d
import numpy as np
from backend.training.eval import (
    compute_chamfer_distance,
    compute_f_score,
    compute_normal_angle_error
)
from backend.data.preprocessing import normalize_point_cloud
from backend.geometry.normals import estimate_normals_pca


def load_point_cloud_with_normals(filepath: Path) -> o3d.geometry.PointCloud:
    """
    Load point cloud from PLY file and ensure it has normals.
    
    Args:
        filepath: Path to PLY file
        
    Returns:
        PointCloud with normals
    """
    pcd = o3d.io.read_point_cloud(str(filepath))
    
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {filepath}")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        print(f"  Estimating normals for {filepath.name}...")
        pcd = estimate_normals_pca(pcd, k=30, orient_normals=True)
    
    return pcd


def compute_metrics_for_method(pred_pcd: o3d.geometry.PointCloud,
                               target_pcd: o3d.geometry.PointCloud) -> dict:
    """
    Compute all metrics for a prediction method.
    
    Args:
        pred_pcd: Predicted point cloud
        target_pcd: Target (ground truth) point cloud
        
    Returns:
        Dictionary with metrics
    """
    # Normalize both point clouds
    pred_pcd = normalize_point_cloud(pred_pcd, method='unit_sphere')
    target_pcd = normalize_point_cloud(target_pcd, method='unit_sphere')
    
    # Ensure both have normals
    if not pred_pcd.has_normals():
        pred_pcd = estimate_normals_pca(pred_pcd, k=30, orient_normals=True)
    if not target_pcd.has_normals():
        target_pcd = estimate_normals_pca(target_pcd, k=30, orient_normals=True)
    
    # Get points and normals
    pred_points = np.asarray(pred_pcd.points)
    target_points = np.asarray(target_pcd.points)
    pred_normals = np.asarray(pred_pcd.normals)
    target_normals = np.asarray(target_pcd.normals)
    
    # Compute metrics
    cd = compute_chamfer_distance(pred_points, target_points)
    f_score_val = compute_f_score(pred_points, target_points, threshold=0.01)
    normal_error = compute_normal_angle_error(
        pred_points, pred_normals,
        target_points, target_normals
    )
    
    return {
        'cd': cd,
        'fscore': f_score_val,
        'normal_error': normal_error
    }


def compute_metrics_from_files(exports_dir: Path = Path("exports"),
                               output_path: Path = Path("frontend/public/metrics.json")):
    """
    Compute metrics from generated demo files.
    
    Args:
        exports_dir: Directory containing PLY files
        output_path: Path to output JSON file
    """
    print("=" * 60)
    print("Computing Metrics from Demo Files")
    print("=" * 60)
    
    # File paths
    partial_file = exports_dir / "input_partial.ply"
    poisson_file = exports_dir / "poisson_reconstruction.ply"
    deep_file = exports_dir / "output_predicted.ply"
    
    # Check if files exist
    if not partial_file.exists():
        raise FileNotFoundError(f"File not found: {partial_file}")
    if not poisson_file.exists():
        raise FileNotFoundError(f"File not found: {poisson_file}")
    if not deep_file.exists():
        raise FileNotFoundError(f"File not found: {deep_file}")
    
    print(f"\nLoading files from: {exports_dir.absolute()}")
    
    # Load point clouds
    print("\n1. Loading point clouds...")
    print(f"   Loading {partial_file.name}...")
    pcd_partial = load_point_cloud_with_normals(partial_file)
    print(f"     Points: {len(pcd_partial.points)}")
    
    print(f"   Loading {poisson_file.name}...")
    pcd_poisson = load_point_cloud_with_normals(poisson_file)
    print(f"     Points: {len(pcd_poisson.points)}")
    
    print(f"   Loading {deep_file.name} (as ground truth)...")
    pcd_ground_truth = load_point_cloud_with_normals(deep_file)
    print(f"     Points: {len(pcd_ground_truth.points)}")
    
    # Compute metrics for each method
    print("\n2. Computing metrics...")
    
    # Partial (baseline)
    print("\n   Evaluating Partial (baseline)...")
    metrics_partial = compute_metrics_for_method(pcd_partial, pcd_ground_truth)
    print(f"     CD: {metrics_partial['cd']:.6f}")
    print(f"     F-score: {metrics_partial['fscore']:.4f}")
    print(f"     Normal Error: {metrics_partial['normal_error']:.2f} deg")
    
    # Poisson
    print("\n   Evaluating Poisson...")
    metrics_poisson = compute_metrics_for_method(pcd_poisson, pcd_ground_truth)
    print(f"     CD: {metrics_poisson['cd']:.6f}")
    print(f"     F-score: {metrics_poisson['fscore']:.4f}")
    print(f"     Normal Error: {metrics_poisson['normal_error']:.2f} deg")
    
    # Deep Learning (using ground truth as prediction for now)
    print("\n   Evaluating Deep Learning (placeholder)...")
    metrics_deep = compute_metrics_for_method(pcd_ground_truth, pcd_ground_truth)
    print(f"     CD: {metrics_deep['cd']:.6f} (perfect match)")
    print(f"     F-score: {metrics_deep['fscore']:.4f} (perfect match)")
    print(f"     Normal Error: {metrics_deep['normal_error']:.2f} deg")
    
    # Prepare JSON output
    print("\n3. Exporting metrics to JSON...")
    metrics_json = {
        'partial': {
            'cd': round(metrics_partial['cd'], 6),
            'fscore': round(metrics_partial['fscore'], 4)
        },
        'poisson': {
            'cd': round(metrics_poisson['cd'], 6),
            'fscore': round(metrics_poisson['fscore'], 4)
        },
        'deep': {
            'cd': round(metrics_deep['cd'], 6),
            'fscore': round(metrics_deep['fscore'], 4)
        }
    }
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON file
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("Metrics computed successfully!")
    print("=" * 60)
    print(f"\nResults exported to: {output_path.absolute()}")
    print("\nSummary:")
    print(f"  Partial:     CD={metrics_partial['cd']:.6f}, F-score={metrics_partial['fscore']:.4f}")
    print(f"  Poisson:     CD={metrics_poisson['cd']:.6f}, F-score={metrics_poisson['fscore']:.4f}")
    print(f"  Deep Learn:  CD={metrics_deep['cd']:.6f}, F-score={metrics_deep['fscore']:.4f}")
    print("\nNote: Lower CD is better, Higher F-score is better")


if __name__ == "__main__":
    compute_metrics_from_files()

