"""
Generate metrics.json file for frontend display.
This script evaluates all methods and exports metrics to JSON.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import open3d as o3d
from backend.data.preprocessing import (
    create_partial_point_cloud,
    normalize_point_cloud
)
from backend.training.eval import (
    evaluate_all_methods,
    export_metrics_json,
    print_results_table
)


def generate_metrics():
    """
    Generate metrics for demo files and export to JSON.
    """
    print("=" * 60)
    print("Generating Metrics for Frontend")
    print("=" * 60)
    
    # Create synthetic test sample (same as demo files)
    print("\nCreating test sample...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    pcd_full = mesh.sample_points_uniformly(number_of_points=3000)
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    
    # Create partial function (same as demo files)
    def create_partial(pcd):
        return create_partial_point_cloud(
            pcd,
            method='mask_angle',
            direction=[1, 0, 0],
            max_angle=45.0
        )
    
    # Create partial point cloud
    pcd_partial = create_partial(pcd_full)
    
    # Evaluate all methods
    print("\nEvaluating all methods...")
    results = evaluate_all_methods(
        pcd_partial=pcd_partial,
        pcd_full=pcd_full,
        model=None,  # No model for now
        device='cpu'
    )
    
    # Print results table
    print_results_table(results)
    
    # Export to JSON
    output_path = project_root / "frontend" / "public" / "metrics.json"
    export_metrics_json(results, str(output_path))
    
    print("\n" + "=" * 60)
    print("Metrics generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    generate_metrics()

