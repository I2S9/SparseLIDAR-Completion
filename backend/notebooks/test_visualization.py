"""
Test script to load and visualize a point cloud.
Run this script to test the visualization functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.data.datasets import load_point_cloud, normalize_point_cloud, get_point_cloud_info
from backend.visualization.visualize_o3d import show_point_cloud


def test_visualization():
    """
    Test loading and visualizing a point cloud.
    """
    print("=" * 50)
    print("Point Cloud Visualization Test")
    print("=" * 50)
    
    # Example: create a simple synthetic point cloud if no file is provided
    # In a real scenario, you would load a KITTI or ShapeNet file
    
    # Check if a file path is provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nLoading point cloud from: {file_path}")
        pcd = load_point_cloud(file_path)
    else:
        print("\nNo file provided. Creating a simple synthetic point cloud for testing...")
        import open3d as o3d
        import numpy as np
        
        # Create a simple sphere point cloud for testing
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        
        print("Created synthetic sphere point cloud (1000 points)")
    
    # Display point cloud info
    info = get_point_cloud_info(pcd)
    print(f"\nPoint Cloud Information:")
    print(f"  Number of points: {info['num_points']}")
    print(f"  Has colors: {info['has_colors']}")
    print(f"  Has normals: {info['has_normals']}")
    if 'centroid' in info:
        print(f"  Centroid: {info['centroid']}")
        print(f"  Extent: {info['extent']}")
    
    # Normalize the point cloud
    print("\nNormalizing point cloud...")
    pcd_normalized = normalize_point_cloud(pcd)
    
    # Display normalized point cloud
    print("\nDisplaying point cloud...")
    print("Controls:")
    print("  - Left mouse button + drag: Rotate")
    print("  - Right mouse button + drag: Pan")
    print("  - Scroll wheel: Zoom")
    print("  - Q or close window: Quit")
    
    show_point_cloud(pcd_normalized, window_name="Point Cloud Visualization")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    test_visualization()

