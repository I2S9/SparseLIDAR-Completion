"""
Test script for preprocessing functions.
Run this to verify preprocessing works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.data.preprocessing import (
    downsample_voxel,
    normalize_point_cloud,
    create_partial_point_cloud,
    add_gaussian_noise,
    mask_points_by_angle
)
from backend.visualization.visualize_o3d import show_point_clouds


def test_preprocessing():
    """
    Test all preprocessing functions.
    """
    print("=" * 60)
    print("Preprocessing Functions Test")
    print("=" * 60)
    
    # Create test point cloud
    print("\n1. Creating test point cloud...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    pcd_original = mesh.sample_points_uniformly(number_of_points=3000)
    print(f"   Original: {len(pcd_original.points)} points")
    
    # Test downsampling
    print("\n2. Testing downsampling...")
    pcd_down = downsample_voxel(pcd_original, voxel_size=0.1)
    print(f"   Downsampled: {len(pcd_down.points)} points")
    
    # Test normalization
    print("\n3. Testing normalization...")
    pcd_norm = normalize_point_cloud(pcd_original, method='unit_sphere')
    points_norm = np.asarray(pcd_norm.points)
    max_dist = np.max(np.linalg.norm(points_norm, axis=1))
    print(f"   Normalized max distance: {max_dist:.4f} (should be ~1.0)")
    
    # Test partial point cloud (mask side)
    print("\n4. Testing partial point cloud (mask side)...")
    pcd_partial = create_partial_point_cloud(
        pcd_original,
        method='mask_side',
        direction=[1, 0, 0],
        mask_ratio=0.5
    )
    print(f"   Partial (mask side): {len(pcd_partial.points)} points")
    
    # Test angle masking
    print("\n5. Testing angle masking (>45°)...")
    pcd_angle = create_partial_point_cloud(
        pcd_original,
        method='mask_angle',
        direction=[1, 0, 0],
        max_angle=45.0
    )
    print(f"   Angle masked: {len(pcd_angle.points)} points")
    
    # Test noise
    print("\n6. Testing Gaussian noise...")
    pcd_noisy = add_gaussian_noise(pcd_original, std=0.01, relative=True)
    print(f"   Noisy: {len(pcd_noisy.points)} points")
    
    # Test complete pipeline
    print("\n7. Testing complete preprocessing pipeline...")
    pcd = pcd_original
    pcd = downsample_voxel(pcd, voxel_size=0.1)
    pcd = create_partial_point_cloud(pcd, method='mask_angle', direction=[1, 0, 0], max_angle=45.0)
    pcd = add_gaussian_noise(pcd, std=0.01, relative=True)
    pcd = normalize_point_cloud(pcd, method='unit_sphere')
    print(f"   Final: {len(pcd.points)} points")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    
    # Visualize comparison
    print("\nDisplaying visualization (original vs preprocessed)...")
    print("Close the window to continue.")
    show_point_clouds([pcd_original, pcd], window_name="Original vs Preprocessed")


if __name__ == "__main__":
    test_preprocessing()

