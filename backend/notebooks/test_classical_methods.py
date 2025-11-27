"""
Test script for classical methods: normal estimation and surface reconstruction.
This script demonstrates the complete pipeline:
1. Load point cloud
2. Estimate normals (PCA)
3. Reconstruct surface (Poisson)
4. Visualize results
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.data.datasets import load_point_cloud
from backend.data import preprocessing
from backend.geometry.normals import estimate_normals_pca
from backend.geometry.reconstruction import poisson_surface_reconstruction, mesh_to_point_cloud
from backend.visualization.visualize_o3d import show_point_clouds, show_point_cloud


def test_classical_methods():
    """
    Test complete pipeline: load → estimate normals → reconstruct → visualize
    """
    print("=" * 60)
    print("Classical Methods Test: Normals + Poisson Reconstruction")
    print("=" * 60)
    
    # Step 1: Load or create point cloud
    print("\n1. Loading point cloud...")
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"   Loading from: {file_path}")
        pcd = load_point_cloud(file_path)
    else:
        print("   Creating synthetic point cloud...")
        # Create a more complex shape for better demonstration
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
        pcd = mesh.sample_points_uniformly(number_of_points=2000)
    
    # Normalize for better visualization
    pcd = preprocessing.normalize_point_cloud(pcd, method='unit_sphere')
    print(f"   Points: {len(pcd.points)}")
    
    # Step 2: Estimate normals using PCA
    print("\n2. Estimating normals using PCA...")
    print("   Computing k-nearest neighbors and local PCA...")
    pcd_with_normals = estimate_normals_pca(pcd, k=30, orient_normals=True)
    
    if pcd_with_normals.has_normals():
        normals = np.asarray(pcd_with_normals.normals)
        print(f"   Normals estimated: {len(normals)}")
        print(f"   Normal sample: {normals[0]}")
    else:
        print("   ERROR: Normals not estimated!")
        return
    
    # Step 3: Poisson Surface Reconstruction
    print("\n3. Performing Poisson Surface Reconstruction...")
    print("   This may take a moment...")
    try:
        mesh = poisson_surface_reconstruction(
            pcd_with_normals,
            depth=9,
            scale=1.1
        )
        print(f"   Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    except Exception as e:
        print(f"   ERROR in reconstruction: {e}")
        return
    
    # Step 4: Convert mesh back to point cloud for comparison
    print("\n4. Converting mesh to point cloud...")
    pcd_reconstructed = mesh_to_point_cloud(mesh, number_of_points=len(pcd.points))
    print(f"   Reconstructed points: {len(pcd_reconstructed.points)}")
    
    # Step 5: Visualize
    print("\n5. Visualization:")
    print("   - Window 1: Original point cloud with normals")
    print("   - Window 2: Original vs Reconstructed comparison")
    print("   Close windows to continue.")
    
    # Visualize original with normals
    print("\n   Showing original point cloud with normals...")
    show_point_cloud(pcd_with_normals, window_name="Original with Normals")
    
    # Visualize comparison
    print("\n   Showing comparison: Original vs Reconstructed...")
    show_point_clouds(
        [pcd, pcd_reconstructed],
        window_name="Original vs Poisson Reconstructed"
    )
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_classical_methods()

