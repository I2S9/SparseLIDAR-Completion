"""
Create demo files for the web interface.
Generates input_partial.ply, poisson_reconstruction.ply, and output_predicted.ply
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import open3d as o3d
import numpy as np
from backend.data.preprocessing import (
    create_partial_point_cloud,
    normalize_point_cloud
)
from backend.geometry.normals import estimate_normals_pca
from backend.geometry.reconstruction import poisson_surface_reconstruction, mesh_to_point_cloud
from backend.visualization.export_ply import save_point_cloud_ply, save_mesh_ply


def create_demo_files():
    """
    Create demo files for the web interface.
    """
    print("=" * 60)
    print("Creating Demo Files for Web Interface")
    print("=" * 60)
    
    # Create a more interesting shape for demo
    print("\n1. Creating base point cloud...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    pcd_full = mesh.sample_points_uniformly(number_of_points=3000)
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    print(f"   Full cloud: {len(pcd_full.points)} points")
    
    # Create partial (input)
    print("\n2. Creating partial input...")
    pcd_partial = create_partial_point_cloud(
        pcd_full,
        method='mask_angle',
        direction=[1, 0, 0],
        max_angle=45.0
    )
    print(f"   Partial cloud: {len(pcd_partial.points)} points")
    
    # Poisson reconstruction
    print("\n3. Creating Poisson reconstruction...")
    pcd_with_normals = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    poisson_mesh = poisson_surface_reconstruction(
        pcd_with_normals,
        depth=9,
        scale=1.1
    )
    print(f"   Poisson mesh: {len(poisson_mesh.vertices)} vertices, {len(poisson_mesh.triangles)} triangles")
    
    # Convert Poisson mesh to point cloud for comparison
    pcd_poisson = mesh_to_point_cloud(poisson_mesh, number_of_points=len(pcd_full.points))
    print(f"   Poisson point cloud: {len(pcd_poisson.points)} points")
    
    # For Deep Learning output, use full as placeholder
    # (In real scenario, this would be model prediction)
    print("\n4. Creating Deep Learning output (placeholder)...")
    pcd_dl = pcd_full  # Placeholder - in real scenario, use model prediction
    print(f"   DL output: {len(pcd_dl.points)} points")
    
    # Export all files
    print("\n5. Exporting files...")
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)
    
    # Export partial input
    save_point_cloud_ply(
        pcd_partial,
        output_dir / "input_partial.ply",
        normalize=True
    )
    
    # Export Poisson reconstruction (as point cloud)
    save_point_cloud_ply(
        pcd_poisson,
        output_dir / "poisson_reconstruction.ply",
        normalize=True
    )
    
    # Export Poisson mesh
    save_mesh_ply(
        poisson_mesh,
        output_dir / "poisson_mesh.ply",
        normalize=True
    )
    
    # Export Deep Learning output
    save_point_cloud_ply(
        pcd_dl,
        output_dir / "output_predicted.ply",
        normalize=True
    )
    
    print("\n" + "=" * 60)
    print("Demo files created successfully!")
    print("=" * 60)
    print(f"\nFiles saved in: {output_dir.absolute()}")
    print("  - input_partial.ply")
    print("  - poisson_reconstruction.ply")
    print("  - poisson_mesh.ply")
    print("  - output_predicted.ply")
    print("\nYou can now use these files in the web interface!")


if __name__ == "__main__":
    create_demo_files()

