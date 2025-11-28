"""
Export predicted point clouds or meshes to PLY format for web visualization.
Always exports in normalized coordinate system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import open3d as o3d
from typing import Union


def save_point_cloud_ply(points: Union[np.ndarray, o3d.geometry.PointCloud], 
                         filename: str,
                         normalize: bool = True) -> None:
    """
    Save point cloud to PLY file.
    Always exports in normalized coordinate system.
    
    Args:
        points: Point cloud as numpy array (Nx3) or Open3D PointCloud
        filename: Output filename (will add .ply if not present)
        normalize: If True, normalize to unit sphere before export
    """
    # Ensure filename has .ply extension
    filename = Path(filename)
    if filename.suffix.lower() != '.ply':
        filename = filename.with_suffix('.ply')
    
    # Convert to Open3D PointCloud if needed
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    else:
        pcd = points
    
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty, cannot export")
    
    # Normalize to unit sphere if requested
    if normalize:
        points_array = np.asarray(pcd.points)
        
        # Center the point cloud
        centroid = np.mean(points_array, axis=0)
        points_centered = points_array - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        
        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points_normalized)
        
        # Preserve normals after normalization (they don't need to be transformed)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Create directory if needed
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to PLY (Open3D automatically includes normals if present)
    success = o3d.io.write_point_cloud(str(filename), pcd, write_ascii=False)
    
    if not success:
        raise IOError(f"Failed to write point cloud to {filename}")
    
    print(f"Point cloud saved: {filename} ({len(pcd.points)} points)")


def save_mesh_ply(mesh: o3d.geometry.TriangleMesh,
                 filename: str,
                 normalize: bool = True) -> None:
    """
    Save triangle mesh to PLY file.
    Always exports in normalized coordinate system.
    
    Args:
        mesh: Open3D TriangleMesh
        filename: Output filename (will add .ply if not present)
        normalize: If True, normalize to unit sphere before export
    """
    # Ensure filename has .ply extension
    filename = Path(filename)
    if filename.suffix.lower() != '.ply':
        filename = filename.with_suffix('.ply')
    
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh is empty, cannot export")
    
    # Normalize to unit sphere if requested
    if normalize:
        vertices = np.asarray(mesh.vertices)
        
        # Center the mesh
        centroid = np.mean(vertices, axis=0)
        vertices_centered = vertices - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(vertices_centered, axis=1))
        if max_dist > 0:
            vertices_normalized = vertices_centered / max_dist
        else:
            vertices_normalized = vertices_centered
        
        # Create new mesh with normalized vertices
        mesh_normalized = o3d.geometry.TriangleMesh()
        mesh_normalized.vertices = o3d.utility.Vector3dVector(vertices_normalized)
        mesh_normalized.triangles = mesh.triangles
        
        # Copy normals if present
        if mesh.has_vertex_normals():
            mesh_normalized.vertex_normals = mesh.vertex_normals
        if mesh.has_triangle_normals():
            mesh_normalized.triangle_normals = mesh.triangle_normals
        
        mesh = mesh_normalized
    
    # Create directory if needed
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to PLY
    success = o3d.io.write_triangle_mesh(str(filename), mesh)
    
    if not success:
        raise IOError(f"Failed to write mesh to {filename}")
    
    print(f"Mesh saved: {filename} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")


def export_results(partial_pcd: o3d.geometry.PointCloud,
                  predicted_pcd: o3d.geometry.PointCloud,
                  poisson_mesh: o3d.geometry.TriangleMesh,
                  output_dir: str = "exports") -> None:
    """
    Export all results to PLY files in normalized coordinate system.
    
    Creates:
    - input_partial.ply
    - output_predicted.ply
    - poisson_reconstruction.ply
    
    Args:
        partial_pcd: Partial input point cloud
        predicted_pcd: Predicted complete point cloud
        poisson_mesh: Poisson reconstructed mesh
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Exporting results to PLY files")
    print("=" * 60)
    
    # Export partial input
    print("\n1. Exporting input_partial.ply...")
    save_point_cloud_ply(
        partial_pcd,
        output_path / "input_partial.ply",
        normalize=True
    )
    
    # Export predicted output
    print("\n2. Exporting output_predicted.ply...")
    save_point_cloud_ply(
        predicted_pcd,
        output_path / "output_predicted.ply",
        normalize=True
    )
    
    # Export Poisson reconstruction
    print("\n3. Exporting poisson_reconstruction.ply...")
    save_mesh_ply(
        poisson_mesh,
        output_path / "poisson_reconstruction.ply",
        normalize=True
    )
    
    print("\n" + "=" * 60)
    print("All exports completed!")
    print("=" * 60)
    print(f"Files saved in: {output_path.absolute()}")


if __name__ == "__main__":
    """
    Example export script.
    """
    import open3d as o3d
    from backend.data.preprocessing import create_partial_point_cloud, normalize_point_cloud
    from backend.geometry.normals import estimate_normals_pca
    from backend.geometry.reconstruction import poisson_surface_reconstruction
    
    print("=" * 60)
    print("Export Example")
    print("=" * 60)
    
    # Create synthetic data
    print("\nCreating synthetic test data...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    pcd_full = mesh.sample_points_uniformly(number_of_points=2000)
    
    # Normalize
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    
    # Create partial
    pcd_partial = create_partial_point_cloud(
        pcd_full,
        method='mask_angle',
        direction=[1, 0, 0],
        max_angle=45.0
    )
    
    # Estimate normals and reconstruct with Poisson
    pcd_with_normals = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    poisson_mesh = poisson_surface_reconstruction(pcd_with_normals, depth=9, scale=1.1)
    
    # For predicted, use full as placeholder (in real scenario, use model prediction)
    pcd_predicted = pcd_full
    
    # Export all results
    export_results(
        partial_pcd=pcd_partial,
        predicted_pcd=pcd_predicted,
        poisson_mesh=poisson_mesh,
        output_dir="exports"
    )
