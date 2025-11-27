"""
Surface reconstruction methods using Open3D.
Includes Poisson Surface Reconstruction and Ball Pivoting.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple


def poisson_surface_reconstruction(pcd: o3d.geometry.PointCloud,
                                   depth: int = 9,
                                   width: int = 0,
                                   scale: float = 1.1,
                                   linear_fit: bool = False) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct surface using Poisson Surface Reconstruction.
    
    Requires point cloud with normals.
    
    Args:
        pcd: Open3D PointCloud with normals
        depth: Maximum depth of the tree (higher = more detail, slower)
        width: Width parameter (0 = auto)
        scale: Scale parameter for reconstruction
        linear_fit: Use linear interpolation
        
    Returns:
        Reconstructed TriangleMesh
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals for Poisson reconstruction")
    
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")
    
    # Perform Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )
    
    return mesh


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh,
                       number_of_points: int = 10000) -> o3d.geometry.PointCloud:
    """
    Convert a triangle mesh to a point cloud by uniform sampling.
    
    Args:
        mesh: Open3D TriangleMesh
        number_of_points: Number of points to sample from mesh
        
    Returns:
        PointCloud sampled from mesh
    """
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh is empty")
    
    # Sample points uniformly from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    
    return pcd


def ball_pivoting_reconstruction(pcd: o3d.geometry.PointCloud,
                                radii: list = None) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct surface using Ball Pivoting Algorithm.
    
    Requires point cloud with normals.
    
    Args:
        pcd: Open3D PointCloud with normals
        radii: List of ball radii to try (if None, auto-compute)
        
    Returns:
        Reconstructed TriangleMesh
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals for Ball Pivoting")
    
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
    
    # Auto-compute radii if not provided
    if radii is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    
    # Perform Ball Pivoting reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    return mesh


def reconstruct_and_convert(pcd: o3d.geometry.PointCloud,
                            method: str = 'poisson',
                            reconstruction_params: Optional[dict] = None,
                            number_of_points: int = 10000) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Reconstruct surface and convert mesh back to point cloud for comparison.
    
    Args:
        pcd: Open3D PointCloud with normals
        method: Reconstruction method ('poisson' or 'ball_pivoting')
        reconstruction_params: Dictionary of parameters for reconstruction
        number_of_points: Number of points to sample from reconstructed mesh
        
    Returns:
        Tuple of (mesh, point_cloud)
    """
    if reconstruction_params is None:
        reconstruction_params = {}
    
    # Reconstruct surface
    if method == 'poisson':
        mesh = poisson_surface_reconstruction(pcd, **reconstruction_params)
    elif method == 'ball_pivoting':
        mesh = ball_pivoting_reconstruction(pcd, **reconstruction_params.get('radii', None))
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    
    # Convert mesh to point cloud
    pcd_reconstructed = mesh_to_point_cloud(mesh, number_of_points=number_of_points)
    
    return mesh, pcd_reconstructed
