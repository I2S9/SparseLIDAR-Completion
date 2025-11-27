"""
PCA-based normal estimation and other classical methods for point clouds.
"""

import numpy as np
import open3d as o3d
from typing import Optional


def get_k_nearest_neighbors(pcd: o3d.geometry.PointCloud, 
                            query_point: np.ndarray,
                            k: int = 30) -> np.ndarray:
    """
    Get k-nearest neighbors of a query point in the point cloud.
    
    Args:
        pcd: Open3D PointCloud object
        query_point: 3D coordinates of query point
        k: Number of neighbors to find
        
    Returns:
        Array of k nearest neighbor indices
    """
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        return np.array([])
    
    # Build KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Find k nearest neighbors
    [_, indices, _] = kdtree.search_knn_vector_3d(query_point, k)
    
    return np.array(indices)


def compute_local_pca(points: np.ndarray) -> tuple:
    """
    Compute PCA on a local set of points.
    
    Args:
        points: Nx3 array of point coordinates
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted by decreasing eigenvalues
        The eigenvector with smallest eigenvalue is the normal direction
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points for PCA")
    
    # Center the points
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Compute covariance matrix
    covariance = np.cov(points_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def estimate_normals_pca(pcd: o3d.geometry.PointCloud, 
                         k: int = 30,
                         orient_normals: bool = True) -> o3d.geometry.PointCloud:
    """
    Estimate normals using PCA on k-nearest neighbors.
    
    For each point:
    1. Find k nearest neighbors
    2. Compute local PCA
    3. Use eigenvector with smallest eigenvalue as normal
    
    Args:
        pcd: Open3D PointCloud object
        k: Number of nearest neighbors for PCA
        orient_normals: If True, orient normals consistently
        
    Returns:
        PointCloud with estimated normals
    """
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        return pcd
    
    # Build KDTree for efficient search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    normals = []
    
    # Estimate normal for each point
    for i, point in enumerate(points):
        # Find k nearest neighbors (including the point itself)
        [_, indices, _] = kdtree.search_knn_vector_3d(point, k)
        
        # Get neighbor points
        neighbor_points = points[indices]
        
        # Compute local PCA
        try:
            eigenvalues, eigenvectors = compute_local_pca(neighbor_points)
            
            # Normal is the eigenvector with smallest eigenvalue
            normal = eigenvectors[:, -1]  # Last column (smallest eigenvalue)
            
            # Ensure normal points outward (heuristic: towards camera/viewpoint)
            # For now, we'll use a simple orientation
            if orient_normals and i == 0:
                # Use first point as reference
                reference_normal = normal
            elif orient_normals and i > 0:
                # Flip if dot product with reference is negative
                if np.dot(normal, reference_normal) < 0:
                    normal = -normal
            
            normals.append(normal)
            
        except ValueError:
            # If PCA fails (not enough points), use default normal
            normals.append([0, 0, 1])
    
    normals = np.array(normals)
    
    # Normalize normals
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms[:, np.newaxis]
    
    # Create new point cloud with normals
    pcd_with_normals = o3d.geometry.PointCloud()
    pcd_with_normals.points = pcd.points
    pcd_with_normals.normals = o3d.utility.Vector3dVector(normals)
    
    # Copy colors if present
    if pcd.has_colors():
        pcd_with_normals.colors = pcd.colors
    
    # Use Open3D's normal orientation if requested
    if orient_normals:
        pcd_with_normals.orient_normals_consistent_tangent_plane(k)
    
    return pcd_with_normals


def estimate_normals_open3d(pcd: o3d.geometry.PointCloud,
                           radius: Optional[float] = None,
                           max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    Estimate normals using Open3D's built-in method (for comparison).
    
    Args:
        pcd: Open3D PointCloud object
        radius: Search radius (if None, uses k-nearest neighbors)
        max_nn: Maximum number of neighbors
        
    Returns:
        PointCloud with estimated normals
    """
    pcd_with_normals = o3d.geometry.PointCloud()
    pcd_with_normals.points = pcd.points
    
    if pcd.has_colors():
        pcd_with_normals.colors = pcd.colors
    
    # Estimate normals
    if radius is None:
        pcd_with_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
        )
    else:
        pcd_with_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
        )
    
    # Orient normals consistently
    pcd_with_normals.orient_normals_consistent_tangent_plane(max_nn)
    
    return pcd_with_normals
