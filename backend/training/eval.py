"""
Evaluation scripts for baselines and the model.
Computes Chamfer Distance, F-score, and Normal Angle Error.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.data.datasets import load_point_cloud
from backend.data.preprocessing import normalize_point_cloud, create_partial_point_cloud
from backend.geometry.normals import estimate_normals_pca
from backend.geometry.reconstruction import poisson_surface_reconstruction, mesh_to_point_cloud
from backend.models.losses import chamfer_distance, f_score


def compute_chamfer_distance(pred_points: np.ndarray, 
                             target_points: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred_points: Nx3 array of predicted points
        target_points: Mx3 array of target points
        
    Returns:
        Chamfer distance (scalar)
    """
    pred_tensor = torch.from_numpy(pred_points.astype(np.float32))
    target_tensor = torch.from_numpy(target_points.astype(np.float32))
    
    cd = chamfer_distance(pred_tensor, target_tensor, reduction='mean')
    return cd.item()


def compute_f_score(pred_points: np.ndarray,
                   target_points: np.ndarray,
                   threshold: float = 0.01) -> float:
    """
    Compute F-score between two point clouds.
    
    Args:
        pred_points: Nx3 array of predicted points
        target_points: Mx3 array of target points
        threshold: Distance threshold
        
    Returns:
        F-score (0-1, higher is better)
    """
    pred_tensor = torch.from_numpy(pred_points.astype(np.float32))
    target_tensor = torch.from_numpy(target_points.astype(np.float32))
    
    f_score_val = f_score(pred_tensor, target_tensor, threshold=threshold, reduction='mean')
    return f_score_val.item()


def compute_normal_angle_error(pred_points: np.ndarray,
                               pred_normals: np.ndarray,
                               target_points: np.ndarray,
                               target_normals: np.ndarray) -> float:
    """
    Compute average normal angle error between predicted and target normals.
    
    For each predicted point, finds nearest target point and computes angle between normals.
    
    Args:
        pred_points: Nx3 array of predicted points
        pred_normals: Nx3 array of predicted normals
        target_points: Mx3 array of target points
        target_normals: Mx3 array of target normals
        
    Returns:
        Average normal angle error in degrees
    """
    pred_tensor = torch.from_numpy(pred_points.astype(np.float32))
    target_tensor = torch.from_numpy(target_points.astype(np.float32))
    pred_normals_tensor = torch.from_numpy(pred_normals.astype(np.float32))
    target_normals_tensor = torch.from_numpy(target_normals.astype(np.float32))
    
    # Find nearest neighbors
    dists = torch.cdist(pred_tensor, target_tensor, p=2)
    nearest_indices = torch.argmin(dists, dim=1)  # N
    
    # Get corresponding target normals
    target_normals_matched = target_normals_tensor[nearest_indices]  # Nx3
    
    # Normalize normals
    pred_normals_norm = pred_normals_tensor / (torch.norm(pred_normals_tensor, dim=1, keepdim=True) + 1e-8)
    target_normals_norm = target_normals_matched / (torch.norm(target_normals_matched, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(pred_normals_norm * target_normals_norm, dim=1)  # N
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  # Clamp for numerical stability
    
    # Convert to angles (in degrees)
    angles = torch.acos(cosine_sim) * 180.0 / np.pi
    
    # Return average angle error
    return torch.mean(angles).item()


def evaluate_partial(pcd_partial: o3d.geometry.PointCloud,
                     pcd_full: o3d.geometry.PointCloud) -> Dict[str, float]:
    """
    Evaluate partial point cloud (baseline: just use partial as prediction).
    
    Args:
        pcd_partial: Partial point cloud
        pcd_full: Full (target) point cloud
        
    Returns:
        Dictionary with metrics
    """
    pred_points = np.asarray(pcd_partial.points)
    target_points = np.asarray(pcd_full.points)
    
    # Compute metrics
    cd = compute_chamfer_distance(pred_points, target_points)
    f_score_val = compute_f_score(pred_points, target_points, threshold=0.01)
    
    # Normal error (if normals available)
    if pcd_partial.has_normals() and pcd_full.has_normals():
        pred_normals = np.asarray(pcd_partial.normals)
        target_normals = np.asarray(pcd_full.normals)
        normal_error = compute_normal_angle_error(pred_points, pred_normals, target_points, target_normals)
    else:
        normal_error = None
    
    return {
        'chamfer_distance': cd,
        'f_score': f_score_val,
        'normal_error': normal_error
    }


def evaluate_pca_normals(pcd_partial: o3d.geometry.PointCloud,
                        pcd_full: o3d.geometry.PointCloud) -> Dict[str, float]:
    """
    Evaluate PCA normals method (baseline: use partial with PCA normals).
    
    Args:
        pcd_partial: Partial point cloud
        pcd_full: Full (target) point cloud
        
    Returns:
        Dictionary with metrics
    """
    # Estimate normals using PCA
    pcd_with_normals = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    
    pred_points = np.asarray(pcd_with_normals.points)
    target_points = np.asarray(pcd_full.points)
    
    # Compute metrics
    cd = compute_chamfer_distance(pred_points, target_points)
    f_score_val = compute_f_score(pred_points, target_points, threshold=0.01)
    
    # Normal error
    if pcd_with_normals.has_normals() and pcd_full.has_normals():
        pred_normals = np.asarray(pcd_with_normals.normals)
        target_normals = np.asarray(pcd_full.normals)
        normal_error = compute_normal_angle_error(pred_points, pred_normals, target_points, target_normals)
    else:
        normal_error = None
    
    return {
        'chamfer_distance': cd,
        'f_score': f_score_val,
        'normal_error': normal_error
    }


def evaluate_poisson(pcd_partial: o3d.geometry.PointCloud,
                    pcd_full: o3d.geometry.PointCloud) -> Dict[str, float]:
    """
    Evaluate Poisson Surface Reconstruction method.
    
    Args:
        pcd_partial: Partial point cloud
        pcd_full: Full (target) point cloud
        
    Returns:
        Dictionary with metrics
    """
    # Estimate normals for Poisson
    pcd_with_normals = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    
    try:
        # Poisson reconstruction
        mesh = poisson_surface_reconstruction(pcd_with_normals, depth=9, scale=1.1)
        
        # Convert mesh to point cloud
        num_points = len(pcd_full.points)
        pcd_reconstructed = mesh_to_point_cloud(mesh, number_of_points=num_points)
        
        pred_points = np.asarray(pcd_reconstructed.points)
        target_points = np.asarray(pcd_full.points)
        
        # Compute metrics
        cd = compute_chamfer_distance(pred_points, target_points)
        f_score_val = compute_f_score(pred_points, target_points, threshold=0.01)
        
        # Normal error (if available)
        if pcd_reconstructed.has_normals() and pcd_full.has_normals():
            pred_normals = np.asarray(pcd_reconstructed.normals)
            target_normals = np.asarray(pcd_full.normals)
            normal_error = compute_normal_angle_error(pred_points, pred_normals, target_points, target_normals)
        else:
            normal_error = None
        
        return {
            'chamfer_distance': cd,
            'f_score': f_score_val,
            'normal_error': normal_error
        }
    except Exception as e:
        print(f"  Error in Poisson reconstruction: {e}")
        return {
            'chamfer_distance': None,
            'f_score': None,
            'normal_error': None
        }


def evaluate_sparse_unet(pcd_partial: o3d.geometry.PointCloud,
                        pcd_full: o3d.geometry.PointCloud,
                        model: Optional[torch.nn.Module] = None,
                        device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate Sparse UNet model (Deep Learning method).
    
    Args:
        pcd_partial: Partial point cloud
        pcd_full: Full (target) point cloud
        model: Trained model (if None, returns placeholder)
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    if model is None:
        # Placeholder: return partial as prediction
        print("  Warning: No model provided, using partial as prediction")
        return evaluate_partial(pcd_partial, pcd_full)
    
    try:
        # Convert to sparse tensor and predict
        # Note: This requires MinkowskiEngine
        # For now, we'll use a placeholder
        pred_points = np.asarray(pcd_partial.points)
        target_points = np.asarray(pcd_full.points)
        
        # TODO: Implement actual model prediction when MinkowskiEngine is available
        # For now, return partial as prediction
        cd = compute_chamfer_distance(pred_points, target_points)
        f_score_val = compute_f_score(pred_points, target_points, threshold=0.01)
        normal_error = None
        
        return {
            'chamfer_distance': cd,
            'f_score': f_score_val,
            'normal_error': normal_error
        }
    except Exception as e:
        print(f"  Error in Sparse UNet evaluation: {e}")
        return {
            'chamfer_distance': None,
            'f_score': None,
            'normal_error': None
        }


def evaluate_all_methods(pcd_partial: o3d.geometry.PointCloud,
                        pcd_full: o3d.geometry.PointCloud,
                        model: Optional[torch.nn.Module] = None,
                        device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """
    Evaluate all methods and return results.
    
    Args:
        pcd_partial: Partial point cloud
        pcd_full: Full (target) point cloud
        model: Optional trained model for Sparse UNet
        device: Device to use
        
    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    print("\nEvaluating methods...")
    print("-" * 60)
    
    # Normalize both point clouds before comparison
    pcd_partial_norm = normalize_point_cloud(pcd_partial, method='unit_sphere')
    pcd_full_norm = normalize_point_cloud(pcd_full, method='unit_sphere')
    
    # Ensure target has normals for normal error computation
    if not pcd_full_norm.has_normals():
        pcd_full_norm = estimate_normals_pca(pcd_full_norm, k=30, orient_normals=True)
    
    # 1. Partial (baseline)
    print("1. Partial point cloud (baseline)...")
    results['Partial'] = evaluate_partial(pcd_partial_norm, pcd_full_norm)
    
    # 2. PCA Normals
    print("2. PCA Normals...")
    results['PCA Normals'] = evaluate_pca_normals(pcd_partial_norm, pcd_full_norm)
    
    # 3. Poisson
    print("3. Poisson Surface Reconstruction...")
    results['Poisson'] = evaluate_poisson(pcd_partial_norm, pcd_full_norm)
    
    # 4. Sparse UNet (DL)
    print("4. Sparse UNet (Deep Learning)...")
    results['Sparse UNet'] = evaluate_sparse_unet(pcd_partial_norm, pcd_full_norm, model, device)
    
    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """
    Print results as a formatted table.
    
    Args:
        results: Dictionary with results for each method
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Prepare table data
    table_data = []
    headers = ["Method", "CD (lower better)", "F-score (higher better)", "Normal Error (lower better)"]
    
    for method, metrics in results.items():
        cd = metrics.get('chamfer_distance')
        f_score_val = metrics.get('f_score')
        normal_error = metrics.get('normal_error')
        
        # Format values
        cd_str = f"{cd:.6f}" if cd is not None else "N/A"
        f_score_str = f"{f_score_val:.4f}" if f_score_val is not None else "N/A"
        normal_error_str = f"{normal_error:.2f} deg" if normal_error is not None else "N/A"
        
        table_data.append([method, cd_str, f_score_str, normal_error_str])
    
    # Print table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\nNote: CD and Normal Error - lower is better; F-score - higher is better")
    print("=" * 60)


def evaluate_on_sample(pcd_full: o3d.geometry.PointCloud,
                       create_partial_fn,
                       model: Optional[torch.nn.Module] = None,
                       device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """
    Evaluate all methods on a single sample.
    
    Args:
        pcd_full: Full point cloud
        create_partial_fn: Function to create partial point cloud
        model: Optional trained model
        device: Device to use
        
    Returns:
        Dictionary with results
    """
    # Create partial point cloud
    pcd_partial = create_partial_fn(pcd_full)
    
    # Evaluate all methods
    results = evaluate_all_methods(pcd_partial, pcd_full, model, device)
    
    return results


if __name__ == "__main__":
    """
    Example evaluation script.
    """
    import open3d as o3d
    from backend.data.preprocessing import create_partial_point_cloud
    
    print("=" * 60)
    print("Point Cloud Completion - Evaluation")
    print("=" * 60)
    
    # Create synthetic test sample
    print("\nCreating test sample...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    pcd_full = mesh.sample_points_uniformly(number_of_points=2000)
    
    # Create partial function
    def create_partial(pcd):
        return create_partial_point_cloud(
            pcd,
            method='mask_angle',
            direction=[1, 0, 0],
            max_angle=45.0
        )
    
    # Evaluate
    results = evaluate_on_sample(
        pcd_full=pcd_full,
        create_partial_fn=create_partial,
        model=None,  # No model for now
        device='cpu'
    )
    
    # Print results
    print_results_table(results)
