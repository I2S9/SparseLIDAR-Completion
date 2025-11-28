"""
Train a simple PointNet autoencoder on demo objects.
Minimal training script for demonstration purposes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
from tqdm import tqdm

from backend.models.simple_ae import create_model
from backend.models.losses import chamfer_distance
from backend.data.preprocessing import normalize_point_cloud
from backend.data.datasets import load_point_cloud
from backend.visualization.export_ply import save_point_cloud_ply


def prepare_point_cloud(pcd: o3d.geometry.PointCloud, num_points: int = 2048) -> torch.Tensor:
    """
    Prepare point cloud for model input.
    Normalize and resample to fixed number of points.
    
    Args:
        pcd: Open3D PointCloud
        num_points: Target number of points
        
    Returns:
        (1, num_points, 3) tensor
    """
    # Normalize
    pcd = normalize_point_cloud(pcd, method='unit_sphere')
    
    # Resample to fixed number of points
    if len(pcd.points) > num_points:
        # Downsample
        pcd = pcd.farthest_point_down_sample(num_points)
    elif len(pcd.points) < num_points:
        # Upsample by duplicating points
        points = np.asarray(pcd.points)
        n_needed = num_points - len(points)
        indices = np.random.choice(len(points), n_needed, replace=True)
        additional_points = points[indices]
        all_points = np.vstack([points, additional_points])
        pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Convert to tensor
    points = np.asarray(pcd.points)
    tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)  # (1, N, 3)
    
    return tensor


def create_training_data():
    """
    Create training data from demo objects.
    Uses sphere, cube, and torus from the demo dataset.
    """
    print("Creating training data...")
    
    import open3d as o3d
    from backend.data.preprocessing import (
        create_partial_point_cloud,
        normalize_point_cloud
    )
    from backend.geometry.normals import estimate_normals_pca
    
    objects = []
    
    # Sphere
    print("  Creating sphere...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=50)
    pcd_sphere = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_sphere = normalize_point_cloud(pcd_sphere, method='unit_sphere')
    pcd_sphere = estimate_normals_pca(pcd_sphere, k=30, orient_normals=True)
    objects.append(pcd_sphere)
    
    # Cube
    print("  Creating cube...")
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    mesh.compute_vertex_normals()
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    pcd_cube = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_cube = normalize_point_cloud(pcd_cube, method='unit_sphere')
    pcd_cube = estimate_normals_pca(pcd_cube, k=30, orient_normals=True)
    objects.append(pcd_cube)
    
    # Torus
    print("  Creating torus...")
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.0,
        tube_radius=0.3,
        radial_resolution=50,
        tubular_resolution=30
    )
    pcd_torus = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_torus = normalize_point_cloud(pcd_torus, method='unit_sphere')
    pcd_torus = estimate_normals_pca(pcd_torus, k=30, orient_normals=True)
    objects.append(pcd_torus)
    
    # Create partial versions for input
    inputs = []
    targets = []
    
    for obj in objects:
        # Create partial input
        partial = create_partial_point_cloud(
            obj,
            method='mask_side',
            direction=[1, 0, 0],
            mask_ratio=0.3
        )
        partial = estimate_normals_pca(partial, k=30, orient_normals=True)
        inputs.append(partial)
        targets.append(obj)
    
    print(f"Created {len(inputs)} training samples")
    
    return inputs, targets


def train_model(model: nn.Module,
                train_inputs: list,
                train_targets: list,
                num_epochs: int = 50,
                batch_size: int = 2,
                lr: float = 0.001,
                device: str = 'cpu'):
    """
    Train the autoencoder model.
    
    Args:
        model: Model to train
        train_inputs: List of partial point clouds (input)
        train_targets: List of full point clouds (target)
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Device: {device}, Batch size: {batch_size}, Learning rate: {lr}")
    
    num_points = model.num_points
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_inputs))
        
        for i in range(0, len(train_inputs), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Skip if batch is too small (need at least 1 sample)
            if len(batch_indices) == 0:
                continue
            
            # Prepare batch
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                input_tensor = prepare_point_cloud(train_inputs[idx], num_points)
                target_tensor = prepare_point_cloud(train_targets[idx], num_points)
                batch_inputs.append(input_tensor)
                batch_targets.append(target_tensor)
            
            # Stack into batch
            batch_input = torch.cat(batch_inputs, dim=0).to(device)  # (B, N, 3)
            batch_target = torch.cat(batch_targets, dim=0).to(device)  # (B, N, 3)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_input)  # (B, num_points, 3)
            
            # Compute loss (Chamfer Distance)
            loss = chamfer_distance(output, batch_target, reduction='mean')
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print("Training completed!")
    return model


def predict_and_export(model: nn.Module,
                      input_pcd: o3d.geometry.PointCloud,
                      output_path: Path,
                      device: str = 'cpu'):
    """
    Predict completion and export to PLY.
    
    Args:
        model: Trained model
        input_pcd: Partial input point cloud
        output_path: Path to save output PLY
        device: Device to use
    """
    model.eval()
    
    # Prepare input
    input_tensor = prepare_point_cloud(input_pcd, model.num_points).to(device)
    
    # Predict
    with torch.no_grad():
        output_tensor = model(input_tensor)  # (1, num_points, 3)
    
    # Convert to numpy
    output_points = output_tensor.cpu().numpy()[0]  # (num_points, 3)
    
    # Create Open3D point cloud
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(output_points)
    
    # Normalize and export
    output_pcd = normalize_point_cloud(output_pcd, method='unit_sphere')
    save_point_cloud_ply(output_pcd, output_path, normalize=True)
    
    print(f"Prediction exported to: {output_path}")


def main():
    """
    Main training and prediction pipeline.
    """
    print("=" * 60)
    print("Simple PointNet Autoencoder Training")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create training data
    train_inputs, train_targets = create_training_data()
    
    # Create model
    print("\nCreating model...")
    num_points = 2048
    model = create_model(num_points=num_points, latent_dim=128, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model = train_model(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        num_epochs=50,
        batch_size=2,
        lr=0.001,
        device=device
    )
    
    # Save model
    model_path = Path("exports") / "simple_ae_model.pth"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Load demo partial input
    print("\nGenerating prediction for demo...")
    partial_file = Path("exports") / "input_partial.ply"
    
    if partial_file.exists():
        pcd_partial = o3d.io.read_point_cloud(str(partial_file))
        
        # Predict
        output_path = Path("exports") / "deep_output.ply"
        predict_and_export(model, pcd_partial, output_path, device=device)
        
        # Also update output_predicted.ply
        output_path_main = Path("exports") / "output_predicted.ply"
        predict_and_export(model, pcd_partial, output_path_main, device=device)
        
        print("\n" + "=" * 60)
        print("Training and prediction completed!")
        print("=" * 60)
        print(f"\nFiles created:")
        print(f"  - {model_path}")
        print(f"  - {output_path}")
        print(f"  - {output_path_main}")
    else:
        print(f"\nWarning: {partial_file} not found. Skipping prediction.")


if __name__ == "__main__":
    main()

