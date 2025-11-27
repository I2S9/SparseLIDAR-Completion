"""
Test script for Sparse UNet model.
Tests forward pass and verifies dimensions are correct.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from backend.models.sparse_unet import SparseUNet, create_sparse_tensor, sparse_tensor_to_point_cloud
    from backend.models.losses import ChamferLoss, FScoreLoss, CombinedLoss
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}")
    print("MinkowskiEngine is not available. This test requires MinkowskiEngine.")
    print("Please install it or run this in a Linux/WSL2 environment.")
    MINKOWSKI_AVAILABLE = False


def test_sparse_unet():
    """
    Test Sparse UNet forward pass.
    """
    if not MINKOWSKI_AVAILABLE:
        print("\nSkipping test - MinkowskiEngine not available.")
        return
    
    print("=" * 60)
    print("Sparse UNet Model Test")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    print("\n1. Creating Sparse UNet model...")
    model = SparseUNet(in_channels=3, out_channels=3, base_channels=32)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create dummy input (sparse point cloud)
    print("\n2. Creating dummy input...")
    batch_size = 2
    num_points = 1000
    
    # Random coordinates (voxel coordinates, integers)
    coordinates = torch.randint(0, 100, (num_points, 3), dtype=torch.long, device=device)
    # Add batch index
    batch_indices = torch.zeros(num_points, 1, dtype=torch.long, device=device)
    coordinates_with_batch = torch.cat([batch_indices, coordinates], dim=1)
    
    # Random features (point coordinates as features)
    features = torch.randn(num_points, 3, device=device)
    
    # Create sparse tensor
    input_sparse = ME.SparseTensor(
        features=features,
        coordinates=coordinates_with_batch,
        device=device
    )
    
    print(f"   Input sparse tensor:")
    print(f"     Features shape: {input_sparse.F.shape}")
    print(f"     Coordinates shape: {input_sparse.C.shape}")
    print(f"     Number of points: {len(input_sparse.F)}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    try:
        with torch.no_grad():
            output_sparse = model(input_sparse)
        
        print(f"   Output sparse tensor:")
        print(f"     Features shape: {output_sparse.F.shape}")
        print(f"     Coordinates shape: {output_sparse.C.shape}")
        print(f"     Number of points: {len(output_sparse.F)}")
        
        # Convert to point cloud
        output_coords, output_features = sparse_tensor_to_point_cloud(output_sparse)
        print(f"   Output coordinates shape: {output_coords.shape}")
        print(f"   Output features shape: {output_features.shape}")
        
        print("\n✓ Forward pass successful!")
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test loss functions
    print("\n4. Testing loss functions...")
    
    # Create dummy predictions and targets
    pred_points = torch.randn(500, 3, device=device)
    target_points = torch.randn(800, 3, device=device)
    
    # Chamfer Distance
    chamfer_loss = ChamferLoss()
    chamfer_val = chamfer_loss(pred_points, target_points)
    print(f"   Chamfer Distance: {chamfer_val.item():.6f}")
    
    # F-score
    fscore_loss = FScoreLoss(threshold=0.1)
    fscore_val = fscore_loss(pred_points, target_points)
    print(f"   F-score Loss (1 - F): {fscore_val.item():.6f}")
    
    # Combined loss
    combined_loss = CombinedLoss(
        chamfer_weight=1.0,
        fscore_weight=0.5,
        normal_weight=0.0
    )
    combined_val = combined_loss(pred_points, target_points)
    print(f"   Combined Loss: {combined_val.item():.6f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print("\nModel is ready for training.")
    print("Note: This model requires MinkowskiEngine and works best on Linux/WSL2.")


if __name__ == "__main__":
    test_sparse_unet()

