"""
Training loop for point cloud completion model.
Includes dataloader, optimizer, scheduler, and logging.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from backend.training.utils import set_seed, save_checkpoint


class Trainer:
    """
    Trainer class for point cloud completion model.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cpu',
                 checkpoint_dir: str = 'checkpoints',
                 use_wandb: bool = False,
                 project_name: str = 'sparse-lidar-completion'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            loss_fn: Loss function
            optimizer: Optimizer (default: Adam)
            scheduler: Optional learning rate scheduler
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        if loss_fn is None:
            from backend.models.losses import ChamferLoss
            self.loss_fn = ChamferLoss().to(device)
        else:
            self.loss_fn = loss_fn.to(device)
        
        # Optimizer (Adam by default)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler (optional)
        self.scheduler = scheduler
        
        # Logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': train_loader.batch_size,
                'device': device
            })
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch}")
        print("-" * 60)
        
        for batch_idx, (partial, full) in enumerate(self.train_loader):
            # Move to device
            partial = partial.to(self.device)
            full = full.to(self.device)
            
            # Forward pass
            # Note: For now, we'll use a simple approach
            # In a real scenario with MinkowskiEngine, we'd convert to sparse tensors
            # For testing without MinkowskiEngine, we'll use a placeholder
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                # Reshape for model: (batch, points, features) -> (batch * points, features)
                batch_size = partial.shape[0]
                num_points = partial.shape[1]
                partial_flat = partial.view(-1, partial.shape[2])
                
                # Forward through model
                pred_flat = self.model(partial_flat)
                
                # Reshape back: (batch * points, features) -> (batch, points, features)
                pred = pred_flat.view(batch_size, num_points, -1)
                
                # For loss, we need to handle variable sizes
                # Take only the valid points (non-padded)
                # Simple approach: use first N points where N is min size
                min_points = min(pred.shape[1], full.shape[1])
                pred_trimmed = pred[:, :min_points, :]
                full_trimmed = full[:, :min_points, :]
                
                # Flatten for loss computation
                pred_flat_loss = pred_trimmed.reshape(-1, pred_trimmed.shape[2])
                full_flat_loss = full_trimmed.reshape(-1, full_trimmed.shape[2])
                
                # Compute loss
                loss = self.loss_fn(pred_flat_loss, full_flat_loss)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Logging
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx:3d}/{len(self.train_loader):3d} | Loss: {loss.item():.6f}")
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'epoch': epoch,
                        'batch': batch_idx
                    })
                    
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self, num_epochs: int = 1, save_every: int = 1):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        print(f"Number of epochs: {num_epochs}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Update learning rate if scheduler is available
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Learning rate: {current_lr:.6f}")
            
            # Log epoch summary
            elapsed_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Time: {elapsed_time:.2f}s")
            
            if self.use_wandb:
                wandb.log({
                    'epoch_avg_loss': avg_loss,
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    filepath=str(checkpoint_path),
                    additional_info={
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                    }
                )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        if self.use_wandb:
            wandb.finish()


def create_synthetic_dataset(num_samples: int = 10, device: str = 'cpu'):
    """
    Create a synthetic dataset for testing (5-10 samples as recommended).
    
    Args:
        num_samples: Number of samples to generate
        device: Device to use
        
    Returns:
        DataLoader
    """
    import open3d as o3d
    from backend.data.preprocessing import create_partial_point_cloud, normalize_point_cloud
    from backend.data.datasets import PointCloudCompletionDataset
    
    # Create synthetic point clouds
    point_clouds = []
    for i in range(num_samples):
        # Create different shapes for variety
        if i % 3 == 0:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        elif i % 3 == 1:
            mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        else:
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=1.0, resolution=20)
        
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        point_clouds.append(pcd)
    
    # Create partial function
    def create_partial(pcd):
        return create_partial_point_cloud(
            pcd,
            method='mask_angle',
            direction=[1, 0, 0],
            max_angle=45.0
        )
    
    # Create dataset
    dataset = PointCloudCompletionDataset(
        point_clouds=point_clouds,
        create_partial_fn=create_partial,
        normalize_fn=lambda pcd: normalize_point_cloud(pcd, method='unit_sphere'),
        device=device
    )
    
    # Import collate function
    from backend.data.datasets import collate_point_clouds
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_point_clouds
    )
    
    return dataloader


def train_minimal(num_batches: int = 20, num_samples: int = 10):
    """
    Minimal training function for testing (20 batches, 5-10 samples).
    
    Args:
        num_batches: Number of batches to train on
        num_samples: Number of samples in dataset
    """
    from backend.training.utils import set_seed
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic dataset (5-10 samples as recommended)
    print(f"\nCreating synthetic dataset with {num_samples} samples...")
    train_loader = create_synthetic_dataset(num_samples=num_samples, device=device)
    
    # Adjust number of batches
    actual_batches = min(num_batches, len(train_loader))
    print(f"Training on {actual_batches} batches")
    
    # Create a simple placeholder model for testing
    # In real scenario, this would be SparseUNet
    class PlaceholderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 64)
            self.linear2 = nn.Linear(64, 64)
            self.linear3 = nn.Linear(64, 3)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # Simple MLP for testing
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            return x
    
    model = PlaceholderModel()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        checkpoint_dir='checkpoints',
        use_wandb=False  # Set to True if W&B is installed
    )
    
    # Calculate epochs needed for num_batches
    batches_per_epoch = len(train_loader)
    num_epochs = (num_batches + batches_per_epoch - 1) // batches_per_epoch
    
    # Train
    trainer.train(num_epochs=num_epochs, save_every=1)
    
    # Verify checkpoint was saved
    checkpoint_path = Path('checkpoints') / f"checkpoint_epoch_{num_epochs:03d}.pth"
    if checkpoint_path.exists():
        print(f"\n[OK] Checkpoint saved successfully: {checkpoint_path}")
    else:
        print(f"\n[WARNING] Checkpoint not found at {checkpoint_path}")


if __name__ == "__main__":
    # Minimal training: 20 batches, 10 samples
    train_minimal(num_batches=20, num_samples=10)
