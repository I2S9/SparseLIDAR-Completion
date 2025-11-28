"""
Shared utilities for training: seeding, checkpoint helpers, logging helpers.
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   additional_info: Optional[Dict[str, Any]] = None):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        additional_info: Optional additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu'):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
