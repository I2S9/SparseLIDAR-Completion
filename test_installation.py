"""
Simple test script to verify the installation.
Run: python test_installation.py
"""

print("Testing installation...")
print("=" * 50)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (CPU mode)")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import open3d as o3d
    print(f"✓ Open3D version: {o3d.__version__}")
except ImportError:
    print("✗ Open3D not installed")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except ImportError:
    print("✗ SciPy not installed")

try:
    import MinkowskiEngine as ME
    print(f"✓ MinkowskiEngine installed")
except ImportError:
    print("⚠ MinkowskiEngine not installed (install after PyTorch)")

print("=" * 50)
print("Installation test complete!")

