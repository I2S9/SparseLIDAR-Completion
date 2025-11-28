# SparseLIDAR-Completion

>  Reconstruct and complete partial LIDAR point clouds using deep learning and classical geometry.

![Demo](demo.gif)

## Project Overview

This project addresses the challenge of completing sparse, incomplete 3D point clouds obtained from LIDAR sensors in real-world conditions (aircraft, drones, autonomous vehicles). It combines classical geometric methods (Poisson Surface Reconstruction, PCA normals) with deep learning (Sparse UNet) to predict missing parts of point clouds.

**Key Features:**
- Classical baselines: PCA normal estimation, Poisson Surface Reconstruction
- Deep Learning: Sparse UNet architecture with MinkowskiEngine
- Comprehensive evaluation: Chamfer Distance, F-score, Normal Angle Error
- Interactive 3D visualization: Web-based Three.js viewer

## Tech Stack

**Backend:**
- Python 3.10+
- PyTorch (with CUDA support)
- MinkowskiEngine (sparse convolutions)
- Open3D (3D processing and visualization)
- NumPy, SciPy, Matplotlib

**Frontend:**
- React 18
- Three.js (3D rendering)
- Vite (build tool)

## Architecture

```
backend/
  data/           # Dataset loading, preprocessing
  geometry/       # Classical methods (PCA, Poisson)
  models/         # Sparse UNet, loss functions
  training/       # Training loop, evaluation
  visualization/  # Open3D helpers, PLY export

frontend/
  src/
    components/   # Three.js point cloud viewer
    utils/        # PLY loaders
```

## Installation

### Backend

1. Create conda environment:
```bash
conda create -n sparse-lidar python=3.10
conda activate sparse-lidar
```

2. Install PyTorch (with CUDA if available):
```bash
# CPU only
pip install torch torchvision torchaudio

# With CUDA (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install MinkowskiEngine (Linux/WSL2 only):
```bash
pip install ninja
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173/`

## Quick Start

### Generate Demo Files

**Option 1: Mini Dataset (Recommended)**

Generate a mini demonstration dataset with 3 simple objects (sphere, cube, torus):
```bash
conda activate sparse-lidar
python backend/notebooks/create_demo_dataset.py
```

This creates:
- Main demo files in `exports/` (used by the web interface)
- Individual object files in `exports/sphere/`, `exports/cube/`, `exports/torus/`

**Option 2: Simple Demo**

Generate a single simple demo object:
```bash
conda activate sparse-lidar
python backend/notebooks/create_demo_files.py
```

Both scripts create PLY files in `exports/` directory.

### Train Simple Deep Learning Model

Train a minimal PointNet autoencoder to generate Deep Learning predictions:

```bash
conda activate sparse-lidar
python backend/notebooks/train_simple_ae.py
```

This will:
- Create training data from 3 simple objects (sphere, cube, torus)
- Train a PointNet autoencoder for 50 epochs
- Generate `output_predicted.ply` with model predictions
- Save the trained model to `exports/simple_ae_model.pth`

**Note:** This is a minimal model for demonstration. For production, use the full Sparse UNet architecture.

### Compute Metrics

After generating demo files, compute evaluation metrics:

```bash
conda activate sparse-lidar
python backend/notebooks/compute_metrics.py
```

This computes Chamfer Distance, F-score, and Normal Angle Error for:
- Partial (baseline)
- Poisson reconstruction
- Deep Learning (placeholder)

Results are exported to `frontend/public/metrics.json` for display in the web interface.

### Run Web Demo

1. Copy files to frontend (Windows PowerShell):
```powershell
if (-not (Test-Path "frontend\public\exports")) { New-Item -ItemType Directory -Path "frontend\public\exports" -Force | Out-Null }
Copy-Item -Path "exports\*.ply" -Destination "frontend\public\exports\" -Force
```

Or on Linux/Mac:
```bash
mkdir -p frontend/public/exports
cp exports/*.ply frontend/public/exports/
```

2. Start web server:
```bash
cd frontend
npm install  # First time only
npm run dev
```

3. Open `http://localhost:5173/` and click "Load Demo Files"

## Usage Examples

### Load and Visualize Point Cloud

```python
from backend.data.datasets import load_point_cloud
from backend.visualization.visualize_o3d import show_point_cloud

pcd = load_point_cloud("data/pointcloud.ply")
show_point_cloud(pcd)
```

### Preprocessing Pipeline

```python
from backend.data.preprocessing import (
    downsample_voxel,
    create_partial_point_cloud,
    normalize_point_cloud
)

pcd = downsample_voxel(pcd, voxel_size=0.05)
pcd = create_partial_point_cloud(pcd, method='mask_angle', max_angle=45.0)
pcd = normalize_point_cloud(pcd, method='unit_sphere')
```

### Classical Methods

```python
from backend.geometry.normals import estimate_normals_pca
from backend.geometry.reconstruction import poisson_surface_reconstruction

pcd_with_normals = estimate_normals_pca(pcd, k=30)
mesh = poisson_surface_reconstruction(pcd_with_normals, depth=9)
```

### Evaluation

```python
from backend.training.eval import evaluate_all_methods, print_results_table

results = evaluate_all_methods(pcd_partial, pcd_full)
print_results_table(results)
```

## Results and Metrics

The evaluation compares four methods:

| Method | Chamfer Distance ↓ | F-score ↑ | Normal Error ↓ |
|--------|-------------------|-----------|----------------|
| Partial | Baseline | - | - |
| PCA Normals | Improved | Improved | ~58° |
| Poisson | Best (classical) | Good | ~62° |
| Sparse UNet | Best (DL) | Best | TBD |

*Note: Metrics depend on dataset and preprocessing. Run evaluation scripts for specific results.*

## Web Demo

The interactive web interface allows you to:
- Compare partial input, Poisson reconstruction, and Deep Learning output
- Switch between individual views or side-by-side comparison
- Interactively rotate, pan, and zoom in 3D
- Load custom PLY files

**Screenshots:** The web interface provides real-time 3D visualization of point clouds with color-coded methods (Red: Partial, Cyan: Poisson, Green: Deep Learning).

## Project Structure

- `backend/data/`: Dataset loading and preprocessing
- `backend/geometry/`: Classical methods (PCA, Poisson)
- `backend/models/`: Deep learning models and losses
- `backend/training/`: Training and evaluation scripts
- `backend/visualization/`: Export and visualization tools
- `frontend/`: Web interface with Three.js

## License

This project is for research and educational purposes.
