# SparseLIDAR-Completion

Reconstruct and complete partial LIDAR point clouds using deep learning and classical geometry.

## Project Overview

This project aims to reconstruct complete 3D shapes from partial, incomplete, and highly sparse LIDAR point clouds, such as those obtained in real-world conditions: sensors mounted on aircraft, drones, autonomous vehicles, or mobile platforms with limited visibility.

## Installation

### Backend

1. Create conda environment:
```bash
conda create -n sparse-lidar python=3.10
conda activate sparse-lidar
```

2. Install PyTorch with CUDA (version must match your GPU)

3. Install MinkowskiEngine matching PyTorch and CUDA

4. Install Open3D

5. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

### Frontend

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

## Project Structure

See `.cursorrules` for detailed architecture.

## Development Roadmap

- Phase 1: Data and Visualization
- Phase 2: Classical Baselines
- Phase 3: Deep Learning Model
- Phase 4: Metrics and Evaluation
- Phase 5: Export and Web Visualization

