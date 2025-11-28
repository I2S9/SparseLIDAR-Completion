"""
Create a mini demonstration dataset with 2-3 simple objects.
Generates input_partial.ply, poisson_reconstruction.ply, and deep_completion.ply
for each object.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import open3d as o3d
import numpy as np
from backend.data.preprocessing import (
    create_partial_point_cloud,
    normalize_point_cloud
)
from backend.geometry.normals import estimate_normals_pca
from backend.geometry.reconstruction import poisson_surface_reconstruction, mesh_to_point_cloud
from backend.visualization.export_ply import save_point_cloud_ply, save_mesh_ply


def create_sphere_object():
    """Create a sphere object with partial view."""
    print("\n" + "=" * 60)
    print("Creating SPHERE object")
    print("=" * 60)
    
    # Create sphere mesh
    print("\n1. Creating base sphere...")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=50)
    pcd_full = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    print(f"   Full cloud: {len(pcd_full.points)} points")
    
    # Create partial (input) - mask one side
    print("\n2. Creating partial input (masking one side)...")
    pcd_partial = create_partial_point_cloud(
        pcd_full,
        method='mask_side',
        direction=[1, 0, 0],  # Mask points on +X side
        mask_ratio=0.3  # Keep 30% of points (the -X side)
    )
    # Estimate normals for partial input
    pcd_partial = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    print(f"   Partial cloud: {len(pcd_partial.points)} points (with normals)")
    
    # Poisson reconstruction
    print("\n3. Creating Poisson reconstruction...")
    poisson_mesh = poisson_surface_reconstruction(
        pcd_partial,
        depth=9,
        scale=1.1
    )
    print(f"   Poisson mesh: {len(poisson_mesh.vertices)} vertices, {len(poisson_mesh.triangles)} triangles")
    
    # Convert Poisson mesh to point cloud
    pcd_poisson = mesh_to_point_cloud(poisson_mesh, number_of_points=len(pcd_full.points))
    pcd_poisson = estimate_normals_pca(pcd_poisson, k=30, orient_normals=True)
    print(f"   Poisson point cloud: {len(pcd_poisson.points)} points (with normals)")
    
    # For Deep Learning output, use full as placeholder
    print("\n4. Creating Deep Learning output (placeholder)...")
    pcd_dl = pcd_full
    pcd_dl = estimate_normals_pca(pcd_dl, k=30, orient_normals=True)
    print(f"   DL output: {len(pcd_dl.points)} points (with normals)")
    
    return {
        'name': 'sphere',
        'partial': pcd_partial,
        'poisson': pcd_poisson,
        'poisson_mesh': poisson_mesh,
        'deep_learning': pcd_dl
    }


def create_cube_object():
    """Create a cube object with partial view."""
    print("\n" + "=" * 60)
    print("Creating CUBE object")
    print("=" * 60)
    
    # Create cube mesh
    print("\n1. Creating base cube...")
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    mesh.compute_vertex_normals()
    # Subdivide for more points
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    pcd_full = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    print(f"   Full cloud: {len(pcd_full.points)} points")
    
    # Create partial (input) - mask one corner
    print("\n2. Creating partial input (masking one corner)...")
    pcd_partial = create_partial_point_cloud(
        pcd_full,
        method='mask_angle',
        direction=[1, 1, 1],  # View from corner
        max_angle=60.0  # Keep points within 60 degrees
    )
    # Estimate normals for partial input
    pcd_partial = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    print(f"   Partial cloud: {len(pcd_partial.points)} points (with normals)")
    
    # Poisson reconstruction
    print("\n3. Creating Poisson reconstruction...")
    poisson_mesh = poisson_surface_reconstruction(
        pcd_partial,
        depth=9,
        scale=1.1
    )
    print(f"   Poisson mesh: {len(poisson_mesh.vertices)} vertices, {len(poisson_mesh.triangles)} triangles")
    
    # Convert Poisson mesh to point cloud
    pcd_poisson = mesh_to_point_cloud(poisson_mesh, number_of_points=len(pcd_full.points))
    pcd_poisson = estimate_normals_pca(pcd_poisson, k=30, orient_normals=True)
    print(f"   Poisson point cloud: {len(pcd_poisson.points)} points (with normals)")
    
    # For Deep Learning output, use full as placeholder
    print("\n4. Creating Deep Learning output (placeholder)...")
    pcd_dl = pcd_full
    pcd_dl = estimate_normals_pca(pcd_dl, k=30, orient_normals=True)
    print(f"   DL output: {len(pcd_dl.points)} points (with normals)")
    
    return {
        'name': 'cube',
        'partial': pcd_partial,
        'poisson': pcd_poisson,
        'poisson_mesh': poisson_mesh,
        'deep_learning': pcd_dl
    }


def create_torus_object():
    """Create a torus object with partial view."""
    print("\n" + "=" * 60)
    print("Creating TORUS object")
    print("=" * 60)
    
    # Create torus mesh
    print("\n1. Creating base torus...")
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.0,
        tube_radius=0.3,
        radial_resolution=50,
        tubular_resolution=30
    )
    pcd_full = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_full = normalize_point_cloud(pcd_full, method='unit_sphere')
    print(f"   Full cloud: {len(pcd_full.points)} points")
    
    # Create partial (input) - mask one side
    print("\n2. Creating partial input (masking one side)...")
    pcd_partial = create_partial_point_cloud(
        pcd_full,
        method='mask_side',
        direction=[1, 0, 0],  # Mask points on +X side
        mask_ratio=0.4  # Keep 40% of points (the -X side)
    )
    # Estimate normals for partial input
    pcd_partial = estimate_normals_pca(pcd_partial, k=30, orient_normals=True)
    print(f"   Partial cloud: {len(pcd_partial.points)} points (with normals)")
    
    # Poisson reconstruction
    print("\n3. Creating Poisson reconstruction...")
    poisson_mesh = poisson_surface_reconstruction(
        pcd_partial,
        depth=9,
        scale=1.1
    )
    print(f"   Poisson mesh: {len(poisson_mesh.vertices)} vertices, {len(poisson_mesh.triangles)} triangles")
    
    # Convert Poisson mesh to point cloud
    pcd_poisson = mesh_to_point_cloud(poisson_mesh, number_of_points=len(pcd_full.points))
    pcd_poisson = estimate_normals_pca(pcd_poisson, k=30, orient_normals=True)
    print(f"   Poisson point cloud: {len(pcd_poisson.points)} points (with normals)")
    
    # For Deep Learning output, use full as placeholder
    print("\n4. Creating Deep Learning output (placeholder)...")
    pcd_dl = pcd_full
    pcd_dl = estimate_normals_pca(pcd_dl, k=30, orient_normals=True)
    print(f"   DL output: {len(pcd_dl.points)} points (with normals)")
    
    return {
        'name': 'torus',
        'partial': pcd_partial,
        'poisson': pcd_poisson,
        'poisson_mesh': poisson_mesh,
        'deep_learning': pcd_dl
    }


def create_demo_dataset(objects=['sphere', 'cube', 'torus']):
    """
    Create a mini demonstration dataset with multiple objects.
    
    Args:
        objects: List of object types to create ('sphere', 'cube', 'torus')
    """
    print("=" * 60)
    print("Creating Mini Demonstration Dataset")
    print("=" * 60)
    print(f"\nObjects to generate: {', '.join(objects)}")
    
    # Create output directory
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)
    
    # Object creation functions
    creators = {
        'sphere': create_sphere_object,
        'cube': create_cube_object,
        'torus': create_torus_object
    }
    
    # Generate all objects
    all_objects = []
    for obj_name in objects:
        if obj_name in creators:
            try:
                obj_data = creators[obj_name]()
                all_objects.append(obj_data)
            except Exception as e:
                print(f"\nERROR: Failed to create {obj_name}: {e}")
                continue
        else:
            print(f"\nWARNING: Unknown object type '{obj_name}', skipping...")
    
    if not all_objects:
        print("\nERROR: No objects were created successfully!")
        return
    
    # Export files
    print("\n" + "=" * 60)
    print("Exporting files...")
    print("=" * 60)
    
    # For the web demo, use the first object (or combine them)
    # Option 1: Use first object only
    main_obj = all_objects[0]
    
    print(f"\nUsing '{main_obj['name']}' as main demo object...")
    
    # Export to frontend/public/scenes/ for web interface
    scenes_dir = Path("frontend/public/scenes")
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting scenes to: {scenes_dir.absolute()}")
    
    for obj_data in all_objects:
        obj_name = obj_data['name']
        scene_dir = scenes_dir / obj_name
        scene_dir.mkdir(exist_ok=True)
        
        # Export with standardized names
        save_point_cloud_ply(
            obj_data['partial'],
            scene_dir / "partial.ply",
            normalize=True
        )
        
        save_point_cloud_ply(
            obj_data['poisson'],
            scene_dir / "poisson.ply",
            normalize=True
        )
        
        save_point_cloud_ply(
            obj_data['deep_learning'],
            scene_dir / "deep.ply",
            normalize=True
        )
        
        # Also export mesh for wireframe
        save_mesh_ply(
            obj_data['poisson_mesh'],
            scene_dir / "poisson_mesh.ply",
            normalize=True
        )
        
        print(f"  ✓ {obj_name}/ (partial.ply, poisson.ply, deep.ply, poisson_mesh.ply)")
    
    # Also export main demo files for backward compatibility
    print(f"\nExporting main demo files (backward compatibility)...")
    save_point_cloud_ply(
        main_obj['partial'],
        output_dir / "input_partial.ply",
        normalize=True
    )
    print(f"  ✓ input_partial.ply")
    
    save_point_cloud_ply(
        main_obj['poisson'],
        output_dir / "poisson_reconstruction.ply",
        normalize=True
    )
    print(f"  ✓ poisson_reconstruction.ply")
    
    save_mesh_ply(
        main_obj['poisson_mesh'],
        output_dir / "poisson_mesh.ply",
        normalize=True
    )
    print(f"  ✓ poisson_mesh.ply")
    
    save_point_cloud_ply(
        main_obj['deep_learning'],
        output_dir / "output_predicted.ply",
        normalize=True
    )
    print(f"  ✓ output_predicted.ply")
    
    print("\n" + "=" * 60)
    print("Dataset created successfully!")
    print("=" * 60)
    print(f"\nFiles saved in:")
    print(f"  - {output_dir.absolute()} (main demo files)")
    print(f"  - {scenes_dir.absolute()} (scene files for web interface)")
    print("\nMain demo files:")
    print("  - input_partial.ply")
    print("  - poisson_reconstruction.ply")
    print("  - poisson_mesh.ply")
    print("  - output_predicted.ply")
    print("\nScene files (for web interface):")
    for obj_data in all_objects:
        print(f"  - scenes/{obj_data['name']}/ (partial.ply, poisson.ply, deep.ply, poisson_mesh.ply)")
    print("\nYou can now use these files in the web interface!")


if __name__ == "__main__":
    # Create dataset with sphere, cube, and torus
    create_demo_dataset(objects=['sphere', 'cube', 'torus'])

