"""
Open3D visualization helpers for point clouds and meshes.
"""

import open3d as o3d
import numpy as np
from typing import Optional, List, Tuple


def show_point_cloud(pcd: o3d.geometry.PointCloud, 
                     window_name: str = "Point Cloud",
                     width: int = 1024,
                     height: int = 768,
                     point_size: float = 1.0,
                     background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    """
    Display a point cloud in an interactive 3D window.
    
    Args:
        pcd: Open3D PointCloud object to display
        window_name: Name of the visualization window
        width: Window width in pixels
        height: Window height in pixels
        point_size: Size of points in the visualization
        background_color: RGB background color (0.0-1.0)
    """
    if len(pcd.points) == 0:
        print("Warning: Point cloud is empty, nothing to display")
        return
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array(background_color)
    
    # Set view point to look at the point cloud
    view_control = vis.get_view_control()
    
    # Get point cloud bounds to set appropriate view
    points = np.asarray(pcd.points)
    if len(points) > 0:
        centroid = np.mean(points, axis=0)
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        max_extent = np.max(extent)
        
        # Set camera to view the point cloud
        view_control.set_lookat(centroid)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.7)
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def show_point_clouds(pcds: List[o3d.geometry.PointCloud],
                     window_name: str = "Point Clouds",
                     width: int = 1024,
                     height: int = 768) -> None:
    """
    Display multiple point clouds in the same window.
    
    Args:
        pcds: List of Open3D PointCloud objects
        window_name: Name of the visualization window
        width: Window width in pixels
        height: Window height in pixels
    """
    if len(pcds) == 0:
        print("Warning: No point clouds to display")
        return
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    
    # Add all point clouds
    for i, pcd in enumerate(pcds):
        vis.add_geometry(pcd)
    
    # Set view point
    view_control = vis.get_view_control()
    
    # Compute overall bounds
    all_points = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        if len(points) > 0:
            all_points.append(points)
    
    if len(all_points) > 0:
        all_points = np.vstack(all_points)
        centroid = np.mean(all_points, axis=0)
        view_control.set_lookat(centroid)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()


def create_colored_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    Create a colored point cloud from numpy arrays.
    
    Args:
        points: Nx3 array of point coordinates
        colors: Optional Nx3 array of RGB colors (0.0-1.0)
        
    Returns:
        Open3D PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError("Colors must have same shape as points")
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd
