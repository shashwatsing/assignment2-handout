#!/usr/bin/env python
import numpy as np
import yaml
import math
import copy
import open3d as o3d
import os
import sys

sys.path.append('../')

from quadrotor_simulator_py.map_tools import OccupancyGrid, Cell, Point

def line_segments_numpy(t, points):
    line_set = []
    for p in points:
        line = generate_lineset(Point().from_numpy(t), Point().from_numpy(p))
        line_set.append(line)
    return line_set

def generate_lineset(start, end):
    points = np.array([start.to_numpy(), end.to_numpy()])
    # Define the line connectivity (indexing into the points array)
    lines = [[0, 1]]  # Connect point 0 to point 1
    
    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)  # Set points
    line_set.lines = o3d.utility.Vector2iVector(lines)    # Define connectivity

    # Optionally, add color to the line
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
    return line_set

def raycells_to_voxels(raycells, og):
    list_of_points = np.zeros((len(raycells), 3))
    for i, r in enumerate(raycells):
        list_of_points[i, :] = og.cell2point(r).to_numpy() + Point(og.resolution/2,
                                                                   og.resolution/2,
                                                                   og.resolution/2).to_numpy()

    return numpy_to_voxel_grid(list_of_points, og.resolution)

def numpy_to_voxel_grid(pts, res):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    voxel_size = res
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def visualize_mesh(meshfile):
    mesh = o3d.io.read_triangle_mesh(meshfile)

    # Check if the mesh was successfully loaded
    if mesh.is_empty():
        print("Failed to load the mesh.")
    else:
        print("Mesh successfully loaded:")

    # Optionally compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    return mesh

def visualize_sensor_triad(Twc):
    triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
    triad.transform(Twc.get_se3())
    return triad
