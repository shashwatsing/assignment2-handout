import os
import unittest
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import cv2
from sklearn.metrics import mean_squared_error

sys.path.append('../')

from quadrotor_simulator_py.sensor_simulator import SensorSimulator
from quadrotor_simulator_py.utils import Pose, Rot3
from quadrotor_simulator_py.map_tools import *

def score_results(res, correct, eps, test_name):
    rmse = math.sqrt(mean_squared_error(res, correct))
    if rmse < eps:
        print('Test ' + test_name + ' passed')
        return 1.
    print('Test ' + test_name + ' failed. RMSE is ' +
          str(rmse) + ' but < ' + str(eps) + ' required.')
    return 0.

def test_ray_mesh_intersection(sensor_simulator, Twb,
                               solutions_filepath, meshfile, visualize=True):
    world_frame_points = sensor_simulator.ray_mesh_intersect(Twb)
    #np.savez(solutions_filepath,
    #         world_frame_points=world_frame_points)

    correct_soln = np.load(solutions_filepath)["world_frame_points"]

    if visualize:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(world_frame_points)

        Twc = Twb.compose(sensor_simulator.Tbc)
        transformation = Twc.get_se3()
        triad = visualize_sensor_triad(Twc)

        line_list = line_segments_numpy(Twc.translation(), world_frame_points)

        mesh = visualize_mesh(meshfile)

        o3d.visualization.draw_geometries([mesh, point_cloud, triad] + line_list)

    return score_results(world_frame_points, correct_soln, 1e-2, 'ray_mesh_intersect')

def test_transform_to_camera_frame(sensor_simulator, Twb,
                                   world_frame_points, solutions_filepath, meshfile, visualize=True):
    camera_frame_points = sensor_simulator.transform_to_camera_frame(Twb, world_frame_points)
    #np.savez(solutions_filepath,
    #         camera_frame_points=camera_frame_points)

    correct_soln = np.load(solutions_filepath)["camera_frame_points"]

    if visualize:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(camera_frame_points)

        T = Pose()
        triad = visualize_sensor_triad(T)

        line_list = line_segments_numpy(np.zeros(3), camera_frame_points)

        mesh = visualize_mesh(meshfile)

        o3d.visualization.draw_geometries([mesh, point_cloud, triad] + line_list)

    return score_results(camera_frame_points, correct_soln, 1e-2, 'camera_frame_points')

def test_project_to_image_plane(sensor_simulator, camera_frame_points, solutions_filepath, visualize=True):
    I = sensor_simulator.project_to_image_plane(camera_frame_points)
    #np.savez(solutions_filepath, I=I)

    correct_soln = np.load(solutions_filepath)["I"]

    if visualize:
        # To display the depth image properly, we need to apply a colormap;
        # however, the colormaps only work with 8-bit images so we must rescale.
        im_gray = cv2.normalize(I, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv_image = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return score_results(I, correct_soln, 1e-2, 'depth image')

def run_tests():

    score = 0.0

    DATA_DIR = "../data/"
    config = "../config/depth_sensor.yaml"
    meshfile = "../mesh/scene1.ply"

    sensor_simulator = SensorSimulator(config, meshfile)

    """ 
    Test using first pose
    """
    Twb = Pose()
    Twb.set_translation(np.array([2, -1, 1.5]))
    Twb.set_rotation(np.eye(3))
    solutions_filepath = "../data/ray_mesh_intersect_solutions_pose1.npz"
    score += test_ray_mesh_intersection(sensor_simulator, Twb, solutions_filepath, meshfile, False)

    world_frame_points = np.load(solutions_filepath)["world_frame_points"]
    solutions_filepath = "../data/camera_frame_solutions_pose1.npz"
    score += test_transform_to_camera_frame(sensor_simulator, Twb, world_frame_points, solutions_filepath, meshfile, False)

    camera_frame_points = np.load(solutions_filepath)["camera_frame_points"]
    solutions_filepath = "../data/depth_image_solutions_pose1.npz"
    score += test_project_to_image_plane(sensor_simulator, camera_frame_points, solutions_filepath, False)

    """ 
    Test using second pose
    """
    Twb = Pose()
    Twb.set_translation(np.array([5, -8, 2.0]))
    R = Rot3().from_euler_zyx([0.0, 0.0, math.pi/2])
    Twb.set_rotation(R.R)
    solutions_filepath = "../data/ray_mesh_intersect_solutions_pose2.npz"
    score += test_ray_mesh_intersection(sensor_simulator, Twb, solutions_filepath, meshfile, False)

    world_frame_points = np.load(solutions_filepath)["world_frame_points"]
    solutions_filepath = "../data/camera_frame_solutions_pose2.npz"
    score += test_transform_to_camera_frame(sensor_simulator, Twb, world_frame_points, solutions_filepath, meshfile, False)

    camera_frame_points = np.load(solutions_filepath)["camera_frame_points"]
    solutions_filepath = "../data/depth_image_solutions_pose2.npz"
    score += test_project_to_image_plane(sensor_simulator, camera_frame_points, solutions_filepath, False)

    return score

if __name__ == "__main__":
    run_tests()
