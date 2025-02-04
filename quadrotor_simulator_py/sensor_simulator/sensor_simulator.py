#!/usr/bin/env python
import numpy as np
import yaml
import math
from plyfile import *

from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils import Pose

class SensorSimulator:

    def __init__(self, yaml_file, meshfile):
        self.cx = 0.0
        self.cy = 0.0
        self.fx = 0.0
        self.fy = 0.0

        self.vertices = []
        self.faces = []
        self.Tbc = Pose()

        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YamlError as exc:
                print(exc)

            self.to_meters = data['depth_camera']['to_meters']
            fov_x_deg = data['depth_camera']['fov']['x']
            fov_y_deg = data['depth_camera']['fov']['y']
            self.size_x = data['depth_camera']['size']['x']
            self.size_y = data['depth_camera']['size']['y']
            self.range_min = data['depth_camera']['range_min']
            self.range_max = data['depth_camera']['range_max']
            self.trimmed_range_max = data['depth_camera']['trimmed_range_max']
            #self.downsample = data['depth_camera']['downsample']

            fov_x_rad = fov_x_deg * math.pi / 180.0
            fov_y_rad = fov_y_deg * math.pi / 180.0

            self.cx = 0.5 * self.size_x
            self.cy = 0.5 * self.size_y
            self.fx = self.cx / np.tan(0.5 * fov_x_rad)
            self.fy = self.cy / np.tan(0.5 * fov_y_rad)

            self.Tbc.set_translation(np.array([data['depth_camera']['offset']['x'],
                                               data['depth_camera']['offset']['y'],
                                               data['depth_camera']['offset']['z']]))
            self.Tbc.set_rotation(Rot3().from_euler_zyx([data['depth_camera']['offset']['roll'],
                                                         data['depth_camera']['offset']['pitch'],
                                                         data['depth_camera']['offset']['yaw']]).R)

            self.load_ply(meshfile)

            self.get_normalized_depth_points()


    def __repr__(self):
        return ('SensorSimulator\n' +
                'cx:\t\t' + str(self.cx) + '\n' +
                'cy:\t\t\t' + str(self.cy) + '\n' +
                'fx:\n' + str(self.fx) + '\n' +
                'fy:\n' + str(self.fy) + '\n' +
                self.rpm_params.__repr__()
                )

    def get_normalized_depth_points(self):
        """
        Pre-calculate normalized vectors representing rays
            of the depth sensor. These are stored and iterated
            over in the ray_mesh_intersect function
        """
        self.depth_points = np.zeros((self.size_x*self.size_y, 3))
        count = 0
        for x in range(0, self.size_x):
            for y in range(0, self.size_y):
                p = np.array([ (x - self.cx) / self.fx, (y - self.cy) / self.fy, 1])
                self.depth_points[count, :] = p / np.linalg.norm(p)
                count += 1

    def load_ply(self, filename):
        """
        Load a PLY file using the 'plyfile' library.
        Returns:
            vertices: Nx3 numpy array of vertex coordinates
            faces:    Mx3 numpy array of triangle vertex indices
        """
        # Read the PLY file
        plydata = PlyData.read(filename)

        # Extract vertex data
        vertex_data = plydata['vertex'].data
        self.vertices = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).astype(np.float32)

        # Extract face (indices) data.
        if 'face' not in plydata:
            raise ValueError("No 'face' element found in the PLY file.")

        face_data = plydata['face'].data['vertex_indices']

        # Convert list-of-lists into a 2D array (Mx3)
        self.faces = np.vstack(face_data).astype(np.int32)

    def ray_triangle_intersect(self, ray_orig, ray_dir, v0, v1, v2):
        """
        Möller–Trumbore ray-triangle intersection.
        ray_orig, ray_dir: 3D vectors (np.array of shape (3,))
        v0, v1, v2:        triangle vertices (np.array of shape (3,))
        Returns:
            t (float) if there is an intersection (distance along the ray),
            or None if no intersection.
        """

        # TODO: Assignment 1.1
        return None

    def ray_mesh_intersect(self, Twb):
        """
        Perform ray-mesh intersection at the pose specified by Twb
            using the mesh loaded in the class instance.
        Twb: Transform that represents body frame wrt world frame
        Returns:
            world_frame_points (nx3 numpy array) world frame points projected into the world frame
        """
        Twc = Twb.compose(self.Tbc)

        t = Twc.translation()
        R = Twc.get_so3()

        world_frame_points = np.zeros((self.size_x * self.size_y, 3))

        ray_orig = np.array(t.T, dtype=np.float32)

        for idx, ray in enumerate(self.depth_points):
            rotated_ray = (R  @ ray.reshape((3,1))).reshape(3)

            # Make sure the direction is normalized
            ray_dir = np.array(rotated_ray.T, dtype=np.float32)
            ray_dir /= np.linalg.norm(ray_dir)

            # Iterate over all triangles and check for intersections
            hit_something = False
            nearest_t = float('inf')
            hit_point = None
            hit_face_index = -1
            world_frame_points[idx,:] = ray_dir * self.range_max + t

            for i, tri in enumerate(self.faces):
                v0 = self.vertices[tri[0]]
                v1 = self.vertices[tri[1]]
                v2 = self.vertices[tri[2]]

                t_hit = self.ray_triangle_intersect(ray_orig, ray_dir, v0, v1, v2)
                if t_hit is not None and t_hit < nearest_t:
                    nearest_t = t_hit
                    hit_something = True
                    hit_face_index = i
                    # Intersection point
                    hit_point = ray_orig + ray_dir * t_hit

                    if nearest_t < self.range_max:
                        world_frame_points[idx,:] = hit_point
        return world_frame_points

    def transform_to_camera_frame(self, Twb, world_frame_points):
        """
        Transform points from world frame to camera frame
        Twb: Transform that represents body frame wrt world frame
        world_frame_points: points generated from ray_mesh_intersect
        Returns:
            camera_frame_points (nx3 numpy array) camera frame points projected into the world frame
        """

        # TODO: Assignment 1.2
        camera_frame_points = np.zeros((self.size_x * self.size_y, 3))
        return camera_frame_points

    def project_to_image_plane(self, camera_frame_points):
        """
        Converts camera frame points to a depth image
        camera_frame_points: points generated from transform_to_camera_frame function
        Returns:
            I (size_y x size_x numpy array) representing depth image
        """

        # TODO: Assignment 1.3
        I = np.zeros((self.size_y, self.size_x))
        return I
