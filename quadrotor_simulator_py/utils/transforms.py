import numpy as np

from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils.quaternion import Quaternion
from quadrotor_simulator_py.utils.pose import Pose


def set_robot_transforms():
    """ The transform that represents the camera in the frame
        of the IMU is given by Tbc. The transform that 
        represents the pose of the robot's body in world frame
        coordinates is given by Twb. Find the transform
        that represents the pose of the camera in the world frame.
        
        Output: 
            Pose object that represents the camera in the world frame
    """

    rpy = np.array([-1.6, 0.0, -1.6])
    tbc = np.array([0.15, 0.0, -0.06])
    Rbc = Rot3().from_euler_zyx(rpy)

    # This transform represents the camera in body frame coordinates
    Tbc = Pose(np.append(tbc, Rbc.to_quat().data))

    rpy = np.array([0., 0., 0.])
    twb = np.array([4., 1., 3.])
    Rwb = Rot3().from_euler_zyx(rpy)

    # This transform represents the body in in world frame coordinates
    Twb = Pose(np.append(twb, Rwb.to_quat().data))

    # TODO: Assignment 1: Problem 1.10
    return Pose()

