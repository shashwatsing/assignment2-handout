import numpy as np

from numpy import arctan2 as atan2
from numpy import arcsin as asin
from numpy import cos as cos
from numpy import sin as sin

from quadrotor_simulator_py.utils.quaternion import Quaternion


class Rotation3:

    def __init__(self, R=None):
        self.R = None

        if R is None:
            self.R = np.eye(3)
        else:
            self.R = R

    def to_euler_zyx(self):
        """ Convert self.R to Z-Y-X euler angles

        Output:
            zyx: 1x3 numpy array containing euler angles.
                The order of angles should be phi, theta, psi, where
                roll == phi, pitch == theta, yaw == psi
        """

        theta = 0.
        phi = 0.
        psi = 0.

        phi = atan2(self.R[2, 1], self.R[2, 2])
        theta = -asin(self.R[2, 0])
        psi = atan2(self.R[1, 0], self.R[0, 0])

        return np.array([phi, theta, psi])
    
    @classmethod
    def from_euler_zyx(self, zyx):
        """Convert euler angle rotation representation to 3x3
                rotation matrix. The input is represented as
                np.array([roll, pitch, yaw]).
        Arg:
            zyx: 1x3 numpy array containing euler angles

        Output:
            Rot: 3x3 rotation matrix (numpy)
        """
        phi = zyx[0]
        theta = zyx[1]
        psi = zyx[2]

        cos_phi = cos(phi)
        sin_phi = sin(phi)
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        cos_psi = cos(psi)
        sin_psi = sin(psi)

        R_x = np.array([[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]])
        R_y = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])
        R_z = np.array([[cos_psi, -sin_psi, 0], [sin_psi, cos_psi, 0], [0, 0, 1]])

        R = np.dot(np.dot(R_z, R_y), R_x)

        return Rotation3(R=R)

    def roll(self):
        """ Extracts the phi component from the rotation matrix

        Output:
            phi: scalar value representing phi
        """

        R = self.R
        phi = atan2(R[2, 1], R[2, 2])
        return phi

    def pitch(self):
        """ Extracts the theta component from the rotation matrix

        Output:
            theta: scalar value representing theta
        """

        theta = -asin(self.R[2, 0])

        return theta

    def yaw(self):
        """ Extracts the psi component from the rotation matrix

        Output:
            theta: scalar value representing psi
        """

        psi = atan2(self.R[1, 0], self.R[0, 0])

        return psi

    @classmethod
    def from_quat(self, q):
        """ Calculates the 3x3 rotation matrix from a quaternion
                parameterized as (w,x,y,z).

        Output:
            Rot: 3x3 rotation matrix represented as numpy matrix
        """

        w = q.w()
        x = q.x()
        y = q.y()
        z = q.z()

        Rot = Rotation3()
        Rot.R = np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]])

        return Rot

    def to_quat(self):
        """ Calculates a quaternion from the class variable
                self.R and returns it

        Output:
            q: An instance of the Quaternion class parameterized
                as [w, x, y, z]
        """

        w = 0.5 * np.sqrt(1 + self.R[0, 0] + self.R[1, 1] + self.R[2, 2])
        x = (self.R[2, 1] - self.R[1, 2]) / (4 * w)
        y = (self.R[0, 2] - self.R[2, 0]) / (4 * w)
        z = (self.R[1, 0] - self.R[0, 1]) / (4 * w)
        return Quaternion([w, x, y, z])
