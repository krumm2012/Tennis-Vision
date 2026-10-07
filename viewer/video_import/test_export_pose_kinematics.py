import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from export_pose_kinematics import angular_velocity, derivative, segment_frame


class KinematicsTests(unittest.TestCase):
    def test_constant_translation_has_correct_velocity(self):
        fps = 25
        time = np.arange(10) / fps
        position = time[:, None] * np.array([[2., -3., 4.]])
        velocity = derivative(position, fps)
        np.testing.assert_allclose(velocity[1:-1], np.tile([2, -3, 4], (8, 1)))
        self.assertTrue(np.isnan(velocity[[0, -1]]).all())

    def test_hidden_frame_invalidates_neighbour_derivatives(self):
        positions = np.arange(21, dtype=float).reshape(7, 3)
        positions[3] = np.nan
        velocities = derivative(positions, 25)
        self.assertTrue(np.isnan(velocities[2:5]).all())
        self.assertTrue(np.isfinite(velocities[[1, 5]]).all())

    def test_known_rotation_and_hidden_frame(self):
        fps = 25
        angles = np.arange(7) / fps * 2
        matrices = Rotation.from_rotvec(angles[:, None] * np.array([[0., 0., 1.]])).as_matrix()
        np.testing.assert_allclose(angular_velocity(matrices, fps), np.tile([0, 0, 2], (6, 1)), atol=1e-12)
        matrices[3] = np.nan
        velocity = angular_velocity(matrices, fps)
        self.assertTrue(np.isnan(velocity[2:4]).all())
        self.assertTrue(np.isfinite(velocity[[0, 1, 4, 5]]).all())

    def test_proxy_frame_is_right_handed_and_degeneracy_is_missing(self):
        lateral = np.array([[1., 1., 0.], [0., 1., 0.]])
        up = np.array([[0., 1., 0.], [0., 1., 0.]])
        matrices = segment_frame(lateral, up)
        np.testing.assert_allclose(matrices[0], np.eye(3))
        self.assertTrue(np.isnan(matrices[1, :, 0]).all())


if __name__ == '__main__':
    unittest.main()
