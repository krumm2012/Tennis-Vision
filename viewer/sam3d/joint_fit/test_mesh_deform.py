import unittest

import numpy as np

from build_full_viewer import deform_hand_vertices


class HandMeshPreviewTests(unittest.TestCase):
    def test_fingertip_follows_joint_while_distant_body_stays_fixed(self):
        joints = np.zeros((21, 3), dtype=float)
        joints[20] = [0, 0, 0]
        for finger in range(5):
            for k in range(4):
                joints[4 * finger + k] = [finger * .04, .08 - k * .02, 0]
        target = joints.copy()
        target[0] += [.02, 0, 0]
        vertices = np.array([[.002, .08, 0], [.002, .06, 0], [1, 1, 1]])
        changed = deform_hand_vertices(vertices, joints, target)
        self.assertGreater(changed[0, 0] - vertices[0, 0], .015)
        np.testing.assert_array_equal(changed[-1], vertices[-1])

    def test_unchanged_joints_leave_mesh_exactly_unchanged(self):
        joints = np.zeros((21, 3))
        vertices = np.array([[.01, .02, .03], [1, 2, 3]])
        np.testing.assert_array_equal(deform_hand_vertices(vertices, joints, joints), vertices)


if __name__ == '__main__':
    unittest.main()
