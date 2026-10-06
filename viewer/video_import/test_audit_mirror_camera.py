import unittest
import numpy as np

from audit_mirror_camera import epipolar_residual, normal_diagnostic, paired_points


class MirrorAuditTests(unittest.TestCase):
    def test_physical_reflected_points_satisfy_epipolar_lines(self):
        n = np.array([.1, -.3, .95]); n /= np.linalg.norm(n)
        x = np.array([1., .2, 8.]); y = x-2*(x@n-11)*n
        size = [2560, 1440]; focal = 2900
        a = x[:2]/x[2]*focal+np.array(size)/2
        b = y[:2]/y[2]*focal+np.array(size)/2
        np.testing.assert_allclose(epipolar_residual(a, b, focal, size, n), 0, atol=1e-9)
        self.assertGreater(np.max(np.abs(epipolar_residual(a, b+[0, 20], focal, size, n))), 1)

    def test_unconfirmed_rim_pairs_excluded(self):
        points = {'tip': [1, 2], 'rim_side': [3, 4]}
        pairs = paired_points({'frames': [{'frame': 1, 'points': points, 'mirror_points': points}]})
        self.assertEqual([r['part'] for r in pairs], ['tip'])

    def test_training_and_reserved_frame_overlap_rejected(self):
        row = {'frame': 30, 'part': 'tip', 'real': [1, 2], 'mirror': [3, 4]}
        with self.assertRaises(ValueError):
            normal_diagnostic([row], {'reserved': [row]}, {}, {})
