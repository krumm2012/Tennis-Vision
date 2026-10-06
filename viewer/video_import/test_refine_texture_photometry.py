import unittest
import numpy as np
from refine_texture_photometry import robust_source_offsets, grid_edges, poisson_correction


class PhotometryTests(unittest.TestCase):
    def test_recovers_source_exposure_and_ignores_invalid_observations(self):
        reference = np.full((60, 3), 100.)
        colors = np.stack([reference + 8, reference - 5])
        colors[:, -1] = 255
        valid = np.ones((2, 60), bool); valid[:, -1] = False
        offsets, counts = robust_source_offsets(colors, valid, reference, np.ones(60, bool), shrinkage=0)
        np.testing.assert_allclose(offsets, [[-8] * 3, [5] * 3])
        np.testing.assert_array_equal(counts, [59, 59])

    def test_low_support_and_extreme_colors_are_bounded(self):
        reference = np.full((60, 3), 100.)
        colors = np.stack([reference + 90, reference - 90])
        valid = np.ones((2, 60), bool); valid[1, 10:] = False
        offsets, _ = robust_source_offsets(colors, valid, reference, np.ones(60, bool), shrinkage=0)
        np.testing.assert_allclose(offsets, [[-12] * 3, [0] * 3])

    def test_packed_uv_islands_are_not_connected(self):
        face_map = np.array([[0, 0, 1, 1]])
        p, q = grid_edges(face_map, np.ones_like(face_map, bool), np.full((2, 3), -1))
        self.assertNotIn((1, 2), list(zip(p, q)))
        self.assertEqual(len(p), 2)

    def test_poisson_reduces_seam_without_changing_distant_pixels(self):
        colors = np.full((12, 40, 3), 100.)
        colors[:, 20:] += 10
        fm = np.zeros((12, 40), int)
        p, q = grid_edges(fm, np.ones_like(fm, bool), np.full((1, 3), -1))
        seam = ((p % 40) == 19) & ((q % 40) == 20)
        delta, info = poisson_correction(colors, fm.shape, p, q, seam)
        self.assertEqual(info['seam_edges'], 12)
        corrected = colors + delta
        self.assertLess(np.abs(corrected[:, 19] - corrected[:, 20]).mean(), 5)
        np.testing.assert_array_equal(corrected[:, :10], colors[:, :10])
        self.assertLessEqual(np.abs(delta).max(), 6)

    def test_no_qualified_seam_is_identity(self):
        fm = np.zeros((5, 5), int)
        p, q = grid_edges(fm, np.ones_like(fm, bool), np.full((1, 3), -1))
        colors = np.random.default_rng(3).uniform(0, 255, (5, 5, 3))
        delta, _ = poisson_correction(colors, fm.shape, p, q, np.zeros(len(p), bool))
        np.testing.assert_array_equal(delta, np.zeros_like(colors))


if __name__ == '__main__': unittest.main()
