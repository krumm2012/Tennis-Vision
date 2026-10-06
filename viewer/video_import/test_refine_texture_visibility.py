import unittest
import numpy as np
from multivideo_texture import valid_points
from refine_texture_visibility import facing_cosine, uv_seam_pairs


class VisibilityTests(unittest.TestCase):
    def test_opposite_surface_is_not_visible(self):
        points = np.array([[-1., -1., 2.], [0., 1., 2.], [1., -1., 2.]])
        faces = np.array([[0, 1, 2], [2, 1, 0]])
        cosine = facing_cosine(points, faces)
        self.assertGreater(cosine[0], .9)
        self.assertLess(cosine[1], -.9)

    def test_reflected_winding_is_restored_once(self):
        points = np.array([[-1., -1., 2.], [0., 1., 2.], [1., -1., 2.]])
        faces = np.array([[0, 1, 2]])
        reflected = points * [-1, 1, 1]
        np.testing.assert_allclose(facing_cosine(points, faces), facing_cosine(reflected, faces, reflected=True))
        self.assertLess(facing_cosine(reflected, faces, reflected=False)[0], 0)

    def test_thin_hand_occlusion_uses_stricter_tolerance(self):
        points = np.array([[0., 0., 2.015], [0., 0., 2.015]])
        mask = np.full((20, 20), 255, np.uint8); depth = np.full((20, 20), 2., np.float32)
        arguments = (points, mask, depth, 10, np.array([20, 20]), np.array([20, 20]))
        original = valid_points(*arguments)[0]
        strict = valid_points(*arguments, depth_tolerance=np.array([.01, .025]))[0]
        np.testing.assert_array_equal(original, [True, True])
        np.testing.assert_array_equal(strict, [False, True])

    def test_zero_area_face_cannot_be_front_facing(self):
        result = facing_cosine(np.zeros((3, 3)), np.array([[0, 1, 2]]))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0], 0)

    def test_uv_seams_pair_the_same_geometric_edge(self):
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        uv_faces = np.array([[0, 1, 2], [3, 4, 5]])
        uv = np.array([[0, 0], [.4, 0], [0, .4], [.6, .6], [1, .6], [1, 1]])
        samples, owners = uv_seam_pairs(faces, uv_faces, uv)
        self.assertEqual(samples.shape, (1, 2, 2))
        np.testing.assert_array_equal(owners, [[0, 1]])
        self.assertTrue((samples[0, 0] < .4).all())
        self.assertTrue((samples[0, 1] >= .6).all())


if __name__ == '__main__': unittest.main()
