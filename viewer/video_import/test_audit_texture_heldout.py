import unittest
import numpy as np
from audit_texture_heldout import regions, validate_sources, stats


class HeldoutPixelTests(unittest.TestCase):
    def test_regions_exclude_gutter_preserve_old_and_include_all_added(self):
        old = np.zeros((2, 3, 4), np.uint8); new = old.copy()
        old[0, :2] = [10, 20, 30, 255]; new[:] = old
        new[1, 0] = [20, 30, 40, 255]; new[1, 2] = [80, 90, 100, 255]
        faces = np.array([[0, 0, 0], [1, 1, -1]])
        r = regions(old, new, faces, 2)
        np.testing.assert_array_equal(r['old'], [0])
        np.testing.assert_array_equal(r['added'], [3])
        np.testing.assert_array_equal(r['unknown'], [2, 4])
        new[0, 0, 0] += 1
        with self.assertRaises(ValueError): regions(old, new, faces, 1)

    def test_heldout_or_invalid_view_provenance_is_rejected(self):
        source = {'clip': np.array([[0, 0]]), 'frame': np.array([[5, 25]]), 'view': np.array([[0, 1]])}
        rows = [{'frames': {'train_frames': [5], 'heldout_frames': [25]}}]
        with self.assertRaises(ValueError): validate_sources(source, rows, np.ones((1, 2), bool))
        source['frame'][:] = 5
        validate_sources(source, rows, np.ones((1, 2), bool))
        source['view'][0, 1] = 2
        with self.assertRaises(ValueError): validate_sources(source, rows, np.ones((1, 2), bool))

    def test_empty_visibility_is_not_zero_error(self):
        self.assertIsNone(stats([])['median'])
        self.assertEqual(stats([np.array([3., 9.])])['median'], 6.)


if __name__ == '__main__': unittest.main()
