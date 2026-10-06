import unittest
import numpy as np
from audit_shoulder_evidence import discovery_frames, support_summary, clothing_edge_review


class ShoulderEvidenceTests(unittest.TestCase):
    def test_discovery_excludes_all_frozen_splits(self):
        frames = discovery_frames(250, [12, 32], [22, 42], [52, 62])
        self.assertEqual(frames[:3], [2, 72, 82])
        self.assertFalse(set(frames) & {12, 22, 32, 42, 52, 62})
        self.assertTrue(all(0 <= f < 250 for f in frames))

    def test_support_cannot_turn_repeated_hair_into_skin(self):
        labels = np.tile([1, 2, 3, 2, 0], (10, 1)).astype(np.uint8)
        labels[5:, 3] = 3
        keys = np.array([[i // 5, i, 0] for i in range(10)])
        s = support_summary(labels, keys)
        np.testing.assert_array_equal(s['strong'], [False, True, True, False, False])
        np.testing.assert_array_equal(s['count'], [0, 10, 10, 10, 0])
        self.assertEqual(s['agreement'][3], .5)
        keys[:, 0] = 0
        self.assertFalse(support_summary(labels, keys)['strong'].any())

    def test_edge_low_confidence_and_color_veto_are_separate(self):
        labels = np.full((20, 20), 2, np.uint8); labels[:, 10:] = 3
        confidence = np.full((20, 20), 255, np.uint8); confidence[:10] = 100
        crop = np.full((20, 20, 3), 200, np.uint8); crop[10:, 10:] = [0, 0, 200]
        roi = np.ones((20, 20), bool); roi[-2:] = False
        band, weak, veto = clothing_edge_review(labels, confidence, crop, roi)
        self.assertTrue(weak.any()); self.assertTrue(veto.any())
        self.assertFalse((weak & veto).any()); self.assertFalse(band[-2:].any())
        self.assertTrue(np.all(~weak | band)); self.assertTrue(np.all(~veto | band))


if __name__ == '__main__':
    unittest.main()
