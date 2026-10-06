import unittest
import numpy as np
from semantic_body_layers import sample_layers, distances
from refine_texture_semantics import verified_classes, consensus_target, repair_colors


class SemanticTests(unittest.TestCase):
    def test_categorical_sampling_never_blends_classes(self):
        label = np.array([[1, 3], [2, 4]], np.uint8)
        confidence = np.full((2, 2), 255, np.uint8)
        values, conf, _ = sample_layers(label, confidence, distances(label),
            np.array([[10, 20], [11, 20], [10, 21], [30, 20]]), [10, 20, 12, 22])
        np.testing.assert_array_equal(values, [1, 3, 2, 0])
        np.testing.assert_array_equal(conf, [1, 1, 1, 0])

    def test_clothing_veto_does_not_invent_skin(self):
        classes = np.array([3, 3, 2, 1], np.uint8)
        colors = np.array([[180, 185, 185], [65, 140, 200], [65, 140, 200], [20, 20, 20]], np.uint8)
        actual = verified_classes(classes, np.ones(4), colors)
        np.testing.assert_array_equal(actual, [3, 0, 2, 1])

    def test_low_confidence_remains_unknown(self):
        np.testing.assert_array_equal(verified_classes(np.array([1, 2, 3]),
            np.array([.4, .7, .84]), np.full((3, 3), 180, np.uint8)), [0, 0, 0])

    def test_hair_observations_do_not_establish_underlying_skin(self):
        labels = np.ones((16, 3), np.uint8)
        labels[:8, 0] = 2
        labels[:4, 1] = 2
        labels[:, 2] = 3
        target, _, _ = consensus_target(labels, np.ones_like(labels, bool),
            np.tile([0, 1], 8), np.array([True, True, False]))
        np.testing.assert_array_equal(target, [2, 0, 0])

    def test_one_clip_cannot_establish_cross_clip_consensus(self):
        labels = np.full((16, 1), 2, np.uint8)
        target, _, _ = consensus_target(labels, np.ones_like(labels, bool),
            np.zeros(16, int), np.ones(1, bool))
        self.assertEqual(target[0], 0)

    def test_unresolved_pixels_and_outside_roi_are_preserved(self):
        baseline = np.full((5, 4), 80, np.uint8); baseline[:, 3] = 255
        before = baseline.copy()
        accepted = repair_colors(baseline, np.array([1, 3]), np.array([[10, 20, 30], [40, 50, 60]]),
                                 np.array([240, 250]), np.array([True, False]))
        np.testing.assert_array_equal(accepted, [1])
        np.testing.assert_array_equal(baseline[[0, 2, 3, 4]], before[[0, 2, 3, 4]])
        np.testing.assert_array_equal(baseline[1], [10, 20, 30, 240])


if __name__ == '__main__': unittest.main()
