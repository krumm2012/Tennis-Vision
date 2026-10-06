import unittest
import numpy as np
from audit_automatic_grip import ratio_distance, hand_axis_distance, evaluate
from fit_racket_pose import project


class AutomaticGripTests(unittest.TestCase):
    def test_planar_ratio_and_resolution_invariance(self):
        for scale in [1, 2]:
            points = {'handle_end': [100 * scale, 100 * scale], 'tip': [785 * scale, 100 * scale]}
            value = ratio_distance(points, [190 * scale, 100 * scale], .685, scale)
            self.assertTrue(value['usable'])
            self.assertAlmostEqual(value['distance_m'], .09)
            self.assertFalse(value['perspective_corrected'])

    def test_missing_endpoint_and_off_axis_do_not_fabricate_a_distance(self):
        self.assertFalse(ratio_distance({'tip': [300, 100]}, [150, 100], .685)['usable'])
        value = ratio_distance({'handle_end': [100, 100], 'tip': [785, 100]}, [190, 120], .685)
        self.assertFalse(value['usable'])

    def test_perspective_hand_axis_recovers_free_distance_with_reversed_axis(self):
        anchor = np.array([.1, .2, 4.])
        axis = np.array([1., .3, .6]);axis /= np.linalg.norm(axis)
        for distance in [.055, .125]:
            butt, tip = project(np.array([anchor - distance * axis, anchor + (.685 - distance) * axis]), 1200, [1280, 720])
            value = hand_axis_distance({'handle_end': butt, 'tip': tip}, anchor, -axis, 1200, [1280, 720])
            self.assertTrue(value['usable'])
            self.assertAlmostEqual(value['distance_m'], distance, places=6)

    def test_evaluation_keeps_missing_endpoint_coverage_and_pixels_separate(self):
        prediction = {'video_sha256': 'v', 'image_size': [1280, 720], 'fps': 25, 'length_m': .685,
                      'frames': [{'racket': None, 'grip_center_original_px': [105, 100],
                                  'ratio': {'usable': False}, 'hand_axis': {'usable': False}}]}
        labels = {'video_sha256': 'v', 'image_size': [1280, 720], 'fps': 25,
                  'frames': [{'frame': 0, 'points': {'grip_center': [100, 100], 'handle_end': [90, 100]}, 'grip_confirmed': {'points': True}}]}
        report = evaluate(prediction, labels)
        self.assertEqual(report['pixel_errors_original_px']['grip_center']['median'], 5.)
        self.assertEqual(report['pixel_errors_original_px']['handle_end']['count'], 0)
        self.assertEqual(report['visible_butt_frames'], 0)
        self.assertEqual(report['annotated_point_coverage']['handle_end']['manual_visible_count'], 1)
        self.assertEqual(report['annotated_point_coverage']['handle_end']['fraction'], 0.)
        self.assertEqual(report['distance_proxy_difference_cm']['ratio']['count'], 0)
        self.assertFalse(report['physical_distance_ground_truth_available'])

    def test_wrong_video_evaluation_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({'video_sha256': 'a'}, {'video_sha256': 'b'})


if __name__ == '__main__':
    unittest.main()
