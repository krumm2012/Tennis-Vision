"""Geometric checks for the racket-to-video fitting contract."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_racket_pose as racket


class RacketPoseTests(unittest.TestCase):
    def test_recovers_known_camera_transform_with_mirror(self):
        frame = 0
        rotation = cv2.Rodrigues(np.array([0.24, -0.41, 0.31]))[0]
        wrist = np.array([0.4, -0.7, 8.1])
        translation = wrist - rotation @ np.array([0, racket.DEFAULT_MODEL['grip_y_m'], 0])
        object_points = racket.model_points(racket.DEFAULT_MODEL)
        camera_points = object_points @ rotation.T + translation
        mirror = {'normal_camera': [0.03, -0.35, 0.936], 'distance_camera_m': 10}
        focal = 1468.6
        real = racket.project(camera_points, focal)
        reflected = racket.project(racket.reflected(camera_points, mirror), focal)
        row = {'frame': frame, 'points': dict(zip(racket.PARTS, real.tolist())),
               'mirror_points': dict(zip(racket.PARTS, reflected.tolist())),
               '_raw_wrist': wrist.tolist()}
        meta = {'source_roots': [[0, 0, 0]], 'focal': [focal]}
        result = racket.fit_frame(row, frame, meta, mirror, racket.DEFAULT_MODEL)
        self.assertEqual(result['status'], 'fitted')
        self.assertFalse(result['ambiguous'])
        self.assertLess(result['real_reprojection_rms_px'], .01)
        self.assertLess(result['mirror_reprojection_rms_px'], .01)
        self.assertLess(np.linalg.norm(np.array(result['translation_camera_m']) - translation), 1e-4)
        self.assertLess(np.linalg.norm(np.array(result['rotation_camera_columns']) - rotation), 1e-4)


    def test_incomplete_landmarks_are_not_fitted(self):
        row = {'frame': 0, 'points': {'handle_end': [100, 100]}, '_raw_wrist': [0, 0, 0]}
        meta = {'source_roots': [[0, 0, 0]], 'focal': [1400]}
        result = racket.fit_frame(row, 0, meta, {}, racket.DEFAULT_MODEL)
        self.assertEqual(result['status'], 'insufficient_landmarks')
        self.assertNotIn('translation_camera_m', result)


if __name__ == '__main__':
    unittest.main()
