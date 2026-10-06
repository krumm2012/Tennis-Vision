import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from audit_racket_evidence_boundaries import NAMES, projections
from correct_reviewed_racket_frame import fit_manual_frame


class ManualCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.model = {'grip_y_m': .045, 'throat_y_m': .345, 'length_m': .685, 'head_half_width_m': .135, 'head_center_y_m': .52}
        self.meta = {'focal': [2937.], 'image_size': [2560, 1440]}
        n = np.array([.035, -.35, .936]); n /= np.linalg.norm(n)
        self.mirror = {'normal_camera': n.tolist(), 'distance_camera_m': 10.6}
        self.r = Rotation.from_rotvec([.6, .3, 1.5]).as_matrix(); self.t = np.array([.5, -.3, 8.])
        self.anchor = self.r@np.array([0, .045, 0])+self.t
        self.row = {'frame': 0, 'face_correspondence_confirmed': False}
        for v in ['points', 'mirror_points']:
            uv = projections(self.r, self.t, self.model, self.meta, 0, self.mirror if v == 'mirror_points' else None)
            self.row[v] = dict(zip(NAMES[:5], uv[:5].tolist()))

    def fit(self, mode):
        return fit_manual_frame(self.row, self.model, self.meta, self.mirror,
                                Rotation.from_rotvec([.62, .32, 1.53]).as_matrix(), self.t+[.03, .02, .1], mode, self.anchor)

    def test_native_resolution_camera_recovers_manual_pixels(self):
        result = self.fit('real_manual')
        self.assertLess(result['metrics']['points']['rms_original_px'], 1e-5)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['validation_role'], 'manual_edit_training_frame_not_heldout')

    def test_independent_unconfirmed_mirror_rim_swap(self):
        points = self.row['mirror_points']; points['rim_side'], points['rim_opposite'] = points['rim_opposite'], points['rim_side']
        result = self.fit('both_manual')
        self.assertLess(result['metrics']['points']['rms_original_px'], 1e-5)
        self.assertLess(result['metrics']['mirror_points']['rms_original_px'], 1e-5)

    def test_fixed_anchor_cannot_move_to_fit_pixels(self):
        result = self.fit('real_manual_fixed_grip')
        self.assertLess(result['frozen_palm_anchor_gap_mm'], 1e-8)
        self.assertLess(result['metrics']['points']['rms_original_px'], 1e-5)
