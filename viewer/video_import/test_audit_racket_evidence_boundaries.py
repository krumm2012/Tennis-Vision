import copy
import unittest
import numpy as np
from audit_racket_evidence_boundaries import projections, residuals


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = {'throat_y_m': .35, 'length_m': .685, 'head_half_width_m': .13,
                      'head_center_y_m': .51, 'grip_y_m': .09}
        self.meta = {'focal': [1000.], 'image_size': [1280, 720]}
        self.uv = projections(np.eye(3), np.array([.3, 0., 3.]), self.model, self.meta, 0)

    def test_reflection_applied_once_in_camera_space(self):
        mirrored = projections(np.eye(3), np.array([.3, 0., 3.]), self.model, self.meta, 0,
                               {'normal_camera': [1., 0., 0.], 'distance_camera_m': 0.})
        np.testing.assert_allclose(mirrored[:, 0], 1280 - self.uv[:, 0])
        np.testing.assert_allclose(mirrored[:, 1], self.uv[:, 1])

    def test_grip_is_projected_from_configured_distance_above_butt(self):
        np.testing.assert_allclose(self.uv[6] - self.uv[0], [0., 30.])

    def test_unoriented_rim_diagnostics_do_not_change_authoritative_rows(self):
        row = {'source': 'manual_review', 'points': {'rim_side': self.uv[4].tolist(), 'rim_opposite': self.uv[3].tolist()}}
        saved = copy.deepcopy(row)
        result = residuals(self.uv, row, 'points', 2.)
        self.assertEqual(result['point_error_canonical_px'], {'rim_side': 0., 'rim_opposite': 0.})
        self.assertIsNone(result['shaft_error_deg']); self.assertEqual(row, saved)
        row['face_correspondence_confirmed'] = True
        self.assertGreater(residuals(self.uv, row, 'points', 2.)['point_error_canonical_px']['rim_side'], 0.)

    def test_missing_handle_is_not_filled_for_shaft_diagnostics(self):
        row = {'points': {'tip': self.uv[2].tolist()}}
        result = residuals(self.uv, row, 'points', 2.)
        self.assertEqual(set(result['point_error_canonical_px']), {'tip'})
        self.assertIsNone(result['shaft_error_deg'])


if __name__ == '__main__': unittest.main()
