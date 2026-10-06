import unittest
import numpy as np
from racket_manual_evidence import view_observations,pixel_terms


class ManualEvidenceTests(unittest.TestCase):
    def test_reviewed_frame_removes_unreviewed_center_and_occluded_mirror(self):
        auto={'points':{'head_center':[200,100],'tip':[220,100]},'confidence':.9}
        review={'points':{'tip':[210,102]},'mirror_points':{}}
        points,weights=view_observations(auto,review,'real',2)
        self.assertEqual(set(points),{'tip'});np.testing.assert_array_equal(points['tip'],[105,51]);self.assertEqual(weights,{'tip':3.})
        self.assertEqual(view_observations(auto,review,'mirror',2),({},{}))

    def test_unconfirmed_rim_order_invariant_but_physical_marking_preserved(self):
        names=['rim_side','rim_opposite'];pred=np.array([[10,20],[30,20]])
        observed={'rim_side':np.array([30,20]),'rim_opposite':np.array([10,20])};weights={k:3 for k in names}
        np.testing.assert_array_equal(pixel_terms(pred,observed,weights,names),0)
        self.assertGreater(np.linalg.norm(pixel_terms(pred,observed,weights,names,rim_confirmed=True)),1)
