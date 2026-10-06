import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from racket_stability import stable_rotations,supported_frames,motion_metrics

class RacketStabilityTests(unittest.TestCase):
    def test_reviewed_rotation_stays_exact_while_neighbors_can_smooth(self):
        matrices=Rotation.from_euler('y',np.linspace(0,80,15)[:,None],degrees=True).as_matrix()
        matrices[7]=matrices[7]@Rotation.from_euler('x',45,degrees=True).as_matrix()
        result=stable_rotations(matrices,np.ones(15),25,fixed_frames=[7])
        np.testing.assert_array_equal(result[7],matrices[7])
        self.assertGreater(np.linalg.norm(result[6]-matrices[6]),.01)
        np.testing.assert_allclose(result.transpose(0,2,1)@result,np.tile(np.eye(3),(15,1,1)),atol=1e-7)

    def test_face_outlier_is_reduced_without_freezing_swing(self):
        # 100 degrees over one second; corrupted central face estimate +65deg.
        truth=Rotation.from_euler('y',np.linspace(0,100,26)[:,None],degrees=True).as_matrix()
        noisy=truth.copy();noisy[12]=truth[12]@Rotation.from_euler('x',65,degrees=True).as_matrix()
        smooth=stable_rotations(noisy,np.ones(26),25)
        self.assertLess(motion_metrics(smooth)['acceleration_max_deg'],12)
        self.assertLess(np.rad2deg(Rotation.from_matrix(truth[12].T@smooth[12]).magnitude()),20)
        self.assertGreater(np.rad2deg(Rotation.from_matrix(smooth[0].T@smooth[-1]).magnitude()),85)
        np.testing.assert_allclose(np.einsum('nji,njk->nik',smooth,smooth),np.tile(np.eye(3),(26,1,1)),atol=1e-7)
    def test_short_bracketed_gaps_completed_long_or_unbounded_hidden(self):
        obs=np.zeros(65,bool);obs[[3,10,40,55]]=True
        support=supported_frames(obs,25,.6)
        self.assertTrue(support[3:11].all());self.assertTrue(support[40:56].all())
        self.assertFalse(support[0:3].any());self.assertFalse(support[11:40].any());self.assertFalse(support[56:].any())

if __name__=='__main__':unittest.main()
