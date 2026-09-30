import unittest
import numpy as np
from stabilize import smooth_display,stabilize_body

class StabilizationTest(unittest.TestCase):
    def test_constant_velocity_and_boundaries(self):
        t=np.array([0,.03,.08,.12,.16]);p=np.zeros((5,3,3));p[:,:,0]=t[:,None]*2
        np.testing.assert_allclose(smooth_display(p,t),p,atol=1e-7)
    def test_noise_reduced_with_bounded_correction(self):
        t=np.arange(40)/25;p=np.zeros((40,3,3));p[:,:,1]=(.004*(-1.)**np.arange(40))[:,None]
        q=smooth_display(p,t)
        self.assertLess(np.linalg.norm(np.diff(q,n=2,axis=0)),.5*np.linalg.norm(np.diff(p,n=2,axis=0)))
        self.assertLessEqual(np.linalg.norm(q-p,axis=-1).max(),.015001)
        np.testing.assert_array_equal(q[[0,-1]],p[[0,-1]].astype('f4'))
    def test_fast_motion_and_hand_preserved(self):
        t=np.arange(9)/25;j=np.zeros((9,70,3));j[:,41,0]=np.array([0,.02,.1,.3,.7,1.1,1.3,1.4,1.42]);j[:,62]=j[:,41]
        v=j[:,[41,62]].copy();q,h=stabilize_body(v,j,t)
        np.testing.assert_allclose(q,v,atol=1e-7);np.testing.assert_allclose(h[:,41],j[:,41],atol=1e-7)
    def test_gap_and_invalid_time(self):
        p=np.zeros((3,2,3));p[1]=1
        np.testing.assert_array_equal(smooth_display(p,[0,.04,1]),p)
        with self.assertRaises(ValueError):smooth_display(p,[0,0,1])
if __name__=='__main__':unittest.main()
