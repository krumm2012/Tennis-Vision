import unittest
import numpy as np
from multiview_constraints import reflect,constrain

class MultiviewTests(unittest.TestCase):
 def test_reflection_is_involution_and_preserves_distances(self):
  rng=np.random.default_rng(6);p=rng.normal(size=(3,70,3));n=np.array([.1,-.3,.95]);n/=np.linalg.norm(n)
  reflected=reflect(p,n,10.)
  np.testing.assert_allclose(reflect(reflected,n,10.),p,atol=1e-13)
  np.testing.assert_allclose(np.linalg.norm(reflected[:,21]-reflected[:,41],axis=-1),np.linalg.norm(p[:,21]-p[:,41],axis=-1))
 def test_depth_bias_is_removed_from_gate_and_never_moves_primary_wrist(self):
  rng=np.random.default_rng(3);primary=rng.normal(scale=.04,size=(30,70,3));primary[:,:,2]+=8
  alternative=primary.copy();alternative[:,:]+=np.array([.1,-.1,.4]);alternative[:,21:41,0]+=.04
  alternative[15,21:41,0]+=.5
  virtual=reflect(alternative,[0,0,1],10.)
  valid=np.ones(30,bool);valid[3]=False
  fused,accepted,body,hand,_=constrain(primary,virtual,valid,[0,0,1],10.,25)
  self.assertEqual(accepted.sum(),28);self.assertFalse(accepted[15]);self.assertLess(body[0],1e-10)
  np.testing.assert_array_equal(fused[:,41],primary[:,41]);np.testing.assert_array_equal(fused[:,:21],primary[:,:21]);np.testing.assert_array_equal(fused[~accepted],primary[~accepted])
  self.assertLessEqual(np.linalg.norm(fused-primary,axis=-1).max(),.00800001)
  self.assertGreater(np.linalg.norm(fused-primary,axis=-1).max(),.007)

if __name__=='__main__':unittest.main()
