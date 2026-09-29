import unittest
import numpy as np
import torch
from solver import rotation6d,shaft_distance,solve
from prepare import continuity_edges

class GeometryTests(unittest.TestCase):
    def test_rotations_cannot_scale_or_reflect_racket(self):
        r=rotation6d(torch.tensor([[1.,2,3,4,1,2],[0.,1,0,1,0,0]]))
        torch.testing.assert_close(r.transpose(1,2)@r,torch.eye(3).expand(2,3,3),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(torch.linalg.det(r),torch.ones(2))
    def test_capsule_distance_is_rigid_motion_invariant(self):
        p=torch.tensor([[[.02,.03,0.],[0.,.05,.02]]]);r=torch.eye(3)[None];g=torch.zeros(1,3)
        radial,axial=shaft_distance(p,r,g);torch.testing.assert_close(radial,torch.full((1,2),.02))
        q=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]]);t=torch.tensor([1.,-3,5]);rr,aa=shaft_distance(p@q.T+t,q[None],t[None]);torch.testing.assert_close(rr,radial);torch.testing.assert_close(aa,axial)
    def test_long_missing_interval_breaks_grasp_continuity(self):
        t=np.arange(12)*.04;q=np.zeros(12);q[[0,11]]=1
        self.assertFalse(continuity_edges(t,q,np.ones(12)).any())
        q[5]=1;self.assertTrue(continuity_edges(t,q,np.ones(12)).all())
    def test_nonmonotonic_timestamps_rejected(self):
        with self.assertRaises(ValueError):continuity_edges([0,.1,.1],[1,1,1],[1,1,1])
    def test_absent_evidence_never_creates_grasp_edges(self):
        self.assertFalse(continuity_edges([0,.04,.08],[0,0,0],[0,0,0]).any())
    def test_solver_preserves_wrist_and_joint_motion_bounds(self):
        n=3;hand=np.zeros((n,21,3));hand[:,:,2]=4
        for t,x in zip([0,4,8,12,16],[.04,.03,.01,-.01,-.03]):
            for k in range(4):hand[:,t+k,:2]=[x,.10-k*.012]
        ring=np.array([[.1,.4,0],[0,.5,0],[-.1,.4,0],[0,.3,0]])
        a=dict(hand=hand,rotation=np.repeat(np.eye(3)[None],n,0),grip=np.tile([0,.05,4.02],(n,1)),ring=ring,model_grip=np.array([0,.045,0]),hand_weight=np.ones(n),observation_weight=np.zeros(n),edge_weight=np.ones(n-1),time=np.arange(n)*.04,ellipse_center=np.tile([320,240],(n,1)),ellipse_radii=np.ones((n,2))*20,ellipse_axes=np.repeat(np.eye(2)[None],n,0),focal=np.ones(n)*500,principal=np.array([320,240]),hand_image=hand[:,:,:2]/hand[:,:,2:]*500+[320,240],grasp_weight=np.ones(n))
        z,_=solve(a,steps=8);self.assertTrue(np.isfinite(z['hand']).all());np.testing.assert_allclose(z['hand'][:,20],hand[:,20],atol=1e-7)
        self.assertLessEqual(np.max(abs(z['hand']-hand)),.025001)
        self.assertLessEqual(np.max(abs(z['hand'][:,[3,7,11,15,19]]-hand[:,[3,7,11,15,19]])),.010001)

if __name__=='__main__':unittest.main()
