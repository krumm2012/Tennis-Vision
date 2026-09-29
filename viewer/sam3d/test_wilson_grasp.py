import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fit_wilson_sequence import palm_frame
from fit_wilson_grip import grip_contact

class GraspTests(unittest.TestCase):
    def hand(self):
        p=np.zeros((70,3));p[28]=[.03,.08,0];p[32]=[.01,.08,0];p[36]=[-.01,.075,0];p[40]=[-.03,.07,0]
        return p
    def test_palm_frame_is_right_handed(self):
        _,h=palm_frame(self.hand());np.testing.assert_allclose(h.T@h,np.eye(3),atol=1e-12);self.assertAlmostEqual(np.linalg.det(h),1)
    def test_grip_follows_palm_under_rigid_motion(self):
        p=self.hand();w,h=palm_frame(p);q=Rotation.from_rotvec([.7,-.4,.2]).as_matrix();translation=np.array([1.,2.,3.]);ww,hh=palm_frame(p@q.T+translation);offset=np.array([.01,.04,.006]);np.testing.assert_allclose(ww+hh@offset,q@(w+h@offset)+translation,atol=1e-12)

class ContactTests(unittest.TestCase):
    def hand(self):
        p=np.zeros((70,3))
        for base,x,y in zip([28,32,36,40],[.03,.01,-.01,-.03],[.08,.08,.075,.07]):
            p[base]=[x,y,0];p[base-1]=[x,y+.025,.01];p[base-3]=[x,y+.025,.035]
        return p
    def test_contact_is_in_finger_corridor_and_shaft_exits_index_side(self):
        p=self.hand();w,h=palm_frame(p);off,axis=grip_contact(p);center=w+h@off;shaft=h@axis
        self.assertGreater(center[1],.075) # excludes the old wrist+4cm anchor
        self.assertGreater(center[2],.01) # off palm surface, toward curled fingers
        self.assertGreater(shaft[0],.9) # shaft crosses fingers, not wrist-to-middle-finger
        butt=center-shaft*.045
        self.assertLess(butt[0],p[40,0]) # butt projects beyond the little finger
        self.assertLess(np.linalg.norm(butt-p[40]),.04)
    def test_contact_is_rigid_motion_equivariant(self):
        p=self.hand();o,a=grip_contact(p);q=Rotation.from_rotvec([.9,.2,-.7]).as_matrix()
        oo,aa=grip_contact(p@q.T+np.array([2,4,3]))
        np.testing.assert_allclose(o,oo,atol=1e-12);np.testing.assert_allclose(a,aa,atol=1e-12)

if __name__=='__main__':unittest.main()
