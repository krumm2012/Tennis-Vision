import unittest
import numpy as np
from audit_racket_roles import ray_pair,project

class RayConsistencyTests(unittest.TestCase):
    def test_exact_reflected_point(self):
        n=np.array([.04,-.35,.935]);n/=np.linalg.norm(n);p=np.array([.8,.2,7.]);d=10.
        mirrored=p-2*(p@n-d)*n
        q=ray_pair(project(p,1500,[2560,1440]),project(mirrored,1500,[2560,1440]),1500,[2560,1440],n,d)
        self.assertLess(q['ray_gap_m'],1e-10)
        self.assertLess(q['real_epipolar_error_px'],1e-9)
        self.assertTrue(q['positive_depths'])
    def test_wrong_mirror_correspondence_is_detectable(self):
        p=np.array([.8,.2,7.]);n=np.array([0,0,1.]);m=p-2*(p@n-10)*n
        q=ray_pair(project(p,1500,[2560,1440]),project(m,1500,[2560,1440])+[0,100],1500,[2560,1440],n,10)
        self.assertGreater(q['ray_gap_m'],.01)
        self.assertGreater(q['real_epipolar_error_px'],10)
    def test_same_vertical_line_epipolar_geometry(self):
        n=np.array([0,0,1.]);p=np.array([.8,.2,7.]);m=p-2*(p@n-10)*n
        r=project(p,1500,[2560,1440]);v=project(m,1500,[2560,1440]);before_r=r.copy();before_v=v.copy()
        ray_pair(r,v,1500,[2560,1440],n,10)
        np.testing.assert_array_equal(r,before_r);np.testing.assert_array_equal(v,before_v)
