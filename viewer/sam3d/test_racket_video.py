import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from fit_racket_video import DEFAULT_MODEL, ring_points, project, fit_shape, head_ellipse, continuous_plane


class VideoRacketGeometryTests(unittest.TestCase):
    def test_known_silhouette_and_wrist_are_recovered(self):
        wrist=np.array([.5,-.6,8.])
        matrix=Rotation.from_rotvec([.4,.2,1.]).as_matrix()
        translation=wrist-matrix@np.array([0,DEFAULT_MODEL['grip_y_m'],0])
        uv=project(ring_points(DEFAULT_MODEL)@matrix.T+translation,1468.)
        center,axes,angle=cv2.fitEllipse(uv.astype(np.float32))
        ellipse=(np.array(center),np.array(axes)/2,np.radians(angle))
        result=fit_shape(ellipse,wrist,project(wrist[None,:],1468.)[0],1468.,DEFAULT_MODEL)
        self.assertLess(result[3],.2)
        self.assertLess(result[4],.01)

    def test_symmetric_plane_does_not_flip_during_interpolation(self):
        previous=Rotation.from_rotvec([.2,.5,.1]).as_matrix()
        flipped=previous@np.diag([-1.,1.,-1.])
        np.testing.assert_allclose(continuous_plane(flipped,previous),previous,atol=1e-10)

    def test_missing_contour_has_no_observation(self):
        self.assertIsNone(head_ellipse([]))


if __name__=='__main__':
    unittest.main()
