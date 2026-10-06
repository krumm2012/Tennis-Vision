import unittest
import numpy as np
from audit_racket_physical_definitions import line_grip_probe, project, camera_projection, camera_sensitivity, epipolar_errors


class PhysicalDefinitionAuditTests(unittest.TestCase):
    def test_grip_distance_recovered_on_known_camera_ray(self):
        anchor = np.array([.4,-.3,8.]); axis = np.array([-.8,.1,.3]); axis /= np.linalg.norm(axis)
        butt = anchor-.095*axis; principal = np.array([1280.,720.])
        result = line_grip_probe(anchor,axis,project(butt,2900.,principal),2900.,principal)
        self.assertAlmostEqual(result['implied_grip_from_butt_m'],.095,places=10)
        self.assertLess(result['ray_shaft_gap_mm'],1e-8)

    def test_plane_distance_cannot_change_epipolar_discrepancy(self):
        q = np.array([.08,-.2,11.,0.,0.,0.]); n = np.array([np.sin(q[0])*np.cos(q[1]),np.sin(q[1]),np.cos(q[0])*np.cos(q[1])])
        x = np.array([[.3,.2,8.]])
        a = project(x,2900.,np.array([1280.,720.]))[0]
        for distance in [10.,11.,12.]:
            q[2] = distance; b,_ = camera_projection(x,np.array([2900.]),q,np.array([1280.,720.]))
            np.testing.assert_allclose(epipolar_errors({'real':a,'mirror':b[0]},2900.,np.array([1280.,720.]),n),0,atol=1e-9)

    def test_heldout_and_manual_pixels_do_not_enter_camera_fit(self):
        rng = np.random.default_rng(4); frames = np.repeat(np.arange(10),10)
        points = rng.uniform([-.5,-.4,7.],[.5,.4,9.],(100,3)); focals = np.full(100,2900.)
        geometry = {'normal_camera':[0.,-.3,np.sqrt(.91)],'distance_camera_m':11.}
        q = np.array([0.,np.arcsin(-.3),11.,0.,0.,0.]); principal = np.array([1280.,720.])
        observed,_ = camera_projection(points,focals,q,principal)
        args = [points,observed,frames,np.ones(100,bool),focals,[2560,1440],geometry,{},np.full(10,2900.)]
        before = camera_sensitivity(*args)
        edited = observed.copy(); edited[frames%5 == 0] += 200
        args[1] = edited; args[7] = {'manual':[{'frame':0,'part':'tip','real':[1400,500],'mirror':[900,20]}]}
        after = camera_sensitivity(*args)
        self.assertFalse(set(before['train_frames']) & set(before['heldout_frames']))
        for name in before['candidates']:
            for key in ['normal_camera','distance_camera_m','focal_scale','principal_offset_original_px']:
                np.testing.assert_allclose(before['candidates'][name][key],after['candidates'][name][key],atol=1e-12)
            self.assertGreater(after['candidates'][name]['body_core_heldout_original_px']['median'],200)


if __name__ == '__main__':
    unittest.main()
