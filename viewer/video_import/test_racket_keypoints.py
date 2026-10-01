import unittest
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from racket_keypoints import extract,triangulate,project
from racket_direction import calibrate_axes,fixed_grip_initial
from racket_landmarks import validate

class RacketDirectionTests(unittest.TestCase):
 def polygon(self,shaft=False):
  mask=np.zeros((120,170),np.uint8);cv2.ellipse(mask,(110,60),(28,22),0,0,360,255,-1)
  if shaft:cv2.rectangle(mask,(12,56),(92,64),255,-1)
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  return contours[0].reshape(-1,2)
 def test_head_only_mask_does_not_invent_occluded_butt_or_throat(self):
  found=extract(self.polygon(),[20,60],.9)
  self.assertNotIn('handle_end',found['points']);self.assertNotIn('throat',found['points']);self.assertFalse(found['face_side_identified']);self.assertGreater(found['shaft_direction_image'][0],.99)
  full=extract(self.polygon(True),[20,60],.9);self.assertIn('handle_end',full['points']);self.assertLess(full['points']['handle_end'][0],20)
 def test_mirror_rays_recover_camera_point_and_reject_parallel_views(self):
  n=np.array([.04,-.1,.994]);n/=np.linalg.norm(n);point=np.array([1.2,.4,5.]);virtual=point-2*(point@n-9)*n
  uv=project(np.array([point,virtual]),1200,[1280,720]);result=triangulate(*uv,1200,[1280,720],n,9)
  self.assertIsNotNone(result);np.testing.assert_allclose(result[0],point,atol=1e-10)
  self.assertIsNone(triangulate([640,360],[640,360],1200,[1280,720],[0,0,1],9))
 def test_hand_local_calibration_uses_heldout_observations_and_preserves_missing_evidence(self):
  count=30;hands=Rotation.from_euler('z',np.linspace(-30,30,count)[:,None],degrees=True).as_matrix();a=np.array([1.,0,0]);b=Rotation.from_euler('z',25,degrees=True).apply(a)
  world=hands@a;seen=hands@b;targets=np.tile([0,0,5.],(count,1));head=project(targets+seen*.46,1200,[1280,720]);focals=np.ones(count)*1200
  corrected,report=calibrate_axes(world,hands,list(seen),list(head),targets,focals,[1280,720]);self.assertTrue(report['accepted']);self.assertLess(report['stereo_heldout_after_deg'],1);self.assertLess(report['image_heldout_after_deg'],1);self.assertFalse(report['physical_grip_bevel_verified'])
  np.testing.assert_allclose(corrected,seen,atol=.002)
  kept,rejected=calibrate_axes(world,hands,[seen[i] if i<4 else None for i in range(count)],list(head),targets,focals,[1280,720]);self.assertFalse(rejected['accepted']);np.testing.assert_array_equal(kept,world)
 def test_signed_face_requires_video_identity_and_visible_marked_sides(self):
  meta={'video_sha256':'same','image_size':[1280,720],'fps':25,'frames':30};value={**meta,'frames':[{'frame':0,'points':{'handle_end':[100,200],'tip':[200,200]},'face_correspondence_confirmed':True}]}
  with self.assertRaises(ValueError):validate(value,meta)
  value['frames'][0].update(points={'handle_end':[100,200],'tip':[200,200],'rim_side':[180,180],'rim_opposite':[180,220]},side_feature='A侧蓝色贴纸')
  self.assertTrue(validate(value,meta)['frames'][0]['face_correspondence_confirmed'])
  value['video_sha256']='other'
  with self.assertRaises(ValueError):validate(value,meta)
 def test_pose_initialization_uses_the_same_fixed_grip_and_stereo_shaft_as_solver(self):
  from fit_racket_pose import DEFAULT_MODEL
  from fit_racket_video import ring_points
  model={**DEFAULT_MODEL,'grip_y_m':.045};model['head_outline']=ring_points(model).tolist()
  R=Rotation.from_euler('xyz',[25,15,70],degrees=True).as_matrix();grip=np.array([.1,.2,5.]);t=grip-R@np.array([0,.045,0]);ring=project(np.array(model['head_outline'])@R.T+t,1200,[1280,720])
  center,radii,angle=cv2.fitEllipse(ring.astype(np.float32));ellipse=(np.array(center),np.array(radii)/2,np.radians(angle))
  point=project((R@np.array([0,model['head_center_y_m'],0])+t)[None],1200,[1280,720])[0]
  fitted=fixed_grip_initial(ellipse,grip,1200,[1280,720],model,{'head_center':point},{'head_center':1.},R[:,1],4.,R[:,1],None)
  self.assertLess(np.linalg.norm(fitted[:,1]-R[:,1]),.03)
  fitted_center=fitted@np.array([0,model['head_center_y_m'],0])+grip-fitted@np.array([0,.045,0]);self.assertLess(np.linalg.norm(project(fitted_center[None],1200,[1280,720])[0]-point),.5)

if __name__=='__main__':unittest.main()
