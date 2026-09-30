import unittest,json,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mirror_calibration import fit,corners,apply

class MirrorCalibrationTests(unittest.TestCase):
 def sample(self):
  size=[960,720];f=850.;world=np.array([[0,0,0],[3.3,0,0],[3.3,0,4.8],[0,0,4.8]])
  r=Rotation.from_euler('x',55,degrees=True).as_matrix();x=world@r.T+[-1.65,1,8];normal=np.array([0,0,1.]);distance=15.;y=x-2*(x@normal-distance)[:,None]*normal
  uv=lambda p:(p[:,:2]/p[:,2:]*f+np.array(size)/2).tolist()
  meta={'image_size':size,'frames':10,'focal':[f]*10,'video_sha256':'sample'}
  bundle={'image_size':size,'video_sha256':'sample','ground':{'points':uv(x),'width':3.3,'length':4.8,'frame':0},'mirror':{'points':uv(y),'frame':0}}
  return bundle,meta
 def test_paired_points_recover_reflection_at_non_default_size(self):
  b,m=self.sample();out=fit(b,m);self.assertLess(out['rms_px'],.1);self.assertAlmostEqual(abs(out['normal_camera'][2]),1,places=3);self.assertAlmostEqual(abs(out['distance_camera_m']),15,places=2)
 def test_wrong_video_and_crossed_corners_rejected(self):
  b,m=self.sample();b['video_sha256']='other'
  with self.assertRaises(ValueError):fit(b,m)
  with self.assertRaises(ValueError):corners({'points':[[0,0],[100,100],[100,0],[0,100]]},[960,720])
 def test_apply_adds_only_mirror_green_channel_and_preserves_real_mask(self):
  import cv2
  bundle,meta=self.sample();roots=np.tile([0,0,8.],(10,1));vertices=np.tile([[-.3,-.5,0],[.3,-.5,0],[.3,.5,0],[-.3,.5,0]],(10,1,1));meta.update(source_roots=roots.tolist(),mask_atlas_grid=[5,2])
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);out=folder/'result';out.mkdir();work=folder/'attempts/0001/work';work.mkdir(parents=True);np.savez(work/'reconstruction.npz',vertices=vertices)
   (folder/'record.json').write_text(json.dumps({'attempt':1}));(out/'mesh_meta.json').write_text(json.dumps(meta))
   atlas=np.zeros((480,1600,3),np.uint8);atlas[:,:,2]=255;cv2.imwrite(str(out/'person_masks_sam2.png'),atlas);(out/'person_masks_sam2_stats.json').write_text(json.dumps([{'real':1.,'mirror':0.}]*10))
   p=(vertices[0,:2,:2]+[0,0])/22*850+[480,360];box=[p[0,0],p[0,1],p[1,0],p[1,1]+850/22];poly=[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]
   evidence={'video_sha256':'sample','frames':[{'persons':[{'box':box,'polygon':poly}]}]*10};(out/'person_candidates.json').write_text(json.dumps(evidence))
   result=apply(folder,bundle);self.assertEqual(result['mask_frames'],10);mask=cv2.imread(str(out/'person_masks_sam2.png'));self.assertTrue(np.all(mask[:,:,2]==255));self.assertGreater(np.count_nonzero(mask[:,:,1]),0)

if __name__=='__main__':unittest.main()
