"""Integration regression: ordinary (non-mirror) videos must complete grip fitting."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from fit_dataset_grip import fit_dataset,grip_contact,palm_frame
from fit_racket_pose import DEFAULT_MODEL,project
from fit_racket_video import ring_points

class GripWithoutMirrorTests(unittest.TestCase):
    def test_real_silhouette_fit_without_mirror_or_roi(self):
        with tempfile.TemporaryDirectory() as space:
            folder=Path(space)/'video';out=folder/'result';work=folder/'attempts/0001/work';out.mkdir(parents=True);work.mkdir(parents=True)
            hand=np.zeros((70,3))
            for base,x,y in zip([28,32,36,40],[.03,.01,-.01,-.03],[.08,.08,.075,.07]):
                hand[base]=[x,y,0];hand[base-1]=[x,y+.025,.01];hand[base-3]=[x,y+.025,.035]
            roots=np.tile([0.,0.,4.],(3,1));np.savez(work/'reconstruction.npz',joints=np.tile(hand,(3,1,1)))
            (folder/'record.json').write_text(json.dumps({'attempt':1}))
            model=dict(DEFAULT_MODEL,grip_y_m=.045);model['head_outline']=ring_points(model).tolist();R=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
            w,H=palm_frame(hand);offset,_=grip_contact(hand);target=w+H@offset+roots[0];t=target-R@np.array([0,.045,0]);polygon=project(ring_points(model)@R.T+t,1200,[1280,720]).tolist()
            rows=[{'frame':i,'status':'fitted','quality':'silhouette_fitted','observed_polygon':polygon,'detection_confidence':.9,'mask_fit_rms_px':1.,'rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist()} for i in range(3)]
            (out/'mesh_meta.json').write_text(json.dumps({'frames':3,'fps':25,'image_size':[1280,720],'video_sha256':'same','source_roots':roots.tolist(),'focal':[1200]*3,'mirror_available':False}))
            (out/'racket_poses.json').write_text(json.dumps({'video_sha256':'same','image_size':[1280,720],'model':model,'frames':rows,'summary':{'hand':'right'}}))
            with contextlib.redirect_stdout(io.StringIO()):fit_dataset(folder)
            fitted=json.loads((out/'racket_poses.json').read_text());self.assertEqual(fitted['summary']['hidden'],0);self.assertEqual(fitted['summary']['mirror_observations'],0)
            self.assertTrue(all(r['palm_anchor_error_m']<1e-8 for r in fitted['frames']))

    def test_confirmed_grip_participates_in_sequence_calibration(self):
        from audit_racket_evidence_boundaries import projections,NAMES
        with tempfile.TemporaryDirectory() as space:
            folder=Path(space);out=folder/'result';work=folder/'attempts/0001/work';out.mkdir();work.mkdir(parents=True)
            hand=np.zeros((70,3))
            for base,x,y in zip([28,32,36,40],[.03,.01,-.01,-.03],[.08,.08,.075,.07]):
                hand[base]=[x,y,0];hand[base-1]=[x,y+.025,.01];hand[base-3]=[x,y+.025,.035]
            roots=np.tile([0.,0.,4.],(3,1));np.savez(work/'reconstruction.npz',joints=np.tile(hand,(3,1,1)))
            (folder/'record.json').write_text(json.dumps({'attempt':1}))
            model=dict(DEFAULT_MODEL,grip_y_m=.045,head_outline=ring_points(DEFAULT_MODEL).tolist())
            R=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
            w,H=palm_frame(hand);offset,_=grip_contact(hand);target=w+H@offset+roots[0];t=target-R@np.array([0,.095,0])
            meta={'frames':3,'fps':25,'image_size':[1280,720],'video_sha256':'same','source_roots':roots.tolist(),'focal':[1200]*3,'mirror_available':False}
            polygon=project(ring_points(model)@R.T+t,1200,[1280,720]).tolist()
            rows=[{'frame':i,'status':'fitted','quality':'silhouette_fitted','observed_polygon':polygon,'detection_confidence':.9,'mask_fit_rms_px':1.,'rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist()} for i in range(3)]
            (out/'mesh_meta.json').write_text(json.dumps(meta));(out/'racket_poses.json').write_text(json.dumps({'video_sha256':'same','image_size':[1280,720],'model':model,'frames':rows,'summary':{'hand':'right'}}))
            uv=projections(R,t,{**model,'grip_y_m':.095},meta,0)
            marks={**meta,'frames':[{'frame':i,'points':dict(zip(NAMES[:5],uv[:5].tolist()))|{'grip_center':uv[6].tolist()},'mirror_points':{},'grip_confirmed':{'points':True}} for i in range(3)]}
            (out/'racket_landmarks.json').write_text(json.dumps(marks))
            with contextlib.redirect_stdout(io.StringIO()):fit_dataset(folder)
            fitted=json.loads((out/'racket_poses.json').read_text());self.assertAlmostEqual(fitted['model']['grip_y_m'],.095,places=5)
            self.assertEqual(fitted['grip_calibration']['frames'],[0,1,2])
            for row in fitted['frames']:
                self.assertIn('grip_center',row['observed_keypoints']);self.assertLess(row['reviewed_landmark_rms_px'],.01)
            # A prescribed distance must survive a complete sequence fit even
            # when the image-only calibration favors a different distance.
            setting={k:meta[k] for k in ['video_sha256','image_size','fps']}
            setting.update(grip_from_butt_m=.09,source='user_specified')
            (out/'racket_grip_settings.json').write_text(json.dumps(setting))
            with contextlib.redirect_stdout(io.StringIO()):fit_dataset(folder)
            prescribed=json.loads((out/'racket_poses.json').read_text())
            self.assertEqual(prescribed['model']['grip_y_m'],.09)
            self.assertEqual(prescribed['model']['grip_position_source'],'user_specified')
            self.assertEqual(prescribed['model']['grip_calibration_frames'],[])
            for row in prescribed['frames']:
                anchor=np.asarray(row['rotation_camera_columns'])@np.array([0,.09,0])+row['translation_camera_m']
                np.testing.assert_allclose(anchor,row['grip_target_camera_m'],atol=1e-8)

if __name__=='__main__':unittest.main()
