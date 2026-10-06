import copy,json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
import numpy as np
from scipy.spatial.transform import Rotation
from racket_landmarks import validate,usable_grip
from racket_grip_calibration import estimate,save,manual_anchor
from audit_racket_evidence_boundaries import projections,NAMES
from run_records import sha256
from server import Handler,Library


class GripCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.meta={'video_sha256':'test-video','image_size':[1280,720],'fps':25,'frames':10,'focal':[1600.]*10,'mirror_available':False}
        ring=[[.13*np.cos(a),.51+.175*np.sin(a),0] for a in np.linspace(0,2*np.pi,32,endpoint=False)]
        self.model={'length_m':.685,'head_half_width_m':.13,'head_center_y_m':.51,'throat_y_m':.35,'grip_y_m':.045,'head_outline':ring,'dimensions_measured':False}
        self.r=Rotation.from_euler('xyz',[.3,.25,-.4]).as_matrix();self.t=np.array([.2,-.3,7.])
        pose={'status':'fitted','rotation_camera_columns':self.r.tolist(),'translation_camera_m':self.t.tolist(),'grip_target_camera_m':(self.r@np.array([0,.045,0])+self.t).tolist()}
        self.poses={**self.meta,'model':self.model,'frames':[{**copy.deepcopy(pose),'frame':i} for i in range(10)],'summary':{}}
        uv=projections(self.r,self.t,{**self.model,'grip_y_m':.095},self.meta,2)
        self.labels={**self.meta,'frames':[{'frame':2,'points':dict(zip(NAMES[:5],uv[:5].tolist()))|{'grip_center':uv[6].tolist()},'mirror_points':{},'grip_confirmed':{'points':True,'mirror_points':False}}]}

    def fixture(self,root):
        root.mkdir(parents=True,exist_ok=True)
        for name,value in [('mesh_meta.json',self.meta),('racket_poses.json',self.poses),('racket_landmarks.json',{**self.labels,'frames':[]})]:
            (root/name).write_text(json.dumps(value))
        (root/'mesh_refined.bin').write_bytes(b'unchanged-body')

    def test_known_axial_grip_recovery_is_image_estimate_not_measurement(self):
        clean=validate(self.labels,self.meta);report=estimate(clean,self.poses,self.meta)
        self.assertTrue(report['calibration_ready']);self.assertAlmostEqual(report['estimated_grip_from_butt_m'],.095,places=6)
        self.assertFalse(report['distance_measured']);self.assertEqual(report['frames'],[2])

    def test_unconfirmed_and_heldout_grips_cannot_calibrate(self):
        labels=copy.deepcopy(self.labels);labels['frames'][0]['grip_confirmed']['points']=False
        self.assertFalse(estimate(validate(labels,self.meta),self.poses,self.meta)['calibration_ready'])
        report=estimate(validate(self.labels,self.meta),self.poses,self.meta,heldout=[2])
        self.assertFalse(report['calibration_ready']);self.assertEqual(report['records'][0]['reason'],'heldout_excluded')

    def test_confirmation_requires_matching_visible_point_and_legacy_is_unchanged(self):
        labels=copy.deepcopy(self.labels);del labels['frames'][0]['points']['grip_center']
        with self.assertRaises(ValueError):validate(labels,self.meta)
        del labels['frames'][0]['grip_confirmed'];clean=validate(labels,self.meta)
        self.assertFalse(usable_grip(clean['frames'][0]));self.assertNotIn('grip_confirmed',clean['frames'][0])

    def test_off_axis_grip_is_rejected(self):
        labels=copy.deepcopy(self.labels);labels['frames'][0]['points']['grip_center'][0]+=40
        self.assertFalse(estimate(validate(labels,self.meta),self.poses,self.meta)['calibration_ready'])

    def test_anchor_uses_reviewed_ray_and_preserves_prior_depth(self):
        prior=np.array([.1,.2,7.]);row=self.labels['frames'][0];anchor=manual_anchor(row,prior,self.meta)
        uv=anchor[:2]/anchor[2]*1600+np.array([640,360])
        np.testing.assert_allclose(uv,row['points']['grip_center']);self.assertEqual(anchor[2],prior[2])

    def test_save_preserves_main_geometry_and_detects_concurrent_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);before={name:sha256(root/name) for name in ['mesh_meta.json','racket_poses.json','mesh_refined.bin']}
            oldsha=sha256(root/'racket_landmarks.json');result=save(root,self.labels,oldsha)
            self.assertTrue(result['calibration_ready']);self.assertEqual(before,{name:sha256(root/name) for name in before})
            candidate=json.loads((root/'racket_poses_grip_calibrated.json').read_text());self.assertFalse(candidate['accepted'])
            self.assertEqual(candidate['summary']['manual_grip_calibrated_frames'],[2])
            for i in [0,1,3,9]:self.assertEqual(candidate['frames'][i]['translation_camera_m'],self.poses['frames'][i]['translation_camera_m'])
            with self.assertRaises(ValueError):save(root,self.labels,oldsha)

    def test_invalid_video_does_not_write_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);before=sha256(root/'racket_landmarks.json');labels=copy.deepcopy(self.labels);labels['video_sha256']='wrong'
            with self.assertRaises(ValueError):save(root,labels)
            self.assertEqual(sha256(root/'racket_landmarks.json'),before)

    def test_prescribed_nine_cm_survives_image_recalibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            setting={k:self.meta[k] for k in ['video_sha256','image_size','fps']}
            setting.update(grip_from_butt_m=.09,source='user_specified')
            (root/'racket_grip_settings.json').write_text(json.dumps(setting))
            report=save(root,self.labels)
            self.assertAlmostEqual(report['estimated_grip_from_butt_m'],.095,places=6)
            self.assertEqual(report['applied_grip_from_butt_m'],.09)
            pose=json.loads((root/'racket_poses_grip_calibrated.json').read_text())
            self.assertEqual(pose['model']['grip_y_m'],.09)
            self.assertEqual(pose['model']['grip_position_source'],'user_specified')
            self.assertEqual(pose['model']['grip_calibration_frames'],[])
            for i,row in enumerate(pose['frames']):
                uv=projections(np.asarray(row['rotation_camera_columns']),np.asarray(row['translation_camera_m']),pose['model'],self.meta,i)
                np.testing.assert_allclose(row['projected_points']['grip_center'],uv[6])
            setting['video_sha256']='wrong'
            (root/'racket_grip_settings.json').write_text(json.dumps(setting))
            before=sha256(root/'racket_landmarks.json')
            with self.assertRaises(ValueError):save(root,self.labels)
            self.assertEqual(sha256(root/'racket_landmarks.json'),before)

    def test_prescribed_grip_works_without_image_distance_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            setting={k:self.meta[k] for k in ['video_sha256','image_size','fps']}
            setting.update(grip_from_butt_m=.09,source='user_specified')
            (root/'racket_grip_settings.json').write_text(json.dumps(setting))
            labels={**self.labels,'frames':[]};report=save(root,labels)
            self.assertTrue(report['calibration_ready']);self.assertFalse(report['image_calibration_ready'])
            self.assertEqual(report['applied_grip_from_butt_m'],.09)

    def test_real_http_endpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ident='1'*32;self.fixture(root/ident/'result')
            (root/ident/'record.json').write_text(json.dumps({'id':ident,'status':'ready','created':0}))
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.library=Library(root,[]);server.origins=set()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                value={**self.labels,'landmarks_sha256':sha256(root/ident/'result/racket_landmarks.json')}
                request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/videos/{ident}/racket-grip-calibration',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:result=json.load(response)
                self.assertTrue(result['saved']);self.assertTrue(result['calibration_ready'])
                self.assertEqual(result['landmarks_sha256'],sha256(root/ident/'result/racket_landmarks.json'))
            finally:server.shutdown();server.server_close();server.library.pool.shutdown();thread.join()


if __name__=='__main__':unittest.main()
