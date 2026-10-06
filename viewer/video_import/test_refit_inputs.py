import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from racket_calibration import validate,measured_model,resize_mesh
from racket_keypoints import role_compatible
from racket_review_frames import choose
from mhr_parameters import capture,stack,available
from refit_readiness import assess


class RefitInputsTests(unittest.TestCase):
    meta={'video_sha256':'own','image_size':[1280,720],'fps':25,'frames':100}
    model={'length_m':.685,'head_half_width_m':.13,'head_center_y_m':.51,
           'throat_y_m':.35,'grip_y_m':.045,'head_outline':[[-.13,.34,0],[.13,.34,0],[0,.685,0]],
           'asset_file':'wilson_mesh.bin','dimensions_measured':False}

    def value(self):
        return {**self.meta,'dimensions_cm':{'length':70,'head_width':28,'head_height':35},'measured':True}

    def test_dimensions_are_video_scoped_and_assumptions_never_become_measurements(self):
        value=self.value();value['measured']=False;clean=validate(value,self.meta)
        self.assertFalse(clean['size_ready']);self.assertFalse(clean['physical_grip_bevel_verified'])
        self.assertEqual(measured_model(self.model,clean),self.model)
        value['video_sha256']='other'
        with self.assertRaises(ValueError):validate(value,self.meta)
        value=self.value();value['dimensions_cm']['head_height']=50
        with self.assertRaises(ValueError):validate(value,self.meta)

    def test_measured_racket_geometry_and_asset_scale_together_without_changing_colors(self):
        calibration=validate(self.value(),self.meta);model=measured_model(self.model,calibration)
        self.assertTrue(model['dimensions_measured']);self.assertFalse(model['grip_position_measured'])
        self.assertAlmostEqual(model['head_center_y_m'],.525)
        values=np.array([[0,0,0,.1,.2,.3],[.13,.34,0,.4,.5,.6],[0,.685,0,.7,.8,.9]],dtype=np.float32)
        scaled=resize_mesh(values,self.model,model)
        np.testing.assert_array_equal(scaled[:,3:],values[:,3:]);self.assertAlmostEqual(float(values[2,1]),.685,places=6)
        self.assertAlmostEqual(float(scaled[1,0]),.14,places=6);self.assertAlmostEqual(float(scaled[2,1]),.70,places=6)
        self.assertEqual(model['asset_file'],'wilson_mesh_directional.bin')

    def test_wrong_role_contour_is_rejected_and_clear_selection_spans_the_clip(self):
        self.assertFalse(role_compatible(np.array([101,20]),np.array([100,400]),np.array([100,0]),1))
        self.assertTrue(role_compatible(np.array([110,410]),np.array([100,400]),np.array([100,0]),1))
        rows=[{'frame':i,'score':10 if i<20 else 1} for i in range(100)]
        selected=choose(rows,25,10)
        self.assertEqual(len(selected),10);self.assertGreater(selected[-1]['frame'],80)

    def test_native_parameter_capture_preserves_layout_and_absent_mirror_frames(self):
        person={'mhr_model_params':np.arange(204),'shape_params':np.arange(45),'expr_params':np.zeros(72)}
        row=capture(person);arrays=stack([row,None,row],'mirror_')
        self.assertEqual(arrays['mirror_mhr_model_params'].shape,(3,204));self.assertFalse(arrays['mirror_mhr_model_params'][1].any())
        self.assertTrue(available(stack([row,row]),2))
        person['mhr_model_params']=np.zeros(200)
        with self.assertRaises(ValueError):capture(person)

    def test_preflight_blocks_missing_native_parameters_without_modifying_published_mesh(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'mesh_meta.json').write_text(json.dumps(self.meta));(out/'mesh_local.bin').write_bytes(b'original')
            value=self.value();value['measured']=False;(out/'racket_dimensions.json').write_text(json.dumps(value))
            kp={'video_sha256':'own','frames':[{'frame':i,'views':{'real':{}}} for i in range(0,100,10)]}
            (out/'racket_review_frames.json').write_text(json.dumps(kp));archive=out/'body.npz';np.savez(archive,joints=np.zeros((100,70,3)))
            result=assess(out,archive,True,True)
            self.assertEqual(result['blocked_by'],['native_mhr_parameters_missing_rerun_required'])
            self.assertEqual(result['validation_source'],'automatic_contours_unverified');self.assertFalse(result['size_measured'])
            self.assertEqual((out/'mesh_local.bin').read_bytes(),b'original')

    def test_native_replay_converts_units_camera_axes_and_retains_gradients(self):
        import torch
        from refit_fullbody import replay,rotation_vector_matrix
        class Head:
            keypoint_mapping=torch.zeros(70,131)
            def mhr(self,shape,params,expression):
                return params[:,:12].reshape(-1,4,3),torch.zeros(len(params),127,8)
        head=Head();head.keypoint_mapping[0,0]=1
        params=torch.nn.Parameter(torch.arange(204,dtype=torch.float32)[None]);v,j=replay(head,torch.zeros(1,45),params,torch.zeros(1,72))
        np.testing.assert_allclose(v.detach().numpy()[0,0],[0,-.01,-.02]);np.testing.assert_allclose(j.detach().numpy()[0,0],[0,-.01,-.02])
        v.sum().backward();self.assertTrue(torch.isfinite(params.grad).all());self.assertNotEqual(float(params.grad[0,1]),0)
        rotation=torch.nn.Parameter(torch.zeros(2,3));r=rotation_vector_matrix(rotation);r.sum().backward();self.assertTrue(torch.isfinite(rotation.grad).all())
        torch.testing.assert_close(r,torch.eye(3)[None].repeat(2,1,1))

    def test_manual_grip_distance_cannot_use_joint_fit_heldout_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'mesh_meta.json').write_text(json.dumps(self.meta))
            points={name:[100,100] for name in ['handle_end','tip','rim_side','rim_opposite']}
            marks={**self.meta,'frames':[{'frame':i,'points':points,'mirror_points':{}} for i in range(0,100,10)]}
            (out/'racket_landmarks.json').write_text(json.dumps(marks))
            (out/'racket_dimensions.json').write_text(json.dumps(self.value()))
            archive=out/'body.npz';np.savez(archive,joints=np.zeros((100,70,3)))
            def check(frames):
                (out/'racket_poses.json').write_text(json.dumps({'model':{'grip_position_source':'manual_image_estimate','grip_calibration_frames':frames}}))
                return assess(out,archive)['blocked_by']
            reason='grip_calibration_includes_heldout_recalibrate_with_training_frames'
            self.assertIn(reason,check([0,10]))
            self.assertNotIn(reason,check([10,20]))
