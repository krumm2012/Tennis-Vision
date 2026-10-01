import unittest
import torch
from grip_objective import grip_terms,directed_image_loss,reliability,hand_geometry,hand_direction_weight


def hand():
    joints=torch.zeros(2,70,3);joints[:,:,2]=3
    for ids,y in [([28,32,36,40],.09),([27,31,35,39],.11),([25,29,33,37],.12)]:
        joints[:,ids,0]=torch.tensor([.03,.01,-.01,-.03]);joints[:,ids,1]=y
    joints[:,[25,29,33,37],2]+=.01
    return joints


class GripObjectiveTests(unittest.TestCase):
    def test_hand_direction_gate_is_continuous_and_preserves_clear_conflicts(self):
        import math
        angles=torch.tensor([0.,25.,34.999,35.001,45.,90.,180.])
        weights=hand_direction_weight(1-torch.cos(angles*math.pi/180))
        torch.testing.assert_close(weights[:2],torch.full((2,),.4))
        torch.testing.assert_close(weights[4:],torch.full((3,),.03))
        self.assertLess(float(abs(weights[2]-weights[3])),.001)
        self.assertTrue(torch.all(weights[1:]<=weights[:-1]))

    def test_detached_fingers_have_restoring_gradient_and_finite_handle_penalty(self):
        j=hand();r=torch.eye(3)[None].repeat(2,1,1);t=torch.tensor([[.2,0.,3.],[.2,0.,3.]],requires_grad=True)
        terms=grip_terms(j,r,t,torch.tensor([0,.045,0]),.013,torch.ones(2))
        terms['finger_distance'].backward();self.assertGreater(float(t.grad[:,0].min()),0)
        self.assertTrue(torch.isfinite(t.grad).all())
        far=t.detach().clone();far[:,1]=1
        self.assertGreater(float(grip_terms(j,r,far,torch.tensor([0,.045,0]),.013,torch.ones(2))['handle_segment']),0)

    def test_directed_axis_rejects_reversed_endpoints(self):
        p=torch.tensor([[0.,0.],[10.,0.]],requires_grad=True)
        self.assertAlmostEqual(float(directed_image_loss(p,p.detach()).detach()),0)
        self.assertAlmostEqual(float(directed_image_loss(p,p.detach().flip(0)).detach()),2)
        directed_image_loss(p,torch.tensor([[0.,0.],[0.,10.]])).backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)

    def test_heldout_landmarks_cannot_change_training_hand_weights(self):
        j=hand();root=torch.zeros(2,3);f=torch.ones(2)*1000;size=torch.tensor([1280.,720.]);marks={'frames':[{'frame':i,'points':{'handle_end':[100,100],'tip':[0,100]}} for i in range(2)]}
        weights=reliability(j,root,f,size,marks,{0});self.assertLess(float(weights[0]),1);self.assertAlmostEqual(float(weights[1]),.1)
        marks['frames'][1]['points']['tip']=[300,400]
        torch.testing.assert_close(weights,reliability(j,root,f,size,marks,{0}))

    def test_mhr_hand_and_racket_both_receive_gradients(self):
        j=hand().requires_grad_();r=torch.eye(3)[None].repeat(2,1,1).requires_grad_();t=torch.tensor([[.05,0.,3.],[.05,0.,3.]],requires_grad=True)
        terms=grip_terms(j,r,t,torch.tensor([0,.045,0]),.013,torch.ones(2));sum(terms.values()).backward()
        for value in [j,r,t]:self.assertTrue(torch.isfinite(value.grad).all());self.assertGreater(float(value.grad.abs().sum()),0)

    def test_joint_solver_cpu_integration_keeps_candidate_private(self):
        import numpy as np
        from refit_fullbody import solve
        j=hand().repeat(3,1,1)
        class Head:
            keypoint_mapping=torch.cat((torch.eye(70),torch.zeros(70,127)),dim=1)
            def mhr(self,shape,params,expression):
                base=hand()[0]*torch.tensor([1.,-1.,-1.])*100
                vertices=base[None].repeat(len(params),1,1)+params[:,3:136].mean(1)[:,None,None]
                return vertices,torch.zeros(len(params),127,8)
        data={'mhr_model_params':np.zeros((6,204),np.float32),'mhr_shape_params':np.zeros((6,45),np.float32),'mhr_expr_params':np.zeros((6,72),np.float32),'source_roots':np.zeros((6,3),np.float32),'joints':j.numpy(),'vertices':j.numpy()}
        meta={'focal':[1000.]*6,'image_size':[1280,720]}
        model={'grip_y_m':.045,'throat_y_m':.35,'length_m':.685,'head_half_width_m':.13,'head_center_y_m':.51}
        poses={'model':model,'frames':[{'rotation_camera_columns':np.eye(3).tolist(),'translation_camera_m':[0.,.055,3.]} for _ in range(6)]}
        marks={'frames':[{'frame':i,'points':{'handle_end':[640,378.3333333],'tip':[640,606.6666667]},'mirror_points':{},'source':'automatic_contour_unverified'} for i in [0,2]]}
        calibration={'dimensions_cm':{},'measured':False};ready={'train_frames':[0],'heldout_frames':[2],'validation_source':'synthetic_fixture'}
        out,report=solve(Head(),data,meta,poses,marks,calibration,ready,steps=2,device='cpu')
        self.assertEqual(report['objective_version'],'confidence_observations_v5_continuous_hand_gate');self.assertFalse(report['published_to_viewer'])
        self.assertEqual(report['heldout_frames'],[2]);self.assertIn('hand_shaft_deg',report['contact_after'])
        for value in out.values():self.assertTrue(np.isfinite(value).all())
        split,_=solve(Head(),data,meta,poses,marks,calibration,ready,steps=2,device='cpu',batch_size=2)
        np.testing.assert_allclose(out['racket_rotation'],split['racket_rotation'],atol=1e-6)
        np.testing.assert_allclose(out['racket_translation'],split['racket_translation'],atol=1e-6)

if __name__=='__main__':unittest.main()
