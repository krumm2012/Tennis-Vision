"""Experimental native MHR + rigid racket joint optimisation in a private output.

Never writes viewer meshes. Native replay must reproduce SAM meshes first.
Measured or explicitly permitted assumed dimensions are required. Reprojection
priors are SAM predictions, not independent body ground truth.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from refit_readiness import assess
from run_records import write, sha256


def rotation_vector_matrix(v):
    import torch
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    theta=torch.linalg.vector_norm(v,dim=-1)
    # sinc is differentiable and well-defined at the zero initial correction.
    a=torch.sinc(theta/torch.pi)[:,None,None]
    b=(.5*torch.sinc(theta/(2*torch.pi))**2)[:,None,None]
    return torch.eye(3,device=v.device)[None]+a*k+b*(k@k)


def replay(head, shape, params, expression):
    """Official MHR head units and keypoint mapping; fixed camera axes [x,-y,-z]."""
    import torch
    vertices,state=head.mhr(shape,params,expression)
    vertices=vertices/100;coords=state[:,:,:3]/100
    mapped=torch.einsum('kv,bvc->bkc',head.keypoint_mapping[:70],torch.cat((vertices,coords),dim=1))
    axes=torch.tensor([1.,-1.,-1.],device=vertices.device)
    return vertices*axes,mapped*axes


def solve(head, data, meta, poses, marks, calibration, readiness, steps=120, device='cuda'):
    import torch
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    params=tensor(data['mhr_model_params']);shape=tensor(data['mhr_shape_params']);expr=tensor(data['mhr_expr_params'])
    roots=tensor(data['source_roots']);j0=tensor(data['joints']);verts0=tensor(data['vertices'])
    count=len(params);focal=tensor(meta['focal']);image_size=tensor(meta['image_size'])
    with torch.no_grad():
        mesh_error=[];joint_error=[]
        for a in range(0,count,16):
            v,j=replay(head,shape[a:a+16],params[a:a+16],expr[a:a+16])
            mesh_error.append(float((v-verts0[a:a+16]).abs().max()))
            joint_error.append(float((j-j0[a:a+16]).abs().max()))
    if max(mesh_error)>1e-4 or max(joint_error)>1e-4:
        raise ValueError('原生 MHR 回放未复现源网格/关节，禁止拟合；检查模型与坐标版本')
    r0=tensor([r['rotation_camera_columns'] for r in poses['frames']]);t0=tensor([r['translation_camera_m'] for r in poses['frames']])
    delta=torch.nn.Parameter(torch.zeros(count,133,device=device))
    racket_delta=torch.nn.Parameter(torch.zeros(count,6,device=device))
    optimizer=torch.optim.Adam([delta,racket_delta],lr=.008)
    model=poses['model'];grip=tensor([0,model['grip_y_m'],0])
    objects=tensor([[0,0,0],[0,model['throat_y_m'],0],[0,model['length_m'],0],
                    [model['head_half_width_m'],model['head_center_y_m'],0],[-model['head_half_width_m'],model['head_center_y_m'],0],[0,model['head_center_y_m'],0]])
    names=['handle_end','throat','tip','rim_side','rim_opposite','head_center'];manual={r['frame']:r for r in marks['frames']}
    train=set(readiness['train_frames']);heldout=set(readiness['heldout_frames'])
    mirror=meta.get('_mirror');n=tensor(mirror['normal_camera']) if mirror else None
    radius=calibration['dimensions_cm'].get('handle_diameter',2.6)/200
    def project(p, f):return p[...,:2]/p[...,2:].clamp_min(.1)*f[...,None,None]+image_size/2
    def current(a,b):
        # Optimise global rotation and full skeletal pose; native translation/scales,
        # expression, identity shape and camera roots stay fixed for this experiment.
        p=torch.cat((params[a:b,:3],params[a:b,3:136]+.25*torch.tanh(delta[a:b]),params[a:b,136:]),dim=1)
        v,j=replay(head,shape[a:b],p,expr[a:b])
        rd=racket_delta[a:b];r=rotation_vector_matrix(rd[:,:3])@r0[a:b];t=t0[a:b]+.05*torch.tanh(rd[:,3:])
        return p,v,j,r,t
    for iteration in range(steps):
        optimizer.zero_grad();total=0.
        for a in range(0,count,16):
            b=min(a+16,count);p,v,j,r,t=current(a,b);world=j+roots[a:b,None]
            baseline=j0[a:b]+roots[a:b,None]
            # Preserve full-body projection and original anatomy as priors.
            loss=torch.nn.functional.smooth_l1_loss(project(world,focal[a:b]),project(baseline,focal[a:b]),beta=8)/20
            loss+=((j-j0[a:b])**2).mean()*40+(delta[a:b]**2).mean()*.03
            bases=world[:,[28,32,36,40]];prox=world[:,[27,31,35,39]]
            forward=torch.nn.functional.normalize(bases.mean(1)-world[:,41],dim=-1)
            across=torch.nn.functional.normalize(bases[:,0]-bases[:,3],dim=-1)
            normal=torch.nn.functional.normalize(torch.cross(across,forward,dim=-1),dim=-1)
            curl=((world[:,[25,29,33,37]]-bases)*normal[:,None]).sum(-1).mean(1)
            contact=(bases+prox).mean(1)*.5+normal*torch.where(curl>=0,radius,-radius)[:,None]
            anchor=torch.einsum('bij,j->bi',r,grip)+t
            loss+=((anchor-contact)**2).mean()*1200
            # Finger-joint proxy distance to grip cylinder, not mesh-surface contact.
            local=torch.einsum('bji,bkj->bki',r,prox-t[:,None]);radial=torch.linalg.vector_norm(local[:,:,[0,2]],dim=-1)
            on_handle=(local[:,:,1]>=0)&(local[:,:,1]<.16)
            loss+=(torch.relu(radius-radial)**2*on_handle).mean()*80
            obj=torch.einsum('bij,kj->bki',r,objects)+t[:,None]
            for i in range(a,b):
                if i not in train:continue
                for view in ('points','mirror_points'):
                    seen=manual[i][view];ids=[k for k,name in enumerate(names) if name in seen]
                    if not ids or (view=='mirror_points' and n is None):continue
                    points=obj[i-a]
                    if view=='mirror_points':points=points-2*((points@n-mirror['distance_camera_m'])[:,None])*n
                    uv=project(points[None],focal[i:i+1])[0]
                    evidence_weight=1. if manual[i].get('source')=='manual_review' else .3
                    loss+=evidence_weight*torch.nn.functional.smooth_l1_loss(uv[ids],tensor([seen[names[k]] for k in ids]),beta=8)/(meta['image_size'][0]/1280*10)
            scaled=loss*(b-a)/count
            if not torch.isfinite(scaled):raise ValueError('联合拟合出现非有限残差；候选未保存')
            scaled.backward();total+=float(scaled.detach())
        temporal=(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()*.5
        temporal+=(racket_delta[2:]-2*racket_delta[1:-1]+racket_delta[:-2]).square().mean()*.5
        temporal.backward();optimizer.step()
        if iteration%20==0:print(f'MHR joint iteration {iteration}: {total:.5f}',flush=True)
    out={};vertices=[];joints=[];parameters=[];rotations=[];translations=[]
    with torch.no_grad():
        for a in range(0,count,16):
            p,v,j,r,t=current(a,min(count,a+16))
            for target,value in [(vertices,v),(joints,j),(parameters,p),(rotations,r),(translations,t)]:target.append(value.cpu().numpy())
    out.update(vertices=np.concatenate(vertices),joints=np.concatenate(joints),mhr_model_params=np.concatenate(parameters),
               racket_rotation=np.concatenate(rotations),racket_translation=np.concatenate(translations),source_roots=data['source_roots'])
    def rms(rot,trans,indices):
        errors=[]
        for i in indices:
            points=objects.cpu().numpy()@rot[i].T+trans[i]
            uv=points[:,:2]/points[:,2:]*meta['focal'][i]+np.asarray(meta['image_size'])/2
            for k,name in enumerate(names):
                if name in manual[i]['points']:errors.append(np.linalg.norm(uv[k]-manual[i]['points'][name]))
        return float(np.median(errors))/(meta['image_size'][0]/1280)
    before=rms(r0.cpu().numpy(),t0.cpu().numpy(),heldout);after=rms(out['racket_rotation'],out['racket_translation'],heldout)
    displacement=np.linalg.norm(out['vertices']-data['vertices'],axis=-1)
    report={'native_replay_mesh_max_m':max(mesh_error),'native_replay_joint_max_m':max(joint_error),
            'heldout_landmark_before_canonical_px':before,'heldout_landmark_after_canonical_px':after,
            'body_displacement_p95_m':float(np.percentile(displacement,95)),
            'numerical_regression_gate_passed':after<before*.9 and np.percentile(displacement,95)<.03,
            'full_body_joint_fit_completed':True,'camera_roots_preserved':True,'shape_and_scale_preserved':True,
            'dimensions_measured':calibration['measured'],'physical_grip_bevel_verified':False,
            'contact_method':'palm corridor and proximal joint / assumed-cylinder proxy; not mesh surface contact',
            'validation_source':readiness['validation_source'],'requires_visual_review':True,'published_to_viewer':False,'steps':steps}
    return out,report


def prepare_observations(result,readiness):
    out=Path(result);observations={}
    if readiness.get('automatic_frames'):
        kp=json.loads((out/'racket_keypoints.json').read_text());selection=json.loads((out/'racket_review_frames.json').read_text())
        if kp['video_sha256']!=readiness['video_sha256'] or sha256(out/'racket_keypoints.json')!=selection['keypoints_sha256']:raise ValueError('自动关键帧观测版本不一致')
        for row in selection['frames']:
            i=row['frame'];source=kp['frames'][i];seen={'frame':i,'points':{},'mirror_points':{},'source':'automatic_contour_unverified'}
            for view,target in [('real','points'),('mirror','mirror_points')]:
                if view in row['views']:seen[target]=source.get(view,{}).get('points',{})
            observations[i]=seen
    if (out/'racket_landmarks.json').exists():
        reviewed=json.loads((out/'racket_landmarks.json').read_text())
        if reviewed['video_sha256']!=readiness['video_sha256']:raise ValueError('人工标注属于其他视频')
        for row in reviewed['frames']:
            existing=observations.setdefault(row['frame'],{'frame':row['frame'],'points':{},'mirror_points':{}})
            for view in ['points','mirror_points']:existing[view].update(row[view])
            existing['source']='manual_review'
    return {'frames':list(observations.values())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-assumed',action='store_true');p.add_argument('--allow-automatic',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--steps',type=int,default=120)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);readiness=assess(a.result,a.archive,a.allow_assumed,a.allow_automatic);write(a.output/'readiness.json',readiness)
    if a.check_only or not readiness['ready']:
        print(json.dumps(readiness));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('需要云 CUDA 运行真实 MHR 重拟合')
    sys.path.insert(0,os.environ['SAM3D_BODY_CODE']);from sam_3d_body import load_sam_3d_body
    weights=Path(os.environ['SAM3D_WEIGHTS']);model,_=load_sam_3d_body(str(weights/'model.ckpt'),device='cuda',mhr_path=str(weights/'assets/mhr_model.pt'))
    model.eval()
    for parameter in model.parameters():parameter.requires_grad=False
    head=model.head_pose;del model;torch.cuda.empty_cache()
    read=lambda name:json.loads((a.result/name).read_text())
    meta=read('mesh_meta.json');meta['_mirror']=read('mirror_geometry.json') if meta.get('mirror_available') else None
    poses=read('racket_poses_directional.json') if (a.result/'racket_poses_directional.json').exists() else read('racket_poses.json')
    if not all(r['status']=='fitted' for r in poses['frames']):raise ValueError('本轮原型要求连续球拍初值；不能伪造长遮挡')
    with np.load(a.archive,allow_pickle=False) as d:data={k:d[k].copy() for k in d.files}
    marks=prepare_observations(a.result,readiness)
    if not np.allclose(data['source_roots'],meta['source_roots'],atol=.002,rtol=0) or not np.allclose(data['focal'],meta['focal'],rtol=.001):raise ValueError('重推理相机与现有标定不一致，需重新打包和标定')
    values,report=solve(head,data,meta,poses,marks,read('racket_dimensions.json'),readiness,a.steps)
    if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('联合拟合候选无效；未保存或发布')
    np.savez_compressed(a.output/'mhr_refit_candidate.npz',**values)
    report.update(video_sha256=meta['video_sha256'],source_archive_sha256=sha256(a.archive),candidate_sha256=sha256(a.output/'mhr_refit_candidate.npz'),code_sha256=sha256(Path(__file__)))
    write(a.output/'refit_report.json',report);print(json.dumps(report))


if __name__=='__main__':main()
