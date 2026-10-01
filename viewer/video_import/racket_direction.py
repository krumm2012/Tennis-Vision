"""Heldout-validated hand-local shaft prior from this video's mirror observations."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def fixed_grip_initial(ellipse,target,focal,size,model,points,weights,hand_axis,hand_weight,stereo_axis,previous):
    # Fit the same fixed grip used by the sequence solver. A free translation seed
    # can fit the ellipse and then become wrong when the palm anchor is enforced.
    from fit_racket_video import initial_rotations
    from fit_wilson_grip import ellipse_residual
    from fit_racket_pose import project,model_points,PARTS
    grip=np.array([0,model['grip_y_m'],0]);ring=np.array(model['head_outline'])
    center=ellipse[0] if ellipse is not None else points.get('head_center')
    if center is None:return previous
    target=np.asarray(target);uv=project(target[None],focal,size)[0]
    seeds=initial_rotations(np.asarray(center),uv,target,focal,model,previous)
    for axis in [hand_axis,stereo_axis]:
        if axis is None:continue
        y=np.asarray(axis);x=np.array([y[1],-y[0],0.]);x-=y*(x@y)
        if np.linalg.norm(x)<1e-8:continue
        x/=np.linalg.norm(x);seeds.append(np.column_stack([x,y,np.cross(x,y)]))
    objects={**dict(zip(PARTS,model_points(model))),'head_center':np.array([0,model['head_center_y_m'],0])}
    def residual(q):
        R=Rotation.from_rotvec(q).as_matrix();t=target-R@grip
        terms=[ellipse_residual(project(ring@R.T+t,focal,size),ellipse)] if ellipse is not None else []
        for name,p in points.items():
            if name in objects:terms.append((project((R@objects[name]+t)[None],focal,size)[0]-p)*weights[name]*(6 if name=='head_center' else 1))
        terms.append((R[:,1]-hand_axis)*hand_weight)
        if stereo_axis is not None:terms.append((R[:,1]-stereo_axis)*12)
        return np.concatenate(terms)
    candidates=[]
    for seed in seeds:
        solved=least_squares(residual,Rotation.from_matrix(seed).as_rotvec(),loss='soft_l1',f_scale=3,max_nfev=45)
        R=Rotation.from_rotvec(solved.x).as_matrix()
        if previous is not None and Rotation.from_matrix(previous.T@R).magnitude()>Rotation.from_matrix(previous.T@R@np.diag([-1.,1.,-1.])).magnitude():R=R@np.diag([-1.,1.,-1.])
        candidates.append((solved.cost,R))
    return min(candidates,key=lambda c:c[0])[1]

def angles(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))

def calibrate_axes(world_axes,palm_matrices,stereo_axes,head_uv,targets,focals,size):
    world=np.asarray(world_axes);hands=np.asarray(palm_matrices);local=np.einsum('nji,nj->ni',hands,world)
    ids=np.array([i for i,a in enumerate(stereo_axes) if a is not None]);report={'accepted':False,'method':'robust fixed rotation in palm coordinates; every fifth frame held out','stereo_samples':len(ids),'physical_grip_bevel_verified':False}
    train=ids[ids%5!=0];test=ids[ids%5==0]
    if len(train)<12 or len(test)<5:return world.copy(),report
    seen=np.array([a if a is not None else [0,0,0] for a in stereo_axes]);stereo_local=np.einsum('nji,nj->ni',hands,seen)
    targets=np.asarray(targets);focals=np.asarray(focals);image_ids=np.array([i for i,p in enumerate(head_uv) if p is not None]);image_test=image_ids[image_ids%5==0]
    head=np.array([p if p is not None else [0,0] for p in head_uv]);grip_uv=targets[:,:2]/targets[:,2:]*focals[:,None]+np.asarray(size)/2;image_direction=head-grip_uv;image_direction/=np.maximum(np.linalg.norm(image_direction,axis=1,keepdims=True),1e-8)
    def values(x):
        rotated=local@Rotation.from_rotvec(x).as_matrix().T;axis=np.einsum('nij,nj->ni',hands,rotated);end=targets+axis*.1
        uv=end[:,:2]/np.maximum(end[:,2:],.1)*focals[:,None]+np.asarray(size)/2-grip_uv;uv/=np.maximum(np.linalg.norm(uv,axis=1,keepdims=True),1e-8)
        return rotated,axis,uv
    def residual(x):
        rotated,_,_=values(x)
        return np.r_[(rotated[train]-stereo_local[train]).ravel(),x*.05]
    fit=least_squares(residual,np.zeros(3),loss='soft_l1',f_scale=.15,max_nfev=100)
    rotated,corrected,uv=values(fit.x);_,_,before_uv=values(np.zeros(3))
    before=float(np.median(angles(local[test],stereo_local[test])));after=float(np.median(angles(rotated[test],stereo_local[test])))
    before_image=float(np.median(angles(before_uv[image_test],image_direction[image_test]))) if len(image_test) else None
    after_image=float(np.median(angles(uv[image_test],image_direction[image_test]))) if len(image_test) else None
    bias=float(np.degrees(np.linalg.norm(fit.x)))
    accepted=bool(bias<60 and after<before*.95 and before_image is not None and after_image<before_image*.9)
    report.update(accepted=accepted,train_frames=train.tolist(),heldout_frames=test.tolist(),bias_rotation_palm=Rotation.from_rotvec(fit.x).as_matrix().tolist(),bias_angle_deg=bias,stereo_heldout_before_deg=before,stereo_heldout_after_deg=after,image_heldout_before_deg=before_image,image_heldout_after_deg=after_image)
    return (corrected if accepted else world.copy()),report
