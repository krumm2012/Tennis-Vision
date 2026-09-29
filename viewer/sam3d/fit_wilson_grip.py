"""Palm-anchored Wilson fitting with robust silhouettes, mirror evidence and temporal priors."""
import json,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from fit_racket_video import head_ellipse,fit_shape,silhouette_rms
from fit_racket_pose import project,model_points,PARTS
from track_racket_masks import select_candidate


def palm_frame(joints):
    p=np.asarray(joints);w=p[41];forward=np.mean(p[[28,32,36,40]],axis=0)-w;forward/=np.linalg.norm(forward)
    across=p[28]-p[40];across-=forward*np.dot(across,forward);across/=np.linalg.norm(across)
    normal=np.cross(across,forward)
    return w,np.column_stack([across,forward,normal])


def grip_contact(joints):
    """Anatomical grasp prior, not a measured grip/bevel classification.

    Shaft exits at the index side; the butt stays by the little finger.
    A 13mm radius puts its center on the flexion side of the MCP/PIP corridor.
    """
    p=np.asarray(joints);w,h=palm_frame(p)
    bases=p[[28,32,36,40]];proximal=p[[27,31,35,39]]
    curl=(p[[25,29,33,37]]-bases).mean(0)@h[:,2]
    side=1. if curl>=0 else -1.
    center=(bases+proximal).mean(0)*.5+h[:,2]*side*.013
    axis=(bases[0]+proximal[0]-bases[3]-proximal[3])*.5
    axis/=np.linalg.norm(axis)
    return (center-w)@h,axis@h


def ellipse_residual(uv,ellipse):
    c,r,a=ellipse;axes=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return np.r_[(np.linalg.norm(((uv-c)@axes)/r,axis=1)-1)*np.sqrt(np.prod(r)),(uv.mean(0)-c)*1.5]


def main():
    root=Path('output/sam3d_cloud');read=lambda n:json.loads((root/n).read_text())
    if not (root/'racket_poses_v1.json').exists():shutil.copy2(root/'racket_poses.json',root/'racket_poses_v1.json')
    old=read('racket_poses_v1.json');meta=read('mesh_meta.json');frames=read('temporal_pose.json')['frames'];detections=read('racket_yolo_candidates.json')['frames'];masks=read('racket_tracked_masks.json')['frames'];model=read('wilson_model.json')['model'];mirror=read('mirror_geometry.json')
    count=len(frames);ring=np.array(model['head_outline']);grip=np.array([0,model['grip_y_m'],0]);roots=np.array(meta['source_roots']);focals=np.array(meta['focal']);palms=[palm_frame(f['raw']) for f in frames];wrists=np.array([a[0] for a in palms])+roots;hands=np.array([a[1] for a in palms]);normal=np.array(mirror['normal_camera']);normal/=np.linalg.norm(normal);plane=mirror['distance_camera_m']
    model=dict(model,grip_y_m=.045);grip=np.array([0.,.045,0.])
    contacts=[grip_contact(f['raw']) for f in frames]
    offsets=np.array([c[0] for c in contacts]);grip_axes=np.array([hands[i]@c[1] for i,c in enumerate(contacts)])
    anatomical=np.median(offsets,axis=0);real=[];mirror_obs=[];confidence=[]
    for i in range(count):
        candidate=select_candidate(detections[i]);score=candidate['confidence'] if candidate else 0.;confidence.append(score)
        e=head_ellipse(masks[i]['polygon']) if masks[i]['mask_valid'] else None
        if candidate and score>.1:
            y=head_ellipse(candidate['polygon'])
            if y is not None and (e is None or np.linalg.norm(e[0]-y[0])>20):e=y
        real.append(e)
        reflected=wrists[i]-2*(wrists[i]@normal-plane)*normal;uv=project(reflected[None,:],focals[i])[0];options=[]
        for c in detections[i]['candidates']:
            if c['confidence']<.1:continue
            h=head_ellipse(c['polygon'])
            if h is not None and np.linalg.norm(h[0]-uv)<100 and (e is None or np.linalg.norm(h[0]-e[0])>60):options.append((c['confidence'],h))
        mirror_obs.append(max(options,key=lambda q:q[0])[1] if options else None)
    # Deterministic, spatially distributed clear candidates; not manually verified ground truth.
    keys=[]
    for start in range(0,count,20):
        options=[i for i in range(start,min(start+20,count)) if real[i] is not None and confidence[i]>.15 and old['frames'][i]['quality']=='silhouette_fitted' and old['frames'][i].get('mask_fit_rms_px',99)<2]
        if options:keys.append(max(options,key=lambda i:confidence[i]))
    previous=None;initial=[]
    for i in range(count):
        target=wrists[i]+hands[i]@offsets[i]
        if real[i] is not None:
            result=fit_shape(real[i],target,project(target[None,:],focals[i])[0],focals[i],model,previous)
            matrix=result[1]
        else:matrix=previous if previous is not None else np.array(old['frames'][i]['rotation_camera_columns'])
        initial.append(matrix);previous=matrix
    initial=np.array(initial)
    # Start away from 180-degree chordal stationary points before joint optimization.
    for i in range(1,count):
        delta=Rotation.from_matrix(initial[i-1].T@initial[i]).as_rotvec();angle=np.linalg.norm(delta)
        if angle>np.radians(30):initial[i]=initial[i-1]@Rotation.from_rotvec(delta*np.radians(30)/angle).as_matrix()
    offset=anatomical
    targets=wrists+np.einsum('nij,nj->ni',hands,offsets)
    # Admit mirror ellipses only when a plausible initial projection supports the association.
    for i,e in enumerate(mirror_obs):
        if e is None:continue
        points=ring@initial[i].T+targets[i]-initial[i]@grip;ref=points-2*(points@normal-plane)[:,None]*normal
        if np.linalg.norm(project(ref,focals[i]).mean(0)-e[0])>35:mirror_obs[i]=None
    accepted=[e is not None for e in real]
    # Per-frame observation residuals plus smooth hand-relative rotations.
    def residual(x):
        rs=Rotation.from_rotvec(x.reshape(count,3)).as_matrix();relative=np.einsum('nji,njk->nik',hands,rs);result=[]
        for i,R in enumerate(rs):
            points=ring@R.T+targets[i]-R@grip
            result.append(ellipse_residual(project(points,focals[i]),real[i]) if accepted[i] else np.zeros(34))
            reflected=points-2*(points@normal-plane)[:,None]*normal
            result.append(.3*ellipse_residual(project(reflected,focals[i]),mirror_obs[i]) if mirror_obs[i] is not None else np.zeros(34))
        # Absolute camera rotation must remain smooth even when SAM fingers jump.
        result=[np.sign(v)*np.sqrt(6*(np.sqrt(1+(v/3)**2)-1)) for v in result]
        # A directed shaft constraint prevents the butt pointing toward the fingers.
        result.append(((rs[:,:,1]-grip_axes)*1).ravel())
        result.append(((rs[1:]-rs[:-1])*6).ravel())
        result.append(((rs[2:]-2*rs[1:-1]+rs[:-2])*4).ravel())
        return np.concatenate(result)
    sparsity=lil_matrix((count*71+(count-1)*9+(count-2)*9,count*3),dtype=int)
    for i in range(count):sparsity[i*68:(i+1)*68,i*3:(i+1)*3]=1
    for i in range(count):sparsity[count*68+i*3:count*68+(i+1)*3,i*3:(i+1)*3]=1
    k=count*71
    for i in range(count-1):sparsity[k+i*9:k+(i+1)*9,i*3:(i+2)*3]=1
    k+=(count-1)*9
    for i in range(count-2):sparsity[k+i*9:k+(i+1)*9,i*3:(i+3)*3]=1
    x=Rotation.from_matrix(initial).as_rotvec().ravel()
    for iteration in range(2):
        fit=least_squares(residual,x,jac_sparsity=sparsity.tocsr(),loss='linear',max_nfev=100,ftol=1e-4);x=fit.x
        rs=Rotation.from_rotvec(x.reshape(count,3)).as_matrix()
        errors=[silhouette_rms(project(ring@rs[i].T+targets[i]-rs[i]@grip,focals[i]),e) if e is not None else None for i,e in enumerate(real)]
        if iteration==0:accepted=[e is not None and errors[i]<8 for i,e in enumerate(real)]
        print('iteration',iteration,'observations',sum(accepted),'cost',fit.cost,flush=True)
    accepted=[accepted[i] and errors[i] is not None and errors[i]<5 for i in range(count)]
    rows=[]
    for i,R in enumerate(rs):
        t=targets[i]-R@grip;observed=accepted[i];uv=project(model_points(model)@R.T+t,focals[i]);error=errors[i]
        jump=float(np.degrees(Rotation.from_matrix(rs[i-1].T@R).magnitude())) if i else 0
        reasons=[]
        if not observed:reasons.append('missing_or_rejected_observation')
        if error is not None and error>3:reasons.append('contour_residual')
        if jump>45:reasons.append('rotation_jump')
        if R[:,1]@grip_axes[i]<np.cos(np.radians(35)):reasons.append('hand_axis_conflict')
        rows.append({'frame':i,'palm_offset_m':offsets[i].tolist(),'grip_axis_error_deg':float(np.degrees(np.arccos(np.clip(R[:,1]@grip_axes[i],-1,1)))),'status':'fitted','ambiguous':True,'quality':'silhouette_fitted' if observed else 'temporal_estimate','source':'wilson_anatomical_grip_v3','rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist(),'grip_target_camera_m':targets[i].tolist(),'palm_anchor_error_m':float(np.linalg.norm(R@grip+t-targets[i])),'wrist_gap_m':float(np.linalg.norm(targets[i]-wrists[i])),'mask_fit_rms_px':round(error,2) if error is not None else None,'detection_confidence':confidence[i],'mirror_observation_used':mirror_obs[i] is not None,'rotation_step_deg':jump,'review_reasons':reasons,'projected_points':dict(zip(PARTS,uv.tolist())),'projected_head_outline':project(ring@R.T+t,focals[i]).tolist()})
    calibration={'keyframes_1based':[i+1 for i in keys],'selection':'automatic clear-candidate selection; not manual ground truth','palm_offset_m':offset.tolist(),'grip_local_m':grip.tolist(),'method':'per-frame MCP/PIP grasp corridor; directed shaft prior; butt 45mm below contact; silhouette fit','requires_manual_confirmation':True,'status':'provisional_anatomical_prior' if np.allclose(offset,anatomical) else 'provisional_image_calibration','dimensions_measured':False,'signed_face_orientation_verified':False}
    out={**old,'model':model,'method':'wilson_anatomical_grip_v3','grasp_calibration':calibration,'frames':rows,'summary':{'frames':count,'silhouette_fitted':sum(accepted),'interpolated':0,'constrained_estimate':count-sum(accepted),'mirror_observations':sum(e is not None for e in mirror_obs),'mono_plane_ambiguity':True,'model_dimensions_measured':False}}
    (root/'wilson_grasp_calibration_v3.json').write_text(json.dumps(calibration,indent=2));(root/'racket_poses_v3.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['summary']))

if __name__=='__main__':main()
