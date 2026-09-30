"""Reuse demo palm grip, real/mirror silhouettes and sparse temporal fitting per video."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'viewer/sam3d'))
from fit_wilson_grip import palm_frame,grip_contact,ellipse_residual
from fit_racket_video import head_ellipse,fit_shape,silhouette_rms
from fit_racket_pose import project,model_points,PARTS
from run_records import write,sha256,now

def select_candidate(frame):
    return frame.get('selected')

def fit_dataset(folder,result=None):
    folder=Path(folder);root=Path(result) if result else folder/'result'
    read=lambda n:json.loads((root/n).read_text())
    baseline=root/'racket_poses_wrist.json'
    if not baseline.exists():shutil.copy2(root/'racket_poses.json',baseline)
    old=read('racket_poses_wrist.json');meta=read('mesh_meta.json');size=meta['image_size']
    if old['video_sha256']!=meta['video_sha256']:raise ValueError('球拍来源与视频不一致')
    if old['summary'].get('hand','right')!='right':raise ValueError('掌内握拍目前仅支持右手')
    attempt=json.loads((folder/'record.json').read_text())['attempt']
    archive=folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:
        frames=[{'raw':p} for p in d['joints']]
    detections=read('racket_candidates.json')['frames'] if (root/'racket_candidates.json').exists() else [{'candidates':[]} for _ in frames]
    if len(detections)!=len(frames) or len(old['frames'])!=len(frames):raise ValueError('球拍观测帧数不一致')
    if (root/'racket_candidates.json').exists() and read('racket_candidates.json')['video_sha256']!=meta['video_sha256']:raise ValueError('镜中球拍观测来源不一致')
    if not any(r['status']=='fitted' for r in old['frames']):raise ValueError('没有可用球拍观测')
    for f in frames:
        p=np.asarray(f['raw'])
        if not np.isfinite(p).all() or np.linalg.norm(p[28]-p[40])<1e-6 or np.linalg.norm(p[[28,32,36,40]].mean(0)-p[41])<1e-6:raise ValueError('掌部关节退化，需复核后重试')
    for i,f in enumerate(detections):
        r=old['frames'][i];f['selected']={'confidence':r.get('detection_confidence',0),'polygon':r['observed_polygon']} if r.get('observed_polygon') else None
    masks=[{'mask_valid':bool(r.get('observed_polygon')),'polygon':r.get('observed_polygon',[])} for r in old['frames']]
    model=old['model']
    mirror_enabled=bool(meta.get('mirror_available')) and (root/'mirror_geometry.json').exists()
    mirror=read('mirror_geometry.json') if mirror_enabled else {'normal_camera':[0,0,1],'distance_camera_m':0}
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
        reflected=wrists[i]-2*(wrists[i]@normal-plane)*normal;uv=project(reflected[None,:],focals[i],size)[0];options=[]
        for c in detections[i]['candidates']:
            if not mirror_enabled or c['confidence']<.1:continue
            h=head_ellipse(c['polygon'])
            if h is not None and np.linalg.norm(h[0]-uv)<100 and (e is None or np.linalg.norm(h[0]-e[0])>60):options.append((c['confidence'],h))
        mirror_obs.append(max(options,key=lambda q:q[0])[1] if options else None)
    # Deterministic, spatially distributed clear candidates; not manually verified ground truth.
    keys=[]
    for start in range(0,count,20):
        options=[i for i in range(start,min(start+20,count)) if real[i] is not None and confidence[i]>.15 and old['frames'][i].get('quality')=='silhouette_fitted' and old['frames'][i].get('mask_fit_rms_px',99)<2]
        if options:keys.append(max(options,key=lambda i:confidence[i]))
    previous=None;initial=[]
    for i in range(count):
        target=wrists[i]+hands[i]@offsets[i]
        if real[i] is not None:
            result=fit_shape(real[i],target,project(target[None,:],focals[i],size)[0],focals[i],model,previous,image_size=size)
            matrix=result[1]
        else:matrix=previous if previous is not None else np.array(next(r['rotation_camera_columns'] for r in old['frames'] if r['status']=='fitted'))
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
        if np.linalg.norm(project(ref,focals[i],size).mean(0)-e[0])>35:mirror_obs[i]=None
    accepted=[e is not None for e in real]
    # Per-frame observation residuals plus smooth hand-relative rotations.
    def residual(x):
        rs=Rotation.from_rotvec(x.reshape(count,3)).as_matrix();result=[]
        for i,R in enumerate(rs):
            points=ring@R.T+targets[i]-R@grip
            result.append(ellipse_residual(project(points,focals[i],size),real[i]) if accepted[i] else np.zeros(34))
            reflected=points-2*(points@normal-plane)[:,None]*normal
            result.append(.3*ellipse_residual(project(reflected,focals[i],size),mirror_obs[i]) if mirror_obs[i] is not None else np.zeros(34))
        # Absolute camera rotation must remain smooth even when SAM fingers jump.
        result=[np.sign(v)*np.sqrt(6*(np.sqrt(1+(v/3)**2)-1)) for v in result]
        # A directed shaft constraint prevents the butt pointing toward the fingers.
        result.append(((rs[:,:,1]-grip_axes)*24).ravel())
        result.append(((rs[1:]-rs[:-1])*20).ravel())
        result.append(((rs[2:]-2*rs[1:-1]+rs[:-2])*12).ravel())
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
        errors=[silhouette_rms(project(ring@rs[i].T+targets[i]-rs[i]@grip,focals[i],size),e) if e is not None else None for i,e in enumerate(real)]
        if iteration==0:
            accepted=[e is not None and errors[i]<8 for i,e in enumerate(real)]
            for i,e in enumerate(mirror_obs):
                if e is None:continue
                points=ring@rs[i].T+targets[i]-rs[i]@grip
                reflected=points-2*(points@normal-plane)[:,None]*normal
                if silhouette_rms(project(reflected,focals[i],size),e)>8:mirror_obs[i]=None
        print('iteration',iteration,'observations',sum(accepted),'cost',fit.cost,flush=True)
    accepted=[accepted[i] and errors[i] is not None and errors[i]<5 for i in range(count)]
    rows=[]
    for i,R in enumerate(rs):
        t=targets[i]-R@grip;observed=accepted[i];uv=project(model_points(model)@R.T+t,focals[i],size);error=errors[i]
        jump=float(np.degrees(Rotation.from_matrix(rs[i-1].T@R).magnitude())) if i else 0
        reasons=[]
        if not observed:reasons.append('missing_or_rejected_observation')
        if error is not None and error>3:reasons.append('contour_residual')
        if jump>45:reasons.append('rotation_jump')
        if R[:,1]@grip_axes[i]<np.cos(np.radians(35)):reasons.append('hand_axis_conflict')
        rows.append({'frame':i,'palm_offset_m':offsets[i].tolist(),'grip_axis_error_deg':float(np.degrees(np.arccos(np.clip(R[:,1]@grip_axes[i],-1,1)))),'status':'fitted','ambiguous':True,'quality':'silhouette_fitted' if observed else 'temporal_estimate','source':'wilson_anatomical_grip_v3','rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist(),'grip_target_camera_m':targets[i].tolist(),'palm_anchor_error_m':float(np.linalg.norm(R@grip+t-targets[i])),'wrist_gap_m':float(np.linalg.norm(targets[i]-wrists[i])),'mask_fit_rms_px':round(error,2) if error is not None else None,'detection_confidence':confidence[i],'mirror_observation_used':mirror_obs[i] is not None,'rotation_step_deg':jump,'review_reasons':reasons,'projected_points':dict(zip(PARTS,uv.tolist())),'projected_head_outline':project(ring@R.T+t,focals[i],size).tolist()})
    calibration={'keyframes_1based':[i+1 for i in keys],'selection':'automatic clear-candidate selection; not manual ground truth','palm_offset_m':offset.tolist(),'grip_local_m':grip.tolist(),'method':'per-frame MCP/PIP grasp corridor; directed shaft prior; butt 45mm below contact; silhouette fit','requires_manual_confirmation':True,'status':'provisional_anatomical_prior' if np.allclose(offset,anatomical) else 'provisional_image_calibration','dimensions_measured':False,'signed_face_orientation_verified':False}
    for i,row in enumerate(rows):
        if old['frames'][i]['status']!='fitted':
            row.update(status='missing_observation',quality='hidden')
        elif not accepted[i]:
            row['quality']='constrained_estimate'
        if old['frames'][i].get('observed_polygon'):
            row['observed_polygon']=old['frames'][i]['observed_polygon']
        row['face_angle_to_camera_deg']=float(np.degrees(np.arccos(np.clip(abs(rs[i,2,2]),0,1))))
    out={**old,'model':model,'method':'wilson_anatomical_grip_v3','grasp_calibration':calibration,'frames':rows,'summary':{'frames':count,'observed':sum(r['status']=='fitted' and r['quality']=='silhouette_fitted' for r in rows),'hidden':sum(r['status']!='fitted' for r in rows),'hand':old['summary'].get('hand','right'),'interpolated':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'constrained_estimate':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'mirror_observations':sum(e is not None for e in mirror_obs),'mono_plane_ambiguity':True,'model_dimensions_measured':False}}
    visible=[r for r in rows if r['status']=='fitted']
    out['summary'].update(palm_anchor_max_mm=max(r['palm_anchor_error_m'] for r in visible)*1000,shaft_error_median_deg=float(np.median([r['grip_axis_error_deg'] for r in visible])),rotation_step_p95_deg=float(np.percentile([r['rotation_step_deg'] for r in visible],95)))
    write(root/'racket_poses.json',out);write(root/'wilson_grasp_calibration.json',calibration);write(root/'wilson_model.json',{'model':model})
    write(root/'grip_fit_manifest.json',{'video_sha256':meta['video_sha256'],'baseline_sha256':sha256(baseline),'poses_sha256':sha256(root/'racket_poses.json'),'code_sha256':sha256(Path(__file__)),'finished_at':now(),'summary':out['summary'],'mirror_enabled':mirror_enabled})
    print(json.dumps(out['summary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);a=p.parse_args();fit_dataset(a.dataset,a.result)
