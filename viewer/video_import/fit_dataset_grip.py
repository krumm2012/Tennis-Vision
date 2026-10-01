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
from racket_stability import smooth_vectors,stable_rotations,supported_frames,motion_metrics

def select_candidate(frame):
    return frame.get('selected')

def fit_dataset(folder,result=None):
    folder=Path(folder);root=Path(result) if result else folder/'result'
    read=lambda n:json.loads((root/n).read_text())
    baseline=root/'racket_poses_wrist.json'
    if not baseline.exists():shutil.copy2(root/'racket_poses.json',baseline)
    old=read('racket_poses_wrist.json');meta=read('mesh_meta.json');pixel_scale=meta['image_size'][0]/1280;size=(np.asarray(meta['image_size'])/pixel_scale).tolist()
    if old['video_sha256']!=meta['video_sha256']:raise ValueError('球拍来源与视频不一致')
    if old['summary'].get('hand','right')!='right':raise ValueError('掌内握拍目前仅支持右手')
    attempt=json.loads((folder/'record.json').read_text())['attempt']
    archive=folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:
        frames=[{'raw':p} for p in d['joints']]
        joint_shape=d['joints'].shape
    constraints=root/'multiview_constraints.npz';multiview_frames=0
    if constraints.exists() and meta.get('multiview_refined_available'):
        report=read('multiview_constraints_report.json')
        if report['constraints_sha256']!=sha256(constraints) or report['archive_sha256']!=sha256(archive):raise ValueError('多视角约束哈希不一致')
        with np.load(constraints,allow_pickle=False) as d:
            if str(d['video_sha256'])!=meta['video_sha256'] or str(d['geometry_sha256'])!=sha256(root/'mirror_geometry.json') or d['joints'].shape!=joint_shape or not np.isfinite(d['joints']).all():raise ValueError('多视角握拍约束来源无效')
            frames=[{'raw':p} for p in d['joints']];multiview_frames=int(d['accepted'].sum())
    detections=read('racket_candidates.json')['frames'] if (root/'racket_candidates.json').exists() else [{'candidates':[]} for _ in frames]
    if len(detections)!=len(frames) or len(old['frames'])!=len(frames):raise ValueError('球拍观测帧数不一致')
    if (root/'racket_candidates.json').exists() and read('racket_candidates.json')['video_sha256']!=meta['video_sha256']:raise ValueError('镜中球拍观测来源不一致')
    if not any(r['status']=='fitted' for r in old['frames']):raise ValueError('没有可用球拍观测')
    for f in frames:
        p=np.asarray(f['raw'])
        if not np.isfinite(p).all() or np.linalg.norm(p[28]-p[40])<1e-6 or np.linalg.norm(p[[28,32,36,40]].mean(0)-p[41])<1e-6:raise ValueError('掌部关节退化，需复核后重试')
    roi=read('racket_roi_candidates.json') if (root/'racket_roi_candidates.json').exists() else None
    if roi and (roi['video_sha256']!=meta['video_sha256'] or len(roi['frames'])!=len(frames)):raise ValueError('原分辨率观测来源不一致')
    if roi:
        for i,f in enumerate(detections):f['candidates']+=roi['frames'][i]['candidates']
    for row in old['frames']:
        if row.get('observed_polygon'):row['observed_polygon']=(np.asarray(row['observed_polygon'])/pixel_scale).tolist()
    for f in detections:
        for c in f['candidates']:
            c['polygon']=(np.asarray(c['polygon'])/pixel_scale).tolist();c['box']=(np.asarray(c['box'])/pixel_scale).tolist()
    for i,f in enumerate(detections):
        r=old['frames'][i];f['selected']={'confidence':r.get('detection_confidence',0),'polygon':r['observed_polygon']} if r.get('observed_polygon') else None
    masks=[{'mask_valid':bool(r.get('observed_polygon')),'polygon':r.get('observed_polygon',[])} for r in old['frames']]
    model=old['model']
    mirror_enabled=bool(meta.get('mirror_available')) and (root/'mirror_geometry.json').exists()
    mirror=read('mirror_geometry.json') if mirror_enabled else {'normal_camera':[0,0,1],'distance_camera_m':0}
    count=len(frames);ring=np.array(model['head_outline']);grip=np.array([0,model['grip_y_m'],0]);roots=np.array(meta['source_roots']);focals=np.array(meta['focal'])/pixel_scale;palms=[palm_frame(f['raw']) for f in frames];wrists=np.array([a[0] for a in palms])+roots;hands=np.array([a[1] for a in palms]);normal=np.array(mirror['normal_camera'],dtype=float);normal/=np.linalg.norm(normal);plane=mirror['distance_camera_m']
    model=dict(model,grip_y_m=.045);grip=np.array([0.,.045,0.])
    contacts=[grip_contact(f['raw']) for f in frames]
    offsets=np.array([c[0] for c in contacts]);grip_axes=np.array([hands[i]@c[1] for i,c in enumerate(contacts)])
    raw_axes=grip_axes.copy();grip_axes=smooth_vectors(grip_axes,meta['fps']);grip_axes/=np.maximum(np.linalg.norm(grip_axes,axis=1,keepdims=True),1e-8)
    anatomical=np.median(offsets,axis=0);real=[];mirror_obs=[];confidence=[];previous_center=None;previous_wrist=None;observation_sources=[];observation_polygons=[]
    for i in range(count):
        candidate=select_candidate(detections[i]);score=candidate['confidence'] if candidate else 0.;confidence.append(score)
        e=head_ellipse(masks[i]['polygon']) if masks[i]['mask_valid'] else None
        if candidate and score>.1:
            y=head_ellipse(candidate['polygon'])
            if y is not None and (e is None or np.linalg.norm(e[0]-y[0])>20):e=y
        # Reassociate raw candidates, including missing legacy wrist-fit frames.
        wrist_uv=project(wrists[i][None,:],focals[i],size)[0];options=[]
        for c in detections[i]['candidates']:
            if c.get('view')=='mirror' or c['confidence']<.08:continue
            h=head_ellipse(c['polygon'])
            if h is None or np.linalg.norm(h[0]-wrist_uv)>140*size[0]/1280:continue
            box=np.asarray(c['box']);distance=np.linalg.norm(np.maximum(np.maximum(box[:2]-wrist_uv,wrist_uv-box[2:]),0))
            if distance>85*size[0]/1280:continue
            prediction=previous_center+(wrist_uv-previous_wrist) if previous_center is not None else h[0]
            score=c['confidence']*np.exp(-distance/70)*np.exp(-np.linalg.norm(h[0]-prediction)/100)
            options.append((score,c,h))
        source='normalized_full_frame';selected_polygon=candidate['polygon'] if candidate else None
        if options:
            _,candidate,e=max(options,key=lambda q:q[0]);confidence[-1]=candidate['confidence'];source=candidate.get('source','normalized_full_frame')
            previous_center=e[0];previous_wrist=wrist_uv;selected_polygon=candidate['polygon']
        observation_sources.append(source if e is not None else None);observation_polygons.append(selected_polygon if e is not None else None)
        real.append(e)
        reflected=wrists[i]-2*(wrists[i]@normal-plane)*normal;uv=project(reflected[None,:],focals[i],size)[0];options=[]
        for c in detections[i]['candidates']:
            if not mirror_enabled or c.get('view')=='real' or c['confidence']<.1:continue
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
    raw_targets=targets.copy();correction=smooth_vectors(targets-wrists,meta['fps'])-(targets-wrists)
    # Filter small palm noise without moving the grip arbitrarily far from the mesh hand.
    correction*=np.minimum(1,.015/np.maximum(np.linalg.norm(correction,axis=1,keepdims=True),1e-8))
    targets+=correction
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
        result.append(((rs[:,:,1]-grip_axes)*4).ravel())
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
    # Keep the solver estimates, then remove high-frequency angular acceleration on SO(3).
    before_stability=motion_metrics(rs);raw_rs=rs.copy()
    support=supported_frames([real[i] is not None and errors[i]<12 or mirror_obs[i] is not None for i in range(count)],meta['fps'],.6)
    rs=stable_rotations(rs,[2.5 if accepted[i] and errors[i]<5 else .7 for i in range(count)],meta['fps'])
    errors=[silhouette_rms(project(ring@rs[i].T+targets[i]-rs[i]@grip,focals[i],size),e) if e is not None else None for i,e in enumerate(real)]
    accepted=[e is not None and errors[i]<5 for i,e in enumerate(real)]
    rows=[]
    for i,R in enumerate(rs):
        t=targets[i]-R@grip;observed=accepted[i];uv=project(model_points(model)@R.T+t,focals[i],size);error=errors[i]
        jump=float(np.degrees(Rotation.from_matrix(rs[i-1].T@R).magnitude())) if i else 0
        reasons=[]
        if not observed:reasons.append('missing_or_rejected_observation')
        if error is not None and error>3:reasons.append('contour_residual')
        if jump>45:reasons.append('rotation_jump')
        if R[:,1]@grip_axes[i]<np.cos(np.radians(35)):reasons.append('hand_axis_conflict')
        rows.append({'frame':i,'palm_offset_m':offsets[i].tolist(),'grip_axis_error_deg':float(np.degrees(np.arccos(np.clip(R[:,1]@grip_axes[i],-1,1)))),'status':'fitted','ambiguous':True,'quality':'silhouette_fitted' if observed else 'temporal_estimate','source':'wilson_anatomical_grip_so3_v4','rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist(),'grip_target_camera_m':targets[i].tolist(),'observation_source':observation_sources[i],'raw_palm_offset_error_m':float(np.linalg.norm(targets[i]-raw_targets[i])),'stability_correction_deg':float(np.degrees(Rotation.from_matrix(raw_rs[i].T@R).magnitude())),'palm_anchor_error_m':float(np.linalg.norm(R@grip+t-targets[i])),'wrist_gap_m':float(np.linalg.norm(targets[i]-wrists[i])),'mask_fit_rms_px':round(error,2) if error is not None else None,'detection_confidence':confidence[i],'mirror_observation_used':mirror_obs[i] is not None,'rotation_step_deg':jump,'review_reasons':reasons,'projected_points':dict(zip(PARTS,uv.tolist())),'projected_head_outline':project(ring@R.T+t,focals[i],size).tolist()})
    calibration={'keyframes_1based':[i+1 for i in keys],'selection':'automatic clear-candidate selection; not manual ground truth','palm_offset_m':offset.tolist(),'grip_local_m':grip.tolist(),'method':'per-frame MCP/PIP grasp corridor; directed shaft prior; butt 45mm below contact; silhouette fit','requires_manual_confirmation':True,'status':'provisional_anatomical_prior' if np.allclose(offset,anatomical) else 'provisional_image_calibration','dimensions_measured':False,'signed_face_orientation_verified':False}
    for i,row in enumerate(rows):
        if not support[i]:
            row.update(status='missing_observation',quality='hidden')
        elif not accepted[i]:
            row['quality']='constrained_estimate'
        if observation_polygons[i] is not None:
            row['observed_polygon']=observation_polygons[i]
        row['face_angle_to_camera_deg']=float(np.degrees(np.arccos(np.clip(abs(rs[i,2,2]),0,1))))
    out={**old,'model':model,'method':'wilson_anatomical_grip_so3_v4','grasp_calibration':calibration,'frames':rows,'summary':{'frames':count,'observed':sum(r['status']=='fitted' and r['quality']=='silhouette_fitted' for r in rows),'hidden':sum(r['status']!='fitted' for r in rows),'hand':old['summary'].get('hand','right'),'interpolated':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'constrained_estimate':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'mirror_observations':sum(e is not None for e in mirror_obs),'mono_plane_ambiguity':True,'model_dimensions_measured':False}}
    out['summary'].update(motion_metrics(rs));out['stability']={'before':before_stability,'after':motion_metrics(rs),'method':'SO3 angular acceleration, confidence weighted; robust hand axes; .6s bracketed gap support','fps':meta['fps']}
    out['observation_resolution']=roi['original_size'] if roi else size
    visible=[r for r in rows if r['status']=='fitted']
    out['summary'].update(palm_anchor_max_mm=max(r['palm_anchor_error_m'] for r in visible)*1000,shaft_error_median_deg=float(np.median([r['grip_axis_error_deg'] for r in visible])),rotation_step_p95_deg=float(np.percentile([r['rotation_step_deg'] for r in visible],95)),multiview_hand_frames=multiview_frames)
    out['residual_pixel_units']='1280px canonical width; resolution-independent thresholds'
    for row in rows:
        row['projected_head_outline']=(np.asarray(row['projected_head_outline'])*pixel_scale).tolist()
        row['projected_points']={k:(np.asarray(v)*pixel_scale).tolist() for k,v in row['projected_points'].items()}
        if row.get('observed_polygon'):row['observed_polygon']=(np.asarray(row['observed_polygon'])*pixel_scale).tolist()
    write(root/'racket_poses.json',out);write(root/'wilson_grasp_calibration.json',calibration);write(root/'wilson_model.json',{'model':model})
    write(root/'grip_fit_manifest.json',{'video_sha256':meta['video_sha256'],'baseline_sha256':sha256(baseline),'poses_sha256':sha256(root/'racket_poses.json'),'code_sha256':sha256(Path(__file__)),'finished_at':now(),'summary':out['summary'],'mirror_enabled':mirror_enabled,'roi_observations_sha256':sha256(root/'racket_roi_candidates.json') if roi else None,'stability':out['stability']})
    print(json.dumps(out['summary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);a=p.parse_args();fit_dataset(a.dataset,a.result)
