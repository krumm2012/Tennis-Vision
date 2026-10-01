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
    if (root/'racket_keypoints.json').exists() and not (root/'racket_poses_reference.json').exists():
        published=read('racket_poses.json')
        if all(r.get('status')=='fitted' and 'grip_target_camera_m' in r for r in published['frames']):
            if published['model'].get('asset_file')=='wilson_mesh_directional.bin':
                shutil.copy2(root/'wilson_mesh_directional.bin',root/'wilson_mesh_reference.bin');published['model']['asset_file']='wilson_mesh_reference.bin'
                write(root/'racket_poses_reference.json',published)
            else:shutil.copy2(root/'racket_poses.json',root/'racket_poses_reference.json')
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
    base_model=old['model'];model=base_model
    dimension_calibration=None
    if (root/'racket_dimensions.json').exists():
        from racket_calibration import validate,measured_model,resize_mesh
        dimension_calibration=validate(read('racket_dimensions.json'),meta);model=measured_model(base_model,dimension_calibration)
        if dimension_calibration['size_ready']:
            template=np.fromfile(root/'wilson_mesh.bin',dtype='<f4');pending=root/'wilson_mesh_directional.pending.bin';resize_mesh(template,base_model,model).tofile(pending);pending.replace(root/'wilson_mesh_directional.bin')
    mirror_enabled=bool(meta.get('mirror_available')) and (root/'mirror_geometry.json').exists()
    mirror=read('mirror_geometry.json') if mirror_enabled else {'normal_camera':[0,0,1],'distance_camera_m':0}
    count=len(frames);ring=np.array(model['head_outline']);grip=np.array([0,model['grip_y_m'],0]);roots=np.array(meta['source_roots']);focals=np.array(meta['focal'])/pixel_scale;palms=[palm_frame(f['raw']) for f in frames];wrists=np.array([a[0] for a in palms])+roots;hands=np.array([a[1] for a in palms]);normal=np.array(mirror['normal_camera'],dtype=float);normal/=np.linalg.norm(normal);plane=mirror['distance_camera_m']
    model=dict(model,grip_y_m=model.get('grip_y_m',.045) if dimension_calibration and dimension_calibration['size_ready'] else .045);grip=np.array([0.,model['grip_y_m'],0.])
    keypoints=read('racket_keypoints.json') if (root/'racket_keypoints.json').exists() else None
    if keypoints and (keypoints['video_sha256']!=meta['video_sha256'] or keypoints['image_size']!=meta['image_size'] or keypoints['fps']!=meta['fps'] or len(keypoints['frames'])!=count):raise ValueError('拍柄关键点与视频不一致')
    manual=read('racket_landmarks.json') if (root/'racket_landmarks.json').exists() else None
    if manual:
        from racket_landmarks import validate
        manual=validate(manual,meta)
    manual_rows={r['frame']:r for r in manual['frames']} if manual else {}
    landmark_names=['handle_end','throat','tip','head_center','rim_side','rim_opposite']
    landmark_objects=np.r_[model_points(model)[:3],[[0,model['head_center_y_m'],0]],model_points(model)[3:]]
    landmark_views=[];stereo_axes=[]
    for i in range(count):
        row=keypoints['frames'][i] if keypoints else {};observations=[]
        for view in ['real','mirror']:
            source=row.get(view,{});points={k:np.array(p)/pixel_scale for k,p in source.get('points',{}).items()};weights={k: min(1.,source.get('confidence',0)*3)*2.5/(source.get('uncertainty_px',{}).get(k,5)/pixel_scale) for k in points}
            reviewed=manual_rows.get(i,{});review_points=reviewed.get('points' if view=='real' else 'mirror_points',{})
            for name,p in review_points.items():points[name]=np.array(p)/pixel_scale;weights[name]=3.
            observations.append((points,weights))
        landmark_views.append(observations);stereo_axes.append(np.array(row['stereo_shaft']['direction_camera']) if row.get('stereo_shaft') else None)
    contacts=[grip_contact(f['raw']) for f in frames]
    offsets=np.array([c[0] for c in contacts]);grip_axes=np.array([hands[i]@c[1] for i,c in enumerate(contacts)])
    raw_axes=grip_axes.copy();anatomical_axes=smooth_vectors(raw_axes,meta['fps']);anatomical_axes/=np.maximum(np.linalg.norm(anatomical_axes,axis=1,keepdims=True),1e-8)
    from racket_direction import calibrate_axes
    grip_axes,axis_calibration=calibrate_axes(grip_axes,hands,stereo_axes,[view[0][0].get('head_center') for view in landmark_views],wrists+np.einsum('nij,nj->ni',hands,offsets),focals,size)
    grip_axes=smooth_vectors(grip_axes,meta['fps']);grip_axes/=np.maximum(np.linalg.norm(grip_axes,axis=1,keepdims=True),1e-8)
    hand_weights=np.ones(count)*4.;prior_image_errors=[None]*count
    for i in range(count):
        points=landmark_views[i][0][0]
        if 'head_center' in points:
            target=wrists[i]+hands[i]@offsets[i];uv=project(np.array([target,target+grip_axes[i]*.1]),focals[i],size)
            a=uv[1]-uv[0];b=points['head_center']-uv[0]
            angle=float(np.degrees(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8),-1,1))))
            prior_image_errors[i]=angle
            if angle>35:hand_weights[i]=.35
        if stereo_axes[i] is not None and stereo_axes[i]@grip_axes[i]<np.cos(np.radians(35)):hand_weights[i]=.35
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
        if keypoints and 'head_center' in landmark_views[i][0][0]:
            from racket_direction import fixed_grip_initial
            matrix=fixed_grip_initial(real[i],target,focals[i],size,model,*landmark_views[i][0],grip_axes[i],hand_weights[i],stereo_axes[i],previous)
        elif real[i] is not None:
            result=fit_shape(real[i],target,project(target[None,:],focals[i],size)[0],focals[i],model,previous,image_size=size)
            matrix=result[1]
        else:matrix=previous if previous is not None else np.array(next(r['rotation_camera_columns'] for r in old['frames'] if r['status']=='fitted'))
        initial.append(matrix);previous=matrix
    initial=np.array(initial)
    if manual_rows:
        import cv2
        for i,review in manual_rows.items():
            ids=[k for k,name in enumerate(landmark_names) if name in review['points']]
            if len(ids)<4:continue
            camera=np.array([[focals[i],0,size[0]/2],[0,focals[i],size[1]/2],[0,0,1.]])
            observed=np.array([review['points'][landmark_names[k]] for k in ids])/pixel_scale
            ok,rotations,translations,_=cv2.solvePnPGeneric(landmark_objects[ids],observed,camera,None,flags=cv2.SOLVEPNP_IPPE)
            if not ok:continue
            target=wrists[i]+hands[i]@offsets[i];options=[]
            for r,t in zip(rotations,translations):
                matrix=Rotation.from_rotvec(r.ravel()).as_matrix();pred=landmark_objects[ids]@matrix.T+target-matrix@grip
                if pred[:,2].min()<=.1:continue
                score=np.linalg.norm(project(pred,focals[i],size)-observed,axis=1).mean();options.append((score,matrix))
            if options:
                matrix=min(options,key=lambda a:a[0])[1]
                if review.get('face_correspondence_confirmed') and np.trace(initial[i].T@matrix)<np.trace((initial[i]@np.diag([-1.,1.,-1.])).T@matrix):initial=initial@np.diag([-1.,1.,-1.])
                initial[i]=matrix
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
    def robust(v):return np.sign(v)*np.sqrt(6*(np.sqrt(1+(v/3)**2)-1))
    def residual(x):
        rs=Rotation.from_rotvec(x.reshape(count,3)).as_matrix();result=[]
        for i,R in enumerate(rs):
            points=ring@R.T+targets[i]-R@grip
            result.append(robust(ellipse_residual(project(points,focals[i],size),real[i])) if accepted[i] else np.zeros(34))
            reflected=points-2*(points@normal-plane)[:,None]*normal
            result.append(robust(.3*ellipse_residual(project(reflected,focals[i],size),mirror_obs[i])) if mirror_obs[i] is not None else np.zeros(34))
            obj=landmark_objects@R.T+targets[i]-R@grip
            for view in range(2):
                observed,weights=landmark_views[i][view];camera=obj if view==0 else obj-2*(obj@normal-plane)[:,None]*normal
                uv=project(camera,focals[i],size);terms=np.zeros((6,2))
                for k,name in enumerate(landmark_names):
                    if name in observed:terms[k]=(uv[k]-observed[name])*weights[name]*(1. if view==0 else .5)*(6. if name=='head_center' else 1.)
                result.append(robust(terms).ravel())
            result.append((R[:,1]-grip_axes[i])*hand_weights[i])
            result.append((R[:,1]-stereo_axes[i])*12 if stereo_axes[i] is not None else np.zeros(3))
        result.append(((rs[1:]-rs[:-1])*20).ravel())
        result.append(((rs[2:]-2*rs[1:-1]+rs[:-2])*12).ravel())
        return np.concatenate(result)
    sparsity=lil_matrix((count*98+(count-1)*9+(count-2)*9,count*3),dtype=int)
    for i in range(count):sparsity[i*98:(i+1)*98,i*3:(i+1)*3]=1
    k=count*98
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
    rs=stable_rotations(rs,[2.5 if accepted[i] and errors[i]<5 else .7 for i in range(count)],meta['fps'],acceleration_strength=8 if keypoints else 5)
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
        rows.append({'frame':i,'palm_offset_m':offsets[i].tolist(),'grip_axis_error_deg':float(np.degrees(np.arccos(np.clip(R[:,1]@grip_axes[i],-1,1)))),'status':'fitted','ambiguous':True,'quality':'silhouette_fitted' if observed else 'temporal_estimate','source':'wilson_directional_landmarks_v5' if keypoints else 'wilson_anatomical_grip_so3_v4','rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist(),'grip_target_camera_m':targets[i].tolist(),'observation_source':observation_sources[i],'raw_palm_offset_error_m':float(np.linalg.norm(targets[i]-raw_targets[i])),'stability_correction_deg':float(np.degrees(Rotation.from_matrix(raw_rs[i].T@R).magnitude())),'palm_anchor_error_m':float(np.linalg.norm(R@grip+t-targets[i])),'wrist_gap_m':float(np.linalg.norm(targets[i]-wrists[i])),'mask_fit_rms_px':round(error,2) if error is not None else None,'detection_confidence':confidence[i],'mirror_observation_used':mirror_obs[i] is not None,'rotation_step_deg':jump,'review_reasons':reasons,'projected_points':dict(zip(PARTS,uv.tolist())),'projected_head_outline':project(ring@R.T+t,focals[i],size).tolist()})
        reviewed=manual_rows.get(i,{});manual_errors=[]
        for name,p in reviewed.get('points',{}).items():
            k=landmark_names.index(name);pred=project((landmark_objects[k]@R.T+t)[None],focals[i],size)[0];manual_errors.append(float(np.linalg.norm(pred-np.asarray(p)/pixel_scale)))
        manual_rms=float(np.sqrt(np.mean(np.square(manual_errors)))) if manual_errors else None
        face_verified=bool(reviewed.get('face_correspondence_confirmed') and manual_rms is not None and manual_rms<4 and np.linalg.norm(uv[3]-uv[4])>3)
        rows[-1].update(hand_prior_weight=float(hand_weights[i]),hand_prior_image_error_deg=prior_image_errors[i],stereo_shaft_used=stereo_axes[i] is not None,normal_camera=R[:,2].tolist(),signed_face_angle_to_camera_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),physical_face_sign_verified=face_verified,reviewed_landmark_rms_px=manual_rms,observed_keypoints={name:(p*pixel_scale).tolist() for name,p in landmark_views[i][0][0].items()})
        rows[-1]['raw_anatomical_axis_error_deg']=float(np.degrees(np.arccos(np.clip(R[:,1]@anatomical_axes[i],-1,1))))
        head=R@np.array([0,model['head_center_y_m'],0])+t
        rows[-1]['projected_face_normal']=(project(np.array([head,head+R[:,2]*.12]),focals[i],size)*pixel_scale).tolist()
        rows[-1]['normal_gauge_status']='manual_marked_side' if face_verified else 'continuous_model_normal_unverified'
        seen=landmark_views[i][0][0].get('head_center')
        if seen is not None:
            pair=project(np.array([targets[i],head]),focals[i],size);a=pair[1]-pair[0];b=seen-pair[0]
            direction_error=float(np.degrees(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8),-1,1))));head_error=float(np.linalg.norm(pair[1]-seen))
            rows[-1].update(shaft_image_error_deg=direction_error,head_center_error_canonical_px=head_error)
            if direction_error>25:reasons.append('shaft_image_conflict')
            if head_error>25:reasons.append('head_center_residual')
        if hand_weights[i]<1 and 'hand_axis_conflict' in reasons:reasons[reasons.index('hand_axis_conflict')]='unreliable_hand_prior_downweighted'
    calibration={'keyframes_1based':[i+1 for i in keys],'selection':'automatic clear-candidate selection; not manual ground truth','palm_offset_m':offset.tolist(),'grip_local_m':grip.tolist(),'method':'per-frame MCP/PIP grasp corridor; directed shaft prior; butt 45mm below contact; silhouette fit','requires_manual_confirmation':True,'status':'provisional_anatomical_prior' if np.allclose(offset,anatomical) else 'provisional_image_calibration','dimensions_measured':bool(model.get('dimensions_measured')),'grip_style':dimension_calibration['grip_style'] if dimension_calibration else 'unknown','physical_grip_bevel_verified':False,'signed_face_orientation_verified':False}
    for i,row in enumerate(rows):
        if not support[i]:
            row.update(status='missing_observation',quality='hidden')
        elif not accepted[i]:
            row['quality']='constrained_estimate'
        if observation_polygons[i] is not None:
            row['observed_polygon']=observation_polygons[i]
        row['face_angle_to_camera_deg']=float(np.degrees(np.arccos(np.clip(abs(rs[i,2,2]),0,1))))
    out={**old,'model':model,'method':'wilson_anatomical_grip_so3_v4','grasp_calibration':calibration,'frames':rows,'summary':{'frames':count,'observed':sum(r['status']=='fitted' and r['quality']=='silhouette_fitted' for r in rows),'hidden':sum(r['status']!='fitted' for r in rows),'hand':old['summary'].get('hand','right'),'interpolated':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'constrained_estimate':sum(r['status']=='fitted' and r['quality']=='constrained_estimate' for r in rows),'mirror_observations':sum(e is not None for e in mirror_obs),'mono_plane_ambiguity':True,'model_dimensions_measured':bool(model.get('dimensions_measured'))}}
    out['summary'].update(motion_metrics(rs));out['stability']={'before':before_stability,'after':motion_metrics(rs),'method':'SO3 angular acceleration, confidence weighted; robust hand axes; .6s bracketed gap support','fps':meta['fps']}
    out['observation_resolution']=roi['original_size'] if roi else size
    visible=[r for r in rows if r['status']=='fitted']
    out['summary'].update(palm_anchor_max_mm=max(r['palm_anchor_error_m'] for r in visible)*1000,shaft_error_median_deg=float(np.median([r['grip_axis_error_deg'] for r in visible])),rotation_step_p95_deg=float(np.percentile([r['rotation_step_deg'] for r in visible],95)),multiview_hand_frames=multiview_frames)
    out['summary'].update(hand_prior_downweighted=int(np.count_nonzero(hand_weights<1)),stereo_shaft_frames=sum(a is not None for a in stereo_axes),face_correspondence_confirmed_frames=sum(r['physical_face_sign_verified'] for r in rows))
    out['method']='wilson_directional_landmarks_v5' if keypoints else out['method']
    out['directional_evidence']=keypoints['summary'] if keypoints else None
    out['hand_axis_calibration']=axis_calibration
    out['summary']['raw_anatomical_axis_error_median_deg']=float(np.median([r['raw_anatomical_axis_error_deg'] for r in visible]))
    out['summary']['direction_review_frames']=sum('shaft_image_conflict' in r['review_reasons'] or 'head_center_residual' in r['review_reasons'] for r in rows)
    out['stability']['acceleration_strength']=8 if keypoints else 5
    out['residual_pixel_units']='1280px canonical width; resolution-independent thresholds'
    for row in rows:
        row['projected_head_outline']=(np.asarray(row['projected_head_outline'])*pixel_scale).tolist()
        row['projected_points']={k:(np.asarray(v)*pixel_scale).tolist() for k,v in row['projected_points'].items()}
        if row.get('observed_polygon'):row['observed_polygon']=(np.asarray(row['observed_polygon'])*pixel_scale).tolist()
    write(root/'racket_poses.json',out);write(root/'wilson_grasp_calibration.json',calibration);write(root/'wilson_model.json',{'model':model})
    publication=None
    if keypoints:
        from racket_quality import gate
        publication=gate(root)
    write(root/'grip_fit_manifest.json',{'video_sha256':meta['video_sha256'],'baseline_sha256':sha256(baseline),'poses_sha256':sha256(root/'racket_poses.json'),'code_sha256':sha256(Path(__file__)),'finished_at':now(),'candidate_summary':out['summary'],'summary':read('racket_poses.json')['summary'],'publication':publication,'mirror_enabled':mirror_enabled,'roi_observations_sha256':sha256(root/'racket_roi_candidates.json') if roi else None,'stability':out['stability']})
    print(json.dumps({'candidate':out['summary'],'publication':publication['status'] if publication else None,'published':read('racket_poses.json')['summary']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);a=p.parse_args();fit_dataset(a.dataset,a.result)
