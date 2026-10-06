"""Image-derived manual grip calibration; keep body/depth and units explicit."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

from racket_landmarks import validate, usable_grip
from correct_reviewed_racket_frame import fit_manual_frame
from audit_racket_evidence_boundaries import projections, NAMES
from run_records import sha256, write


def settings(root, meta):
    path=Path(root)/'racket_grip_settings.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if any(value.get(k)!=meta[k] for k in ['video_sha256','image_size','fps']):
        raise ValueError('固定握点设置属于其他视频')
    g=value.get('grip_from_butt_m')
    if isinstance(g,bool) or not isinstance(g,(int,float)) or not np.isfinite(g) or not .01<=g<=.20:
        raise ValueError('固定握点距离须为1–20cm')
    if value.get('source')!='user_specified':raise ValueError('固定握点设置需明确用户来源')
    return value


def apply_settings(calibration, value):
    """Preserve image diagnostics while respecting a prescribed grasp position."""
    result=copy.deepcopy(calibration)
    if value:
        result.update(image_calibration_ready=result['calibration_ready'],calibration_ready=True,
                      status='user_fixed_grip',applied_grip_from_butt_m=float(value['grip_from_butt_m']),
                      grip_position_source='user_specified',fixed_grip_requested=True,distance_measured=False)
    return result


def estimate(labels, poses, meta, heldout=()):
    """Infer axial grip position from real reviewed pixels, never held-out rows.

    Five racket landmarks estimate a rigid pose independently of the hand.
    Then the reviewed grip pixel constrains a one-dimensional handle position.
    Asset dimensions and intrinsics remain estimates; this is not measurement.
    """
    excluded=set(heldout); records=[]; scale=meta['image_size'][0]/1280
    required=set(NAMES[:5])
    for row in labels['frames']:
        i=row['frame']
        if not usable_grip(row):continue
        item={'frame':i,'used':False}
        if i in excluded:
            records.append({**item,'reason':'heldout_excluded'});continue
        if not required<=set(row['points']):
            records.append({**item,'reason':'need_five_visible_real_racket_points'});continue
        pose=poses['frames'][i]
        if pose.get('status')!='fitted':
            records.append({**item,'reason':'missing_pose_initialization'});continue
        r0=np.asarray(pose['rotation_camera_columns']);t0=np.asarray(pose['translation_camera_m'])
        clean=copy.deepcopy(row);clean['points'].pop('grip_center');clean['grip_confirmed']={}
        fitted=fit_manual_frame(clean,poses['model'],meta,None,r0,t0,'real_manual',np.asarray(pose.get('grip_target_camera_m',t0+r0@np.array([0,poses['model']['grip_y_m'],0]))))
        r=np.asarray(fitted['rotation_camera_columns']);t=np.asarray(fitted['translation_camera_m']);seen=np.asarray(row['points']['grip_center'])
        def error(g):
            p=t+r@np.array([0,g,0]);uv=p[:2]/p[2]*meta['focal'][i]+np.array(meta['image_size'])/2
            return float(np.sum((uv-seen)**2))
        fit=minimize_scalar(error,bounds=(.01,.20),method='bounded',options={'xatol':1e-9})
        residual=float(np.sqrt(fit.fun));rms=fitted['metrics']['points']['rms_original_px']
        good=fitted['fit_converged'] and fit.success and .0101<fit.x<.1999 and residual<=4*scale and rms<=8*scale
        records.append({**item,'used':bool(good),'reason':'image_estimate' if good else 'grip_or_racket_reprojection_conflict',
            'grip_from_butt_m':float(fit.x),'grip_residual_original_px':residual,'racket_rms_original_px':rms})
    samples=[r['grip_from_butt_m'] for r in records if r['used']]
    distance=float(np.median(samples)) if samples else None
    spread=float(np.max(np.abs(np.asarray(samples)-distance))) if samples else None
    consistent=bool(samples and spread<=.025)
    status=('image_estimate_single_frame' if len(samples)==1 else 'image_estimate_multi_frame') if consistent else 'inconsistent_grip_estimates' if samples else 'no_usable_confirmed_grip'
    return {'status':status,'calibration_ready':consistent,'estimated_grip_from_butt_m':distance,
        'sample_spread_max_m':spread,'frames':[r['frame'] for r in records if r['used']], 'records':records,
        'heldout_excluded':sorted(excluded),'distance_measured':False,'dimensions_measured':bool(poses['model'].get('dimensions_measured')),
        'camera_calibrated':False,'mirror_used_to_estimate_distance':False,'accepted':False,
        'definition':'Center of the hand grasp region projected onto the handle axis; not wrist or butt cap',
        'limits':'Distance inferred from real reviewed pixels and estimated asset/camera. Mirror grip reported separately; uncertain mirror calibration not used to estimate distance. Hand depth remains an estimated prior, not measured contact.'}


def manual_anchor(row, prior, meta):
    """Apply a confirmed real grip ray at prior depth, without editing the body."""
    prior=np.asarray(prior,float)
    if not usable_grip(row):return prior.copy()
    uv=np.asarray(row['points']['grip_center']);i=row['frame']
    return np.r_[(uv-np.asarray(meta['image_size'])/2)/meta['focal'][i]*prior[2],prior[2]]


def candidate(labels,poses,meta,mirror,calibration):
    result=copy.deepcopy(poses);model=result['model'];edited=[]
    if calibration['calibration_ready']:
        prescribed=calibration.get('fixed_grip_requested',False)
        model.update(grip_y_m=calibration.get('applied_grip_from_butt_m',calibration['estimated_grip_from_butt_m']),grip_position_measured=False,
            grip_position_source='user_specified' if prescribed else 'manual_image_estimate',
            grip_calibration_frames=[] if prescribed else calibration['frames'])
        grip=np.array([0,model['grip_y_m'],0])
        for pose in result['frames']:
            if pose.get('status')=='fitted':
                pose['grip_target_camera_m']=(np.asarray(pose['rotation_camera_columns'])@grip+pose['translation_camera_m']).tolist()
        for row in labels['frames']:
            i=row['frame']
            if i not in calibration['frames']:continue
            old=poses['frames'][i];pose=result['frames'][i];r=np.array(old['rotation_camera_columns']);t=np.array(old['translation_camera_m'])
            prior=np.array(old.get('grip_target_camera_m',r@np.array([0,poses['model']['grip_y_m'],0])+t));anchor=manual_anchor(row,prior,meta)
            fit=fit_manual_frame(row,model,meta,mirror,r,t,'real_manual_fixed_grip',anchor)
            r=np.array(fit['rotation_camera_columns']);t=np.array(fit['translation_camera_m'])
            uv=projections(r,t,model,meta,i);ring=np.asarray(model['head_outline'])@r.T+t
            pose.update(rotation_camera_columns=r.tolist(),translation_camera_m=t.tolist(),grip_target_camera_m=anchor.tolist(),
                projected_points=dict(zip(NAMES,uv.tolist())),projected_head_outline=(ring[:,:2]/ring[:,2:]*meta['focal'][i]+np.array(meta['image_size'])/2).tolist(),
                source='manual_grip_image_calibration_candidate',observed_keypoints=row['points'],quality='constrained_estimate',normal_camera=r[:,2].tolist(),
                face_angle_to_camera_deg=float(np.degrees(np.arccos(np.clip(abs(r[2,2]),0,1)))),physical_face_sign_verified=False,
                manual_grip_metrics=fit['metrics'],manual_grip_anchor_shift_mm=float(np.linalg.norm(anchor-prior)*1000),
                review_reasons=['manual_grip_training_edit','depth_prior_unmeasured','body_contact_unvalidated','temporal_unvalidated'])
            for stale in ['grip_axis_error_deg','projected_face_normal','reviewed_landmark_rms_px']:pose.pop(stale,None)
            edited.append(i)
    result.update(method='manual_grip_image_calibration_preview',accepted=False,grip_calibration=calibration)
    # Cached projections must follow the applied model on every frame, including
    # frames whose rigid pose was left unchanged by the manual correction.
    for i,pose in enumerate(result['frames']):
        if pose.get('status')!='fitted':continue
        r=np.asarray(pose['rotation_camera_columns']);t=np.asarray(pose['translation_camera_m'])
        uv=projections(r,t,model,meta,i);ring=np.asarray(model['head_outline'])@r.T+t
        pose.update(projected_points=dict(zip(NAMES,uv.tolist())),
                    projected_head_outline=(ring[:,:2]/ring[:,2:]*meta['focal'][i]+np.asarray(meta['image_size'])/2).tolist(),
                    grip_target_camera_m=(r@np.array([0,model['grip_y_m'],0])+t).tolist())
    result['summary']['manual_grip_calibrated_frames']=edited
    if len(result['frames'])>=3:
        from racket_stability import motion_metrics
        result['summary'].update(motion_metrics(np.array([p['rotation_camera_columns'] for p in result['frames']])))
    return result


def save(result,value,expected_sha=None):
    root=Path(result);path=root/'racket_landmarks.json'
    if expected_sha is not None and (sha256(path) if path.exists() else None)!=expected_sha:
        raise ValueError('标注已被其他页面修改，请重新加载后再保存')
    meta=json.loads((root/'mesh_meta.json').read_text());poses=json.loads((root/'racket_poses.json').read_text())
    if poses['video_sha256']!=meta['video_sha256'] or len(poses['frames'])!=meta['frames']:raise ValueError('握点校准与球拍来源不一致')
    labels=validate(value,meta);calibration=apply_settings(estimate(labels,poses,meta),settings(root,meta))
    mirror=json.loads((root/'mirror_geometry.json').read_text()) if meta.get('mirror_available') else None
    if mirror and mirror['video_sha256']!=meta['video_sha256']:raise ValueError('镜面来源不一致')
    preview=candidate(labels,poses,meta,mirror,calibration)
    calibration.update(video_sha256=meta['video_sha256'],image_size=meta['image_size'],fps=meta['fps'],
        source_poses_sha256=sha256(root/'racket_poses.json'),source_meta_sha256=sha256(root/'mesh_meta.json'),
        source_mirror_sha256=sha256(root/'mirror_geometry.json') if mirror else None,
        validation_role='manual_training_edits_not_heldout_accuracy',body_modified=False,main_poses_modified=False,code_sha256=sha256(Path(__file__)))
    preview['grip_calibration']=calibration
    # Complete calculation before writing annotations; a fit failure preserves
    # current annotations and every published body/racket file.
    write(path,labels);calibration['landmarks_sha256']=sha256(path)
    write(root/'racket_grip_calibration.json',calibration);write(root/'racket_poses_grip_calibrated.json',preview)
    return {'saved':True,'landmarks_sha256':sha256(path),**calibration}
