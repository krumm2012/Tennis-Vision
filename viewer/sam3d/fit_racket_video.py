"""First-version monocular racket fitting from tracked silhouettes and SAM wrists.

Plane tilt remains ambiguous. Interpolated frames are explicitly labelled and
silhouette residuals must not be interpreted as independent 3D accuracy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from fit_racket_pose import DEFAULT_MODEL, PARTS, model_points, project
from track_racket_masks import select_candidate


def ring_points(model):
    theta=np.linspace(0,2*np.pi,32,endpoint=False)
    return np.column_stack((model['head_half_width_m']*np.cos(theta),
                            model['head_center_y_m']+(model['length_m']-model['head_center_y_m'])*np.sin(theta),
                            np.zeros(len(theta))))


def silhouette_rms(uv,ellipse):
    center,radii,angle=ellipse
    axes=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    error=(np.linalg.norm(((uv-center)@axes)/radii,axis=1)-1)*np.sqrt(np.prod(radii))
    return float(np.sqrt(np.mean(error**2)))


def head_ellipse(polygon):
    points = np.asarray(polygon, np.float32)
    if len(points) < 5:
        return None
    lo = np.floor(points.min(axis=0)).astype(int)-8
    hi = np.ceil(points.max(axis=0)).astype(int)+8
    if np.any(hi-lo > 350):
        return None
    mask = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), np.uint8)
    cv2.fillPoly(mask, [np.round(points-lo).astype(np.int32)], 255)
    # Remove the narrow handle before fitting the broad racket head.
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 12 or cv2.contourArea(contour) < 80:
        return None
    center, axes, angle = cv2.fitEllipse(contour)
    center = np.array(center)+lo
    radii = np.array(axes)/2
    if min(radii)<2.5 or max(radii)>85 or min(radii)/max(radii)<.08:
        return None
    return center, radii, np.radians(angle)


def initial_rotations(center, wrist_uv, wrist_camera, focal, model, previous):
    direction = center-wrist_uv
    magnitude = np.linalg.norm(direction)
    direction /= max(magnitude, 1e-6)
    projected_length = focal/wrist_camera[2]*(model['head_center_y_m']-model['grip_y_m'])
    planar = min(.999, magnitude/max(projected_length, 1e-6))
    rotations = [previous] if previous is not None else []
    for sign in [-1,1]:
        y = np.r_[direction*planar, sign*np.sqrt(1-planar**2)]
        x = np.array([direction[1], -direction[0], 0.])
        z = np.cross(x,y)
        base = np.column_stack((x,y,z))
        for twist in [-1.1, 0., 1.1]:
            rotations.append(base @ Rotation.from_rotvec([0,twist,0]).as_matrix())
    return rotations


def continuous_plane(matrix, previous):
    # An untextured symmetric head has the same silhouette after a 180-degree
    # turn about its shaft. Pick its representation before interpolation.
    if previous is None:return matrix
    alternatives=[matrix,matrix@np.diag([-1.,1.,-1.])]
    return min(alternatives,key=lambda r:Rotation.from_matrix(previous.T@r).magnitude())


def fit_shape(ellipse, wrist_camera, wrist_uv, focal, model, previous=None):
    center, radii, angle = ellipse
    uv_axes = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
    ring = ring_points(model)
    grip = np.array([0,model['grip_y_m'],0])
    local_center = np.array([0,model['head_center_y_m'],0])

    def values(params):
        matrix = Rotation.from_rotvec(params[:3]).as_matrix()
        translation = params[3:]
        uv = project(ring@matrix.T+translation, focal)
        normalized = ((uv-center)@uv_axes)/radii
        contour_error = (np.linalg.norm(normalized,axis=1)-1)*np.sqrt(np.prod(radii))
        projected_center = project((matrix@local_center+translation)[None,:],focal)[0]
        wrist_error = matrix@grip+translation-wrist_camera
        return matrix, translation, contour_error, projected_center-center, wrist_error

    def residual(params):
        _,_,edge,offset,wrist_error = values(params)
        return np.r_[edge, offset*2, wrist_error*100]

    candidates=[]
    for initial in initial_rotations(center,wrist_uv,wrist_camera,focal,model,previous):
        start=np.r_[Rotation.from_matrix(initial).as_rotvec(), wrist_camera-initial@grip]
        fit=least_squares(residual,start,max_nfev=65,method='lm')
        matrix,t,edge,offset,wrist_error=values(fit.x)
        error=float(np.sqrt(np.mean(edge**2)))
        gap=float(np.linalg.norm(wrist_error))
        distance=float(np.linalg.norm(offset))
        matrix=continuous_plane(matrix,previous)
        motion=Rotation.from_matrix(previous.T@matrix).magnitude() if previous is not None else 0.
        score=error+distance*.3+gap*25+motion*.8
        candidates.append((score,matrix,t,error,gap,distance))
    return min(candidates,key=lambda c:c[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'))
    args=parser.parse_args()
    read=lambda name:json.loads((args.data/name).read_text())
    masks=read('racket_tracked_masks.json')['frames']
    detections=read('racket_yolo_candidates.json')['frames']
    temporal=read('temporal_pose.json');meta=read('mesh_meta.json')
    model=dict(DEFAULT_MODEL)
    rows=[];poses={};previous=None
    wrists=np.array([np.array(meta['source_roots'][i])+f['raw'][41]
                     for i,f in enumerate(temporal['frames'])])
    for i,row in enumerate(masks):
        detection=select_candidate(detections[i])
        polygon=row['polygon'];source='sam2_mask'
        ellipse=head_ellipse(polygon) if row['mask_valid'] else None
        # A tracked mask can drift; a clear fresh detection can correct it.
        yolo_ellipse=head_ellipse(detection['polygon']) if detection else None
        if yolo_ellipse is not None and detection['confidence']>=.15:
            if ellipse is None or np.linalg.norm(ellipse[0]-yolo_ellipse[0])>25:
                ellipse=yolo_ellipse;source='yolo_mask'
        record={'frame':i,'detection_confidence':detection['confidence'] if detection else 0.,
                'mask_source':source,'quality':'unresolved'}
        if ellipse is not None:
            best=fit_shape(ellipse,wrists[i],np.array(temporal['frames'][i]['image'][41]),meta['focal'][i],model,previous)
            _,matrix,t,error,gap,offset=best
            record.update(mask_fit_rms_px=round(error,2),wrist_gap_m=round(gap,3),
                          head_ellipse={'center':ellipse[0].round(2).tolist(), 'radii':ellipse[1].round(2).tolist(),'angle':float(ellipse[2])})
            if error<5 and gap<.22 and offset<10 and t[2]>1:
                poses[i]=(matrix,t);previous=matrix;record['quality']='silhouette_fitted'
        rows.append(record)
        if i%25==24:print(f'{i+1}/250 shape fits; accepted {len(poses)}',flush=True)
    if len(poses)<2:
        raise RuntimeError('Too few valid shape fits; refusing to fabricate a full track')
    indices=np.array(sorted(poses));rotations=Rotation.from_matrix([poses[i][0] for i in indices])
    slerp=Slerp(indices,rotations)
    offsets=np.array([poses[i][1]+poses[i][0]@np.array([0,model['grip_y_m'],0])-wrists[i] for i in indices])
    all_rotations=[];all_offsets=[]
    for i in range(len(rows)):
        clamped=np.clip(i,indices[0],indices[-1]);matrix=slerp([clamped]).as_matrix()[0]
        offset=np.array([np.interp(i,indices,offsets[:,k]) for k in range(3)])
        all_rotations.append(matrix);all_offsets.append(offset)
    # Three-frame orientation/offset averaging suppresses per-frame fitting noise.
    for i,row in enumerate(rows):
        nearby=range(max(0,i-1),min(len(rows),i+2))
        matrix=Rotation.from_matrix([all_rotations[j] for j in nearby]).mean().as_matrix()
        offset=np.mean([all_offsets[j] for j in nearby],axis=0)
        t=wrists[i]+offset-matrix@np.array([0,model['grip_y_m'],0])
        projected_ring=project(ring_points(model)@matrix.T+t,meta['focal'][i])
        if 'head_ellipse' in row:
            observed=row['head_ellipse'];ellipse=(np.array(observed['center']),np.array(observed['radii']),observed['angle'])
            error=silhouette_rms(projected_ring,ellipse)
            if i in poses and error>row['mask_fit_rms_px']+1.5:
                matrix,t=poses[i]
                projected_ring=project(ring_points(model)@matrix.T+t,meta['focal'][i])
                offset=t+matrix@np.array([0,model['grip_y_m'],0])-wrists[i]
                error=silhouette_rms(projected_ring,ellipse)
            row['mask_fit_rms_px']=round(error,2)
        if i not in poses:
            row['quality']='interpolated'
            row['nearest_observation_frames']=int(np.min(abs(indices-i)))
        row.update(status='fitted',ambiguous=True,source='automatic_'+row['quality'],
                   translation_camera_m=t.round(7).tolist(),rotation_camera_columns=matrix.round(8).tolist(),
                   wrist_gap_m=round(float(np.linalg.norm(offset)),3))
        uv=project(model_points(model)@matrix.T+t,meta['focal'][i])
        row['projected_points']={name:uv[k].round(2).tolist() for k,name in enumerate(PARTS)}
        row['projected_head_outline']=projected_ring.round(2).tolist()
    for i,row in enumerate(rows):
        reasons=[]
        if row['quality']=='interpolated':reasons.append('missing_observation')
        if row.get('mask_fit_rms_px',0)>3:reasons.append('contour_residual')
        if row['detection_confidence']<.03:reasons.append('weak_detection')
        if i:
            prev=np.array(rows[i-1]['rotation_camera_columns'])
            current=np.array(row['rotation_camera_columns'])
            jump=float(np.degrees(Rotation.from_matrix(prev.T@current).magnitude()))
            row['rotation_step_deg']=round(jump,2)
            if jump>45:reasons.append('rotation_jump')
        row['review_reasons']=reasons
    summary={'frames':len(rows),'silhouette_fitted':len(poses),'interpolated':len(rows)-len(poses),
             'mono_plane_ambiguity':True,'model_dimensions_measured':False}
    out={'version':1,'video_id':'30.56','fps':25,'image_size':[1280,720],
         'coordinate_system':'SAM_source_camera_X_right_Y_down_Z_forward_m',
         'model':model,'method':'SAM2_silhouette_soft_SAM_wrist_temporal_first_version',
         'summary':summary,'frames':rows}
    (args.data/'racket_poses.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
