"""Version-bound manual observations and lower-weight dense contour evidence."""
import json
from pathlib import Path
from run_records import sha256


def automatic_rows(result,video_sha,heldout=()):
    root=Path(result);path=root/'racket_keypoints.json'
    if not path.exists():return {}
    kp=json.loads(path.read_text());meta=json.loads((root/'mesh_meta.json').read_text())
    selection=json.loads((root/'racket_review_frames.json').read_text())
    if kp['video_sha256']!=video_sha or kp.get('fps')!=meta['fps'] or kp.get('image_size')!=meta['image_size'] or len(kp['frames'])!=meta['frames'] or selection.get('keypoints_sha256')!=sha256(path):raise ValueError('自动观测与视频/版本不一致')
    scale=meta['image_size'][0]/1280;rows={}
    for i,source in enumerate(kp['frames']):
        if i in heldout:continue
        row={'frame':i,'source':'automatic_contour_unverified','points':{},'mirror_points':{},'weights':{},'face_correspondence_confirmed':False}
        for view,target in [('real','points'),('mirror','mirror_points')]:
            obs=source.get(view,{})
            confidence=float(obs.get('confidence',0))
            if confidence<.2:continue
            points=obs.get('points',{})
            if len(points)<2:continue
            row[target]=points
            row['weights'][target]={name:min(.3,.3*confidence)/max(1.,float(obs.get('uncertainty_px',{}).get(name,12))/scale/3) for name in points}
        if row['points'] or row['mirror_points']:rows[i]=row
    return rows


def assemble(result,readiness):
    root=Path(result);heldout=set(readiness['heldout_frames'])
    rows=automatic_rows(root,readiness['video_sha256'],heldout) if readiness.get('automatic_evidence_permitted') else {}
    path=root/'racket_landmarks.json'
    if path.exists():
        from racket_landmarks import validate
        meta=json.loads((root/'mesh_meta.json').read_text())
        reviewed=validate(json.loads(path.read_text()),meta)
        for row in reviewed['frames']:
            # A reviewed frame is authoritative, including missing/occluded points.
            rows[row['frame']]={**row,'weights':{view:{name:1. for name in row[view]} for view in ['points','mirror_points']}}
    # Old automatic-only preflights retain their selected heldout points for metrics.
    if readiness['validation_source']=='automatic_contours_unverified':
        for i,row in automatic_rows(root,readiness['video_sha256']).items():
            if i in heldout and i not in rows:rows[i]=row
    return {'frames':[rows[i] for i in sorted(rows)]}
