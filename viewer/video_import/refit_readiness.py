"""Explicit prerequisites for measured grip contact and native full-body refitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from mhr_parameters import available
from racket_calibration import validate as dimensions
from racket_landmarks import validate as landmarks
from run_records import write, sha256


def assess(result, archive, allow_assumed=False, allow_automatic=False):
    out = Path(result); archive = Path(archive)
    meta = json.loads((out/'mesh_meta.json').read_text())
    size = dimensions(json.loads((out/'racket_dimensions.json').read_text()), meta) if (out/'racket_dimensions.json').exists() else None
    marks = landmarks(json.loads((out/'racket_landmarks.json').read_text()), meta)['frames'] if (out/'racket_landmarks.json').exists() else []
    required = {'handle_end','tip','rim_side','rim_opposite'}
    clear = [r['frame'] for r in marks if required <= set(r['points'])]
    mirror = [r['frame'] for r in marks if {'tip','rim_side','rim_opposite'} <= set(r['mirror_points'])]
    with np.load(archive, allow_pickle=False) as d:
        native = available(d, meta['frames'])
        if native and ('video_sha256' not in d or str(d['video_sha256'])!=meta['video_sha256']):raise ValueError('原生人体参数与视频不一致')
    automatic=[]
    if allow_automatic and (out/'racket_review_frames.json').exists():
        selected=json.loads((out/'racket_review_frames.json').read_text())
        if selected['video_sha256']!=meta['video_sha256']:raise ValueError('自动清晰帧属于其他视频')
        automatic=[r['frame'] for r in selected['frames'] if 'real' in r['views']]
    provisional=allow_automatic and len(automatic)>=8
    reasons = []
    complete=bool(size and {'length','head_width','head_height'}<=set(size['dimensions_cm']))
    if not size or not (size['size_ready'] or allow_assumed and complete):reasons.append('measured_racket_dimensions_missing')
    if len(clear) < 6 and not provisional:reasons.append('need_six_clear_reviewed_racket_frames')
    if len(clear)>1 and (max(clear)-min(clear))/meta['fps'] < min(1., meta['frames']/meta['fps']*.5):reasons.append('reviewed_frames_need_temporal_diversity')
    if meta.get('mirror_available') and len(mirror)<2 and not provisional:reasons.append('need_two_reviewed_mirror_racket_frames')
    if not native:reasons.append('native_mhr_parameters_missing_rerun_required')
    evidence=clear if len(clear)>=6 else automatic
    heldout=evidence[::5]
    pose_path=out/'racket_poses_directional.json' if (out/'racket_poses_directional.json').exists() else out/'racket_poses.json'
    if pose_path.exists():
        model=json.loads(pose_path.read_text()).get('model',{})
        if model.get('grip_position_source')=='manual_image_estimate' and set(model.get('grip_calibration_frames',[]))&set(heldout):
            reasons.append('grip_calibration_includes_heldout_recalibrate_with_training_frames')
    training=set(evidence)|{r['frame'] for r in marks}
    if allow_automatic and (out/'racket_keypoints.json').exists():
        from refit_observations import automatic_rows
        training.update(automatic_rows(out,meta['video_sha256'],heldout))
    return {'schema_version':1,'video_sha256':meta['video_sha256'],'ready':not reasons,
            'blocked_by':reasons,'reviewed_real_frames':clear,'reviewed_mirror_frames':mirror,
            'automatic_evidence_permitted':bool(allow_automatic),'automatic_frames':automatic,'validation_source':'reviewed_landmarks' if len(clear)>=6 else 'automatic_contours_unverified',
            'train_frames':sorted(training-set(heldout)), 'heldout_frames':heldout,
            'assumed_dimensions_permitted':bool(allow_assumed),'size_measured':bool(size and size['size_ready']),'native_mhr_parameters_available':native,
            'grip_style':size['grip_style'] if size else 'unknown','physical_grip_bevel_verified':False,
            'camera_calibrated':False,'full_body_joint_fit_completed':False,
            'archive_sha256':sha256(archive)}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();r=assess(a.result,a.archive);write(a.result/'fullbody_refit_readiness.json',r)
    print(json.dumps(r,ensure_ascii=False))
