"""Read-only grip-definition and mirror/intrinsic sensitivity diagnostics.

The grip sweep fits reviewed pixels; it is a compatibility probe, not a
measurement or held-out pose evaluation. Camera fits use body training frames
only. No viewer, human annotation or calibration is modified.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from audit_mirror_camera import BODY_PAIRS, CORE, normal, paired_points, stats
from correct_reviewed_racket_frame import fit_manual_frame
from racket_landmarks import validate
from run_records import sha256, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sam3d'))
from fit_wilson_grip import palm_frame, grip_contact


def project(points, focal, principal):
    points = np.asarray(points)
    if np.any(points[..., 2] <= .1):
        raise ValueError('Nonpositive camera depth')
    return points[..., :2]/points[..., 2:]*np.asarray(focal)[..., None]+principal


def line_grip_probe(anchor, axis, butt_pixel, focal, principal):
    """Closest ray/shaft solution; an assumed axis/depth makes this a proxy."""
    ray = np.r_[(np.asarray(butt_pixel)-principal)/focal, 1.]
    values = np.linalg.lstsq(np.column_stack([axis, ray]), anchor, rcond=None)[0]
    butt = np.asarray(anchor)-values[0]*axis
    return {'implied_grip_from_butt_m': float(values[0]),
            'ray_depth_m': float(values[1]),
            'ray_shaft_gap_mm': float(np.linalg.norm(butt-values[1]*ray)*1000)}


def camera_projection(points, focals, q, principal):
    n = normal(q); reflected = points-2*(points@n-q[2])[:, None]*n
    uv = reflected[:, :2]/np.maximum(reflected[:, 2:], .1)*focals[:, None]*np.exp(q[3])+principal+q[4:6]
    return uv, reflected


def epipolar_errors(pair, focal, principal, plane_normal):
    n = np.asarray(plane_normal)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    e = cross@(np.eye(3)-2*np.outer(n, n))
    a = np.r_[(np.asarray(pair['real'])-principal)/focal, 1.]
    b = np.r_[(np.asarray(pair['mirror'])-principal)/focal, 1.]
    lr, lm = e@b, e.T@a
    value = a@lr*focal
    return np.abs([value/np.linalg.norm(lr[:2]), value/np.linalg.norm(lm[:2])])


def camera_sensitivity(points, observed, frames, core, focals, size, geometry, manual_groups, frame_focals):
    points, observed = np.asarray(points), np.asarray(observed)
    frames, core = np.asarray(frames), np.asarray(core, bool)
    focals, principal = np.asarray(focals), np.asarray(size)/2
    train = core & (frames % 5 != 0)
    heldout = core & (frames % 5 == 0)
    if train.sum() < 40 or heldout.sum() < 10:
        raise ValueError('Insufficient disjoint body calibration samples')
    n = np.asarray(geometry['normal_camera'])
    q0 = np.array([math.atan2(n[0], n[2]), math.asin(n[1]), geometry['distance_camera_m'], 0., 0., 0.])
    candidates = {'baseline': q0}
    bounds = np.array([np.radians(5), np.radians(5), 2., math.log(1.15), size[0]*.05, size[1]*.05])
    details = {}
    for mode, ids in [('plane_only', [0, 1, 2]), ('plane_focal', [0, 1, 2, 3]), ('plane_focal_principal', list(range(6)))]:
        def unpack(x):
            q = q0.copy(); q[ids] = x
            return q
        def residual(x):
            uv, reflected = camera_projection(points[train], focals[train], unpack(x), principal)
            return np.r_[(uv-observed[train]).ravel(), np.minimum(reflected[:, 2]-.1, 0)*1000]
        fit = least_squares(residual, q0[ids], bounds=(q0[ids]-bounds[ids], q0[ids]+bounds[ids]), loss='soft_l1', f_scale=6., max_nfev=300)
        candidates[mode] = unpack(fit.x)
        details[mode] = {'converged': bool(fit.success), 'bounds_hit': bool(np.any(np.abs(fit.active_mask) > 0))}
    results = {}
    for mode, q in candidates.items():
        uv, reflected = camera_projection(points, focals, q, principal)
        errors = np.linalg.norm(uv-observed, axis=1)
        groups = {}
        for name, pairs in manual_groups.items():
            rows = [{'frame': row['frame'], 'part': row['part'], 'symmetric_epipolar_error_original_px':
                epipolar_errors(row, frame_focals[row['frame']]*np.exp(q[3]), principal+q[4:6], normal(q)).tolist()} for row in pairs]
            groups[name] = {'records': rows, **stats([max(r['symmetric_epipolar_error_original_px']) for r in rows])}
        # Real roundtrip change is only a self-consistency diagnostic against
        # frozen SAM geometry, not independent body-image accuracy.
        real_shift = np.linalg.norm(project(points, focals*np.exp(q[3]), principal+q[4:6])-project(points, focals, principal), axis=1)
        results[mode] = {**details.get(mode, {}), 'normal_camera': normal(q).tolist(), 'distance_camera_m': float(q[2]),
            'focal_scale': float(np.exp(q[3])), 'principal_offset_original_px': q[4:6].tolist(),
            'body_train_original_px': stats(errors[train]), 'body_core_heldout_original_px': stats(errors[heldout]),
            'body_arms_heldout_original_px': stats(errors[(~core)&(frames % 5 == 0)]),
            'native_real_projection_change_original_px': stats(real_shift), 'manual_pair_groups': groups,
            'minimum_reflected_depth_m': float(reflected[:, 2].min()), 'accepted': False}
    return {'train_frames': sorted(set(frames[train].tolist())), 'heldout_frames': sorted(set(frames[heldout].tolist())),
            'manual_labels_used_for_fitting': False, 'candidates': results,
            'limits': 'Frozen SAM body is itself estimated; focal/principal changes may compensate body errors. No distortion calibration, measured mirror or independent camera target. Mirror distance cannot repair epipolar point-line discrepancy.'}


def audit(dataset, staged, native, body_candidate, labels_path, new_labels_path, output):
    if output.exists():
        raise ValueError('Use fresh audit directory')
    live = dataset/'result'
    paths = {'video': dataset/'source.mp4', 'poses': live/'racket_poses.json', 'live_meta': live/'mesh_meta.json',
        'live_plane': live/'mirror_geometry.json', 'meta': staged/'mesh_meta.json', 'plane': staged/'mirror_geometry.json',
        'mirror_pose': staged/'mirror_pose.json', 'labels': labels_path, 'new_labels': new_labels_path,
        'native': native, 'body_candidate': body_candidate}
    data = {name: json.loads(p.read_text()) for name, p in paths.items() if p.suffix == '.json'}
    identity = sha256(paths['video']); meta = data['meta']; live_meta = data['live_meta']; poses = data['poses']
    if any(v['video_sha256'] != identity for v in data.values()):
        raise ValueError('Video identity mismatch')
    old, new = validate(data['labels'], meta), validate(data['new_labels'], meta)
    with np.load(native, allow_pickle=False) as archive:
        if str(archive['video_sha256']) != identity or not np.array_equal(archive['focal'], meta['focal']) or not np.array_equal(archive['source_roots'], meta['source_roots']):
            raise ValueError('Frozen camera/native mismatch')
        joints, roots = archive['joints'].copy(), archive['source_roots'].copy()
    report_path = body_candidate.with_name('refit_report.json')
    provenance = json.loads(report_path.read_text())
    if provenance['video_sha256'] != identity or provenance['candidate_sha256'] != sha256(body_candidate):
        raise ValueError('Body candidate identity mismatch')
    paths['body_report'] = report_path
    with np.load(body_candidate, allow_pickle=False) as archive:
        body_joints, body_roots = archive['joints'].copy(), archive['source_roots'].copy()
        preview_path = live/'joint_preview_manifest.json'
        preview = json.loads(preview_path.read_text())
        if preview['candidate_sha256'] != sha256(body_candidate) or any(sha256(live/name) != value for name,value in preview['files'].items()):
            raise ValueError('Viewer/body preview manifest mismatch')
        displayed = np.memmap(live/'mesh_refined.bin', dtype='<f4', mode='r', shape=archive['vertices'].shape)
        display_error = max(float(np.max(np.abs(displayed[i]+np.asarray(live_meta['source_roots'][i])-(archive['vertices'][i]+body_roots[i])))) for i in range(len(body_roots)))
        if display_error > 1e-5:
            raise ValueError('Viewer world mesh differs from body candidate')
        paths['preview_manifest'] = preview_path
        if not np.allclose(archive['racket_rotation'], [r['rotation_camera_columns'] for r in poses['frames']]) or not np.allclose(archive['racket_translation'], [r['translation_camera_m'] for r in poses['frames']]):
            raise ValueError('Viewer differs from body candidate')
    rows, sweep = [], {}
    principal = np.array(meta['image_size'])/2
    for label in old['frames']:
        i = label['frame']; pose = poses['frames'][i]
        r, t, anchor = np.array(pose['rotation_camera_columns']), np.array(pose['translation_camera_m']), np.array(pose['grip_target_camera_m'])
        w, hand = palm_frame(body_joints[i]); offset, axis = grip_contact(body_joints[i]); corridor = w+hand@offset+body_roots[i]
        record = {'frame': i, 'ui_frame': i+1, 'points_original_px': {'butt_model': project(t, meta['focal'][i], principal).tolist(),
            'grip_model': project(anchor, meta['focal'][i], principal).tolist(), 'wrist_model': project(w+body_roots[i], meta['focal'][i], principal).tolist(),
            'mcp_pip_corridor': project(corridor, meta['focal'][i], principal).tolist()},
            'anchor_vs_mcp_pip_prior_gap_mm': float(np.linalg.norm(anchor-corridor)*1000),
            'hand_anatomy_measured': False}
        if 'handle_end' in label['points']:
            butt = np.array(label['points']['handle_end'])
            record.update(manual_butt_original_px=butt.tolist(), manual_butt_to_grip_original_px=float(np.linalg.norm(butt-record['points_original_px']['grip_model'])),
                line_probe=line_grip_probe(anchor, r[:, 1], butt, meta['focal'][i], principal))
        rows.append(record)
        if {'handle_end','throat','tip','rim_side','rim_opposite'} <= set(label['points']):
            # Every frame is fitted to its labels; these results are conditional
            # compatibility tests, not held-out pose accuracy.
            sweep[str(i)] = {str(g): fit_manual_frame(label, {**poses['model'], 'grip_y_m': g}, live_meta, data['live_plane'], r, t, 'real_manual_fixed_grip', anchor)
                for g in [.045, .065, .08, .10, .12]}
    points, observed, frames, core, focals = [], [], [], [], []
    for row in data['mirror_pose']['frames']:
        i = row['frame']
        if row['image'] is None: continue
        for sam, coco in BODY_PAIRS:
            point = np.array(row['image'][coco])
            if point[2] < .65: continue
            points.append(joints[i, sam]+roots[i]); observed.append(point[:2]); frames.append(i); core.append(sam in CORE); focals.append(meta['focal'][i])
    groups = {'old_regression': paired_points(old), 'new_regression': paired_points(new)}
    camera = camera_sensitivity(points, observed, frames, core, focals, meta['image_size'], data['plane'], groups, meta['focal'])
    output.mkdir(parents=True)
    selected = {17, 100, 185, 30, 80}; cap = cv2.VideoCapture(str(paths['video']))
    by_frame = {r['frame']: r for r in rows}
    for label in old['frames']+new['frames']:
        i = label['frame']
        if i not in selected: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, image = cap.read()
        if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != i+1: raise ValueError('Source decode mismatch')
        for view in ['points', 'mirror_points']:
            if not label[view]: continue
            coords = list(label[view].values())
            if view == 'points' and i in by_frame: coords += list(by_frame[i]['points_original_px'].values())
            coords = np.array(coords); lo = np.maximum(np.floor(coords.min(0)).astype(int)-35, 0); hi = np.minimum(np.ceil(coords.max(0)).astype(int)+35, meta['image_size'])
            raw = image[lo[1]:hi[1], lo[0]:hi[0]].copy(); overlay = raw.copy()
            for name, uv in label[view].items():
                p = tuple(np.rint(np.array(uv)-lo).astype(int)); cv2.circle(overlay, p, 3, (60,230,70), -1)
                cv2.putText(overlay, {'handle_end':'butt?', 'throat':'string-bottom?', 'tip':'tip', 'rim_side':'rim1', 'rim_opposite':'rim2'}[name], (p[0]+4,p[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .3, (60,230,70), 1)
            if view == 'points' and i in by_frame:
                for name, color in [('grip_model',(255,210,30)),('butt_model',(20,190,255)),('wrist_model',(230,80,210))]:
                    uv = by_frame[i]['points_original_px'][name]; p = tuple(np.rint(np.array(uv)-lo).astype(int))
                    cv2.drawMarker(overlay,p,color,cv2.MARKER_CROSS,10,1)
            # Enlargement is explicitly pixel replication, not extra evidence.
            tile = np.concatenate([raw, overlay], axis=1)
            cv2.imwrite(str(output/f'frame_{i:04d}_{view}.png'), cv2.resize(tile,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
    cap.release()
    result = {'status':'physical_definitions_and_camera_sensitivity_audited_not_accepted', 'video_sha256': identity,
        'image_size':meta['image_size'], 'fps':meta['fps'], 'definitions': {
            'handle_end':'3D model origin at butt-cap axial center, not wrist or grip center; 2D visible edge is not automatically the axial center',
            'grip':'Estimated MCP/PIP corridor plus assumed 13mm radial offset; currently 45mm from butt, not measured',
            'throat':'Asset uses minimum string-bed y (line-bed lower center); UI previously only said throat, which is ambiguous with V-junction/bridge',
            'tip':'Model axial head extreme; visible rim pixel vs planar center needs consistent definition'},
        'body_display_world_max_error_m':display_error,
        'grip_records':rows,'grip_parameter_sweep':sweep,'grip_sweep_role':'labels_fitted_compatibility_probe_not_heldout_accuracy',
        'new_labels_used_in_grip_sweep':False, 'camera_sensitivity':camera, 'labels_modified':False,'main_viewer_modified':False,
        'source_files':{name:{'path':str(p.resolve()),'sha256':sha256(p)} for name,p in paths.items()},
        'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in ['audit_racket_physical_definitions.py','correct_reviewed_racket_frame.py','audit_mirror_camera.py']},
        'limits':'No measured hand grip location, racket dimensions, camera intrinsics, distortion or mirror. Crops alone cannot certify occluded mirror butt. New labels are viewed regression evidence, not globally unseen validation.'}
    write(output/'physical_definition_report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','staged','native','body-candidate','labels','new-labels','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset,args.staged,args.native,args.body_candidate,args.labels,args.new_labels,args.output)
    print(json.dumps({name:{'body_core_heldout':row['body_core_heldout_original_px']['median'],'focal_scale':row['focal_scale'],'principal_offset':row['principal_offset_original_px'],'new_pairs':row['manual_pair_groups']['new_regression']['median']} for name,row in report['camera_sensitivity']['candidates'].items()},ensure_ascii=False))
