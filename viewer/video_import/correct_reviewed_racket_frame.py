"""Fit explicit manual-frame edits separately from held-out joint-fit validation."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_racket_evidence_boundaries import NAMES, projections, residuals
from racket_landmarks import validate, usable_grip
from run_records import sha256, write


def fit_manual_frame(row, model, meta, mirror, initial_r, initial_t, mode, anchor):
    if mode not in ['real_manual', 'both_manual', 'real_manual_fixed_grip']:
        raise ValueError('Unknown manual correction mode')
    if not set(NAMES[:5]) <= set(row['points']):
        raise ValueError('Five visible reviewed real points required')
    views = ['points', 'mirror_points'] if mode == 'both_manual' else ['points']
    if mode == 'both_manual' and (not mirror or not row.get('mirror_points')):
        raise ValueError('Paired mirror evidence required')
    frame = row['frame']; grip = np.array([0., model['grip_y_m'], 0.])
    fixed = mode == 'real_manual_fixed_grip'; anchor = np.asarray(anchor)
    ids = {v: [k for k, name in enumerate(NAMES) if name in row[v] and (name!='grip_center' or usable_grip(row,v))] for v in views}
    initial_q = Rotation.from_matrix(initial_r).as_rotvec()
    # Rim sides are unordered unless a physical marking was explicitly confirmed.
    flexible = [v for v in views if not row.get('face_correspondence_confirmed') and {'rim_side', 'rim_opposite'} <= set(row[v])]
    solutions = []
    for swaps in itertools.product([False, True], repeat=len(flexible)):
        seen = {v: dict(row[v]) for v in views}
        for v, swap in zip(flexible, swaps):
            if swap:
                seen[v]['rim_side'], seen[v]['rim_opposite'] = seen[v]['rim_opposite'], seen[v]['rim_side']
        observed = {v: np.array([seen[v][NAMES[k]] for k in ids[v]]) for v in views}
        def unpack(q):
            r = Rotation.from_rotvec(q[:3]).as_matrix()
            t = anchor-r@grip if fixed else q[3:]
            return r, t
        def error(q):
            r, t = unpack(q)
            errors = []
            for view in views:
                objects = np.array([[0, 0, 0], [0, model['throat_y_m'], 0], [0, model['length_m'], 0],
                                    [model['head_half_width_m'], model['head_center_y_m'], 0],
                                    [-model['head_half_width_m'], model['head_center_y_m'], 0], [0, model['head_center_y_m'], 0], [0, model['grip_y_m'], 0]])
                camera = objects@r.T+t
                if view == 'mirror_points':
                    n = np.asarray(mirror['normal_camera'])
                    camera -= 2*(camera@n-mirror['distance_camera_m'])[:, None]*n
                # Finite penalties keep invalid trial iterates away from camera.
                uv = camera[:, :2]/np.maximum(camera[:, 2:], .1)*meta['focal'][frame]+np.asarray(meta['image_size'])/2
                errors.extend([(uv[ids[view]]-observed[view]).ravel(), np.minimum(camera[:, 2]-.1, 0)*1000])
            return np.concatenate(errors)
        seeds = [initial_q] if fixed else [np.r_[initial_q, initial_t]]
        if not fixed:
            obj = np.array([[0, 0, 0], [0, model['throat_y_m'], 0], [0, model['length_m'], 0],
                            [model['head_half_width_m'], model['head_center_y_m'], 0], [-model['head_half_width_m'], model['head_center_y_m'], 0]], float)
            focal = meta['focal'][frame]; width, height = meta['image_size']
            camera = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
            points = np.array([seen['points'][name] for name in NAMES[:5]])
            ok, rotations, translations, _ = cv2.solvePnPGeneric(obj, points, camera, None, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                seeds.extend(np.r_[r.ravel(), t.ravel()] for r, t in zip(rotations, translations))
        seeds += [np.r_[Rotation.from_matrix(initial_r@Rotation.from_rotvec([0, np.pi, 0]).as_matrix()).as_rotvec(),
                        [] if fixed else initial_t]]
        for seed in seeds:
            fit = least_squares(error, seed, max_nfev=500, loss='linear', ftol=1e-10, xtol=1e-10, gtol=1e-10)
            r, t = unpack(fit.x)
            try:
                for v in ['points', 'mirror_points'] if mirror else ['points']:
                    projections(r, t, model, meta, frame, mirror if v == 'mirror_points' else None)
            except ValueError:
                continue
            angle = float(np.degrees(Rotation.from_matrix(r@initial_r.T).magnitude()))
            solutions.append({'cost': float(fit.cost), 'r': r, 't': t, 'angle': angle, 'converged': bool(fit.success)})
    if not solutions:
        raise ValueError('No positive-depth rigid pose')
    minimum = min(s['cost'] for s in solutions)
    # Only numerically tied projection solutions use continuity for display.
    best = min([s for s in solutions if s['cost'] <= minimum+1e-6], key=lambda s: s['angle'])
    metrics = {}
    for v in ['points', 'mirror_points'] if mirror else ['points']:
        result = residuals(projections(best['r'], best['t'], model, meta, frame, mirror if v == 'mirror_points' else None), row, v, 1.)
        errors = result.pop('point_error_canonical_px')
        metrics[v] = {**result, 'point_error_original_px': errors, 'median_original_px': float(np.median(list(errors.values()))) if errors else None,
                      'rms_original_px': float(np.sqrt(np.mean(np.square(list(errors.values()))))) if errors else None}
    return {'mode': mode, 'rotation_camera_columns': best['r'].tolist(), 'translation_camera_m': best['t'].tolist(),
            'fit_converged': best['converged'], 'fit_cost': best['cost'], 'rotation_change_deg': best['angle'],
            'translation_change_mm': float(np.linalg.norm(best['t']-initial_t)*1000),
            'frozen_palm_anchor_gap_mm': float(np.linalg.norm(best['r']@grip+best['t']-anchor)*1000),
            'metrics': metrics, 'physical_face_sign_verified': False, 'accepted': False,
            'validation_role': 'manual_edit_training_frame_not_heldout', 'fit_views': views}


def correct(dataset, labels_path, frame, output):
    if output.exists():
        raise ValueError('Use fresh correction output')
    source = dataset/'result'; poses_path = source/'racket_poses.json'; meta_path = source/'mesh_meta.json'; mirror_path = source/'mirror_geometry.json'
    poses = json.loads(poses_path.read_text()); meta = json.loads(meta_path.read_text()); mirror = json.loads(mirror_path.read_text())
    labels = validate(json.loads(labels_path.read_text()), meta)
    if sha256(dataset/'source.mp4') != meta['video_sha256'] or poses['video_sha256'] != meta['video_sha256'] or mirror['video_sha256'] != meta['video_sha256']:
        raise ValueError('Source identity mismatch')
    row = next(r for r in labels['frames'] if r['frame'] == frame); baseline = poses['frames'][frame]
    r0 = np.asarray(baseline['rotation_camera_columns']); t0 = np.asarray(baseline['translation_camera_m']); anchor = np.asarray(baseline['grip_target_camera_m'])
    results = {mode: fit_manual_frame(row, poses['model'], meta, mirror, r0, t0, mode, anchor)
               for mode in ['real_manual', 'both_manual', 'real_manual_fixed_grip']}
    output.mkdir(parents=True)
    result = {'status': 'manual_correction_candidates_not_accepted', 'frame': frame, 'frame_1based': frame+1, 'video_sha256': meta['video_sha256'],
              'results': results, 'old_heldout_role_revoked_for_these_candidates': True, 'human_labels_modified': False,
              'mirror_camera_changed': False, 'body_pose_changed': False, 'main_viewer_changed': False,
              'source_files': {str(p.resolve()): sha256(p) for p in [poses_path, meta_path, mirror_path, labels_path, dataset/'source.mp4']},
              'code_sha256': sha256(Path(__file__)),
              'limits': 'This frame is fitted to reviewed pixels and cannot count as independent validation. Fixed shape/dimensions/camera are estimates. Planar poses and unsigned rim sides remain ambiguous. Frozen palm anchor is a prior, not skin-surface contact. No temporal repair or full-body optimization.'}
    write(output/'manual_correction_report.json', result)
    cap = cv2.VideoCapture(str(dataset/'source.mp4')); cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read(); actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))); cap.release()
    if not ok or actual != frame+1:
        raise ValueError('Decode timeline mismatch')
    candidates = {'v9_baseline': {'rotation_camera_columns': r0.tolist(), 'translation_camera_m': t0.tolist()}, **results}
    for name, candidate in candidates.items():
        r = np.asarray(candidate['rotation_camera_columns']); t = np.asarray(candidate['translation_camera_m']); tiles = []
        for view in ['points', 'mirror_points']:
            ref = mirror if view == 'mirror_points' else None
            uv = projections(r, t, poses['model'], meta, frame, ref)
            ring = np.array(poses['model']['head_outline'])@r.T+t
            if ref:
                n = np.asarray(ref['normal_camera']); ring -= 2*(ring@n-ref['distance_camera_m'])[:, None]*n
            ring = ring[:, :2]/ring[:, 2:]*meta['focal'][frame]+np.array(meta['image_size'])/2
            canvas = image.copy(); cv2.polylines(canvas, [np.rint(ring).astype(np.int32)], True, (25, 205, 250), 2)
            cv2.polylines(canvas, [np.rint(uv[[0, 1, 2]]).astype(np.int32)], False, (25, 205, 250), 2)
            for p in row[view].values():
                cv2.circle(canvas, tuple(np.rint(p).astype(int)), 5, (40, 235, 50), -1)
            points = np.r_[uv, np.array(list(row[view].values()))]; lo = np.maximum(points.min(0).astype(int)-50, 0); hi = np.minimum(points.max(0).astype(int)+50, np.array(meta['image_size']))
            crop = canvas[lo[1]:hi[1], lo[0]:hi[0]]; scale = min(560/crop.shape[1], 300/crop.shape[0]); crop = cv2.resize(crop, None, fx=scale, fy=scale)
            tile = np.full((340, 580, 3), 22, np.uint8); tile[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(tile, f'UI{frame+1} {name} {view}', (8, 324), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1); tiles.append(tile)
        cv2.imwrite(str(output/(name+'.png')), np.concatenate(tiles, axis=1))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'labels', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True, help='Zero-based frame to use as manual edit training evidence')
    args = parser.parse_args(); report = correct(args.dataset, args.labels, args.frame, args.output)
    print(json.dumps(report['results'], ensure_ascii=False))
