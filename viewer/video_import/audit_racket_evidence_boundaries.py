"""Fixed-input, per-frame evidence/rotation audit with source-video crops.

No training, interpolation, causal attribution or physical face assignment.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from run_records import sha256, write


NAMES = ['handle_end', 'throat', 'tip', 'rim_side', 'rim_opposite', 'head_center', 'grip_center']


def projections(rotation, translation, model, meta, frame, mirror=None):
    objects = np.array([[0, 0, 0], [0, model['throat_y_m'], 0], [0, model['length_m'], 0],
                        [model['head_half_width_m'], model['head_center_y_m'], 0],
                        [-model['head_half_width_m'], model['head_center_y_m'], 0],
                        [0, model['head_center_y_m'], 0], [0, model['grip_y_m'], 0]])
    points = objects @ rotation.T + translation
    if mirror:
        normal = np.array(mirror['normal_camera'])
        points -= 2 * (points @ normal - mirror['distance_camera_m'])[:, None] * normal
    if np.any(points[:, 2] <= .1):
        raise ValueError('racket behind camera')
    return points[:, :2] / points[:, 2:] * meta['focal'][frame] + np.array(meta['image_size']) / 2


def residuals(uv, row, view, scale):
    seen = dict(row.get(view, {}))
    if not row.get('grip_confirmed',{}).get(view,False):seen.pop('grip_center',None)
    if not row.get('face_correspondence_confirmed') and {'rim_side', 'rim_opposite'} <= set(seen):
        a = np.array([seen['rim_side'], seen['rim_opposite']])
        if np.linalg.norm(uv[[3, 4]] - a[::-1], axis=1).sum() < np.linalg.norm(uv[[3, 4]] - a, axis=1).sum():
            seen['rim_side'], seen['rim_opposite'] = seen['rim_opposite'], seen['rim_side']
    error = {name: float(np.linalg.norm(uv[NAMES.index(name)] - p) / scale) for name, p in seen.items() if name in NAMES}
    angle = None
    if {'handle_end', 'tip'} <= set(seen):
        a = uv[2] - uv[0]; b = np.array(seen['tip']) - seen['handle_end']
        if np.linalg.norm(b) >= 4:
            angle = float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8), -1, 1))))
    return {'point_error_canonical_px': error, 'shaft_error_deg': angle}


def audit(run, video, centers, radius, output):
    if output.exists():
        raise ValueError('use a new audit output directory')
    report = json.loads((run / 'refit_report.json').read_text())
    candidate = run / 'mhr_refit_candidate.npz'; staged = run.parent / 'staged/result'
    observations = run.parent / 'observations_snapshot.json'
    if sha256(candidate) != report['candidate_sha256'] or sha256(video) != report['video_sha256']:
        raise ValueError('video/candidate identity mismatch')
    meta = json.loads((staged / 'mesh_meta.json').read_text())
    if meta['video_sha256'] != report['video_sha256']:
        raise ValueError('metadata video differs')
    poses = json.loads((staged / 'racket_poses_directional.json').read_text()); model = poses['model']
    mirror = json.loads((staged / 'mirror_geometry.json').read_text())
    obs = {r['frame']: r for r in json.loads(observations.read_text())['frames']}
    with np.load(candidate, allow_pickle=False) as data:
        rotation = data['racket_rotation']; translation = data['racket_translation']
    if len(rotation) != meta['frames']:
        raise ValueError('timeline differs')
    velocity = rotation[1:] @ rotation[:-1].transpose(0, 2, 1)
    step = np.r_[0., np.degrees(Rotation.from_matrix(velocity).magnitude())]
    accel = np.r_[0., np.degrees(Rotation.from_matrix(velocity[1:] @ velocity[:-1].transpose(0, 2, 1)).magnitude()), 0.]
    scale = meta['image_size'][0] / 1280; records = []; train = set(report['train_frames'])
    uv_by_frame = {}
    for i in range(len(rotation)):
        row = obs.get(i, {}); rec = {'frame': i, 'frame_1based': i + 1, 'time_s': i / meta['fps'],
            'source': row.get('source', 'missing'), 'is_training_frame': i in train,
            'is_heldout_frame': i in report['heldout_frames'], 'step_deg': float(step[i]), 'acceleration_deg_frame2': float(accel[i]), 'views': {}}
        uv_by_frame[i] = {}
        for view in ['points', 'mirror_points']:
            uv = projections(rotation[i], translation[i], model, meta, i, mirror if view == 'mirror_points' else None)
            uv_by_frame[i][view] = uv
            seen = row.get(view, {}); weights = row.get('weights', {}).get(view, {})
            rec['views'][view] = {'points': list(seen), 'weights': weights,
                'training_mean_point_weight': float(np.mean([weights.get(name, 1. if row.get('source') == 'manual_review' else .3) for name in seen])) if seen and i in train else 0.,
                **residuals(uv, row, view, scale)}
        records.append(rec)
    output.mkdir(parents=True); cap = cv2.VideoCapture(str(video)); sheets = []
    try:
        for center in centers:
            tiles = [[], []]
            for i in range(max(0, center - radius), min(len(rotation), center + radius + 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, image = cap.read()
                if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != i + 1:
                    raise ValueError('source frame decode mismatch')
                row = obs.get(i, {})
                for v, view in enumerate(['points', 'mirror_points']):
                    uv = uv_by_frame[i][view]; seen = row.get(view, {}); canvas = image.copy()
                    cv2.polylines(canvas, [np.rint(uv[[0, 1, 2]]).astype(np.int32)], False, (255, 130, 30), 3)
                    for name, p in seen.items():
                        if name not in NAMES: continue
                        cv2.circle(canvas, tuple(np.rint(p).astype(int)), 7, (40, 235, 40), -1)
                        cv2.line(canvas, tuple(np.rint(p).astype(int)), tuple(np.rint(uv[NAMES.index(name)]).astype(int)), (50, 50, 245), 2)
                    positions = np.concatenate([uv, np.array(list(seen.values())).reshape(-1, 2)])
                    low = np.maximum(positions.min(0).astype(int) - 90, 0)
                    high = np.minimum(positions.max(0).astype(int) + 90, np.array(canvas.shape[1::-1]))
                    crop = canvas[low[1]:high[1], low[0]:high[0]]
                    tile = np.zeros((320, 330, 3), np.uint8)
                    if crop.size:
                        ratio = min(330 / crop.shape[1], 265 / crop.shape[0])
                        resized = cv2.resize(crop, None, fx=ratio, fy=ratio); tile[:resized.shape[0], :resized.shape[1]] = resized
                    cv2.putText(tile, f'f{i+1} {"real" if v==0 else "mirror"} acc {accel[i]:.1f}', (5, 284), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
                    cv2.putText(tile, f'{records[i]["source"]} n={len(seen)}', (5, 307), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                    tiles[v].append(tile)
            path = output / f'center_f{center+1}_source_sheet.jpg'
            cv2.imwrite(str(path), np.concatenate([np.concatenate(row, axis=1) for row in tiles], axis=0))
            sheets.append({'center': center, 'path': str(path.resolve()), 'sha256': sha256(path)})
    finally:
        cap.release()
    result = {'status': 'diagnostic_evidence_not_causal_validation', 'candidate_sha256': sha256(candidate),
        'video_sha256': sha256(video), 'observations_sha256': sha256(observations),
        'input_sha256': {name: sha256(staged / name) for name in ['mesh_meta.json', 'mirror_geometry.json', 'racket_poses_directional.json']},
        'code_sha256': sha256(Path(__file__)), 'records': records, 'sheets': sheets, 'published': False,
        'limits': 'Automatic points unverified; unordered rims resolved only for diagnostics. Camera/plane/size are frozen estimates. Heldout points are diagnostics only. Blue shaft = fitted, green points = observations, red = residual. Crops fit to tile; not uniform scale across frames. Acceleration is SO(3) per-frame difference, not causal proof.'}
    write(output / 'evidence_boundary_report.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'video', 'output']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--centers', type=int, nargs='+', default=[105, 115, 182], help='zero-based frames')
    p.add_argument('--radius', type=int, default=3)
    a = p.parse_args(); r = audit(a.run, a.video, a.centers, a.radius, a.output)
    print(json.dumps({'status': r['status'], 'sheets': r['sheets']}))
