"""Fit a rigid tennis racket to reviewed video landmarks in the SAM camera frame.

Only annotated frames are solved. A planar racket viewed by one camera may have
two plausible orientations, so every result carries an ambiguity flag.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


PARTS = ('handle_end', 'throat', 'tip', 'rim_side', 'rim_opposite')
DEFAULT_MODEL = {
    'length_m': 0.685,
    'throat_y_m': 0.345,
    'head_half_width_m': 0.135,
    'head_center_y_m': 0.52,
    'grip_y_m': 0.115,
}


def model_points(model):
    return np.array([
        [0, 0, 0],
        [0, model['throat_y_m'], 0],
        [0, model['length_m'], 0],
        [model['head_half_width_m'], model['head_center_y_m'], 0],
        [-model['head_half_width_m'], model['head_center_y_m'], 0],
    ], np.float64)


def project(points, focal):
    depth = points[:, 2]
    if np.any(depth <= 0):
        return np.full((len(points), 2), 1e5)
    return np.column_stack((640 + focal * points[:, 0] / depth,
                            360 + focal * points[:, 1] / depth))


def transform(object_points, params):
    rotation = cv2.Rodrigues(params[:3])[0]
    return object_points @ rotation.T + params[3:6], rotation


def reflected(points, mirror):
    normal = np.asarray(mirror['normal_camera'], np.float64)
    normal /= np.linalg.norm(normal)
    distance = mirror['distance_camera_m']
    return points - 2 * ((points @ normal - distance)[:, None] * normal)


def fit_frame(row, frame, meta, mirror, model):
    points = row['points']
    if not all(name in points for name in PARTS):
        return {'frame': frame, 'status': 'insufficient_landmarks',
                'reason': '五个真实球拍点都需标出，遮挡点不能猜测'}
    observed = np.asarray([points[name] for name in PARTS], np.float64)
    if observed.shape != (5, 2) or not np.isfinite(observed).all():
        return {'frame': frame, 'status': 'invalid_landmarks'}
    obj = model_points(model)
    focal = float(meta['focal'][frame])
    camera = np.array([[focal, 0, 640], [0, focal, 360], [0, 0, 1]], np.float64)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        obj, observed, camera, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return {'frame': frame, 'status': 'pnp_failed'}

    # The mirror is an additional observation, not a second independent camera.
    mirror_points = row.get('mirror_points') or {}
    mirror_ids = [i for i, name in enumerate(PARTS) if name in mirror_points]
    mirror_seen = np.asarray([mirror_points[PARTS[i]] for i in mirror_ids], np.float64).reshape(-1, 2)
    if not np.isfinite(mirror_seen).all():
        return {'frame': frame, 'status': 'invalid_mirror_landmarks'}
    wrist = np.asarray(meta['source_roots'][frame], np.float64)
    wrist += np.asarray(row['_raw_wrist'], np.float64)
    grip = np.array([[0, model['grip_y_m'], 0]], np.float64)

    def measures(params):
        camera_points, _ = transform(obj, params)
        real = project(camera_points, focal)
        mirror_uv = project(reflected(camera_points[mirror_ids], mirror), focal) if mirror_ids else np.empty((0, 2))
        grip_camera = transform(grip, params)[0][0]
        return real, mirror_uv, grip_camera, camera_points

    def residual(params):
        real, mirror_uv, grip_camera, _ = measures(params)
        # The approximate mirror plane is downweighted relative to real pixels.
        return np.concatenate(((real - observed).ravel(),
                               ((mirror_uv - mirror_seen) * .35).ravel(),
                               (grip_camera - wrist) * 35))

    candidates = []
    for rotation, translation in zip(rotations, translations):
        initial = np.r_[rotation.ravel(), translation.ravel()]
        result = least_squares(residual, initial, max_nfev=150, method='lm')
        real, mirror_uv, grip_camera, camera_points = measures(result.x)
        real_error = float(np.sqrt(np.mean(np.sum((real - observed) ** 2, axis=1))))
        mirror_error = float(np.sqrt(np.mean(np.sum((mirror_uv - mirror_seen) ** 2, axis=1)))) if mirror_ids else None
        wrist_gap = float(np.linalg.norm(grip_camera - wrist))
        score = real_error + (mirror_error * .35 if mirror_error is not None else 0) + max(0, wrist_gap - .15) * 35
        _, matrix = transform(obj, result.x)
        candidates.append({'params': result.x, 'rotation': matrix, 'real': real,
                           'mirror': mirror_uv, 'real_error': real_error,
                           'mirror_error': mirror_error, 'wrist_gap': wrist_gap,
                           'score': score, 'depth': float(np.min(camera_points[:, 2]))})
    candidates.sort(key=lambda candidate: candidate['score'])
    best = candidates[0]
    ambiguous = False
    if len(candidates) > 1:
        relative = candidates[0]['rotation'].T @ candidates[1]['rotation']
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        ambiguous = angle > 25 and candidates[1]['score'] - best['score'] < 2.0
    accepted = best['depth'] > 0 and best['real_error'] <= 8 and best['wrist_gap'] <= .35
    if mirror_ids:
        accepted = accepted and best['mirror_error'] <= 20
    result = {
        'frame': frame, 'status': 'fitted' if accepted else 'rejected',
        'ambiguous': bool(ambiguous), 'real_reprojection_rms_px': round(best['real_error'], 2),
        'mirror_reprojection_rms_px': round(best['mirror_error'], 2) if mirror_ids else None,
        'wrist_gap_m': round(best['wrist_gap'], 3),
        'mirror_landmarks': len(mirror_ids), 'source': row.get('source', 'manual'),
        'observed_points': {name: points[name] for name in PARTS},
        'projected_points': {name: best['real'][i].round(2).tolist() for i, name in enumerate(PARTS)},
    }
    if accepted:
        result['translation_camera_m'] = best['params'][3:6].round(7).tolist()
        result['rotation_camera_columns'] = best['rotation'].round(8).tolist()
    return result


def run(annotations, temporal, meta, mirror, model):
    if annotations.get('video_id') != '30.56' or annotations.get('image_size') != [1280, 720] or annotations.get('fps') != 25:
        raise ValueError('球拍标注的视频 ID、尺寸或帧率与当前 SAM 数据不一致')
    if len(temporal['frames']) != meta['frames']:
        raise ValueError('SAM 网格和姿态的帧数不一致')
    rows = []
    seen = set()
    for source in annotations['frames']:
        frame = source['frame']
        if not isinstance(frame, int) or frame < 0 or frame >= meta['frames'] or frame in seen:
            raise ValueError(f'标注帧号重复或越界：{frame}')
        seen.add(frame)
        source = dict(source, _raw_wrist=temporal['frames'][frame]['raw'][41])
        rows.append(fit_frame(source, frame, meta, mirror, model))
    rows.sort(key=lambda row: row['frame'])
    return {'version': 1, 'video_id': '30.56', 'fps': 25, 'image_size': [1280, 720],
            'coordinate_system': 'SAM_source_camera_X_right_Y_down_Z_forward_m',
            'model': model, 'method': 'planar_IPPE_real_pixels_optional_mirror_soft_wrist',
            'frames': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--data', type=Path, default=Path('output/sam3d_cloud'))
    parser.add_argument('--out', type=Path, default=Path('output/sam3d_cloud/racket_poses.json'))
    parser.add_argument('--length-m', type=float, default=DEFAULT_MODEL['length_m'])
    parser.add_argument('--head-half-width-m', type=float, default=DEFAULT_MODEL['head_half_width_m'])
    args = parser.parse_args()
    ratio = args.length_m / DEFAULT_MODEL['length_m']
    model = {name: (value * ratio if name.endswith('_y_m') else value)
             for name, value in DEFAULT_MODEL.items()}
    model['length_m'] = args.length_m
    model['head_half_width_m'] = args.head_half_width_m
    if not .5 <= model['length_m'] <= .85 or not .08 <= model['head_half_width_m'] <= .2:
        parser.error('球拍长度或拍头半宽超出合理范围')
    read = lambda name: json.loads((args.data / name).read_text())
    result = run(json.loads(args.annotations.read_text()), read('temporal_pose.json'),
                 read('mesh_meta.json'), read('mirror_geometry.json'), model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    fitted = sum(row['status'] == 'fitted' for row in result['frames'])
    ambiguous = sum(row['ambiguous'] for row in result['frames'] if row['status'] == 'fitted')
    print(f'已拟合 {fitted}/{len(result["frames"])} 个关键帧；其中 {ambiguous} 帧存在单目歧义。输出：{args.out}')


if __name__ == '__main__':
    main()
