"""Export the raw Viewer pose and rigid racket for downstream motion analysis."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


RACKET_NAMES = ['handle_end', 'grip_center', 'throat', 'tip', 'head_center',
                'rim_side', 'rim_opposite']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False,
                                   allow_nan=False, separators=(',', ':')) + '\n')


def derivative(values, fps):
    """Central difference only where both neighbours and centre are present."""
    out = np.full_like(values, np.nan, dtype=float)
    good = np.isfinite(values).reshape(len(values), -1).all(axis=1)
    keep = good[:-2] & good[1:-1] & good[2:]
    out[1:-1][keep] = (values[2:][keep] - values[:-2][keep]) * fps / 2
    return out


def unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.full_like(v, np.nan), where=length > 1e-8)


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def segment_frame(lateral, up):
    y = unit(up)
    x = unit(lateral - np.sum(lateral * y, axis=1)[:, None] * y)
    z = unit(np.cross(x, y))
    return np.stack([x, y, z], axis=-1)


def angular_velocity(matrices, fps):
    """Spatial SO(3) velocity at interval midpoints; never crosses a hidden frame."""
    out = np.full((len(matrices) - 1, 3), np.nan)
    good = np.isfinite(matrices).all(axis=(1, 2))
    keep = good[:-1] & good[1:]
    delta = matrices[1:][keep] @ matrices[:-1][keep].transpose(0, 2, 1)
    if len(delta):
        out[keep] = Rotation.from_matrix(delta).as_rotvec() * fps
    return out


def csv_rows(path, columns, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(clean(row))


def export(result, archive, output):
    result, archive, output = map(Path, (result, archive, output))
    meta = json.loads((result / 'mesh_meta.json').read_text())
    racket = json.loads((result / 'racket_poses.json').read_text())
    labels = json.loads(Path(__file__).with_name('mhr70_names.json').read_text())
    names = labels['names']
    n, fps = meta['frames'], float(meta['fps'])
    for key in ['video_sha256', 'fps', 'image_size']:
        if racket[key] != meta[key]:
            raise ValueError(f'Racket identity mismatch: {key}')
    if len(racket['frames']) != n:
        raise ValueError('Racket timeline mismatch')
    archive_manifest = None
    with np.load(archive, allow_pickle=False) as source:
        if 'video_sha256' in source:
            if str(source['video_sha256']) != meta['video_sha256']:
                raise ValueError('Body video identity mismatch')
        else:
            # Earlier reconstructions bind video identity in their sidecar.
            archive_manifest = archive.with_name('multiview_manifest.json')
            manifest = json.loads(archive_manifest.read_text())
            if (manifest['source_sha256'] != meta['video_sha256'] or
                    any(manifest[k] != meta[k] for k in ['frames', 'fps', 'image_size'])):
                raise ValueError('Body archive manifest identity mismatch')
        local = source['joints'].astype(float)
        roots = np.asarray(meta['source_roots'])
        if local.shape != (n, 70, 3) or not np.isfinite(local).all():
            raise ValueError('Expected finite MHR70 joints for every frame')
        if not np.array_equal(roots, source['source_roots']):
            raise ValueError('Body source roots differ from Viewer')
        mesh = np.memmap(result / 'mesh_local.bin', dtype='<f4', mode='r',
                         shape=source['vertices'].shape)
        if not np.array_equal(mesh, source['vertices']):
            raise ValueError('Archive mesh differs from raw Viewer geometry')
    body = local + roots[:, None, :]
    model = racket['model']
    obj = np.array([[0, 0, 0], [0, model['grip_y_m'], 0],
                    [0, model['throat_y_m'], 0], [0, model['length_m'], 0],
                    [0, model['head_center_y_m'], 0],
                    [model['head_half_width_m'], model['head_center_y_m'], 0],
                    [-model['head_half_width_m'], model['head_center_y_m'], 0]])
    points = np.full((n, len(obj), 3), np.nan)
    rotations = np.full((n, 3, 3), np.nan)
    translations = np.full((n, 3), np.nan)
    for i, row in enumerate(racket['frames']):
        if row['frame'] != i:
            raise ValueError('Racket frame indices are not contiguous')
        if row['status'] != 'fitted':
            continue
        R = np.asarray(row['rotation_camera_columns'], dtype=float)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError(f'Invalid rotation at frame {i}')
        t = np.asarray(row['translation_camera_m'], dtype=float)
        if not np.isfinite(t).all():
            raise ValueError('Invalid translation')
        rotations[i], translations[i] = R, t
        points[i] = obj @ R.T + t
        if 'grip_target_camera_m' in row and not np.allclose(points[i, 1], row['grip_target_camera_m'], atol=1e-6):
            raise ValueError('Grip anchor differs from rendered rigid pose')
    velocity_body = derivative(body, fps)
    velocity_racket = derivative(points, fps)
    acceleration_body = derivative(velocity_body, fps)
    acceleration_racket = derivative(velocity_racket, fps)
    hip = (body[:, 9] + body[:, 10]) / 2
    shoulder = (body[:, 5] + body[:, 6]) / 2
    spine = shoulder - hip
    pelvis_R = segment_frame(body[:, 10] - body[:, 9], spine)
    trunk_R = segment_frame(body[:, 6] - body[:, 5], spine)
    frame_matrices = {'pelvis_proxy': pelvis_R, 'trunk_proxy': trunk_R, 'racket': rotations}
    angular = {key: angular_velocity(value, fps) for key, value in frame_matrices.items()}
    # A two-endpoint limb has no measurable twist about its longitudinal axis.
    for name, a, b in [('upper_arm', 6, 8), ('forearm', 8, 41)]:
        u = unit(body[:, b] - body[:, a])
        angular[name + '_swing_only'] = np.cross(u, derivative(u, fps))
    lateral_hip, lateral_trunk = pelvis_R[:, :, 0], trunk_R[:, :, 0]
    separation = np.degrees(np.arctan2(
        np.sum(unit(spine) * np.cross(lateral_hip, lateral_trunk), axis=1),
        np.sum(lateral_hip * lateral_trunk, axis=1)))
    scalar = {
        'pelvis_trunk_separation_deg': separation,
        'right_elbow_flexion_deg': 180 - angle(body[:, 6] - body[:, 8], body[:, 41] - body[:, 8]),
        'left_knee_flexion_deg': 180 - angle(body[:, 9] - body[:, 11], body[:, 13] - body[:, 11]),
        'right_knee_flexion_deg': 180 - angle(body[:, 10] - body[:, 12], body[:, 14] - body[:, 12]),
        'right_wrist_speed_m_s': np.linalg.norm(velocity_body[:, 41], axis=1),
        'racket_tip_speed_m_s': np.linalg.norm(velocity_racket[:, 3], axis=1),
        'racket_head_center_speed_m_s': np.linalg.norm(velocity_racket[:, 4], axis=1),
        'wrist_grip_distance_m': np.linalg.norm(body[:, 41] - points[:, 1], axis=1),
    }
    output.mkdir(parents=True, exist_ok=True)
    sources = [result / name for name in ['mesh_meta.json', 'racket_poses.json', 'mesh_local.bin', 'video.mp4']]
    sources += [archive, Path(__file__), Path(__file__).with_name('mhr70_names.json')]
    if archive_manifest is not None:
        sources.append(archive_manifest)
    if sha256(result / 'video.mp4') != meta['video_sha256']:
        raise ValueError('Normalized video hash mismatch')
    application = result / 'grip_application_report.json'
    accepted = json.loads(application.read_text()).get('accepted', False) if application.exists() else False
    if application.exists():
        sources.append(application)
    metadata = {
        'schema_version': 1, 'dataset_id': result.parent.name, 'video_id': racket.get('video_id'),
        **{k: meta[k] for k in ['video_sha256', 'frames', 'fps', 'image_size']},
        'duration_s': n / fps, 'last_frame_time_s': (n - 1) / fps,
        'body_joint_names': names, 'body_joint_labels': labels,
        'racket_keypoint_names': RACKET_NAMES, 'racket_model': model,
        'coordinate_system': {'name': 'source_camera', 'unit': 'model_meter',
                              'x': 'image right', 'y': 'image down', 'z': 'forward away from camera',
                              'handedness': 'right', 'origin': 'camera optical center',
                              'body_camera_equation': 'joints_local + source_root',
                              'racket_camera_equation': 'R @ point_local + translation',
                              'rotation_storage': 'nested matrix rows; columns are local axes in camera space',
                              'ground_frame_calibrated': False, 'metric_scale_measured': False},
        'pose_mode': 'raw; exact archive corresponding to current Viewer mesh_local.bin',
        'racket_accepted': accepted,
        'quality_counts': dict(Counter(row.get('quality', row['status']) for row in racket['frames'])),
        'missing_policy': 'Hidden racket = null in JSON / NaN in NPZ / blank CSV. No added interpolation.',
        'derivatives': {'position': 'central difference at frame time, fps/2*(next-previous)',
                        'angular': 'spatial log(R_next @ R_current.T)*fps, interval midpoint',
                        'filter': 'none; original fitted racket temporal processing retained',
                        'endpoints': 'null; derivatives also null adjacent to gaps',
                        'units': {'linear_velocity': 'model m/s', 'linear_acceleration': 'model m/s^2',
                                  'angular_velocity': 'rad/s', 'joint_angles': 'deg'}},
        'kinematic_definitions': {'pelvis_trunk_frames': 'proxy triads: x=right-minus-left hip/shoulder projected normal to spine, y=hip-center to shoulder-center, z=x cross y',
                                 'pelvis_trunk_separation': 'signed angle between proxy lateral axes about spine; not ground-plane rotation',
                                 'limb_angular_velocity': 'cross(unit segment, derivative); longitudinal twist unavailable',
                                 'flexion': '180 minus internal 3D joint angle; zero means straight'},
        'limits': ['Pose and racket are estimated and this clip is not accepted.',
                   'Asset dimensions, 9cm grip position and camera scale are not measured.',
                   'Rim labels and model normal do not identify physical racket face A/B.',
                   'Raw finite differences may amplify pose jitter at 25 fps.',
                   'No impact frame or stroke phases provided; full-clip peaks are not a kinetic-chain verdict.',
                   'No force, torque, mechanical power, energy transfer, COM or ground reaction forces measured.'],
        'sources': [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in sources],
    }
    write_json(output / 'metadata.json', metadata)
    rows = []
    for i, row in enumerate(racket['frames']):
        rows.append({'frame_index': i, 'frame_number': i + 1, 'time_s': i / fps,
                     'body_keypoints_camera_m': body[i],
                     'body_keypoints_root_relative_m': local[i], 'source_root_camera_m': roots[i],
                     'body_confidence': None, 'racket_keypoints_camera_m': points[i],
                     'racket_keypoints_root_relative_m': points[i] - roots[i],
                     'racket_rotation_local_to_camera': rotations[i],
                     'racket_translation_camera_m': translations[i],
                     'racket_shaft_unit_camera': rotations[i, :, 1],
                     'racket_model_normal_unit_camera': rotations[i, :, 2],
                     'racket_quality': row, 'kinematics': {k: v[i] for k, v in scalar.items()}})
    write_json(output / 'pose3d.json', {'metadata': metadata, 'frames': rows})
    np.savez_compressed(output / 'pose3d.npz', time_s=np.arange(n) / fps,
                        body_joint_names=np.asarray(names), racket_keypoint_names=np.asarray(RACKET_NAMES),
                        body_camera_m=body, body_root_relative_m=local, source_roots_camera_m=roots,
                        racket_camera_m=points, racket_root_relative_m=points - roots[:, None, :],
                        racket_rotation_local_to_camera=rotations, racket_translation_camera_m=translations,
                        body_velocity_camera_m_s=velocity_body, racket_velocity_camera_m_s=velocity_racket,
                        body_acceleration_camera_m_s2=acceleration_body, racket_acceleration_camera_m_s2=acceleration_racket,
                        angular_interval_time_s=(np.arange(n - 1) + .5) / fps,
                        **{k + '_angular_velocity_rad_s': v for k, v in angular.items()},
                        **{k + '_rotation_local_to_camera': v for k, v in frame_matrices.items() if k != 'racket'},
                        racket_quality=np.asarray([r.get('quality', r['status']) for r in racket['frames']]),
                        **scalar)
    def point_rows():
        for i in range(n):
            for kind, labels_, p, v in [('body', names, body, velocity_body),
                                       ('racket', RACKET_NAMES, points, velocity_racket)]:
                for j, name in enumerate(labels_):
                    yield {'frame_index': i, 'time_s': i / fps, 'kind': kind, 'name': name,
                           **dict(zip(['x_m', 'y_m', 'z_m'], p[i, j])),
                           **dict(zip(['vx_m_s', 'vy_m_s', 'vz_m_s'], v[i, j])),
                           'quality': 'model_estimated' if kind == 'body' else racket['frames'][i].get('quality', racket['frames'][i]['status'])}
    csv_rows(output / 'keypoints3d.csv', ['frame_index', 'time_s', 'kind', 'name', 'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s', 'quality'], point_rows())
    csv_rows(output / 'kinematics.csv', ['frame_index', 'time_s', *scalar],
             ({'frame_index': i, 'time_s': i / fps, **{k: v[i] for k, v in scalar.items()}} for i in range(n)))
    angular_rows = []
    peaks = {}
    for key, v in angular.items():
        times = (np.arange(n - 1) + .5) / fps if len(v) == n - 1 else np.arange(n) / fps
        speed = np.linalg.norm(v, axis=1)
        finite = np.isfinite(speed)
        if finite.any():
            i = int(np.nanargmax(speed))
            peaks[key] = {'time_s': float(times[i]), 'angular_speed_rad_s': speed[i],
                          'angular_speed_deg_s': np.degrees(speed[i]), 'scope': 'full_clip'}
        for i, xyz in enumerate(v):
            angular_rows.append({'segment': key, 'time_s': times[i], 'sample_index': i,
                                 **dict(zip(['wx_rad_s', 'wy_rad_s', 'wz_rad_s'], xyz)),
                                 'angular_speed_deg_s': np.degrees(speed[i])})
    csv_rows(output / 'angular_velocity.csv', ['segment', 'sample_index', 'time_s', 'wx_rad_s', 'wy_rad_s', 'wz_rad_s', 'angular_speed_deg_s'], angular_rows)
    write_json(output / 'kinematics_summary.json', {'scope': 'full_clip_no_impact_alignment', 'peaks': peaks,
                                                  'limits': metadata['limits'], 'quality_counts': metadata['quality_counts']})
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = export(args.result, args.archive, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'frames': report['frames'],
                      'body_joints': len(report['body_joint_names']), 'racket_keypoints': len(RACKET_NAMES),
                      'quality_counts': report['quality_counts']}, ensure_ascii=False))
