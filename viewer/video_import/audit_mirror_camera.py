"""Frozen-camera, mirror correspondence and VFR timing audit; never publish a fit."""
import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from audit_racket_roles import ray_pair
from run_records import sha256, write
from source_frame_alignment import probe_timeline, fps_filter_indices

BODY_PAIRS = [(5, 6), (6, 5), (7, 8), (8, 7), (41, 9), (62, 10),
              (9, 12), (10, 11), (11, 14), (12, 13), (13, 16), (14, 15)]
CORE = [5, 6, 9, 10, 11, 12, 13, 14]
OPPOSITE_MHR = dict([(5, 6), (6, 5), (7, 8), (8, 7), (41, 62), (62, 41),
                     (9, 10), (10, 9), (11, 12), (12, 11), (13, 14), (14, 13)])
OLD_RESERVED = {17, 185}


def stats(values):
    a = np.asarray(values, float).ravel()
    return {'count': len(a), 'median': float(np.median(a)) if len(a) else None,
            'p90': float(np.percentile(a, 90)) if len(a) else None,
            'p95': float(np.percentile(a, 95)) if len(a) else None,
            'max': float(np.max(a)) if len(a) else None}


def normal(q):
    return np.array([math.sin(q[0])*math.cos(q[1]), math.sin(q[1]), math.cos(q[0])*math.cos(q[1])])


def epipolar_residual(a, b, focal, size, n):
    """Signed symmetric point-to-line distances, in native image pixels.

    Translation magnitude cancels here; this audit cannot estimate plane distance.
    """
    n = np.asarray(n, float) / np.linalg.norm(n)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    e = cross @ (np.eye(3) - 2*np.outer(n, n))
    a = np.r_[(np.asarray(a)-np.asarray(size)/2)/focal, 1.]
    b = np.r_[(np.asarray(b)-np.asarray(size)/2)/focal, 1.]
    lr, lm = e@b, e.T@a
    value = float(a@lr)*focal
    return np.array([value/max(np.linalg.norm(lr[:2]), 1e-12), value/max(np.linalg.norm(lm[:2]), 1e-12)])


def paired_points(labels):
    return [{'frame': r['frame'], 'part': k, 'real': r['points'][k], 'mirror': r['mirror_points'][k]}
            for r in labels['frames'] for k in sorted(set(r['points']) & set(r['mirror_points']))
            if not k.startswith('rim_')]


def normal_diagnostic(train, groups, meta, geometry):
    if not train or set(r['frame'] for r in train) & {r['frame'] for rows in groups.values() for r in rows}:
        raise ValueError('Missing training pairs or overlapping evaluation frames')
    n = np.asarray(geometry['normal_camera'])
    q0 = np.array([math.atan2(n[0], n[2]), math.asin(n[1])])
    def residual(q):
        return np.concatenate([epipolar_residual(r['real'], r['mirror'], meta['focal'][r['frame']], meta['image_size'], normal(q))
                               for r in train]) / math.sqrt(2)
    fit = least_squares(residual, q0, bounds=(q0-np.radians(5), q0+np.radians(5)), loss='soft_l1', f_scale=6)
    candidate = normal(fit.x)
    result = {'status': 'diagnostic_only_not_accepted', 'normal_camera': candidate.tolist(),
              'normal_change_deg': float(np.degrees(np.arccos(np.clip(n@candidate, -1, 1)))),
              'train_frames': sorted(set(r['frame'] for r in train)), 'train_pairs': len(train),
              'focal_changed': False, 'distance_changed': False, 'groups': {}}
    for name, rows in {'train': train, **groups}.items():
        records = []
        for r in rows:
            errors = {key: float(np.max(np.abs(epipolar_residual(r['real'], r['mirror'], meta['focal'][r['frame']], meta['image_size'], plane))))
                      for key, plane in [('baseline', n), ('candidate', candidate)]}
            records.append({'frame': r['frame'], 'part': r['part'], 'error_original_px': errors})
        result['groups'][name] = {'records': records, **{key: stats([r['error_original_px'][key] for r in records]) for key in ['baseline', 'candidate']}}
    result['limits'] = 'Five reviewed paired landmarks across three training frames cannot establish calibrated intrinsics or a globally correct plane. New six frames and old 17/185 excluded from fitting. Rim correspondence unconfirmed and excluded.'
    return result


def audit(staged, native, dataset, labels_path, protocol_path, output):
    if output.exists():
        raise ValueError('Use fresh audit output')
    paths = {name: staged/name for name in ['mesh_meta.json', 'mirror_geometry.json', 'mirror_pose.json', 'racket_landmarks.json', 'racket_roi_candidates.json']}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    meta, geometry, body, old, roi = [data[name] for name in paths]
    labels = json.loads(labels_path.read_text()); protocol = json.loads(protocol_path.read_text())
    video = dataset/'source.mp4'; original = dataset/'original.video'
    identity = sha256(video)
    if any(d['video_sha256'] != identity for d in [meta, geometry, body, old, roi, labels, protocol]):
        raise ValueError('Video identity mismatch')
    if roi['original_sha256'] != sha256(original) or labels['image_size'] != meta['image_size'] or labels['fps'] != meta['fps']:
        raise ValueError('Original source or label coordinates changed')
    if not {r['frame'] for r in labels['frames']} <= set(protocol['reserved_frames']):
        raise ValueError('New labels outside frozen reserved frames')
    with np.load(native, allow_pickle=False) as d:
        if str(d['video_sha256']) != identity:
            raise ValueError('Native identity mismatch')
        arrays = {k: d[k].copy() for k in ['joints', 'source_roots', 'focal', 'joints2d', 'mirror_joints', 'mirror_roots', 'mirror_focal', 'mirror_joints2d', 'mirror_valid']}
    count = meta['frames']; n = np.asarray(geometry['normal_camera']); distance = geometry['distance_camera_m']
    if not np.allclose(np.linalg.norm(n), 1, atol=1e-8) or len(arrays['joints']) != count:
        raise ValueError('Plane normalization or frame count mismatch')
    if not np.array_equal(arrays['focal'], meta['focal']) or not np.array_equal(arrays['source_roots'], meta['source_roots']):
        raise ValueError('Frozen camera differs from native archive')
    coordinates = {}
    for prefix in ['', 'mirror_']:
        valid = arrays['mirror_valid'] if prefix else np.ones(count, bool)
        points = arrays[prefix+'joints'] + arrays['mirror_roots' if prefix else 'source_roots'][:, None, :]
        uv = points[valid, :, :2]/points[valid, :, 2:] * arrays[prefix+'focal'][valid, None, None] + np.asarray(meta['image_size'])/2
        coordinates[prefix+'projection_roundtrip_original_px'] = stats(np.linalg.norm(uv-arrays[prefix+'joints2d'][valid], axis=-1))
        coordinates[prefix+'focal_px'] = stats(arrays[prefix+'focal'][valid])
    reflection = np.eye(3)-2*np.outer(n, n)
    coordinates.update(reflection_determinant=float(np.linalg.det(reflection)), reflection_squared_max_error=float(np.max(np.abs(reflection@reflection-np.eye(3)))), mirror_missing_frames=np.flatnonzero(~arrays['mirror_valid']).tolist())
    body_records = []
    for row in body['frames']:
        i = row['frame']
        if row['image'] is None:
            continue
        for sam, coco in BODY_PAIRS:
            observation = np.asarray(row['image'][coco])
            if observation[2] < .65:
                continue
            x = arrays['joints'][i, sam] + arrays['source_roots'][i]
            y = x - 2*(x@n-distance)*n
            uv = y[:2]/y[2]*meta['focal'][i]+np.asarray(meta['image_size'])/2
            opposite = arrays['joints'][i, OPPOSITE_MHR[sam]] + arrays['source_roots'][i]
            opposite -= 2*(opposite@n-distance)*n
            wrong_uv = opposite[:2]/opposite[2]*meta['focal'][i]+np.asarray(meta['image_size'])/2
            body_records.append({'frame': i, 'mhr_id': sam, 'coco_id': coco, 'error_original_px': float(np.linalg.norm(uv-observation[:2])), 'unswapped_mapping_error_original_px': float(np.linalg.norm(wrong_uv-observation[:2])), 'core': sam in CORE, 'heldout': i%5 == 0})
    body_summary = {group: stats([r['error_original_px'] for r in body_records if select(r)]) for group, select in {
        'core_heldout': lambda r: r['core'] and r['heldout'], 'arms_heldout': lambda r: not r['core'] and r['heldout'],
        'right_wrist_heldout': lambda r: r['mhr_id'] == 41 and r['heldout'], 'left_wrist_heldout': lambda r: r['mhr_id'] == 62 and r['heldout']}.items()}
    if len(body_records) != geometry['correspondences'] or not np.isclose(body_summary['core_heldout']['median'], geometry['core_heldout_median_px'], atol=1e-5):
        raise ValueError('Frozen mirror fit could not be reproduced')
    correspondence = {key: stats([r[key] for r in body_records if r['core'] and r['heldout']]) for key in ['error_original_px', 'unswapped_mapping_error_original_px']}
    original_times, original_size, original_end = probe_timeline(original); normalized_times, normalized_size, _ = probe_timeline(video)
    mapping = fps_filter_indices(original_times, normalized_times, meta['fps'], original_end)
    if len(mapping) != count or normalized_size != meta['image_size']:
        raise ValueError('Video timeline differs from native frames')
    old_mapping = np.array([r['source_frame'] for r in roi['frames']])
    mismatches = np.flatnonzero(mapping != old_mapping)
    # Independently replay the actual ffmpeg filter and compare pre/post-filter
    # checksums. This also checks drops, duplicates and the final EOF frame.
    replay = subprocess.run(['ffmpeg', '-nostdin', '-v', 'info', '-i', str(original), '-map', '0:v:0', '-an',
                             '-vf', "scale='trunc(min(2560,iw)/2)*2':-2,showinfo,fps=25,showinfo", '-f', 'null', '-'],
                            capture_output=True, text=True, check=True, timeout=60)
    checksums = {1: {}, 3: {}}
    for line in replay.stderr.splitlines():
        match = re.search(r'Parsed_showinfo_([13])[^\]]*\]\s+n:\s*(\d+).* checksum:([0-9A-F]+)', line)
        if match:
            checksums[int(match[1])][int(match[2])] = match[3]
    if len(checksums[1]) != len(original_times) or len(checksums[3]) != count or any(checksums[1][int(mapping[i])] != checksums[3][i] for i in range(count)):
        raise ValueError('Frame mapping disagrees with actual ffmpeg fps filter')
    selected = sorted(set([0, 2, 30, 80, 100, 105, 115, 130, 155, 174, 180, 182, 185, 205, 210, 230, 248]))
    image_matches = []; sources = {}; targets = {}
    for path, needed, destination in [(original, set(mapping[selected]) | set(old_mapping[selected]), sources), (video, set(selected), targets)]:
        cap = cv2.VideoCapture(str(path))
        try:
            for i in range(max(needed)+1):
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Missing decoded source frame')
                if i in needed:
                    destination[i] = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        finally:
            cap.release()
    for i in selected:
        def mae(index):
            return float(np.mean(np.abs(targets[i].astype(float)-sources[index].astype(float))))
        image_matches.append({'frame': i, 'legacy_source_frame': int(old_mapping[i]), 'pts_source_frame': int(mapping[i]),
                              'legacy_image_mae_8bit': mae(old_mapping[i]), 'pts_image_mae_8bit': mae(mapping[i])})
    training = [r for r in paired_points(old) if r['frame'] not in OLD_RESERVED | set(protocol['reserved_frames'])]
    groups = {'old_heldout': [r for r in paired_points(old) if r['frame'] in OLD_RESERVED], 'new_reserved': paired_points(labels)}
    alternative = normal_diagnostic(training, groups, meta, geometry)
    manual = []
    for name, rows in {'train': training, **groups}.items():
        for row in rows:
            manual.append({'group': name, 'frame': row['frame'], 'part': row['part'], **ray_pair(row['real'], row['mirror'], meta['focal'][row['frame']], meta['image_size'], n, distance)})
    output.mkdir(parents=True)
    files = {**paths, 'native': native, 'original': original, 'normalized': video, 'labels': labels_path, 'protocol': protocol_path}
    result = {'status': 'camera_correspondence_audit_complete_no_fit_accepted', 'video_sha256': identity, 'coordinates': coordinates,
              'body_summary_original_px': body_summary, 'coco_mapping_core_heldout_original_px': correspondence, 'body_records': body_records, 'manual_pairs': manual,
              'plane_normal_diagnostic': alternative, 'timeline': {'normalization': 'server scale + fps=25 default round=near',
                  'original_frames': len(original_times), 'normalized_frames': len(normalized_times), 'original_size': original_size,
                  'legacy_mismatch_count': len(mismatches), 'legacy_mismatch_frames': mismatches.tolist(), 'mapping': mapping.tolist(),
                  'actual_ffmpeg_filter_checksum_agreement_frames': count,
                  'selected_image_comparisons': image_matches, 'original_interval_ms': stats(np.diff(np.array(original_times, float))*1000)},
              'inputs': {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in files.items()},
              'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in ['audit_mirror_camera.py', 'source_frame_alignment.py', 'audit_racket_roles.py', 'racket_observations.py']},
              'published': False, 'new_labels_used_for_training': False,
              'limits': 'Coordinate consistency is not geometry truth. Intrinsics and plane distance remain unmeasured, lens distortion uncalibrated. Epipolar discrepancy cannot distinguish point mismatch from wrong plane/intrinsics. Sparse normal diagnostic worsens held-out data. Legacy observations not regenerated, historical runs retained.'}
    write(output/'mirror_camera_report.json', result)
    (output/'ffmpeg_filter_replay.log').write_text(replay.stderr)
    write(output/'original_timestamps.json', [str(t) for t in original_times]); write(output/'normalized_timestamps.json', [str(t) for t in normalized_times])
    snapshot = output/'code_snapshot'; snapshot.mkdir()
    for name in result['code_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['staged', 'native', 'dataset', 'labels', 'protocol', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); report = audit(a.staged, a.native, a.dataset, a.labels, a.protocol, a.output)
    print(json.dumps({'coordinates': report['coordinates'], 'body': report['body_summary_original_px'],
                      'legacy_timing_mismatches': report['timeline']['legacy_mismatch_count'], 'normal_diagnostic': report['plane_normal_diagnostic']}, ensure_ascii=False))
