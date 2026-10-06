"""Read-only automatic grip-distance regression against frozen human pixels.

Predictions use original inference joints and freshly decoded racket contours.
Manual labels and the prescribed grip distance are never prediction inputs.
Centimeters remain model-dependent estimates until a measured reference exists.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from racket_keypoints import extract, role_compatible
from racket_observations import crop_region
from source_frame_alignment import source_frame_indices
from run_records import sha256, write, now

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'viewer/sam3d'))
from fit_wilson_grip import palm_frame, grip_contact
from fit_racket_pose import project


def summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0, 'median': None, 'p95': None, 'max': None}
    return {'count': len(values), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def ratio_distance(points, grip_uv, length_m, pixel_scale=1.):
    """An explicit equal-depth approximation, never metric ground truth."""
    if not {'handle_end', 'tip'} <= set(points):
        return {'usable': False, 'reason': 'missing_visible_butt_or_tip'}
    butt, tip = np.asarray(points['handle_end']), np.asarray(points['tip'])
    line = tip - butt
    span = np.linalg.norm(line)
    if span < 12 * pixel_scale:
        return {'usable': False, 'reason': 'short_projected_shaft'}
    fraction = float((np.asarray(grip_uv) - butt) @ line / (span * span))
    gap = float(np.linalg.norm(np.asarray(grip_uv) - butt - fraction * line))
    distance = fraction * length_m
    usable = .01 <= distance <= .20 and gap <= 6 * pixel_scale
    return {'usable': bool(usable), 'reason': 'equal_depth_proxy' if usable else 'off_axis_or_distance_range',
            'distance_m': float(distance), 'off_axis_original_px': gap,
            'shaft_span_original_px': float(span), 'perspective_corrected': False}


def hand_axis_distance(points, anchor, axis, focal, size, pixel_scale=1.):
    """Infer a free axial distance from a butt pixel and an automatic hand line."""
    if 'handle_end' not in points or 'tip' not in points:
        return {'usable': False, 'reason': 'missing_visible_butt_or_tip'}
    anchor = np.asarray(anchor, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    grip_uv = project(anchor[None], focal, size)[0]
    direction = project((anchor + axis * .05)[None], focal, size)[0] - grip_uv
    if np.linalg.norm(direction) < pixel_scale:
        return {'usable': False, 'reason': 'hand_axis_near_optical_axis'}
    if direction @ (np.asarray(points['tip']) - grip_uv) < 0:
        axis = -axis
    butt = np.asarray(points['handle_end'])

    def error(distance):
        uv = project((anchor - axis * distance)[None], focal, size)[0]
        return float(np.sum((uv - butt) ** 2))

    result = minimize_scalar(error, bounds=(.01, .20), method='bounded', options={'xatol': 1e-9})
    residual = float(np.sqrt(result.fun))
    usable = result.success and .0101 < result.x < .1999 and residual <= 4 * pixel_scale
    return {'usable': bool(usable), 'reason': 'automatic_hand_axis_prior' if usable else 'hand_axis_or_butt_conflict',
            'distance_m': float(result.x), 'butt_residual_original_px': residual,
            'perspective_corrected': True, 'axis_measured': False}


def predict(dataset, model_path, out, device):
    """This function intentionally has no manual-label or pose-file inputs."""
    from ultralytics import YOLO
    result = dataset / 'result'
    meta = json.loads((result / 'mesh_meta.json').read_text())
    attempt = json.loads((dataset / 'record.json').read_text())['attempt']
    archive = dataset / 'attempts' / f'{attempt:04d}' / 'work/reconstruction.npz'
    with np.load(archive, allow_pickle=False) as data:
        joints = data['joints'].copy()
        roots = data['source_roots'].copy()
        focal = data['focal'].copy()
    if len(joints) != meta['frames']:
        raise ValueError('Automatic hand timeline differs')
    np.testing.assert_allclose(focal, meta['focal'], atol=1e-6)
    np.testing.assert_allclose(roots, meta['source_roots'], atol=1e-6)
    dimensions = json.loads((result / 'racket_dimensions.json').read_text())
    for key in ['video_sha256', 'image_size', 'fps']:
        if dimensions[key] != meta[key]:
            raise ValueError('Dimension source identity differs')
    length_m = dimensions['dimensions_cm']['length'] / 100
    source = dataset / 'original.video'
    if not source.exists():
        source = dataset / 'source.mp4'
    if sha256(dataset / 'source.mp4') != meta['video_sha256']:
        raise ValueError('Normalized video hash differs')
    indices, original_size, normalized_size = source_frame_indices(source, dataset / 'source.mp4', meta['fps'])
    if len(indices) != meta['frames'] or normalized_size != meta['image_size']:
        raise ValueError('Automatic image timeline differs')
    size = np.asarray(meta['image_size'])
    image_scale = np.asarray(original_size) / size
    pixel_scale = size[0] / 1280
    mirror = json.loads((result / 'mirror_geometry.json').read_text()) if meta.get('mirror_available') else None
    detector = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(source))
    rows = []
    previous = -1
    try:
        for i, index in enumerate(indices):
            if index != previous + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                raise ValueError('Decoded frame does not match timestamp mapping')
            previous = int(index)
            wrist, hand = palm_frame(joints[i])
            offset, local_axis = grip_contact(joints[i])
            anchor = wrist + hand @ offset + roots[i]
            axis = hand @ local_axis
            uv = project(anchor[None], focal[i], size)[0]
            other = None
            if mirror:
                normal = np.asarray(mirror['normal_camera'])
                reflected = anchor - 2 * (anchor @ normal - mirror['distance_camera_m']) * normal
                other = project(reflected[None], focal[i], size)[0]
            radius = np.clip(focal[i] / max(anchor[2], .1) * .95, 100 * pixel_scale, 230 * pixel_scale)
            x0, y0, x1, y1 = crop_region(uv * image_scale, radius * max(image_scale), original_size)
            options = []
            if x1 - x0 >= 80 and y1 - y0 >= 80:
                prediction = detector.predict(image[y0:y1, x0:x1], classes=[38], conf=.035,
                                              imgsz=960, device=device, verbose=False)[0]
                if prediction.masks is not None:
                    for k, box in enumerate(prediction.boxes):
                        polygon = (prediction.masks.xy[k] + [x0, y0]) / image_scale
                        confidence = float(box.conf[0])
                        found = extract(polygon / pixel_scale, uv / pixel_scale, confidence)
                        if found is None:
                            continue
                        center = np.asarray(found['points']['head_center']) * pixel_scale
                        if not role_compatible(center, uv, other, pixel_scale):
                            continue
                        distance = np.linalg.norm(center - uv) / pixel_scale
                        if distance > 150:
                            continue
                        found['points'] = {name: (np.asarray(p) * pixel_scale).tolist() for name, p in found['points'].items()}
                        found['polygon'] = polygon.tolist()
                        options.append((confidence * np.exp(-distance / 160), found))
            found = max(options, key=lambda entry: entry[0])[1] if options else None
            points = found['points'] if found else {}
            ratios = ratio_distance(points, uv, length_m, pixel_scale)
            hand_distance = hand_axis_distance(points, anchor, axis, focal[i], size, pixel_scale)
            if found and found['confidence'] < .1:
                for value in [ratios, hand_distance]:
                    value.update(usable=False, reason='low_detection_confidence')
            rows.append({'frame': i, 'source_frame': int(index), 'grip_center_world_m': anchor.tolist(),
                         'grip_center_original_px': uv.tolist(), 'hand_axis_world': axis.tolist(),
                         'racket': found, 'ratio': ratios, 'hand_axis': hand_distance})
            if (i + 1) % 25 == 0 or i + 1 == len(indices):
                print(f'Automatic grip: {i+1}/{len(indices)}; racket {sum(bool(r["racket"]) for r in rows)}', flush=True)
    finally:
        cap.release()
    prediction = {'video_sha256': meta['video_sha256'], 'image_size': meta['image_size'], 'fps': meta['fps'],
                  'frames': rows, 'created_at': now(), 'manual_labels_used': False,
                  'prescribed_grip_used': False, 'current_racket_poses_used': False,
                  'body_source': 'original_automatic_inference_joints', 'dimensions_measured': dimensions['measured'],
                  'length_m': length_m, 'code_sha256': sha256(Path(__file__)),
                  'inputs_sha256': {'archive': sha256(archive), 'video': sha256(source),
                                   'normalized_video': sha256(dataset / 'source.mp4'), 'detector': sha256(model_path),
                                   'meta': sha256(result / 'mesh_meta.json'),
                                   'dimensions': sha256(result / 'racket_dimensions.json'),
                                   'mirror_role_exclusion': sha256(result / 'mirror_geometry.json') if mirror else None},
                  'gates': {'detector_confidence': .1, 'ratio_off_axis_canonical_px': 6,
                            'hand_axis_butt_residual_canonical_px': 4, 'distance_range_m': [.01, .20]},
                  'limits': 'Ratio assumes equal depth. Hand-axis estimate depends on automatic hand depth/axis and a 13mm grasp-radius prior. Mirror only excludes the opposite role. No physical centimeter ground truth.'}
    write(out / 'predictions.json', prediction)
    return prediction


def evaluate(prediction, labels):
    """Annotations enter only here, after automatic predictions are frozen."""
    for key in ['video_sha256', 'image_size', 'fps']:
        if prediction[key] != labels[key]:
            raise ValueError('Evaluation annotation identity differs')
    errors = {'grip_center': [], 'handle_end': [], 'tip': [], 'head_center': [], 'throat': []}
    denominators = dict.fromkeys(errors, 0)
    compared = []
    for row in labels['frames']:
        i = row['frame']
        if not isinstance(i, int) or not 0 <= i < len(prediction['frames']):
            raise ValueError('Evaluation frame outside prediction timeline')
        auto = prediction['frames'][i]
        points = auto['racket']['points'] if auto['racket'] else {}
        item = {'frame': i, 'frame_1based': i + 1, 'pixel_errors_original_px': {}}
        if row.get('grip_confirmed', {}).get('points') and 'grip_center' in row['points']:
            gap = float(np.linalg.norm(np.asarray(auto['grip_center_original_px']) - row['points']['grip_center']))
            errors['grip_center'].append(gap)
            denominators['grip_center'] += 1
            item['pixel_errors_original_px']['grip_center'] = gap
            reference = ratio_distance(row['points'], row['points']['grip_center'], prediction['length_m'], prediction['image_size'][0] / 1280)
            item['human_pixel_ratio_reference'] = reference
            if reference.get('usable'):
                for method in ['ratio', 'hand_axis']:
                    estimate = auto[method]
                    if estimate['usable']:
                        item[method + '_difference_from_human_pixel_proxy_cm'] = abs(estimate['distance_m'] - reference['distance_m']) * 100
        for name in errors:
            if name != 'grip_center' and name in row['points']:
                denominators[name] += 1
            if name != 'grip_center' and name in points and name in row['points']:
                gap = float(np.linalg.norm(np.asarray(points[name]) - row['points'][name]))
                errors[name].append(gap)
                item['pixel_errors_original_px'][name] = gap
        compared.append(item)
    count = len(prediction['frames'])
    return {'status': 'regression_diagnostics_not_physical_cm_acceptance', 'accepted': False,
            'frames': count, 'racket_detection_frames': sum(bool(r['racket']) for r in prediction['frames']),
            'visible_butt_frames': sum(bool(r['racket']) and 'handle_end' in r['racket']['points'] for r in prediction['frames']),
            'ratio_usable_frames': sum(r['ratio']['usable'] for r in prediction['frames']),
            'hand_axis_usable_frames': sum(r['hand_axis']['usable'] for r in prediction['frames']),
            'pixel_errors_original_px': {name: summary(values) for name, values in errors.items()},
            'annotated_point_coverage': {name: {'manual_visible_count': denominators[name],
                                              'predicted_count': len(values),
                                              'fraction': len(values) / denominators[name] if denominators[name] else None}
                                         for name, values in errors.items()},
            'distance_estimates_cm': {method: summary([r[method]['distance_m'] * 100 for r in prediction['frames'] if r[method]['usable']]) for method in ['ratio', 'hand_axis']},
            'distance_proxy_difference_cm': {method: summary([r[method + '_difference_from_human_pixel_proxy_cm'] for r in compared if method + '_difference_from_human_pixel_proxy_cm' in r]) for method in ['ratio', 'hand_axis']},
            'records': compared, 'labels_sha256': None, 'physical_distance_ground_truth_available': False,
            'limits': 'Human labels are comparison pixels, not measured centimeters. Conditional pixel errors exclude missing endpoints; coverage must be reported beside errors. Images were inspected in previous diagnostics; not a new blind test.'}


def run(dataset, model, out, device):
    if out.exists():
        raise ValueError('Use a new output directory')
    root = dataset / 'result'
    protected = {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.suffix == '.json'}
    protected['mesh_refined.bin'] = sha256(root / 'mesh_refined.bin')
    out.mkdir(parents=True)
    write(out / 'protected_inputs.json', protected)
    prediction = predict(dataset, model, out, device)
    # The prediction artifact exists before any manual coordinates are read.
    labels = json.loads((root / 'racket_landmarks.json').read_text())
    report = evaluate(prediction, labels)
    report.update(labels_sha256=sha256(root / 'racket_landmarks.json'), predictions_sha256=sha256(out / 'predictions.json'),
                  video_sha256=prediction['video_sha256'], protected_inputs_sha256=protected)
    if any(sha256(root / name) != value for name, value in protected.items()):
        raise ValueError('Live source changed during audit; freeze and rerun evaluation')
    write(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['records', 'protected_inputs_sha256']}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    run(args.dataset, args.model, args.output, args.device)
