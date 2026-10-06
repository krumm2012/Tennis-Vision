"""Project frozen atlas texel centers into excluded frames; never train or publish.

All added/unknown texels are evaluated. Old texels use a declared deterministic
UV-order stride. Errors are appearance residuals under estimated MHR geometry,
not independent geometric truth. Real and reflected views remain separate.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from multivideo_texture import (load_data, uv_lookup, views, visibility,
                               valid_points, sample, validate_fusion_group)
from run_records import sha256, write


def regions(primary, candidate, face_map, old_stride):
    if old_stride < 1 or primary.shape != candidate.shape:
        raise ValueError('invalid atlas shapes or sampling stride')
    old = (primary[:, :, 3] > 0) & (face_map >= 0)
    new = (candidate[:, :, 3] > 0) & (face_map >= 0)
    if np.any(old & ~new) or not np.array_equal(primary[old], candidate[old]):
        raise ValueError('candidate changed existing observed RGBA')
    return {'old': np.flatnonzero(old)[::old_stride],
            'added': np.flatnonzero(new & ~old),
            'unknown': np.flatnonzero((face_map >= 0) & ~new)}


def validate_sources(source, rows, known):
    for clip in np.unique(source['clip'][known]):
        if clip < 0 or clip >= len(rows):
            raise ValueError('invalid provenance clip')
        selected = known & (source['clip'] == clip)
        if not set(source['frame'][selected].tolist()) <= set(rows[clip]['frames']['train_frames']):
            raise ValueError('atlas provenance contains non-training frames')
        if not set(source['view'][selected].tolist()) <= {0, 1}:
            raise ValueError('invalid provenance view')


def stats(values):
    x = np.concatenate(values) if values else np.empty(0)
    return {'observations': int(len(x)), 'mean': float(x.mean()) if len(x) else None,
            'median': float(np.median(x)) if len(x) else None,
            'p95': float(np.percentile(x, 95)) if len(x) else None}


def evaluate(batch, group_manifest, layout, primary_path, candidate_path, output, old_stride=64):
    if output.exists():
        raise ValueError('audit output exists; use a new directory')
    batch_doc = json.loads(batch.read_text())
    identity, rows = validate_fusion_group(group_manifest, batch_doc['clips'], layout)
    reports = [json.loads((p / 'texture_report.json').read_text()) for p in [primary_path, candidate_path]]
    assets = {}
    for folder, report in zip([primary_path, candidate_path], reports):
        if report['layout_sha256'] != sha256(layout):
            raise ValueError('atlas layout differs')
        for name in ['body_texture_rgba.png', 'texture_sources.npz', 'appearance_mesh.npz']:
            actual = sha256(folder / name)
            if actual != report['artifact_sha256'][name]:
                raise ValueError('atlas artifact hash differs')
            assets[str((folder / name).resolve())] = actual
    if reports[0]['inputs'] != reports[1]['inputs']:
        raise ValueError('baseline and candidate geometry inputs differ')
    for r in reports:
        if len(r['inputs']) != len(rows):
            raise ValueError('clip count differs')
        for entry, row in zip(r['inputs'], rows):
            if entry['archive_sha256'] != row['inputs']['native_archive']['sha256'] or entry['video_sha256'] != row['inputs']['video']['sha256']:
                raise ValueError('atlas inputs differ from validation group')
    primary = cv2.imread(str(primary_path / 'body_texture_rgba.png'), cv2.IMREAD_UNCHANGED)
    candidate = cv2.imread(str(candidate_path / 'body_texture_rgba.png'), cv2.IMREAD_UNCHANGED)
    if primary is None or candidate is None or primary.shape[2] != 4 or candidate.shape[2] != 4:
        raise ValueError('RGBA atlas required')
    with np.load(layout, allow_pickle=False) as rig:
        faces = rig['faces']; uv = rig['uv']; uv_faces = rig['uv_faces']
    face_map, bary = uv_lookup(uv.astype(np.float32), uv_faces.astype(np.int32), primary.shape[0])
    for folder, atlas in [(primary_path, primary), (candidate_path, candidate)]:
        with np.load(folder / 'texture_sources.npz', allow_pickle=False) as source:
            if not np.array_equal(face_map, source['face']):
                raise ValueError('UV face map differs')
            validate_sources(source, rows, (atlas[:, :, 3] > 0) & (face_map >= 0))
    selected = regions(primary, candidate, face_map, old_stride)
    pixels = {name: np.array(np.unravel_index(ids, face_map.shape)).T for name, ids in selected.items()}
    weights = {name: bary[p[:, 0], p[:, 1]] for name, p in pixels.items()}
    face_ids = {name: face_map[p[:, 0], p[:, 1]] for name, p in pixels.items()}
    atlas_colors = {name: candidate[p[:, 0], p[:, 1], :3].astype(np.float32) for name, p in pixels.items()}
    errors = {(view, name): [] for view in [0, 1] for name in selected}
    observed = {(view, name): np.zeros(len(selected[name]), bool) for view in [0, 1] for name in selected}
    frames = []; evidence = []
    output.mkdir(parents=True)
    for clip, (entry, row) in enumerate(zip(batch_doc['clips'], rows)):
        folder = Path(entry['folder']); data = load_data(folder / 'attempts/0001/work/reconstruction.npz')
        if not np.array_equal(data['faces'], faces):
            raise ValueError('reconstruction topology differs')
        meta = json.loads((folder / 'result/mesh_meta.json').read_text())
        geometry = folder / 'result/mirror_geometry.json'
        mirror = json.loads(geometry.read_text()) if geometry.exists() else None
        gain = np.array(reports[0]['validation'][clip]['gain_bgr'])
        if not np.array_equal(gain, reports[1]['validation'][clip]['gain_bgr']):
            raise ValueError('exposure gains differ')
        cap = cv2.VideoCapture(str(folder / 'source.mp4')); worst = None
        try:
            for index in row['frames']['heldout_frames']:
                if index in row['frames']['train_frames']:
                    raise ValueError('training/heldout overlap')
                cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = cap.read()
                if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index + 1:
                    raise ValueError('heldout frame decode mismatch')
                for view, camera, mask, other in views(data, meta, mirror, index, 'independent'):
                    _, depth, eroded, resolution = visibility(camera, faces, mask, meta['focal'][index], meta['image_size'], other)
                    tri = camera[faces]; centers = tri.mean(1)
                    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
                    facing = abs((normals * -centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)).sum(1))
                    record = {'clip': folder.name, 'frame': index, 'view': 'real' if view == 0 else 'mirror', 'regions': {}}
                    for name in selected:
                        points = (camera[faces[face_ids[name]]] * weights[name][:, :, None]).sum(1)
                        valid, xy, _ = valid_points(points, eroded, depth, meta['focal'][index], np.array(meta['image_size']), resolution)
                        valid &= facing[face_ids[name]] > .3
                        observed[view, name] |= valid
                        values = np.empty(0, np.float32)
                        if name != 'unknown' and valid.any():
                            truth = np.clip(sample(image, xy[valid]).astype(np.float32) * gain, 0, 255)
                            values = np.abs(atlas_colors[name][valid] - truth).mean(1)
                            errors[view, name].append(values)
                        record['regions'][name] = {'visible_texels': int(valid.sum()), 'error_bgr_mae_0_255': stats([values])}
                        if name == 'added' and view == 0 and len(values) >= 50 and (worst is None or values.mean() > worst[0]):
                            worst = (float(values.mean()), image.copy(), xy[valid].copy(), values.copy(), index)
                    frames.append(record)
        finally:
            cap.release()
        if worst is not None:
            mean, image, xy, values, index = worst
            overlay = image.copy()
            for location in np.arange(0, len(xy), max(1, len(xy) // 700)):
                color = (40, 210, 40) if values[location] < 20 else (20, 140, 255) if values[location] < 40 else (40, 40, 240)
                cv2.circle(overlay, tuple(np.rint(xy[location]).astype(int)), 3, color, -1)
            cv2.putText(overlay, f'{folder.name} heldout frame {index + 1} / added MAE {mean:.2f}', (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            path = output / f'{folder.name}_added_worst_real.jpg'; cv2.imwrite(str(path), overlay)
            evidence.append({'clip': folder.name, 'frame': index, 'path': str(path.resolve()), 'sha256': sha256(path), 'selection': 'largest mean added-region residual among heldout real frames with >=50 samples; diagnostic, not representative average'})
        print('Heldout atlas audit', folder.name, flush=True)
    summary = {}
    for view, label in [(0, 'real'), (1, 'mirror')]:
        summary[label] = {name: {'sampled_texels': len(selected[name]), 'unique_visible_texels': int(observed[view, name].sum()), 'error_bgr_mae_0_255': stats(errors[view, name])} for name in selected}
    result = {'status': 'heldout_pixel_audit_completed_needs_visual_review', 'group': identity,
              'input_sha256': {str(batch.resolve()): sha256(batch), str(group_manifest.resolve()): sha256(group_manifest), str(layout.resolve()): sha256(layout), **assets},
              'code_sha256': sha256(Path(__file__)), 'projection_code_sha256': sha256(Path(__file__).with_name('multivideo_texture.py')),
              'old_uv_order_stride': old_stride, 'added_and_unknown_sampling': 'all template texel centers',
              'gains': 'fixed TRAIN-only gains from baseline; not estimated from heldout',
              'summary': summary, 'frames': frames, 'evidence': evidence,
              'old_region_baseline_candidate_identical': True, 'published': False,
              'limits': ['Errors are appearance residuals under estimated MHR geometry, masks and camera; not independent geometric truth.',
                         'Samples across frames/texels are correlated; no confidence interval or calibrated acceptance threshold.',
                         'All heldout times excluded from atlas provenance; reflected-view geometry remains an independent-estimation assumption.',
                         'Legacy observation-cache generation binding was not retrospectively verified.',
                         'Atlas texel centers evaluated without display filtering; gutter RGB and alpha blending are excluded.']}
    write(output / 'heldout_pixel_report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['batch', 'group-manifest', 'layout', 'primary', 'candidate', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--old-stride', type=int, default=64)
    a = parser.parse_args()
    result = evaluate(a.batch, a.group_manifest, a.layout, a.primary, a.candidate, a.output, a.old_stride)
    print(json.dumps(result['summary']))
