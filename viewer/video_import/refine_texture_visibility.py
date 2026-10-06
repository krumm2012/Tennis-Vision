"""Frozen-geometry, TRAIN-only one-sided texture correspondence experiment.

Mask/depth observations remain automatic estimates. A reflected independent mesh
reverses winding once. Backfacing samples are not valid texture observations even
when they fall within the old 25 mm depth threshold on a thin hand.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from multivideo_texture import (load_data, views, visibility, sample, project,
                               validate_fusion_group, validate_cache_split, fuse)
from blender_uv_diagnostics import atlas_centers, metrics
from run_records import sha256, write


def regions(rest, faces):
    center = rest[faces].mean(axis=1)
    h = -center[:, 1]
    hands = abs(center[:, 0]) > .48
    shoulders_back = ((h > 1.28) & (h < 1.56) & (abs(center[:, 0]) < .35)) | (
        (h > .90) & (h < 1.50) & (abs(center[:, 0]) < .25) & (center[:, 2] > -.04))
    return {'all': np.ones(len(faces), bool), 'hands_proxy': hands,
            'shoulders_back_proxy': shoulders_back & ~hands}


def facing_cosine(camera, faces, reflected=False):
    tri = camera[faces]; center = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dot = (normal * -center).sum(axis=1)
    cosine = dot / np.maximum(np.linalg.norm(normal, axis=1) * np.linalg.norm(center, axis=1), 1e-12)
    return -cosine if reflected else cosine


def uv_seam_pairs(faces, uv_faces, uv):
    """Fixed samples halfway from a shared edge into its two face centers.

    Sampling exactly on an island boundary mixes transparent gutter pixels and
    yields too few supported pairs. These interior samples are a seam proxy,
    not identical physical points or an independent color ground truth.
    """
    edges = {}; pairs = []; owners = []
    for i, face in enumerate(faces):
        for k in range(3):
            a, b = sorted((int(face[k]), int(face[(k + 1) % 3])))
            slots = [int(np.where(face == v)[0][0]) for v in [a, b]]
            ids = uv_faces[i, slots]
            point = .5 * uv[ids].mean(axis=0) + .5 * uv[uv_faces[i]].mean(axis=0)
            if (a, b) in edges:
                j, old_ids, old_point = edges[(a, b)]
                if not np.array_equal(ids, old_ids):
                    pairs.append([old_point, point]); owners.append([j, i])
            else: edges[(a, b)] = (i, ids, point)
    return np.asarray(pairs), np.asarray(owners)


def prepare(batch, layout, observations, group, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((batch / 'batch_manifest.json').read_text())
    _, rows = validate_fusion_group(group, manifest['clips'], layout)
    with np.load(layout, allow_pickle=False) as rig:
        faces = rig['faces']; area_regions = regions(rig['rest_vertices'], faces)
    tolerance = np.where(area_regions['hands_proxy'], .01, .025).astype(np.float32)
    np.save(output / 'face_depth_tolerance.npy', tolerance, allow_pickle=False)
    np.savez_compressed(output / 'regions.npz', **area_regions)
    audit = []
    for entry, row in zip(manifest['clips'], rows):
        folder = Path(entry['folder']); name = folder.name
        path = observations / (name + '_observations.npz')
        with np.load(path, allow_pickle=False) as old:
            validate_cache_split(old, row)
            cache = {k: old[k] for k in old.files}
        meta = json.loads((folder / 'result/mesh_meta.json').read_text())
        archive = folder / 'attempts/0001/work/reconstruction.npz'
        data = load_data(archive)
        mirror = json.loads((folder / 'result/mirror_geometry.json').read_text())
        reliable = np.zeros(len(cache['heldout_face']), bool)
        diagnostics = []
        for index in sorted(set(cache['all_frame'].tolist()) | set(cache['heldout_frame'].tolist())):
            for view, camera, mask, other in views(data, meta, mirror, index, 'independent'):
                cosine = facing_cosine(camera, faces, reflected=bool(view))
                _, depth, eroded, res = visibility(camera, faces, mask, meta['focal'][index], meta['image_size'], other)
                centers = camera[faces].mean(axis=1)
                xy = project(centers, meta['focal'][index], meta['image_size'])
                gap = abs(centers[:, 2] - sample(depth, xy * res / meta['image_size']))
                valid = (cosine > .3) & (gap < tolerance)
                selected = np.where((cache['all_frame'] == index) & (cache['all_view'] == view))[0]
                original = np.zeros(len(faces), bool)
                if len(selected):
                    original = cache['all_score'][selected[0]] > 0
                    cache['all_score'][selected[0], ~valid] = 0
                heldout = (cache['heldout_frame'] == index) & (cache['heldout_view'] == view)
                reliable[heldout] = valid[cache['heldout_face'][heldout]]
                if heldout.any(): original[cache['heldout_face'][heldout]] = True
                clearance = sample(cv2.distanceTransform((mask > 234).astype(np.uint8), cv2.DIST_L2, 5),
                                   xy * np.array(mask.shape[::-1]) / meta['image_size']) * meta['image_size'][0] / mask.shape[1]
                for region, region_mask in area_regions.items():
                    ids = original & region_mask
                    diagnostics.append({'clip': name, 'frame': int(index), 'view': int(view),
                                        'split': 'heldout' if heldout.any() else 'train', 'region': region,
                                        'old_supported': int(ids.sum()), 'backfacing': int((ids & (cosine <= 0)).sum()),
                                        'rejected': int((ids & ~valid).sum()),
                                        'depth_gap_mm': metrics(gap[ids] * 1000),
                                        'mask_clearance_native_px': metrics(clearance[ids])})
                # Source-pixel evidence at a TRAIN frame; no warped or enlarged crops.
                if index == 80:
                    cap = cv2.VideoCapture(str(folder / 'source.mp4')); cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read(); cap.release()
                    if not ok: raise ValueError('Source frame decode failed')
                    canvas = image.copy()
                    mask_big = cv2.resize((mask > 234).astype(np.uint8), tuple(meta['image_size']), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, (40, 220, 40), 1)
                    for region in ['hands_proxy', 'shoulders_back_proxy']:
                        ids = np.flatnonzero(original & area_regions[region])
                        for face in ids[::3]:
                            p = tuple(np.round(xy[face]).astype(int))
                            cv2.circle(canvas, p, 1, (30, 90, 245) if not valid[face] else (240, 190, 30), -1)
                        if len(ids):
                            low = np.maximum(0, xy[ids].min(0).astype(int) - 25)
                            high = np.minimum(meta['image_size'], xy[ids].max(0).astype(int) + 26)
                            cv2.imwrite(str(output / f'{name}_f080_v{view}_{region}.png'), canvas[low[1]:high[1], low[0]:high[0]])
        best = cache['all_score'].argmax(axis=0); ids = np.arange(len(faces))
        cache.update(score=cache['all_score'][best, ids], frame=cache['all_frame'][best],
                     view=cache['all_view'][best], color=cache['all_color'][best, ids], heldout_reliable=reliable)
        target = output / path.name; np.savez_compressed(target, **cache)
        identity = json.loads(path.with_suffix('.json').read_text())
        # This derivative has current bindings, but does not erase legacy provenance.
        identity.update(mesh_meta_sha256=sha256(folder / 'result/mesh_meta.json'),
                        observation_cache_sha256=sha256(target), derivative_code_sha256=sha256(Path(__file__)),
                        parent_cache_sha256=sha256(path), parent_binding='legacy_generation_metadata_unverified',
                        visibility_policy='front cosine > .3, mirror handedness corrected; hand depth 10mm, other 25mm')
        write(target.with_suffix('.json'), identity)
        write(output / (name + '_geometry_mask_audit.json'), diagnostics)
        audit.extend(diagnostics)
        print('VISIBILITY', name, flush=True)
    summary = []
    for view in [0, 1]:
        for region in area_regions:
            rows = [r for r in audit if r['view'] == view and r['region'] == region and r['split'] == 'train']
            count = sum(r['old_supported'] for r in rows)
            summary.append({'view': view, 'region': region, 'original_observations': count,
                            'backfacing_fraction': sum(r['backfacing'] for r in rows) / max(count, 1),
                            'rejected_fraction': sum(r['rejected'] for r in rows) / max(count, 1)})
    write(output / 'audit_summary.json', {'train': summary, 'geometry_changed': False,
          'mask_native_resolution': [512, 288], 'original_video_resolution': [2560, 1440],
          'regions': 'Rest-template coordinate proxies, not manually annotated anatomical parts',
          'policy': 'Frozen before heldout evaluation; no pose/shape fitting, no mask relabeling',
          'source_layout_sha256': sha256(layout), 'group_sha256': sha256(group)})


def evaluate(layout, baseline, candidate, observations, output):
    with np.load(layout, allow_pickle=False) as rig:
        uv, uv_faces, faces = rig['uv'], rig['uv_faces'], rig['faces']
        area_regions = regions(rig['rest_vertices'], faces)
    atlas = [cv2.imread(str(p / 'body_texture_rgba.png'), -1) for p in (baseline, candidate)]
    if atlas[0].shape != atlas[1].shape: raise ValueError('Fixed resolution required')
    values = [atlas_centers(a, uv, uv_faces) for a in atlas]
    reports = [json.loads((p / 'texture_report.json').read_text()) for p in (baseline, candidate)]
    all_rows = []; pooled = {}
    for v in reports[0]['validation']:
        name = v['video']; other = next(x for x in reports[1]['validation'] if x['video'] == name)
        if v['gain_bgr'] != other['gain_bgr']: raise ValueError('Exposure must stay fixed in visibility A/B')
        with np.load(observations / (name + '_observations.npz'), allow_pickle=False) as cache:
            ids = cache['heldout_face']; truth = cache['heldout_color'] * np.asarray(v['gain_bgr'])
            old_supported = values[0][ids, 3] > 240
            new_supported = values[1][ids, 3] > 240
            errors = [abs(val[ids, :3] - truth).mean(axis=1) for val in values]
            for view in [0, 1]:
                for region, selection in area_regions.items():
                    for domain in ['original_fixed', 'front_facing_fixed']:
                        keep = old_supported & selection[ids] & (cache['heldout_view'] == view)
                        if domain == 'front_facing_fixed': keep &= cache['heldout_reliable']
                        base = errors[0][keep]; cand = np.where(new_supported[keep], errors[1][keep], 255.)
                        common = keep & new_supported
                        row = {'clip': name, 'view': view, 'region': region, 'domain': domain,
                               'baseline_fixed': metrics(base), 'candidate_fixed': metrics(cand),
                               'lost_samples': int((keep & ~new_supported).sum()),
                               'baseline_common': metrics(errors[0][common]), 'candidate_common': metrics(errors[1][common])}
                        all_rows.append(row)
                        pool = pooled.setdefault((view, region, domain), [[], [], [], [], 0])
                        for dest, a in zip(pool[:4], [base, cand, errors[0][common], errors[1][common]]): dest.append(a)
                        pool[4] += row['lost_samples']
    summary = []
    for (view, region, domain), arrays in pooled.items():
        summary.append({'view': view, 'region': region, 'domain': domain,
                        **{key: metrics(np.concatenate(a)) for key, a in zip(['baseline_fixed', 'candidate_fixed', 'baseline_common', 'candidate_common'], arrays[:4])},
                        'lost_samples': arrays[4]})
    with np.load(baseline / 'texture_sources.npz', allow_pickle=False) as s:
        surface = s['face'] >= 0
        coverage = {'baseline': float((atlas[0][:, :, 3][surface] > 0).mean()), 'candidate': float((atlas[1][:, :, 3][surface] > 0).mean()),
                    'lost_texels': int((surface & (atlas[0][:, :, 3] > 0) & (atlas[1][:, :, 3] == 0)).sum())}
    pairs, owners = uv_seam_pairs(faces, uv_faces, uv)
    seam_values = []
    for image in atlas:
        points = pairs.copy(); points[:, :, 1] = 1 - points[:, :, 1]
        seam_values.append(sample(image, (points * (len(image) - 1) - .5).reshape(-1, 2)).reshape(-1, 2, 4))
    seam_report = []
    for region, mask in area_regions.items():
        selected = mask[owners].any(axis=1) & (seam_values[0][:, :, 3] > 240).all(axis=1)
        valid = (seam_values[1][:, :, 3] > 240).all(axis=1)
        errors = [abs(value[:, 0, :3].astype(float) - value[:, 1, :3]).mean(axis=1) for value in seam_values]
        seam_report.append({'region': region, 'baseline_fixed': metrics(errors[0][selected]),
                            'candidate_fixed': metrics(np.where(valid[selected], errors[1][selected], 255.)),
                            'baseline_common': metrics(errors[0][selected & valid]),
                            'candidate_common': metrics(errors[1][selected & valid]),
                            'lost_seam_pairs': int((selected & ~valid).sum())})
    write(output / 'heldout_comparison.json', {'summary': summary, 'per_clip': all_rows, 'coverage': coverage,
          'uv_chart_seam_color_MAE_0_255': seam_report,
          'seam_metric_scope': 'Fixed interior points halfway from each geometric seam edge to adjacent face centers; color discontinuity proxy, may include real texture edges and shading',
          'fixed_denominator_loss_penalty': 255, 'heldout_frames_unchanged': True,
          'limits': 'Old backfacing heldout colors may themselves sample the opposite surface. Front-facing subset is fixed by geometry/mask, before inspecting candidates; both domains retained. Not independent geometric ground truth.'})
    print(json.dumps({'coverage': coverage, 'hands': [r for r in summary if r['region'] == 'hands_proxy' and r['domain'] == 'front_facing_fixed']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'fuse', 'evaluate'])
    for name in ['batch', 'layout', 'observations', 'group', 'output', 'baseline']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'prepare': prepare(a.batch, a.layout, a.observations, a.group, a.output / 'observations')
    if a.stage == 'fuse':
        gains = [v['gain_bgr'] for v in json.loads((a.baseline / 'texture_report.json').read_text())['validation']]
        fuse(a.batch, a.layout, a.output / 'candidate', size=2048, step=5, mirror_source='independent', selection='consistency',
             observation_root=a.output / 'observations', pixel_fallback=True, fallback_max_candidates=3,
             group_manifest=a.group, face_depth_tolerance=np.load(a.output / 'observations/face_depth_tolerance.npy'), fixed_exposure_gains=gains)
    if a.stage == 'evaluate': evaluate(a.layout, a.baseline, a.output / 'candidate', a.output / 'observations', a.output)
