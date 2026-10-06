"""Conservative semantic source repair for shoulders/back; source pixels only.

Hair is an occluder on the torso, not a valid torso material. Unresolved pixels
are retained in the display candidate and explicitly marked, with a separate
strict-support mask. No unseen skin is synthesized and no geometry is altered.
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from multivideo_texture import (load_data, views, visibility, valid_points, sample, uv_lookup,
                               project, validate_fusion_group, validate_cache_split)
from refine_texture_visibility import regions, facing_cosine, evaluate
from semantic_body_layers import sample_layers, distances
from blender_uv_diagnostics import atlas_centers, metrics
from audit_texture_heldout import validate_sources
from run_records import write, sha256

POLICY = {'scope': 'shoulders_back_proxy; automatic masks, not human labels',
          'min_observation_confidence': .85, 'min_boundary_distance_native_px': 3.,
          'min_target_observations': 8, 'min_target_clips': 2, 'min_target_agreement': .8,
          'white_upper_max_saturation': 65, 'white_upper_min_value': 75,
          'source_candidates': 8, 'source_depth_tolerance_m': .025,
          'display_unresolved': 'retain baseline and flag; strict evidence support excludes confident hair',
          'no_color_correction_or_seam_blending': True,
          'gates': {'semantic_visible_mean_tolerance': .1, 'per_clip_mean_tolerance': .5,
                    'minimum_hair_repair_fraction': .5, 'coverage_loss': 0.,
                    'outside_roi_unchanged': True, 'visual_review_required': True},
          'heldout_frames_0based': [25, 125, 225], 'fresh_frames_0based': [137, 213]}


def checked_semantic(path, name, frame, view, video_hash, model_hash):
    record = json.loads(path.with_suffix('.json').read_text())
    if (record['clip'], record['frame'], record['view'], record['video_sha256'], record['model_sha256']) != (name, frame, view, video_hash, model_hash):
        raise ValueError('Semantic source identity mismatch')
    if record['cache_sha256'] != sha256(path): raise ValueError('Semantic cache changed')
    with np.load(path) as data: return {k: data[k] for k in data.files}


def verified_classes(labels, confidence, colors):
    """Outfit-specific white-shirt veto; uncertain cloth never becomes skin by rule."""
    result = labels.copy()
    hsv = cv2.cvtColor(np.asarray(colors, np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    false_cloth = (result == 3) & ((hsv[:, 1] > POLICY['white_upper_max_saturation']) |
                                 (hsv[:, 2] < POLICY['white_upper_min_value']))
    result[false_cloth | (confidence < POLICY['min_observation_confidence'])] = 0
    return result


def consensus_target(labels, reliable, clips, roi):
    votes = np.stack([((labels == group) & reliable).sum(0) for group in [2, 3]])
    total = votes.sum(0); winner = votes.argmax(0) + 2
    agreement = votes.max(0) / np.maximum(total, 1)
    clip_support = np.zeros(len(roi), int)
    for clip in np.unique(clips):
        clip_support += ((labels[clips == clip] == winner) & reliable[clips == clip]).any(0)
    valid = roi & (total >= POLICY['min_target_observations']) & (clip_support >= POLICY['min_target_clips']) & (agreement >= POLICY['min_target_agreement'])
    return np.where(valid, winner, 0).astype(np.uint8), total, agreement


def repair_colors(baseline, locations, colors, alpha, valid):
    """Mutate only explicitly validated locations; missing alternatives retain baseline."""
    accepted = locations[valid]
    baseline[accepted, :3] = colors[valid]; baseline[accepted, 3] = alpha[valid]
    return accepted


def build(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    policy_path = out / 'policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != POLICY: raise ValueError('Frozen policy differs')
    if (out / 'candidate').exists(): raise ValueError('Candidate already exists; use new output')
    write(policy_path, POLICY); shutil.copy2(__file__, out / Path(__file__).name)
    batch = json.loads(args.batch.read_text()); _, rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    report = json.loads((args.baseline / 'texture_report.json').read_text())
    if report['appearance_group']['manifest_sha256'] != sha256(args.group): raise ValueError('Group changed')
    for k in ['body_texture_rgba.png', 'texture_sources.npz', 'appearance_mesh.npz']:
        if sha256(args.baseline / k) != report['artifact_sha256'][k]: raise ValueError('Baseline changed')
    with np.load(args.layout) as rig:
        faces, uv, uv_faces, rest = rig['faces'], rig['uv'], rig['uv_faces'], rig['rest_vertices']
    roi = regions(rest, faces)['shoulders_back_proxy']
    model_hash = json.loads((args.semantic / 'model_binding.json').read_text())['model_sha256']
    all_labels = []; all_confidence = []; all_distance = []; all_scores = []; keys = []; hashes = {}
    for clip, (entry, row) in enumerate(zip(batch['clips'], rows)):
        name = Path(entry['folder']).name
        cache_path = args.observations / f'{name}_observations.npz'
        hashes[str(cache_path.resolve())] = sha256(cache_path)
        with np.load(cache_path) as cache:
            validate_cache_split(cache, row)
            for i, (frame, view) in enumerate(zip(cache['all_frame'], cache['all_view'])):
                frame, view = int(frame), int(view)
                path = args.semantic / f'{name}_f{frame:03}_v{view}.npz'
                d = checked_semantic(path, name, frame, view, row['inputs']['video']['sha256'], model_hash)
                labels = verified_classes(d['face_label'], d['face_confidence'], cache['all_color'][i])
                all_labels.append(labels); all_confidence.append(d['face_confidence']); all_distance.append(d['face_distance'])
                all_scores.append(cache['all_score'][i]); keys.append((clip, frame, view))
                hashes[str(path.resolve())] = sha256(path)
        print('CONSENSUS', name, flush=True)
    labels = np.stack(all_labels); confidence = np.stack(all_confidence)
    distance = np.stack(all_distance); scores = np.stack(all_scores)
    keys = np.asarray(keys); reliable = (scores > 0) & (distance >= POLICY['min_boundary_distance_native_px'])
    target, count, agreement = consensus_target(labels, reliable, keys[:, 0], roi)
    eligible = reliable & (labels == target) & (target > 0)
    rank_score = np.where(eligible, scores * confidence, 0)
    ranked = np.argsort(-rank_score, axis=0, kind='stable')[:POLICY['source_candidates']].astype(np.int32)
    ranked[np.take_along_axis(rank_score, ranked, axis=0) <= 0] = -1
    np.savez_compressed(out / 'train_consensus.npz', target=target, count=count, agreement=agreement,
                        roi=roi, labels=labels, reliable=reliable, keys=keys, ranked=ranked)
    baseline = cv2.imread(str(args.baseline / 'body_texture_rgba.png'), -1)
    with np.load(args.baseline / 'texture_sources.npz') as s: sources = {k: s[k] for k in s.files}
    validate_sources(sources, rows, (sources['face'] >= 0) & (baseline[:, :, 3] > 0))
    fm, bary = uv_lookup(uv.astype(np.float32), uv_faces.astype(np.int32), baseline.shape[0])
    if not np.array_equal(fm, sources['face']): raise ValueError('UV rasterization changed')
    inside = (fm >= 0) & roi[np.maximum(fm, 0)] & (baseline[:, :, 3] > 0)
    ys, xs = np.where(inside); fids = fm[ys, xs]; weights = bary[ys, xs]
    old_rgba = baseline[ys, xs].copy(); candidate_rgba = old_rgba.copy()
    old_label = np.zeros(len(ys), np.uint8); old_conf = np.zeros(len(ys), np.float32)
    old_distance = np.zeros(len(ys), np.float32)
    source_clip = sources['clip'][ys, xs]; source_frame = sources['frame'][ys, xs]; source_view = sources['view'][ys, xs]
    active_clip = None; data = meta = mirror = cap = image = None; active_frame = None
    def context(key):
        nonlocal active_clip, data, meta, mirror, cap, image, active_frame
        clip, frame, view = map(int, key)
        folder = Path(batch['clips'][clip]['folder']); name = folder.name
        if active_clip != clip:
            if cap is not None: cap.release()
            data = load_data(folder / 'attempts/0001/work/reconstruction.npz')
            meta = json.loads((folder / 'result/mesh_meta.json').read_text()); mirror = json.loads((folder / 'result/mirror_geometry.json').read_text())
            cap = cv2.VideoCapture(str(folder / 'source.mp4')); active_clip = clip; active_frame = None
        if active_frame != frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
            if not ok: raise ValueError('Cannot decode TRAIN frame')
            active_frame = frame
        semantic_path = args.semantic / f'{name}_f{frame:03}_v{view}.npz'
        semantic = checked_semantic(semantic_path, name, frame, view, rows[clip]['inputs']['video']['sha256'], model_hash)
        for actual, camera, mask, other in views(data, meta, mirror, frame, 'independent'):
            if actual == view: return clip, frame, view, camera, mask, other, semantic
        raise ValueError('Source view unavailable')
    try:
        for key in keys:
            selected = np.flatnonzero((source_clip == key[0]) & (source_frame == key[1]) & (source_view == key[2]))
            if not len(selected): continue
            clip, frame, view, camera, mask, other, sem = context(key)
            points = (camera[faces[fids[selected]]] * weights[selected, :, None]).sum(1)
            xy = project(points, meta['focal'][frame], meta['image_size'])
            lab, conf, dist = sample_layers(sem['labels'], sem['confidence'], distances(sem['labels']), xy, sem['box'])
            lab = verified_classes(lab, conf, sample(image, xy))
            old_label[selected], old_conf[selected], old_distance[selected] = lab, conf, dist
        hair = (old_label == 1) & (old_conf >= POLICY['min_observation_confidence'])
        disagreement = (target[fids] > 0) & (old_label > 0) & (old_label != target[fids]) & (old_distance >= POLICY['min_boundary_distance_native_px'])
        requested = hair | disagreement
        repairable = requested & (target[fids] > 0)
        repaired = np.zeros(len(ys), bool); new_label = old_label.copy()
        new_clip, new_frame, new_view = source_clip.copy(), source_frame.copy(), source_view.copy()
        rank_used = np.full(len(ys), -1, np.int8)
        for rank, candidate_ids in enumerate(ranked):
            ids = candidate_ids[fids]
            missing = repairable & ~repaired & (ids >= 0)
            for source_id in np.unique(ids[missing]):
                selected = np.flatnonzero(missing & (ids == source_id))
                clip, frame, view, camera, mask, other, sem = context(keys[source_id])
                points = (camera[faces[fids[selected]]] * weights[selected, :, None]).sum(1)
                _, depth, eroded, resolution = visibility(camera, faces, mask, meta['focal'][frame], meta['image_size'], other)
                ok, xy, mask_confidence = valid_points(points, eroded, depth, meta['focal'][frame], np.asarray(meta['image_size']), resolution)
                ok &= facing_cosine(camera, faces, bool(view))[fids[selected]] > .3
                raw = sample(image, xy)
                lab, conf, dist = sample_layers(sem['labels'], sem['confidence'], distances(sem['labels']), xy, sem['box'])
                lab = verified_classes(lab, conf, raw)
                ok &= (lab == target[fids[selected]]) & (conf >= POLICY['min_observation_confidence']) & (dist >= POLICY['min_boundary_distance_native_px'])
                color = np.clip(raw.astype(float) * report['validation'][clip]['gain_bgr'], 0, 255).astype(np.uint8)
                accepted = repair_colors(candidate_rgba, selected, color, (mask_confidence * 255).astype(np.uint8), ok)
                repaired[accepted] = True; new_label[accepted] = lab[ok]
                new_clip[accepted] = clip; new_frame[accepted] = frame; new_view[accepted] = view; rank_used[accepted] = rank
            print('REPAIR_RANK', rank, int(repaired.sum()), flush=True)
    finally:
        if cap is not None: cap.release()
    candidate = baseline.copy(); candidate[ys, xs] = candidate_rgba
    dest = out / 'candidate'; dest.mkdir()
    cv2.imwrite(str(dest / 'body_texture_rgba.png'), candidate)
    sources['clip'][ys, xs] = new_clip; sources['frame'][ys, xs] = new_frame; sources['view'][ys, xs] = new_view
    if 'fallback_rank' in sources: sources['fallback_rank'][ys[repaired], xs[repaired]] = 0
    validate_sources(sources, rows, (fm >= 0) & (candidate[:, :, 3] > 0))
    np.savez_compressed(dest / 'texture_sources.npz', **sources)
    np.savez_compressed(out / 'semantic_texel_audit.npz', ys=ys, xs=xs, face=fids, old_label=old_label,
        new_label=new_label, requested=requested, hair=hair, repaired=repaired, rank=rank_used,
        old_clip=source_clip, old_frame=source_frame, old_view=source_view)
    repaired_mask = np.zeros(fm.shape, np.uint8); repaired_mask[ys[repaired], xs[repaired]] = 255
    unresolved_mask = np.zeros(fm.shape, np.uint8); unresolved_mask[ys[hair & ~repaired], xs[hair & ~repaired]] = 255
    cv2.imwrite(str(out / 'repaired_mask.png'), repaired_mask)
    cv2.imwrite(str(out / 'unresolved_hair_mask.png'), unresolved_mask)
    strict = (candidate[:, :, 3] > 0) & (fm >= 0) & (unresolved_mask == 0)
    cv2.imwrite(str(out / 'strict_semantic_support.png'), strict.astype(np.uint8) * 255)
    candidate_report = dict(report); candidate_report['status'] = 'semantic_source_repair_candidate_not_accepted'
    candidate_report['semantic_policy_sha256'] = sha256(policy_path)
    candidate_report['artifact_sha256'] = {k: sha256(dest / k) for k in ['body_texture_rgba.png', 'texture_sources.npz']}
    write(dest / 'texture_report.json', candidate_report)
    summary = {'roi_faces': int(roi.sum()), 'target_faces': int((target > 0).sum()), 'roi_observed_texels': len(ys),
        'requested_repair': int(requested.sum()), 'hair_on_torso_flagged': int(hair.sum()),
        'repaired': int(repaired.sum()), 'hair_repaired': int((hair & repaired).sum()),
        'unresolved_hair': int((hair & ~repaired).sum()),
        'hair_repair_fraction': float((hair & repaired).sum() / max(1, hair.sum())),
        'baseline_observed_coverage': float(((baseline[:, :, 3] > 0) & (fm >= 0)).sum() / (fm >= 0).sum()),
        'candidate_observed_coverage': float(((candidate[:, :, 3] > 0) & (fm >= 0)).sum() / (fm >= 0).sum()),
        'candidate_strict_semantic_support_fraction': float(strict.sum() / (fm >= 0).sum()),
        'strict_support_limit': 'only flags confident hair within shoulder/back proxy; not proof of all other texels correctness',
        'outside_roi_identical': bool(np.array_equal(candidate[~inside], baseline[~inside])),
        'geometry_changed': False, 'unseen_skin_synthesized': False,
        'semantic_sources_sha256': hashes, 'model_sha256': model_hash,
        'baseline_texture_sha256': sha256(args.baseline / 'body_texture_rgba.png')}
    write(out / 'repair_summary.json', summary)
    # Unmasked historical regression retained; it includes hair-occluded observations.
    evaluate(args.layout, args.baseline, dest, args.observations, dest)
    print(json.dumps({k: v for k, v in summary.items() if k != 'semantic_sources_sha256'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['batch', 'group', 'layout', 'baseline', 'semantic', 'observations', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    build(p.parse_args())
