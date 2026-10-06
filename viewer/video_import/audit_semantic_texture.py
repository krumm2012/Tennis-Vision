"""Frozen, paired RGB and automatic semantic correspondence audit.

Hair-occluded raw residuals are retained separately from class-matched visible
surface residuals. Model-to-model semantic consistency is not ground-truth IoU.
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from multivideo_texture import load_data, views, visibility, valid_points, sample, validate_fusion_group
from refine_texture_visibility import facing_cosine
from semantic_body_layers import sample_layers, distances
from refine_texture_semantics import checked_semantic, verified_classes, POLICY
from blender_uv_diagnostics import atlas_centers, metrics
from run_records import write, sha256


def evaluate(args):
    out = args.root / 'semantic_validation'; out.mkdir(exist_ok=False)
    write(out / 'protocol.json', {'policy_sha256': sha256(args.root / 'policy.json'),
        'baseline_texture_sha256': sha256(args.baseline / 'body_texture_rgba.png'),
        'candidate_texture_sha256': sha256(args.root / 'candidate/body_texture_rgba.png'),
        'code_sha256': sha256(Path(__file__)), 'clothing_boundary_band_native_px': 5,
        'domains': ['raw_visible_roi', 'semantic_matched_visible', 'clothing_boundary_band', 'hair_occluded'],
        'warning': 'Automatic labels define observational subsets, not manual geometric/segmentation truth'})
    shutil.copy2(__file__, out / Path(__file__).name)
    batch = json.loads(args.batch.read_text()); _, rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    with np.load(args.layout) as rig: faces, uv, uv_faces = rig['faces'], rig['uv'], rig['uv_faces']
    with np.load(args.root / 'train_consensus.npz') as c: target, roi = c['target'], c['roi']
    base = cv2.imread(str(args.baseline / 'body_texture_rgba.png'), -1)
    cand = cv2.imread(str(args.root / 'candidate/body_texture_rgba.png'), -1)
    atlas = [atlas_centers(x, uv, uv_faces) for x in [base, cand]]
    supported = atlas[0][:, 3] > 240
    candidate_supported = atlas[1][:, 3] > 240
    # Categorical nearest-neighbor atlas samples at the fixed UV face centers.
    xy_uv = uv[uv_faces].mean(1); xy_uv[:, 1] = 1 - xy_uv[:, 1]
    xy_uv = np.clip(np.rint(xy_uv * (len(base) - 1) - .5).astype(int), 0, len(base)-1)
    labels = []
    with np.load(args.root / 'semantic_texel_audit.npz') as a:
        for key in ['old_label', 'new_label']:
            image = np.zeros(base.shape[:2], np.uint8); image[a['ys'], a['xs']] = a[key]
            labels.append(image[xy_uv[:, 1], xy_uv[:, 0]])
    labels[1][~candidate_supported] = 0
    report = json.loads((args.baseline / 'texture_report.json').read_text())
    model_hash = json.loads((args.semantic / 'model_binding.json').read_text())['model_sha256']
    pooled = {}; records = []; semantic = {}; exclusions = []
    input_bindings = {}
    for clip, (entry, row) in enumerate(zip(batch['clips'], rows)):
        folder = Path(entry['folder']); name = folder.name
        archive = folder / 'attempts/0001/work/reconstruction.npz'
        data = load_data(archive); meta = json.loads((folder / 'result/mesh_meta.json').read_text())
        mirror = json.loads((folder / 'result/mirror_geometry.json').read_text())
        for path in [archive, folder / 'source.mp4', folder / 'result/mesh_meta.json', folder / 'result/mirror_geometry.json']:
            input_bindings[str(path.resolve())] = sha256(path)
        cap = cv2.VideoCapture(str(folder / 'source.mp4'))
        for split, frames in [('heldout', POLICY['heldout_frames_0based']), ('fresh', POLICY['fresh_frames_0based'])]:
            if set(frames) & set(row['frames']['train_frames']): raise ValueError('TRAIN leaked into evaluation')
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
                if not ok: raise ValueError('Decode failed')
                for view, camera, mask, other in views(data, meta, mirror, frame, 'independent'):
                    path = args.semantic / f'{name}_f{frame:03}_v{view}.npz'
                    sem = checked_semantic(path, name, frame, view, row['inputs']['video']['sha256'], model_hash)
                    input_bindings[str(path.resolve())] = sha256(path)
                    _, depth, eroded, resolution = visibility(camera, faces, mask, meta['focal'][frame], meta['image_size'], other)
                    valid, xy, _ = valid_points(camera[faces].mean(1), eroded, depth, meta['focal'][frame], np.asarray(meta['image_size']), resolution)
                    valid &= (facing_cosine(camera, faces, bool(view)) > .3) & supported & roi
                    raw = sample(image, xy); truth = raw.astype(float) * report['validation'][clip]['gain_bgr']
                    group, confidence, distance = sample_layers(sem['labels'], sem['confidence'], distances(sem['labels']), xy, sem['box'])
                    group = verified_classes(group, confidence, raw)
                    domains = {
                        'raw_visible_roi': valid,
                        'semantic_matched_visible': valid & (target > 0) & (group == target) & (distance >= 3),
                        'clothing_boundary_band': valid & np.isin(group, [2, 3]) & (distance <= 5),
                        'hair_occluded': valid & (group == 1),
                    }
                    errors = [np.abs(a[:, :3] - truth).mean(1) for a in atlas]
                    errors[1] = np.where(candidate_supported, errors[1], 255.)
                    for domain, selection in domains.items():
                        result = dict(clip=name, frame=frame, split=split, view=view, domain=domain)
                        result['lost_supported_samples'] = int((selection & ~candidate_supported).sum())
                        for label, err in zip(['baseline', 'candidate'], errors):
                            result[label] = metrics(err[selection])
                            pooled.setdefault((split, view, domain, label), []).append(err[selection])
                        records.append(result)
                    exclusions.append(dict(clip=name, frame=frame, view=view, split=split,
                        raw_visible=int(valid.sum()), semantic_matched=int(domains['semantic_matched_visible'].sum()),
                        hair_occluded=int(domains['hair_occluded'].sum()), unknown_label=int((valid & (group == 0)).sum()),
                        ambiguous_train_target=int((valid & (target == 0)).sum())))
                    for label, prediction in zip(['baseline', 'candidate'], labels):
                        keep = domains['semantic_matched_visible']
                        metrics_acc = semantic.setdefault((split, view, label), np.zeros(6, np.int64))
                        metrics_acc += [int(keep.sum()), int((keep & (prediction != group)).sum()),
                            int((keep & (prediction == 3) & (group == 3)).sum()),
                            int((keep & (prediction == 3) & (group != 3)).sum()),
                            int((keep & (prediction != 3) & (group == 3)).sum()),
                            int((keep & (prediction == 1)).sum())]
        cap.release(); print('VALIDATE_SEMANTIC', name, flush=True)
    summary = []
    for split in ['heldout', 'fresh']:
        for view in [0, 1]:
            for domain in domains:
                summary.append(dict(split=split, view=view, domain=domain,
                    **{name: metrics(np.concatenate(pooled[(split, view, domain, name)])) for name in ['baseline', 'candidate']}))
    semantic_summary = []
    for (split, view, label), a in semantic.items():
        semantic_summary.append(dict(split=split, view=view, variant=label, n=int(a[0]),
            mismatch_rate=float(a[1]/max(1,a[0])), clothing_face_F1=float(2*a[2]/max(1,2*a[2]+a[3]+a[4])),
            hair_on_visible_surface_rate=float(a[5]/max(1,a[0]))))
    write(out / 'comparison.json', {'summary': summary, 'per_frame': records,
        'semantic_consistency_not_ground_truth': semantic_summary, 'fixed_domain_counts': exclusions,
        'source_sha256': input_bindings, 'geometry_changed': False,
        'fixed_denominator': 'baseline alpha >240; candidate loss penalized255, not excluded',
        'limits': 'Same automatic parser is used for source filtering and semantic evaluation; keep raw-domain RGB and rendered/source visual review. Fresh frames were used in previous photometry evaluation, not independent of all prior project work.'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'baseline', 'batch', 'group', 'layout', 'semantic']: p.add_argument('--'+key, type=Path, required=True)
    evaluate(p.parse_args())
