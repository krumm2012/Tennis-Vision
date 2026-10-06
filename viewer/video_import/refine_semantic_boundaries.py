"""Native-pixel rejection of ambiguous white straps and hair/material boundaries.

This outfit-specific policy only demotes labels to unknown. It cannot recover
unseen surfaces or establish that a bright line is clothing. Discovery frames
remain separate from the frozen texture TRAIN and evaluation splits.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from multivideo_texture import load_data, views, project, validate_fusion_group
from refine_texture_visibility import regions
from semantic_body_layers import distances, sample_layers, PALETTE_BGR
from run_records import sha256, write

POLICY = dict(version='white_outfit_boundary_guard_v1', mode='reject_to_unknown_only',
    bright_ridge_kernel_px=9, bright_ridge_min=25, neutral_max_saturation=80,
    neutral_min_value=120, max_distance_to_predicted_upper_px=40,
    ridge_margin_px=1, hair_margin_px=2, confidence_min=.85,
    scope='frozen shoulder/back projected bounding rectangle',
    development='Four source-pixel failures from 20261004 discovery; not independent truth.',
    no_new_skin_or_clothing_labels=True, no_heldout_parameter_tuning=True)


def boundary_guard(image, labels, confidence, roi):
    if image.shape[:2] != labels.shape or labels.shape != confidence.shape or labels.shape != roi.shape:
        raise ValueError('Native image and masks must share coordinates')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    k = POLICY['bright_ridge_kernel_px']
    ridge = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if np.any(labels == 3):
        near_cloth = cv2.distanceTransform((labels != 3).astype(np.uint8), cv2.DIST_L2, 5) <= POLICY['max_distance_to_predicted_upper_px']
    else:
        near_cloth = np.zeros(labels.shape, bool)
    neutral = (hsv[:, :, 1] <= POLICY['neutral_max_saturation']) & (hsv[:, :, 2] >= POLICY['neutral_min_value'])
    white_ridge = roi & near_cloth & neutral & (ridge >= POLICY['bright_ridge_min']) & (labels == 2)
    white_guard = (cv2.dilate(white_ridge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (labels == 2) & roi
    hair_near = cv2.dilate((labels == 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hair_guard = roi & hair_near & np.isin(labels, [2, 3])
    # Parser uncertainty never constitutes an inferred skin/cloth correction.
    weak = roi & np.isin(labels, [2, 3]) & (confidence.astype(float) / 255 < POLICY['confidence_min'])
    rejected = white_guard | hair_guard | weak
    out, certainty = labels.copy(), confidence.copy()
    out[rejected] = 0; certainty[rejected] = 0
    return out, certainty, dict(white_ridge=white_guard, hair_boundary=hair_guard,
                                weak_material=weak, rejected=rejected)


def refine(args):
    root = args.output; root.mkdir(parents=True, exist_ok=False)
    write(root / 'boundary_policy.json', POLICY); shutil.copy2(__file__, root / Path(__file__).name)
    batch = json.loads(args.batch.read_text())
    _, rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    with np.load(args.layout) as rig:
        faces = rig['faces']; roi_faces = regions(rig['rest_vertices'], faces)['shoulders_back_proxy']
    source_dirs = {'train': args.previous / 'semantic_train', 'eval': args.previous / 'semantic_eval',
                   'discovery': args.discovery / 'semantic'}
    model_binding = json.loads((source_dirs['train'] / 'model_binding.json').read_text())
    for split in source_dirs:
        dest = root / ('semantic_' + split); dest.mkdir()
        write(dest / 'model_binding.json', dict(model_binding,
            boundary_policy_sha256=sha256(root / 'boundary_policy.json'),
            base_parser_only=model_binding['code_sha256'], refinement_code_sha256=sha256(Path(__file__))))
    bindings = {str(p.resolve()): sha256(p) for p in [args.batch, args.group, args.layout, Path(__file__)]}
    records = []
    for clip, (entry, row) in enumerate(zip(batch['clips'], rows)):
        folder = Path(entry['folder']); name = folder.name
        archive = folder / 'attempts/0001/work/reconstruction.npz'
        data = load_data(archive)
        meta_path = folder / 'result/mesh_meta.json'; mirror_path = folder / 'result/mirror_geometry.json'
        meta = json.loads(meta_path.read_text()); mirror = json.loads(mirror_path.read_text())
        for p in [archive, meta_path, mirror_path, folder / 'source.mp4']:
            bindings[str(p.resolve())] = sha256(p)
        cap = cv2.VideoCapture(str(folder / 'source.mp4'))
        tasks = []
        for split, source in source_dirs.items():
            for p in sorted(source.glob(name + '_f*_v*.npz')):
                rec = json.loads(p.with_suffix('.json').read_text())
                if rec['cache_sha256'] != sha256(p): raise ValueError('Source semantic cache changed')
                frame = rec['frame']
                if split == 'train' and frame not in row['frames']['train_frames']: raise ValueError('TRAIN leak')
                if split == 'eval' and frame in row['frames']['train_frames']: raise ValueError('Evaluation leak')
                if split == 'discovery' and frame in set(row['frames']['train_frames'] + row['frames']['heldout_frames'] + [137, 213]):
                    raise ValueError('Discovery overlaps frozen split')
                tasks.append((frame, rec['view'], split, p, rec))
        active = None; image = None
        for frame, view, split, source, old_record in sorted(tasks):
            if active != frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1: raise ValueError('Decode mismatch')
                active = frame
            camera = next(cam for v, cam, _, _ in views(data, meta, mirror, frame, 'independent') if v == view)
            with np.load(source) as z: old = {k: z[k] for k in z.files}
            x, y, X, Y = old['box']; crop = image[y:Y, x:X]
            xy = project(camera[faces].mean(1), meta['focal'][frame], meta['image_size'])
            local = xy[roi_faces] - [x, y]
            lo = np.clip(np.floor(local.min(0)).astype(int)-5, 0, crop.shape[1::-1])
            hi = np.clip(np.ceil(local.max(0)).astype(int)+6, 0, crop.shape[1::-1])
            roi = np.zeros(old['labels'].shape, bool); roi[lo[1]:hi[1], lo[0]:hi[0]] = True
            labels, conf, flags = boundary_guard(crop, old['labels'], old['confidence'], roi)
            target = root / ('semantic_' + split) / source.name
            fl, fc, fd = sample_layers(labels, conf, distances(labels), xy, old['box'])
            np.savez_compressed(target, labels=labels, confidence=conf, raw_labels=old['raw_labels'],
                box=old['box'], face_label=fl, face_confidence=fc, face_distance=fd,
                **flags)
            record = dict(clip=name, frame=int(frame), view=int(view), split=split,
                source_cache_sha256=sha256(source), cache_sha256=sha256(target),
                video_sha256=row['inputs']['video']['sha256'], model_sha256=model_binding['model_sha256'],
                boundary_policy_sha256=sha256(root / 'boundary_policy.json'),
                counts={k: int(v.sum()) for k, v in flags.items()})
            write(target.with_suffix('.json'), record); records.append(record)
            if (name, frame, view) in [('48.43',102,1), ('49.53',102,1), ('50.03',182,0), ('48.43',125,1), ('48.53',125,1)]:
                marked = crop.copy(); marked[flags['white_ridge']] = [0,230,255]; marked[flags['hair_boundary']] = [255,100,220]
                panel = np.concatenate([crop, cv2.addWeighted(crop,.55,PALETTE_BGR[old['labels']],.45,0),
                    cv2.addWeighted(crop,.55,PALETTE_BGR[labels],.45,0), marked], axis=1)
                cv2.imwrite(str(root / (source.stem + '_boundary.png')), panel)
        cap.release(); print('BOUNDARY', name, len(records), flush=True)
    write(root / 'boundary_records.json', records); write(root / 'input_binding.json', bindings)
    summary = {s: dict(views=sum(r['split']==s for r in records),
        **{k: sum(r['counts'][k] for r in records if r['split']==s) for k in ['white_ridge','hair_boundary','weak_material','rejected']}) for s in source_dirs}
    write(root / 'boundary_summary.json', summary); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['batch','group','layout','previous','discovery','output']:
        p.add_argument('--'+key,type=Path,required=True)
    refine(p.parse_args())
