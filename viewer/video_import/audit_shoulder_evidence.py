"""Search disjoint source frames for unresolved shoulder evidence; never bake it.

Counts describe automatic observations of a frozen set of UV texels, not
independent pixels, anatomical ground truth, or improved texture accuracy.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from multivideo_texture import (load_data, views, visibility, valid_points, sample,
                               project, uv_lookup, validate_fusion_group)
from refine_texture_visibility import facing_cosine
from refine_texture_semantics import verified_classes
from semantic_body_layers import Parser, distances, sample_layers
from run_records import sha256, write


def discovery_frames(length, train, heldout, used_evaluation=(137, 213)):
    excluded = set(train) | set(heldout) | set(used_evaluation)
    return [f for f in range(2, length, 10) if f not in excluded]


def support_summary(labels, keys):
    """Require consistent skin/cloth observations across clips, preserving conflicts."""
    votes = np.stack([(labels == k).sum(0) for k in (2, 3)])
    total = votes.sum(0)
    winner = votes.argmax(0) + 2
    agreement = votes.max(0) / np.maximum(total, 1)
    clips = sum(((labels[keys[:, 0] == c] == winner).any(0)).astype(int)
                for c in np.unique(keys[:, 0]))
    strong = (total >= 8) & (clips >= 2) & (agreement >= .8)
    return dict(count=total, winner=winner.astype(np.uint8), agreement=agreement,
                clips=clips, strong=strong)


def clothing_edge_review(labels, confidence, crop, roi_mask):
    """Uncertain native-pixel band at the predicted skin/upper-clothes interface."""
    kernel = np.ones((11, 11), np.uint8)
    near_skin = cv2.dilate((labels == 2).astype(np.uint8), kernel) > 0
    near_cloth = cv2.dilate((labels == 3).astype(np.uint8), kernel) > 0
    band = near_skin & near_cloth & roi_mask
    weak = band & (confidence.astype(float) / 255 < .85)
    checked = verified_classes(labels.ravel(), confidence.ravel() / 255,
                               crop.reshape(-1, 3)).reshape(labels.shape)
    veto = band & (labels == 3) & (confidence.astype(float) / 255 >= .85) & (checked == 0)
    return band, weak, veto


def search(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'semantic').mkdir()
    shutil.copy2(__file__, out / Path(__file__).name)
    batch = json.loads(args.batch.read_text())
    _, rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    with np.load(args.layout) as rig:
        faces, uv, uv_faces = rig['faces'], rig['uv'], rig['uv_faces']
    with np.load(args.previous / 'train_consensus.npz') as c:
        old_count, old_target, roi = c['count'], c['target'], c['roi']
    with np.load(args.previous / 'semantic_texel_audit.npz') as a:
        unresolved = a['hair'] & ~a['repaired']
        ys, xs, fids = a['ys'][unresolved], a['xs'][unresolved], a['face'][unresolved]
    shape = cv2.imread(str(args.previous / 'unresolved_hair_mask.png'), 0).shape
    fm, bary = uv_lookup(uv.astype(np.float32), uv_faces.astype(np.int32), shape[0])
    if not np.array_equal(fm[ys, xs], fids):
        raise ValueError('UV layout changed')
    weights = bary[ys, xs]
    parser = Parser(args.model)
    binding = {str(p.resolve()): sha256(p) for p in [args.group, args.layout,
        args.previous / 'train_consensus.npz', args.previous / 'semantic_texel_audit.npz',
        args.model / 'onnx/model.onnx', Path(__file__),
        Path(__file__).with_name('semantic_body_layers.py'),
        Path(__file__).with_name('refine_texture_semantics.py')]}
    protocol = dict(role='additional discovery only; not TRAIN and not accuracy evaluation',
        frame_rule='0-based frames 2,12,22,...; exclude original TRAIN, heldout, and 137/213',
        confidence=.85, inside_class_distance_native_px=3, facing_cosine=.3,
        depth_tolerance_m=.025, native_edge_band_radius_px=5,
        hypothesis_support=dict(observations=8, clips=2, agreement=.8),
        original_texture_unchanged=True, geometry_unchanged=True,
        limitations=['Same automatic parser; confidence is not calibrated.',
                    'UV texels can oversample the same native pixel.',
                    'Temporal samples and same-subject clips are correlated.',
                    'Candidate frames are discovery data, never fresh evaluation.'])
    write(out / 'protocol.json', protocol)
    records, keylist, all_labels, all_valid = [], [], [], []
    for clip, (entry, row) in enumerate(zip(batch['clips'], rows)):
        folder = Path(entry['folder']); name = folder.name
        paths = [folder / 'source.mp4', folder / 'result/mesh_meta.json',
                 folder / 'result/mirror_geometry.json', folder / 'attempts/0001/work/reconstruction.npz']
        binding.update({str(p.resolve()): sha256(p) for p in paths})
        data = load_data(paths[3]); meta = json.loads(paths[1].read_text())
        mirror = json.loads(paths[2].read_text())
        frames = discovery_frames(len(data['vertices']), row['frames']['train_frames'], row['frames']['heldout_frames'])
        cap = cv2.VideoCapture(str(paths[0]))
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1:
                raise ValueError('Source frame decode mismatch')
            for view, camera, mask, other in views(data, meta, mirror, frame, 'independent'):
                yy, xx = np.where(mask > 234)
                if not len(xx):
                    raise ValueError('Person mask empty')
                size = np.asarray(meta['image_size']); scale = size / mask.shape[::-1]
                lo = np.maximum(0, np.floor(np.array([xx.min(), yy.min()]) * scale).astype(int) - 20)
                hi = np.minimum(size, np.ceil(np.array([xx.max()+1, yy.max()+1]) * scale).astype(int) + 20)
                crop = image[lo[1]:hi[1], lo[0]:hi[0]]; box = np.r_[lo, hi]
                labels, conf, raw = parser.infer(crop)
                distance = distances(labels)
                _, depth, eroded, resolution = visibility(camera, faces, mask, meta['focal'][frame], size, other)
                points = (camera[faces[fids]] * weights[:, :, None]).sum(1)
                valid, xy, _ = valid_points(points, eroded, depth, meta['focal'][frame], size, resolution)
                valid &= facing_cosine(camera, faces, bool(view))[fids] > .3
                group, certainty, dist = sample_layers(labels, conf, distance, xy, box)
                group = verified_classes(group, certainty, sample(image, xy))
                accepted = valid & (certainty >= .85) & (dist >= 3)
                evidence = np.where(accepted, group, 0).astype(np.uint8)
                clear = np.isin(evidence, [2, 3])
                # A local ROI for edge review; no claim of manually verified anatomy.
                roi_xy = project(camera[faces[roi]].mean(1), meta['focal'][frame], size) - lo
                bound_lo = np.maximum(0, np.floor(roi_xy.min(0)).astype(int) - 5)
                bound_hi = np.minimum(crop.shape[1::-1], np.ceil(roi_xy.max(0)).astype(int) + 6)
                roi_mask = np.zeros(labels.shape, bool)
                roi_mask[bound_lo[1]:bound_hi[1], bound_lo[0]:bound_hi[0]] = True
                band, weak, veto = clothing_edge_review(labels, conf, crop, roi_mask)
                native_points = np.rint(xy[clear]).astype(int)
                name_key = f'{name}_f{frame:03}_v{view}'
                cache = out / 'semantic' / (name_key + '.npz')
                np.savez_compressed(cache, labels=labels, confidence=conf, raw_labels=raw,
                    box=box, xy=xy.astype(np.float32), evidence=evidence, valid=valid,
                    edge_band=band, edge_weak=weak, edge_veto=veto)
                record = dict(key=name_key, clip=name, clip_index=clip, frame=frame, view=view,
                    native_crop_xyxy=box.tolist(), unresolved_geometrically_visible=int(valid.sum()),
                    clear_candidate_texels=int(clear.sum()),
                    clear_candidate_native_pixels=len(np.unique(native_points, axis=0)),
                    newly_seen_old_zero_texels=int((clear & (old_count[fids] == 0)).sum()),
                    still_hair_texels=int((evidence == 1).sum()),
                    edge_band_pixels=int(band.sum()), low_confidence_edge_pixels=int(weak.sum()),
                    white_cloth_veto_pixels=int(veto.sum()),
                    sharpness_native=float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()),
                    cache_sha256=sha256(cache), role='discovery')
                write(cache.with_suffix('.json'), record)
                records.append(record); keylist.append([clip, frame, view])
                all_labels.append(evidence); all_valid.append(valid)
            print('DISCOVERY', name, frame, 'views', len(records), flush=True)
        cap.release()
    evidence = np.stack(all_labels); keys = np.asarray(keylist)
    support = support_summary(evidence, keys)
    seen = support['count'] > 0
    np.savez_compressed(out / 'additional_evidence.npz', labels=evidence, keys=keys,
        geometry_visible=np.stack(all_valid), ys=ys, xs=xs, face=fids,
        old_count=old_count[fids], old_target=old_target[fids], **support)
    summary = dict(clips=len(rows), frames=len(set(tuple(k[:2]) for k in keylist)), views=len(records),
        unresolved_texels=len(ys), unresolved_faces=len(np.unique(fids)),
        any_clear_candidate_texels=int(seen.sum()), still_no_clear_candidate_texels=int((~seen).sum()),
        old_zero_observation_texels=int((old_count[fids] == 0).sum()),
        old_zero_now_clear_candidate_texels=int((seen & (old_count[fids] == 0)).sum()),
        strong_automatic_hypothesis_texels=int(support['strong'].sum()),
        conflicting_skin_cloth_texels=int(((evidence == 2).any(0) & (evidence == 3).any(0)).sum()),
        mean_edge_low_confidence_fraction=float(sum(r['low_confidence_edge_pixels'] for r in records) /
                                                max(1, sum(r['edge_band_pixels'] for r in records))),
        texture_baked=False, interpretation='Automatic discovery support only; visual correspondence review is required.')
    write(out / 'summary.json', summary); write(out / 'frames.json', records)
    write(out / 'input_binding.json', binding)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['batch', 'group', 'layout', 'previous', 'model', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    search(p.parse_args())
