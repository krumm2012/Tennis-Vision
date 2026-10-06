"""Confirm a frozen texture candidate on frames absent from TRAIN/development cache."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from blender_uv_diagnostics import atlas_centers, metrics
from multivideo_texture import load_data, views, visibility, valid_points, sample, validate_fusion_group
from refine_texture_visibility import facing_cosine, regions
from run_records import sha256, write


def run(root, batch, group, layout):
    output = root / 'fresh_frames'; output.mkdir(exist_ok=False)
    frames = [137, 213]
    candidates = ['baseline', 'poisson_only']
    write(output / 'protocol.json', {
        'frames_0based': frames, 'candidate_texture_sha256': {n: sha256(root / n / 'body_texture_rgba.png') for n in candidates},
        'code_sha256': sha256(Path(__file__)), 'frame_selection': 'Fixed indices before decoding; absent from both original cache splits',
        'geometry': 'frozen estimated MHR, not independent geometry truth',
        'roi_mean_tolerance': .1, 'per_clip_roi_tolerance': .5,
        'no_further_candidate_tuning': True})
    shutil.copy2(__file__, output / Path(__file__).name)
    batch_doc = json.loads(batch.read_text()); _, rows = validate_fusion_group(group, batch_doc['clips'], layout)
    with np.load(layout) as r:
        faces, uv, uv_faces = r['faces'], r['uv'], r['uv_faces']; area = regions(r['rest_vertices'], faces)
    with np.load(root / 'correction_fields.npz') as c: area['eligible_faces'] = c['selected_faces']
    values = [atlas_centers(cv2.imread(str(root / n / 'body_texture_rgba.png'), -1), uv, uv_faces) for n in candidates]
    supported = (values[0][:, 3] > 240) & (values[1][:, 3] > 240)
    gains = json.loads((root / 'baseline/texture_report.json').read_text())['validation']
    pooled = {}; results = []; evidence = []; bindings = {}
    for entry, row, gain in zip(batch_doc['clips'], rows, gains):
        folder = Path(entry['folder']); name = folder.name
        if set(frames) & (set(row['frames']['train_frames']) | set(row['frames']['heldout_frames'])):
            raise ValueError('Fresh test frame was in a texture cache split')
        archive = folder / 'attempts/0001/work/reconstruction.npz'
        data = load_data(archive); meta = json.loads((folder / 'result/mesh_meta.json').read_text())
        mirror = json.loads((folder / 'result/mirror_geometry.json').read_text())
        for f in [archive, folder / 'source.mp4', folder / 'result/mesh_meta.json', folder / 'result/mirror_geometry.json']:
            bindings[str(f.resolve())] = sha256(f)
        cap = cv2.VideoCapture(str(folder / 'source.mp4'))
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1: raise ValueError('Frame decode mismatch')
            for view, camera, mask, other in views(data, meta, mirror, frame, 'independent'):
                _, depth, eroded, resolution = visibility(camera, faces, mask, meta['focal'][frame], meta['image_size'], other)
                centers = camera[faces].mean(1)
                valid, xy, confidence = valid_points(centers, eroded, depth, meta['focal'][frame],
                    np.array(meta['image_size']), resolution, np.where(area['hands_proxy'], .01, .025))
                valid &= (facing_cosine(camera, faces, bool(view)) > .3) & supported
                truth = sample(image, xy).astype(float) * gain['gain_bgr']
                errors = [np.abs(v[:, :3] - truth).mean(1) for v in values]
                for region, selected in area.items():
                    keep = valid & selected
                    result = dict(clip=name, frame=frame, view=view, region=region)
                    for label, error in zip(candidates, errors):
                        result[label] = metrics(error[keep])
                        pooled.setdefault((view, region, label), []).append(error[keep])
                    results.append(result)
                for region in ['shoulders_back_proxy', 'hands_proxy']:
                    ids = np.flatnonzero(valid & area[region])
                    if not len(ids): continue
                    lo = np.maximum(0, np.floor(xy[ids].min(0)).astype(int) - 20)
                    hi = np.minimum(meta['image_size'], np.ceil(xy[ids].max(0)).astype(int) + 21)
                    raw = image[lo[1]:hi[1], lo[0]:hi[0]].copy(); overlay = raw.copy()
                    for face in ids[::8]:
                        point = tuple(np.round(xy[face] - lo).astype(int))
                        cv2.circle(overlay, point, 1, (255, 150, 0), -1)
                    path = output / f'{name}_f{frame}_v{view}_{region}.png'
                    cv2.imwrite(str(path), np.concatenate([raw, overlay], axis=1))
                    evidence.append(dict(clip=name, frame=frame, view=view, region=region, crop_native_xyxy=[*lo.tolist(), *hi.tolist()], path=str(path.resolve())))
        cap.release(); print('FRESH', name, flush=True)
    summary = [dict(view=view, region=region, **{name: metrics(np.concatenate(pooled.get((view, region, name), [np.array([])]))) for name in candidates})
               for view in [0, 1] for region in area]
    write(output / 'comparison.json', dict(summary=summary, per_frame=results, evidence=evidence,
          input_sha256=bindings, protocol_sha256=sha256(output / 'protocol.json'),
          scope='New texture-evaluation frames; same camera session/person and estimated geometry, not a new capture or geometric truth'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'batch', 'group', 'layout']: p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); run(a.root, a.batch, a.group, a.layout)
