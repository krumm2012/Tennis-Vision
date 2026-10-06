"""Export editable UV guides and compare two resolutions at heldout surface points."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from multivideo_texture import sample
from run_records import sha256, write


def atlas_centers(atlas, uv, uv_faces):
    xy = uv[uv_faces].mean(axis=1).copy()
    xy[:, 1] = 1 - xy[:, 1]
    # Match the UV rasterizer's pixel centers, not a resized baseline image.
    xy = xy * (atlas.shape[0] - 1) - .5
    return sample(atlas, xy)


def metrics(values):
    return {'n': int(len(values)), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))} if len(values) else {'n': 0}


def build(layout, candidate, baseline, observations, group, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(layout, allow_pickle=False) as rig:
        uv, uv_faces, faces = rig['uv'], rig['uv_faces'], rig['faces']
        vertices = rig['rest_vertices']
    rgba = cv2.imread(str(candidate / 'body_texture_rgba.png'), cv2.IMREAD_UNCHANGED)
    old = cv2.imread(str(baseline / 'body_texture_rgba.png'), cv2.IMREAD_UNCHANGED)
    reports = [json.loads((p / 'texture_report.json').read_text()) for p in (baseline, candidate)]
    for path, report in zip((baseline, candidate), reports):
        if report['layout_sha256'] != sha256(layout) or report['artifact_sha256']['body_texture_rgba.png'] != sha256(path / 'body_texture_rgba.png'):
            raise ValueError('Input binding mismatch')
    if reports[0]['inputs'] != reports[1]['inputs']:
        raise ValueError('Comparison requires the same frozen inputs')
    size = rgba.shape[0]
    xy = np.round(uv * (size - 1)).astype(np.int32); xy[:, 1] = size - 1 - xy[:, 1]
    wire = np.zeros((size, size, 4), np.uint8)
    cv2.polylines(wire, [p.reshape(-1, 1, 2) for p in xy[uv_faces]], True, (255, 255, 255, 150), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / 'UV_wire_4096.png'), wire)
    with np.load(candidate / 'texture_sources.npz', allow_pickle=False) as sources:
        face_map = sources['face']; observed = (rgba[:, :, 3] > 0) & (face_map >= 0)
        support = np.zeros_like(rgba); support[face_map >= 0] = (110, 0, 255, 255)
        support[observed] = (80, 190, 20, 255)
        cv2.imwrite(str(output / 'UV_support_4096.png'), support)
        cv2.imwrite(str(output / 'UV_observed_mask_4096.png'), observed.astype(np.uint8) * 255)
        source_view = np.zeros_like(rgba)
        source_view[observed & (sources['view'] == 0)] = (230, 140, 20, 255)
        source_view[observed & (sources['view'] == 1)] = (30, 170, 240, 255)
        cv2.imwrite(str(output / 'UV_source_view_4096.png'), source_view)
        # Validate each 4K observed texel against the explicit TRAIN split.
        rows = json.loads(group.read_text())['clips']
        from audit_texture_heldout import validate_sources
        validate_sources(sources, rows, observed)
        coverage = float(observed.sum() / (face_map >= 0).sum())
    a, b = uv[uv_faces[:, 1]] - uv[uv_faces[:, 0]], uv[uv_faces[:, 2]] - uv[uv_faces[:, 0]]
    uv_area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    tri = vertices[faces]; surface_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    valid = (uv_area > 1e-12) & (surface_area > 1e-12)
    density = np.sqrt(uv_area[valid] / surface_area[valid]) * size
    sampled = [atlas_centers(atlas, uv, uv_faces) for atlas in [old, rgba]]
    per_clip = []; pooled = {view: [[], []] for view in [0, 1]}
    for row, gain_row in zip(rows, reports[1]['validation']):
        name = gain_row['video']
        with np.load(observations / (name + '_observations.npz'), allow_pickle=False) as cache:
            if not set(cache['heldout_frame'].tolist()) <= set(row['frames']['heldout_frames']):
                raise ValueError('Heldout cache not bound to split')
            ids = cache['heldout_face']; actual = cache['heldout_color'].astype(float) * gain_row['gain_bgr']
            common = (sampled[0][ids, 3] > 240) & (sampled[1][ids, 3] > 240)
            errors = [abs(s[ids, :3] - actual).mean(axis=1) for s in sampled]
            for view in [0, 1]:
                keep = common & (cache['heldout_view'] == view)
                values = [e[keep] for e in errors]
                per_clip.append({'clip': name, 'view': 'real' if view == 0 else 'mirror',
                                 'baseline_2048': metrics(values[0]), 'candidate_4096': metrics(values[1])})
                for dest, value in zip(pooled[view], values): dest.append(value)
    validation = {('real' if v == 0 else 'mirror'): {'baseline_2048': metrics(np.concatenate(e[0])),
                  'candidate_4096': metrics(np.concatenate(e[1]))} for v, e in pooled.items()}
    result = {'uv_vertices': len(uv), 'body_vertices': len(vertices), 'triangles': len(faces),
              'atlas_size': size, 'coverage_within_uv_surface': coverage,
              'degenerate_uv_triangles': int((uv_area <= 1e-12).sum()),
              'texel_density_per_estimated_metre': metrics(density),
              'heldout_surface_center_BGR_MAE_0_255': validation, 'per_clip': per_clip,
              'comparison_scope': 'Same heldout face centers, common observed alpha >240; frozen estimated geometry/camera and cached colors; not independent geometric truth or full-pixel seam evaluation',
              'source_binding_limit': 'Legacy observations lack generation-time metadata binding; current hashes and TRAIN provenance checked',
              'no_accuracy_claim_from_resolution': True,
              'source_sha256': {'layout': sha256(layout), 'candidate': sha256(candidate / 'body_texture_rgba.png'), 'baseline': sha256(baseline / 'body_texture_rgba.png')}}
    write(output / 'uv_quality_report.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['layout', 'candidate', 'baseline', 'observations', 'group', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); r = build(a.layout, a.candidate, a.baseline, a.observations, a.group, a.output)
    print(json.dumps({k: r[k] for k in ['coverage_within_uv_surface', 'degenerate_uv_triangles', 'heldout_surface_center_BGR_MAE_0_255']}, indent=2))
