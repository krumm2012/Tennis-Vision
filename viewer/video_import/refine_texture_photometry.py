"""Isolated TRAIN-only shoulder photometry and screened-Poisson ablation.

This bounded source-offset experiment is not a full MVS seam-leveling solver or
intrinsic decomposition. No geometry, source assignment or observed alpha changes.
"""
import argparse
import json
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from audit_texture_heldout import validate_sources
from blender_uv_diagnostics import atlas_centers, metrics
from multivideo_texture import face_neighbors, validate_fusion_group, validate_cache_split
from refine_texture_visibility import regions, evaluate
from run_records import sha256, write


POLICY = {
    'scope': 'shoulders_back_proxy only; automatic correspondence, not human ground truth',
    'min_train_views': 8, 'min_train_clips': 3, 'max_train_mad': 14.,
    'max_baseline_consensus_difference': 35., 'source_min_faces': 40,
    'source_offset_limit': 12., 'source_shrinkage_faces': 50.,
    'boundary_taper_px': 3., 'poisson_radius_px': 3,
    'poisson_screen': .15, 'poisson_limit': 6., 'max_seam_jump': 30.,
    'max_consensus_edge_difference': 12.,
    'gates': {'coverage_loss': 0., 'roi_heldout_mean_tolerance': .1,
              'whole_heldout_mean_tolerance': .05, 'per_clip_roi_tolerance': .5,
              'minimum_source_seam_reduction': .05, 'detail_ratio_min': .95,
              'detail_ratio_max': 1.05, 'visual_review_required': True},
    'heldout_role': 'development regression, repeatedly used; not independent final acceptance',
}


def robust_source_offsets(colors, valid, reference, selected, min_faces=40, limit=12., shrinkage=50.):
    """Per-source additive BGR offsets from TRAIN face correspondence only."""
    offsets = np.zeros((len(colors), 3), np.float32)
    counts = np.zeros(len(colors), np.int32)
    for i in range(len(colors)):
        keep = valid[i] & selected & np.isfinite(reference).all(axis=1)
        counts[i] = keep.sum()
        if counts[i] >= min_faces:
            delta = np.median(reference[keep] - colors[i, keep], axis=0)
            offsets[i] = np.clip(delta * counts[i] / (counts[i] + shrinkage), -limit, limit)
    return offsets, counts


def grid_edges(face_map, eligible, neighbors):
    """Never connect packed UV islands merely because their pixels touch."""
    h, w = face_map.shape
    indices = np.arange(h * w).reshape(h, w)
    pairs = [(indices[:, :-1].ravel(), indices[:, 1:].ravel()),
             (indices[:-1].ravel(), indices[1:].ravel())]
    p, q = np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])
    f = face_map.ravel(); e = eligible.ravel()
    keep = e[p] & e[q] & (f[p] >= 0) & (f[q] >= 0)
    p, q = p[keep], q[keep]
    adjacent = (f[p] == f[q]) | (neighbors[f[p]] == f[q, None]).any(axis=1)
    return p[adjacent], q[adjacent]


def poisson_correction(colors, shape, p, q, seams, radius=3, screen=.15, limit=6.):
    """Preserve input gradients except frozen source-seam edges; solve a narrow band.

    Delta has a screened Laplacian, zero boundary outside the band. Spatial
    propagation follows only the physical surface edges supplied by grid_edges.
    """
    colors = np.asarray(colors, np.float64).reshape(-1, 3)
    touched = np.zeros(len(colors), bool)
    touched[p[seams]] = True; touched[q[seams]] = True
    for _ in range(radius):
        propagation = touched[p] | touched[q]
        touched[p[propagation]] = True; touched[q[propagation]] = True
    ids = np.flatnonzero(touched)
    correction = np.zeros_like(colors)
    if not len(ids):
        return correction.reshape(*shape, 3), {'unknowns': 0, 'seam_edges': 0}
    index = np.full(len(colors), -1, np.int32); index[ids] = np.arange(len(ids))
    relevant = touched[p] | touched[q]
    a, b = p[relevant], q[relevant]; seam = seams[relevant]
    ia, ib = index[a], index[b]
    both = (ia >= 0) & (ib >= 0)
    degree = np.full(len(ids), screen, np.float64)
    np.add.at(degree, ia[ia >= 0], 1.)
    np.add.at(degree, ib[ib >= 0], 1.)
    matrix = sparse.diags(degree, format='csr') + sparse.coo_matrix(
        (-np.ones(2 * both.sum()),
         (np.r_[ia[both], ib[both]], np.r_[ib[both], ia[both]])),
        shape=(len(ids), len(ids))).tocsr()
    rhs = np.zeros((len(ids), 3))
    # Zero target gradient on qualified seam edges; elsewhere preserve gradients.
    jump = colors[a[seam]] - colors[b[seam]]
    np.add.at(rhs, ia[seam], -jump)
    np.add.at(rhs, ib[seam], jump)
    convergence = []
    for channel in range(3):
        solution, info = cg(matrix, rhs[:, channel], rtol=1e-7, atol=1e-9, maxiter=600)
        if info: raise ValueError(f'Poisson solver did not converge: {info}')
        convergence.append(float(np.linalg.norm(matrix @ solution - rhs[:, channel])))
        correction[ids, channel] = np.clip(solution, -limit, limit)
    return correction.reshape(*shape, 3), {
        'unknowns': len(ids), 'seam_edges': int(seams.sum()), 'residual_norm': convergence,
        'limit': limit, 'scope': 'source transitions inside UV charts; chart seams evaluated separately'}


def nanmedian(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'gate_policy.json', POLICY)  # Frozen before loading any heldout colors.
    shutil.copy2(__file__, out / Path(__file__).name)
    report = json.loads((args.baseline / 'texture_report.json').read_text())
    batch = json.loads(args.batch.read_text())
    _, group_rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    if sha256(args.group) != report['appearance_group']['manifest_sha256']:
        raise ValueError('Baseline group binding mismatch')
    if sha256(args.layout) != report['layout_sha256']:
        raise ValueError('Baseline layout binding mismatch')
    for name in ['body_texture_rgba.png', 'texture_sources.npz', 'appearance_mesh.npz']:
        if sha256(args.baseline / name) != report['artifact_sha256'][name]:
            raise ValueError('Baseline artifact changed')
    with np.load(args.layout) as rig:
        faces, uv, uv_faces = rig['faces'], rig['uv'], rig['uv_faces']
        region = regions(rig['rest_vertices'], faces)['shoulders_back_proxy']
    with np.load(args.baseline / 'texture_sources.npz') as source:
        sources = {k: source[k] for k in source.files}
    baseline = cv2.imread(str(args.baseline / 'body_texture_rgba.png'), -1)
    if baseline.shape != (2048, 2048, 4): raise ValueError('Fixed 2K baseline expected')
    fm = sources['face']; known = (baseline[:, :, 3] > 0) & (fm >= 0)
    validate_sources(sources, group_rows, known)
    center_colors = atlas_centers(baseline, uv, uv_faces)
    training = []; clip_targets = []; clip_counts = []; source_keys = []; bindings = {}
    for clip, gain_row in enumerate(report['validation']):
        name = gain_row['video']; path = args.observations / (name + '_observations.npz')
        bindings[str(path.resolve())] = sha256(path)
        with np.load(path) as cache:
            validate_cache_split(cache, group_rows[clip])
            # The previous experiment's front-facing filter is used to fit offsets,
            # not to delete or replace any baseline texel.
            valid = cache['all_score'] > 0
            color = cache['all_color'].astype(np.float32) * gain_row['gain_bgr']
            color = color.astype(np.float32)
            training.append((color, valid))
            clip_targets.append(nanmedian(np.where(valid[:, :, None], color, np.nan)))
            clip_counts.append(valid.sum(axis=0))
            source_keys.extend((clip, int(f), int(v)) for f, v in zip(cache['all_frame'], cache['all_view']))
        print('TRAIN', name, flush=True)
    reference = nanmedian(np.stack(clip_targets))  # Each clip has equal influence.
    colors = np.concatenate([c for c, _ in training]); valid = np.concatenate([v for _, v in training])
    mad = nanmedian(np.where(valid, np.abs(colors - reference).mean(axis=2), np.nan))
    selected = (region & (valid.sum(axis=0) >= POLICY['min_train_views']) &
                ((np.array(clip_counts) >= 2).sum(axis=0) >= POLICY['min_train_clips']) &
                (mad <= POLICY['max_train_mad']) & (center_colors[:, 3] > 240) &
                (np.abs(center_colors[:, :3] - reference).mean(axis=1) <= POLICY['max_baseline_consensus_difference']) &
                (reference.min(axis=1) > 30) & (reference.max(axis=1) < 240))
    offsets, counts = robust_source_offsets(colors, valid, reference, selected,
        POLICY['source_min_faces'], POLICY['source_offset_limit'], POLICY['source_shrinkage_faces'])
    write(out / 'source_offsets.json', [dict(clip=c, frame=f, view=v, faces=int(n), delta_bgr=d.tolist())
          for (c, f, v), n, d in zip(source_keys, counts, offsets)])
    correction = np.zeros((*fm.shape, 3), np.float32)
    eligible = np.zeros(fm.shape, bool); label = np.full(fm.shape, -1, np.int32)
    for i, ((clip, frame, view), offset) in enumerate(zip(source_keys, offsets)):
        take = known & (sources['clip'] == clip) & (sources['frame'] == frame) & (sources['view'] == view)
        label[take] = i
        take &= selected[np.maximum(fm, 0)] & valid[i, np.maximum(fm, 0)]
        if counts[i] < POLICY['source_min_faces']: continue
        correction[take] = offset; eligible[take] = True
    # Inward taper protects the boundary of this deliberately small experiment.
    taper = np.clip(cv2.distanceTransform(eligible.astype(np.uint8), cv2.DIST_L2, 5) /
                    POLICY['boundary_taper_px'], 0, 1)
    correction *= taper[:, :, None]
    color_only = baseline.copy()
    color_only[:, :, :3] = np.clip(np.rint(baseline[:, :, :3].astype(float) + correction), 0, 255).astype(np.uint8)
    neighbors = face_neighbors(faces)
    p, q = grid_edges(fm, eligible, neighbors)
    flat = baseline[:, :, :3].reshape(-1, 3).astype(float)
    seam = ((label.ravel()[p] != label.ravel()[q]) &
            (np.abs(flat[p] - flat[q]).mean(axis=1) <= POLICY['max_seam_jump']) &
            (np.abs(reference[fm.ravel()[p]] - reference[fm.ravel()[q]]).mean(axis=1) <= POLICY['max_consensus_edge_difference']))
    delta, poisson_info = poisson_correction(color_only[:, :, :3], fm.shape, p, q, seam,
        POLICY['poisson_radius_px'], POLICY['poisson_screen'], POLICY['poisson_limit'])
    combined = color_only.copy()
    combined[:, :, :3] = np.clip(np.rint(color_only[:, :, :3].astype(float) + delta), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / 'eligible_mask.png'), eligible.astype(np.uint8) * 255)
    np.savez_compressed(out / 'correction_fields.npz', color=correction, poisson=delta.astype(np.float32),
                        selected_faces=selected, eligible=eligible, train_consensus=reference, train_mad=mad)
    write(out / 'training_summary.json', {'region_faces': int(region.sum()), 'eligible_faces': int(selected.sum()),
          'eligible_texels': int(eligible.sum()), 'eligible_uv_fraction': float(eligible.sum() / (fm >= 0).sum()),
          'sources_with_fitted_offset': int((counts >= POLICY['source_min_faces']).sum()),
          'poisson': poisson_info, 'input_sha256': bindings, 'policy_sha256': sha256(out / 'gate_policy.json'),
          'fitting_reads': 'all_color/all_score/all_frame/all_view only; no heldout colors',
          'limit': 'automatic region and front-facing correspondence; legacy cache generation metadata unverified'})
    # Freeze all candidate pixels before any heldout evaluation.
    for name, atlas in [('baseline', baseline), ('color_only', color_only), ('color_poisson', combined)]:
        dest = out / name; dest.mkdir()
        cv2.imwrite(str(dest / 'body_texture_rgba.png'), atlas)
        shutil.copy2(args.baseline / 'texture_sources.npz', dest / 'texture_sources.npz')
        r = dict(report)
        r.update(status='isolated_photometry_candidate_not_accepted', photometry=name,
                 photometry_policy_sha256=sha256(out / 'gate_policy.json'),
                 baseline_texture_sha256=sha256(args.baseline / 'body_texture_rgba.png'))
        r['artifact_sha256'] = {k: sha256(dest / k) for k in ['body_texture_rgba.png', 'texture_sources.npz']}
        write(dest / 'texture_report.json', r)
    boundary = np.zeros(fm.size, bool); boundary[p[seam]] = True; boundary[q[seam]] = True
    # Detail proxy on internal, same-source edges at least two grid steps from seams.
    for _ in range(2):
        near = boundary[p] | boundary[q]; boundary[p[near]] = True; boundary[q[near]] = True
    detail = (~boundary[p] & ~boundary[q] & (label.ravel()[p] == label.ravel()[q]) &
              (taper.ravel()[p] == 1) & (taper.ravel()[q] == 1))
    base_detail = np.abs(flat[p[detail]] - flat[q[detail]]).mean()
    summaries = {}
    for name, atlas in [('baseline', baseline), ('color_only', color_only), ('color_poisson', combined)]:
        a = atlas[:, :, :3].reshape(-1, 3).astype(float)
        summaries[name] = {
            'source_boundary_BGR_MAE': metrics(np.abs(a[p[seam]] - a[q[seam]]).mean(axis=1)),
            'interior_detail_ratio': float(np.abs(a[p[detail]] - a[q[detail]]).mean() / max(base_detail, 1e-9)),
            'detail_edges': int(detail.sum()), 'alpha_identical': bool(np.array_equal(atlas[:, :, 3], baseline[:, :, 3])),
            'outside_eligible_identical': bool(np.array_equal(atlas[~eligible], baseline[~eligible])),
            'changed_texels': int(np.any(atlas != baseline, axis=2).sum()),
            'max_channel_change': int(np.abs(atlas.astype(int) - baseline.astype(int)).max()),
        }
        if name != 'baseline':
            evaluate(args.layout, out / 'baseline', out / name, args.observations, out / name)
    write(out / 'image_metrics.json', summaries)
    write(out / 'input_bindings.json', {
        'baseline': {str((args.baseline / f).resolve()): sha256(args.baseline / f)
                     for f in ['texture_report.json', 'body_texture_rgba.png', 'texture_sources.npz', 'appearance_mesh.npz']},
        'layout': sha256(args.layout), 'group': sha256(args.group), 'code': sha256(Path(__file__)),
        'command_arguments': {k: str(v.resolve()) for k, v in vars(args).items()}})
    print(json.dumps({'summary': summaries, 'output': str(out)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['baseline', 'observations', 'layout', 'group', 'batch', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
