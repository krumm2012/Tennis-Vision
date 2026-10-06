"""Diagnostic follow-up: remove color correction, freeze all original seam settings."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from refine_texture_photometry import grid_edges, poisson_correction
from multivideo_texture import face_neighbors
from refine_texture_visibility import evaluate
from run_records import write, sha256


def run(root, layout, observations):
    out = root / 'poisson_only'; out.mkdir(exist_ok=False)
    policy = json.loads((root / 'gate_policy.json').read_text())
    write(out / 'diagnostic_protocol.json', {
        'reason': 'Initial color correction worsened heldout error; isolate seam effect without changing its thresholds.',
        'development_only': True, 'independent_confirmation': False,
        'frozen_policy_sha256': sha256(root / 'gate_policy.json'),
        'source_code_sha256': sha256(Path(__file__)), 'policy': policy})
    shutil.copy2(__file__, out / Path(__file__).name)
    baseline = cv2.imread(str(root / 'baseline/body_texture_rgba.png'), -1)
    with np.load(root / 'baseline/texture_sources.npz') as s:
        fm = s['face']; clip, frame, view = s['clip'], s['frame'], s['view']
    with np.load(root / 'correction_fields.npz') as c:
        eligible, reference = c['eligible'], c['train_consensus']
    with np.load(layout) as rig: neighbors = face_neighbors(rig['faces'])
    p, q = grid_edges(fm, eligible, neighbors)
    source_change = ((clip.ravel()[p] != clip.ravel()[q]) | (frame.ravel()[p] != frame.ravel()[q]) |
                     (view.ravel()[p] != view.ravel()[q]))
    flat = baseline[:, :, :3].reshape(-1, 3).astype(float)
    seams = (source_change & (np.abs(flat[p] - flat[q]).mean(1) <= policy['max_seam_jump']) &
             (np.abs(reference[fm.ravel()[p]] - reference[fm.ravel()[q]]).mean(1) <= policy['max_consensus_edge_difference']))
    delta, solver = poisson_correction(baseline[:, :, :3], fm.shape, p, q, seams,
        policy['poisson_radius_px'], policy['poisson_screen'], policy['poisson_limit'])
    candidate = baseline.copy()
    candidate[:, :, :3] = np.clip(np.rint(candidate[:, :, :3].astype(float) + delta), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / 'body_texture_rgba.png'), candidate)
    shutil.copy2(root / 'baseline/texture_sources.npz', out / 'texture_sources.npz')
    report = json.loads((root / 'baseline/texture_report.json').read_text())
    report.update(photometry='poisson_only_diagnostic', status='diagnostic_not_accepted')
    report['artifact_sha256'] = {k: sha256(out / k) for k in ['body_texture_rgba.png', 'texture_sources.npz']}
    write(out / 'texture_report.json', report)
    write(out / 'solver.json', solver)
    evaluate(layout, root / 'baseline', out, observations, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'layout', 'observations']: p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); run(a.root, a.layout, a.observations)
