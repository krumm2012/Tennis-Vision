"""Read-only CUDA replay audit of existing SAM3D real/mirror native archives.

No inference, fitting, source archive updates, or viewer publication occur here.
Mirror parameters belong to the flipped inference image: replay first, then
negate camera x, preserving anatomical joint IDs, as generate_multiview does.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def statistics(values):
    x = np.concatenate(values) if values else np.empty(0)
    finite = x[np.isfinite(x)]
    return {'count': int(x.size), 'nonfinite': int(x.size-finite.size),
            'max_m': float(finite.max()) if finite.size else None,
            'p95_m': float(np.percentile(finite, 95)) if finite.size else None,
            'median_m': float(np.median(finite)) if finite.size else None}


def audit_view(head, data, prefix, count, batch_size, replay):
    import torch
    mirror = bool(prefix)
    valid = data['mirror_valid'].astype(bool) if mirror else np.ones(count, bool)
    indices = np.flatnonzero(valid)
    keys = {field: prefix+field for field in ['mhr_model_params', 'mhr_shape_params',
                                             'mhr_expr_params', 'vertices', 'joints']}
    layouts = {'mhr_model_params': (count, 204), 'mhr_shape_params': (count, 45),
               'mhr_expr_params': (count, 72), 'vertices': (count, 18439, 3),
               'joints': (count, 70, 3)}
    report = {'total_frames': count, 'observed_frames': int(valid.sum()),
              'missing_frames_0based': np.flatnonzero(~valid).tolist(),
              'parameter_layout_valid': all(data[key].shape == layouts[name] for name, key in keys.items()),
              'nonfinite_source_values': {name: int((~np.isfinite(data[key])).sum()) for name, key in keys.items()},
              'missing_frame_placeholder_nonzero': {name: int(np.count_nonzero(data[key][~valid])) for name, key in keys.items()},
              'coordinate_conversion': 'official replay axes x,-y,-z; then camera x reflected' if mirror else 'official replay axes x,-y,-z'}
    if not report['parameter_layout_valid'] or any(report['nonfinite_source_values'].values()):
        report['status'] = 'invalid_source'
        return report
    mesh_norm, joint_norm, max_component, frames = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            tensor = lambda key: torch.as_tensor(data[key][ids], device='cuda', dtype=torch.float32)
            v, j = replay(head, tensor(keys['mhr_shape_params']), tensor(keys['mhr_model_params']), tensor(keys['mhr_expr_params']))
            if mirror:
                v[..., 0] *= -1
                j[..., 0] *= -1
            dv = v.cpu().numpy()-data[keys['vertices']][ids]
            dj = j.cpu().numpy()-data[keys['joints']][ids]
            ve, je = np.linalg.norm(dv, axis=-1), np.linalg.norm(dj, axis=-1)
            mesh_norm.append(ve.reshape(-1)); joint_norm.append(je.reshape(-1))
            max_component.extend([float(np.abs(dv).max()), float(np.abs(dj).max())])
            frames.extend({'frame': int(i), 'mesh_max_m': float(ve[k].max()), 'joint_max_m': float(je[k].max())} for k, i in enumerate(ids))
    report.update(mesh_euclidean_error=statistics(mesh_norm), joint_euclidean_error=statistics(joint_norm),
                  replay_abs_component_max_m=max(max_component) if max_component else None, per_frame=frames,
                  existing_refit_replay_gate_threshold_m=1e-4)
    report['replay_gate_passed'] = bool(indices.size and report['replay_abs_component_max_m'] <= 1e-4
                                      and not report['mesh_euclidean_error']['nonfinite']
                                      and not report['joint_euclidean_error']['nonfinite'])
    report['status'] = 'replay_matches_observed_frames' if report['replay_gate_passed'] else 'replay_mismatch'
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    if a.batch_size < 1 or a.batch_size > 32:
        raise ValueError('batch-size must be between 1 and 32')
    request = json.loads(a.request.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (a.root/'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = json.loads((a.root/'worker_config.json').read_text())['env']
        os.environ.update(env)
        sys.path.insert(0, str(a.root/'code'))
        sys.path.insert(0, env['SAM3D_BODY_CODE'])
        import torch
        from refit_fullbody import replay
        from sam_3d_body import load_sam_3d_body
        if not torch.cuda.is_available():
            raise RuntimeError('This audit requires real CUDA execution')
        weights = Path(env['SAM3D_WEIGHTS']); rig_path = weights/'assets/mhr_model.pt'
        report = {'audit_id': request['audit_id'], 'kind': 'read_only_native_mhr_replay',
                  'new_sam_inference': False, 'fullbody_refit': False, 'published_to_viewer': False,
                  'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'script_sha256': digest(__file__), 'request_sha256': digest(a.request),
                  'replay_script_sha256': digest(a.root/'code/refit_fullbody.py'),
                  'mhr_model_sha256': digest(rig_path), 'sam3d_checkpoint_sha256': digest(weights/'model.ckpt'),
                  'runtime': {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'python': sys.version, 'platform': platform.platform(),
                              'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()},
                  'clips': []}
        rig = torch.jit.load(str(rig_path), map_location='cpu')
        faces = rig.character_torch.mesh.faces.detach().numpy().copy(); del rig
        model, _ = load_sam_3d_body(str(weights/'model.ckpt'), device='cuda', mhr_path=str(rig_path))
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad = False
        head = model.head_pose; del model; torch.cuda.empty_cache()
        for item in request['clips']:
            job = item['remote_job']
            if len(job) != 32 or any(ch not in '0123456789abcdef' for ch in job):
                raise ValueError('invalid job id')
            folder = a.root/'jobs'/job; archive = folder/'work/reconstruction.npz'
            state = json.loads((folder/'status.json').read_text())
            manifest = json.loads((folder/'run_manifest.json').read_text())
            row = {'video': item['video'], 'job_id': job, 'original_sha256': item['original_sha256'],
                   'archive_sha256': digest(archive), 'normalized_video_sha256': digest(folder/'source.mp4'),
                   'remote_manifest_sha256': digest(folder/'run_manifest.json'), 'remote_status': state['status'],
                   'generation_git_commit': manifest['git_commit'], 'source_hash_matches_request': False}
            row['source_hash_matches_request'] = bool(row['archive_sha256'] == item['archive_sha256'] == manifest['artifact_sha256']['reconstruction.npz']
                                                      and row['normalized_video_sha256'] == item['normalized_video_sha256'] == manifest['source_sha256'])
            if state['status'] != 'ready' or not row['source_hash_matches_request']:
                row['status'] = 'source_identity_mismatch'
            else:
                with np.load(archive, allow_pickle=False) as z:
                    names = ['vertices','joints','faces','mirror_valid','video_sha256']
                    names += [prefix+name for prefix in ['', 'mirror_'] for name in ['mhr_model_params','mhr_shape_params','mhr_expr_params']]
                    names += ['mirror_vertices','mirror_joints']
                    data = {k:z[k].copy() for k in names}
                count = len(data['vertices'])
                row.update(frames=count, topology_matches_provisioned_model=bool(np.array_equal(data['faces'], faces)),
                           topology_sha256=hashlib.sha256(data['faces'].astype('<i8').tobytes()).hexdigest(),
                           embedded_video_hash_matches=bool(str(data['video_sha256']) == row['normalized_video_sha256']),
                           frame_count_matches_manifest=bool(count == manifest['multiview']['frames']))
                row['real'] = audit_view(head, data, '', count, a.batch_size, replay)
                row['mirror'] = audit_view(head, data, 'mirror_', count, a.batch_size, replay)
                row['status'] = 'verified_replay' if all([row['topology_matches_provisioned_model'], row['embedded_video_hash_matches'], row['frame_count_matches_manifest'], row['real'].get('replay_gate_passed'), row['mirror'].get('replay_gate_passed')]) else 'audit_requires_review'
                del data
            save(a.output/(Path(item['video']).stem+'.json'), row)
            report['clips'].append(row)
            save(a.output/'native_audit_report.json', report)
            print(json.dumps({'video':row['video'], 'status':row['status'],
                              'real_abs_max_m':row.get('real',{}).get('replay_abs_component_max_m'),
                              'mirror_abs_max_m':row.get('mirror',{}).get('replay_abs_component_max_m'),
                              'mirror_missing':len(row.get('mirror',{}).get('missing_frames_0based',[]))}), flush=True)
        report.update(status='complete' if all(r['status']=='verified_replay' for r in report['clips']) else 'requires_review',
                      seconds=time.monotonic()-started, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      metric_limits='Replay consistency validates stored parameters and coordinate/topology versions; not body shape, hand grip, mirror geometry or pixel accuracy against ground truth.')
        save(a.output/'native_audit_report.json', report)
        print('Native replay audit complete:', report['status'], flush=True)


if __name__ == '__main__':
    main()
