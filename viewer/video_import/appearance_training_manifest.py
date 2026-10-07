"""Build hash-bound offline inputs for future dynamic MHR appearance training.

Group spec v1 is {schema_version: 1, layout_sha256, clips: [...]}. Every clip
requires clip_id, scene_id, person_id, outfit_id, racket_id, group_id and video /
mesh_meta file references {path, sha256}. native_archive and native_manifest are
nullable references: absence means needs_input, never another clip's archive.
Optional train_frames / heldout_frames must occur together (zero based normalized
video indices); otherwise sample every 5 frames and reserve every 25th frame.
Optional top-level source_index / batch_manifest references bind existing batch
schemas. Optional per-clip racket_assets are separate file references.
Paths resolve relative to the spec. CLI output is one new JSON file; no overwrite.
No inference, baking, rendering, GPU work, or mirror observations are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from run_records import sha256

IDENTITIES = ('scene_id', 'person_id', 'outfit_id', 'racket_id', 'group_id')
PARAMETERS = {'mhr_model_params': 204, 'mhr_shape_params': 45, 'mhr_expr_params': 72}


def reference(value, base):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str):
        raise ValueError('file reference requires path and sha256')
    path = Path(value['path'])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    actual = sha256(path)
    if actual != value.get('sha256'):
        raise ValueError(f'input hash mismatch: {path}')
    return {'path': str(path), 'sha256': actual}


def read_json(ref):
    return json.loads(Path(ref['path']).read_text())


def array_headers(path):
    """Inspect array shapes without materializing per-frame geometry or masks."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError('object arrays are forbidden')
                result[name[:-4]] = shape
    return result


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or min(size) <= 0:
            raise ValueError('video cannot be read')
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if list(image.shape[1::-1]) != size:
                raise ValueError('video dimensions vary')
            count += 1
        if count == 0 or count != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError('decoded video frame count mismatch')
        return {'frames': count, 'fps': float(fps), 'image_size': size,
                'frame_count_check': 'sequential decode matches container count'}
    finally:
        cap.release()


def frame_split(clip, count):
    explicit = 'train_frames' in clip or 'heldout_frames' in clip
    if explicit and not all(k in clip for k in ('train_frames', 'heldout_frames')):
        raise ValueError('train_frames and heldout_frames must occur together')
    train = clip['train_frames'] if explicit else [i for i in range(0, count, 5) if i % 25]
    heldout = clip['heldout_frames'] if explicit else list(range(0, count, 25))
    for name, frames in [('train', train), ('heldout', heldout)]:
        if (not isinstance(frames, list) or not frames or
                any(type(i) is not int or not 0 <= i < count for i in frames) or
                len(set(frames)) != len(frames)):
            raise ValueError(f'invalid {name} frames (nonempty unique zero-based indices required)')
    if set(train) & set(heldout):
        raise ValueError('TRAIN/HELDOUT overlap')
    return {'train_frames': sorted(train), 'heldout_frames': sorted(heldout),
            'index_space': 'normalized video; both lists include only real view',
            'rule': 'explicit' if explicit else 'sample every 5 frames; reserve every 25th frame'}


def validate_layout(path):
    with np.load(path, allow_pickle=False) as data:
        faces, uv, uv_faces, rest = (data[k] for k in ('faces', 'uv', 'uv_faces', 'rest_vertices'))
        for name, a, width, limit in [('faces', faces, 3, len(rest)), ('uv_faces', uv_faces, 3, len(uv))]:
            if (a.ndim != 2 or a.shape[1] != width or not a.size or
                    not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() >= limit):
                raise ValueError(f'invalid layout {name}')
        if (rest.shape != (18439, 3) or uv.ndim != 2 or uv.shape[1] != 2 or
                faces.shape != uv_faces.shape or not np.isfinite(rest).all() or
                not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1))):
            raise ValueError('invalid native MHR topology/UV layout')
        model_hash = str(data['model_sha256']) if 'model_sha256' in data else None
        return faces, {'vertices': len(rest), 'faces': len(faces), 'uv_vertices': len(uv),
                       'topology_sha256': hashlib.sha256(faces.astype('<i8').tobytes()).hexdigest(),
                       'model_sha256': model_hash,
                       'uv_layout_check': 'finite unit-square UV and corresponding face row count/index bounds',
                       'semantic_uv_correspondence': 'provided layout; not independently rederived from model'}


def validate_archive(ref, remote_ref, video, meta, timeline, faces):
    count = timeline['frames']
    remote = read_json(remote_ref)
    if (remote.get('status') != 'ready' or remote.get('source_sha256') != video['sha256'] or
            remote.get('artifact_sha256', {}).get('reconstruction.npz') != ref['sha256']):
        raise ValueError('native manifest source/archive hash mismatch or not ready')
    if 'frames' in remote.get('multiview', {}) and remote['multiview']['frames'] != count:
        raise ValueError('native manifest frame count mismatch')
    shapes = array_headers(ref['path'])
    expected = {'vertices': (count, 18439, 3), 'source_roots': (count, 3), 'focal': (count,)}
    for key, shape in expected.items():
        if shapes.get(key) != shape:
            raise ValueError(f'native {key} frame count/shape mismatch')
    missing = [k for k in PARAMETERS if k not in shapes]
    with np.load(ref['path'], allow_pickle=False) as data:
        if not np.array_equal(data['faces'], faces):
            raise ValueError('native topology mismatch with UV layout')
        if 'video_sha256' not in data or str(data['video_sha256']) != video['sha256']:
            raise ValueError('native embedded video hash mismatch')
        focal = data['focal']
        if not np.isfinite(focal).all() or np.any(focal <= 0):
            raise ValueError('native camera focal invalid')
        if 'focal' in meta and not np.allclose(focal, meta['focal'], rtol=0, atol=1e-4):
            raise ValueError('native camera focal mismatch with metadata')
        if not np.isfinite(data['source_roots']).all():
            raise ValueError('native source roots invalid')
        if 'source_roots' in meta and not np.allclose(data['source_roots'], meta['source_roots'], rtol=0, atol=1e-4):
            raise ValueError('native source roots mismatch with metadata')
        for key, width in PARAMETERS.items():
            if key in shapes and (shapes[key] != (count, width) or not np.isfinite(data[key]).all()):
                raise ValueError(f'native MHR parameter shape/nonfinite mismatch: {key}')
    return {'status': 'needs_input' if missing else 'available', 'missing_parameters': missing,
            'parameter_shapes': {k: list(shapes[k]) for k in PARAMETERS if k in shapes},
            'topology_matches_layout': True, 'focal_matches_meta': 'focal' in meta,
            'replay_verified_by_this_manifest': False,
            'geometry_finite_scan_performed': False}


def validate_index(index, clip, refs, timeline, batch=False):
    matches = [r for r in index['clips'] if (Path(r['folder']) / 'source.mp4').resolve() == Path(refs['video']['path'])]
    if len(matches) != 1:
        raise ValueError('clip does not uniquely match existing batch/index source path')
    row = matches[0]
    comparisons = {'normalized_sha256': refs['video']['sha256'], 'frames': timeline['frames'],
                   'fps': timeline['fps'], 'resolution': timeline['image_size']}
    if refs['native_archive']:
        comparisons['archive_sha256'] = refs['native_archive']['sha256']
    for key, value in comparisons.items():
        if key in row and row[key] != value:
            raise ValueError(f'existing source index mismatch: {key}')
    if batch and (index.get('status') != 'ready' or row.get('status') != 'ready'):
        raise ValueError('batch/clip not ready')


def build_manifest(group_spec, layout):
    group_spec, layout = Path(group_spec).resolve(), Path(layout).resolve()
    spec = json.loads(group_spec.read_text())
    if spec.get('schema_version') != 1 or not spec.get('clips'):
        raise ValueError('group spec v1 requires nonempty clips')
    groups, ids = {}, set()
    for clip in spec['clips']:
        if clip.get('view', 'real') != 'real':
            raise ValueError('only real view inputs are supported; mirror paths must remain separate')
        for key in ('clip_id',) + IDENTITIES:
            if not isinstance(clip.get(key), str) or not clip[key].strip():
                raise ValueError(f'explicit nonempty {key} required')
        if clip['clip_id'] in ids:
            raise ValueError('duplicate clip_id')
        ids.add(clip['clip_id'])
        identity = tuple(clip[k] for k in ('scene_id', 'person_id', 'outfit_id'))
        prior = groups.setdefault(clip['group_id'], identity)
        if prior != identity:
            raise ValueError('mixed scene/person/outfit within body texture group')
    layout_ref = reference({'path': str(layout), 'sha256': spec.get('layout_sha256')}, group_spec.parent)
    faces, layout_info = validate_layout(layout)
    indexes = {k: reference(spec[k], group_spec.parent) for k in ('source_index', 'batch_manifest') if k in spec}
    index_data = {k: read_json(v) for k, v in indexes.items()}
    rows, sources, frame_roles = [], {}, {}
    for clip in spec['clips']:
        refs = {k: reference(clip[k], group_spec.parent) if clip.get(k) is not None else None
                for k in ('video', 'mesh_meta', 'native_archive', 'native_manifest')}
        if refs['video'] is None or refs['mesh_meta'] is None:
            raise ValueError('video and mesh_meta references required')
        meta = read_json(refs['mesh_meta']); timeline = video_info(refs['video']['path'])
        missing = [k for k in ('video_sha256', 'image_size', 'frames', 'fps', 'vertices', 'faces', 'focal', 'source_roots') if k not in meta]
        if 'video_sha256' in meta and meta['video_sha256'] != refs['video']['sha256']:
            raise ValueError('metadata video hash mismatch')
        for key in ('frames', 'image_size'):
            if key in meta and meta[key] != timeline[key]:
                raise ValueError(f'metadata {key} mismatch')
        if 'fps' in meta and not np.isclose(meta['fps'], timeline['fps'], rtol=0, atol=1e-3):
            raise ValueError('metadata fps mismatch')
        if ('vertices' in meta and meta['vertices'] != layout_info['vertices'] or
                'faces' in meta and meta['faces'] != layout_info['faces']):
            raise ValueError('metadata topology count mismatch')
        count = timeline['frames']
        for key, shape in [('focal', (count,)), ('source_roots', (count, 3))]:
            if key in meta:
                a = np.asarray(meta[key])
                if a.shape != shape or not np.isfinite(a).all() or (key == 'focal' and np.any(a <= 0)):
                    raise ValueError(f'metadata camera {key} invalid')
        split = frame_split(clip, count)
        # Duplicate video files cannot change identity or leak heldout frames across aliases.
        source = refs['video']['sha256']; identity = tuple(clip[k] for k in IDENTITIES)
        if source in sources and sources[source] != identity:
            raise ValueError('same video assigned conflicting identity/group')
        sources[source] = identity
        roles = frame_roles.setdefault(source, {'train': set(), 'heldout': set()})
        if (roles['train'] & set(split['heldout_frames']) or roles['heldout'] & set(split['train_frames'])):
            raise ValueError('TRAIN/HELDOUT leakage across duplicate video aliases')
        roles['train'].update(split['train_frames']); roles['heldout'].update(split['heldout_frames'])
        native = {'status': 'needs_input', 'missing_parameters': list(PARAMETERS),
                  'replay_verified_by_this_manifest': False}
        if refs['native_archive'] and refs['native_manifest']:
            native = validate_archive(refs['native_archive'], refs['native_manifest'], refs['video'], meta, timeline, faces)
        elif refs['native_archive'] or refs['native_manifest']:
            raise ValueError('native_archive and native_manifest must be supplied together')
        for key, index in index_data.items():
            validate_index(index, clip, refs, timeline, batch=key == 'batch_manifest')
        racket = [reference(v, group_spec.parent) for v in clip.get('racket_assets', [])]
        rows.append({'clip_id': clip['clip_id'], **{k: clip[k] for k in IDENTITIES},
                     'group_identity': {k: clip[k] for k in ('group_id', 'scene_id', 'person_id', 'outfit_id')},
                     'status': 'needs_input' if missing or native['status'] != 'available' else 'verified_offline_inputs',
                     'missing_meta_fields': missing, 'inputs': refs, 'video': timeline, 'native_mhr': native,
                     'frames': split, 'racket': {'racket_id': clip['racket_id'], 'assets': racket,
                                                'training_target': 'separate asset; excluded from body material'},
                     'camera': {'projection': 'u=fx*x/z+width/2; v=fy*y/z+height/2; fx=fy=focal',
                                'axes': 'camera x right, y image-down, z forward; metres assumed by existing producer',
                                'geometry': 'archive vertices + source_roots, real view only',
                                'native_replay_conversion': 'official replay x,-y,-z; not replayed here',
                                'focal_reference': 'native_archive:focal and mesh_meta:focal',
                                'coordinate_calibration_verified': False},
                     'mirror': {'included_in_training': False, 'review_verified_by_this_manifest': False},
                     'lighting': {'verified': False, 'model': 'unspecified; exposure/illumination separation pending'}})
    return {'schema_version': 1, 'kind': 'dynamic_mhr_offline_appearance_inputs',
            'status': 'needs_input' if any(r['status'] == 'needs_input' for r in rows) else 'verified_offline_inputs',
            'group_spec': {'path': str(group_spec), 'sha256': sha256(group_spec)},
            'generator_sha256': sha256(__file__), 'layout': {**layout_ref, **layout_info},
            'source_records': indexes,
            'groups': [{'group_id': gid, 'scene_id': v[0], 'person_id': v[1], 'outfit_id': v[2],
                        'identity_source': 'explicit user labels; not automatic recognition',
                        'body_material_share_scope': 'this group only'} for gid, v in groups.items()],
            'clips': rows, 'training_executed': False,
            'limits': ['Input integrity only; no GPU replay or visual/geometry acceptance.',
                       'No mirror evidence or existing mirror review is promoted to an acceptance claim.',
                       'No lighting fit, UV baking, material optimization, or Viewer publication.']}


def verify_fusion_group(manifest, batch_clips):
    """Bind selected existing batch folders to one explicitly labelled body group.

    Returns group_identity. Rechecks referenced file hashes, but does not validate
    fusion observation-cache splits: the caller must match its step/heldout rule
    to each clip's frames lists before using cached observations.
    """
    if manifest.get('kind') != 'dynamic_mhr_offline_appearance_inputs':
        raise ValueError('unsupported appearance input manifest')
    selected = []
    for entry in batch_clips:
        source = (Path(entry['folder']) / 'source.mp4').resolve()
        matches = [r for r in manifest['clips'] if Path(r['inputs']['video']['path']).resolve() == source]
        if len(matches) != 1:
            raise ValueError('fusion source missing or ambiguous in manifest')
        selected.append(matches[0])
    identities = {tuple(r[k] for k in ('group_id', 'scene_id', 'person_id', 'outfit_id')) for r in selected}
    if len(identities) != 1:
        raise ValueError('fusion requires one explicit scene/person/outfit/group identity')
    reference(manifest['layout'], Path('.'))
    for row in selected:
        if row['status'] != 'verified_offline_inputs':
            raise ValueError('fusion group needs_input')
        for ref in row['inputs'].values():
            if ref is not None:
                reference(ref, Path('.'))
        frame_split(row['frames'], row['video']['frames'])
    return dict(zip(('group_id', 'scene_id', 'person_id', 'outfit_id'), next(iter(identities))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-spec', type=Path, required=True)
    p.add_argument('--layout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new manifest JSON path; refuses overwrite')
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('output already exists; choose a new manifest path')
    report = build_manifest(a.group_spec, a.layout)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'clips': len(report['clips']), 'output': str(a.output.resolve())}))


if __name__ == '__main__':
    main()
