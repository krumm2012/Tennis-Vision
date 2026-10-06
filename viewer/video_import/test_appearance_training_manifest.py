"""Synthetic offline integrity tests; no cloud, model loading or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import cv2
import numpy as np

from appearance_training_manifest import build_manifest, sha256, verify_fusion_group
from multivideo_texture import fuse, validate_fusion_group, validate_cache_split, validate_cache_binding, fallback_candidates


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.faces = np.array([[0, 1, 2]], np.int32)
        self.layout = self.root / 'layout.npz'
        np.savez(self.layout, faces=self.faces, uv=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                 uv_faces=self.faces, rest_vertices=np.zeros((18439, 3)), model_sha256='a'*64)
        self.video = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (32, 24))
        for i in range(6):
            writer.write(np.full((24, 32, 3), i*20, np.uint8))
        writer.release()
        self.meta = self.root / 'meta.json'
        self.metadata = {'video_sha256': sha256(self.video), 'frames': 6, 'vertices': 18439,
                         'faces': 1, 'fps': 25., 'image_size': [32, 24],
                         'focal': [50.]*6, 'source_roots': [[0., 0., 0.]]*6}
        self.write(self.meta, self.metadata)
        self.archive = self.root / 'reconstruction.npz'
        self.arrays = {'vertices': np.zeros((6, 18439, 3), np.float32),
                       'faces': self.faces, 'focal': np.full(6, 50.), 'source_roots': np.zeros((6, 3)),
                       'video_sha256': sha256(self.video), 'mhr_model_params': np.zeros((6, 204)),
                       'mhr_shape_params': np.zeros((6, 45)), 'mhr_expr_params': np.zeros((6, 72))}
        self.remote = self.root / 'remote_manifest.json'
        self.write_archive()
        self.clip = {'clip_id': 'a', 'scene_id': 'scene', 'person_id': 'person', 'outfit_id': 'outfit',
                     'racket_id': 'racket', 'group_id': 'group', 'video': self.ref(self.video),
                     'mesh_meta': self.ref(self.meta), 'native_archive': self.ref(self.archive),
                     'native_manifest': self.ref(self.remote), 'racket_assets': [],
                     'train_frames': [1, 2, 3, 4], 'heldout_frames': [0, 5]}
        self.spec = {'schema_version': 1, 'layout_sha256': sha256(self.layout), 'clips': [self.clip]}

    def write(self, path, obj):
        path.write_text(json.dumps(obj))

    def ref(self, path):
        return {'path': str(path), 'sha256': sha256(path)}

    def write_archive(self):
        np.savez_compressed(self.archive, **self.arrays)
        self.write(self.remote, {'status': 'ready', 'source_sha256': sha256(self.video),
                                'artifact_sha256': {'reconstruction.npz': sha256(self.archive)},
                                'multiview': {'frames': 6}})

    def refresh_archive(self):
        self.write_archive()
        self.clip['native_archive'] = self.ref(self.archive)
        self.clip['native_manifest'] = self.ref(self.remote)

    def build(self):
        p = self.root / 'spec.json'
        self.write(p, self.spec)
        return build_manifest(p, self.layout)

    def test_available_native_and_explicit_split(self):
        report = self.build()
        self.assertEqual(report['status'], 'verified_offline_inputs')
        row = report['clips'][0]
        self.assertEqual(row['frames']['train_frames'], [1, 2, 3, 4])
        self.assertFalse(row['mirror']['included_in_training'])
        self.assertFalse(row['native_mhr']['replay_verified_by_this_manifest'])
        self.assertFalse(row['lighting']['verified'])
        self.assertEqual(verify_fusion_group(report, [{'folder': str(self.root)}])['group_id'], 'group')

    def test_fusion_binds_actual_folder_assets(self):
        report = self.build()
        result = self.root / 'result'; result.mkdir()
        work = self.root / 'attempts/0001/work'; work.mkdir(parents=True)
        shutil.copyfile(self.meta, result / 'mesh_meta.json')
        shutil.copyfile(self.archive, work / 'reconstruction.npz')
        shutil.copyfile(self.remote, work / 'remote_manifest.json')
        path = self.root / 'training.json'; self.write(path, report)
        identity, rows = validate_fusion_group(path, [{'folder': str(self.root)}], self.layout)
        self.assertEqual(identity['group_id'], 'group')
        validate_cache_split({'all_frame': [1, 2], 'heldout_frame': [0, 5]}, rows[0])
        # Manifest refs remain unchanged; the actual fusion folder must still bind.
        self.write(result / 'mesh_meta.json', {'unrelated': True})
        with self.assertRaisesRegex(ValueError, 'fusion asset differs'):
            validate_fusion_group(path, [{'folder': str(self.root)}], self.layout)

    def test_cached_heldout_cannot_become_training(self):
        row = self.build()['clips'][0]
        with self.assertRaisesRegex(ValueError, 'non-TRAIN'):
            validate_cache_split({'all_frame': [1, 5], 'heldout_frame': [0]}, row)
        with self.assertRaisesRegex(ValueError, 'heldout differs'):
            validate_cache_split({'all_frame': [1], 'heldout_frame': [2]}, row)

    def test_fallback_refuses_missing_group_before_writing(self):
        output = self.root / 'must_not_exist'
        with self.assertRaisesRegex(ValueError, 'group manifest'):
            fuse(self.root, self.layout, output, pixel_fallback=True)
        self.assertFalse(output.exists())

    def test_cache_bytes_and_generation_metadata_are_bound(self):
        result = self.root / 'result'; result.mkdir()
        shutil.copyfile(self.meta, result / 'mesh_meta.json')
        cache = self.root / 'observations.npz'; np.savez(cache, all_frame=[1, 2])
        identity = {'mesh_meta_sha256': sha256(result / 'mesh_meta.json'),
                    'observation_cache_sha256': sha256(cache)}
        self.assertEqual(validate_cache_binding(identity, cache, self.root)['status'], 'verified')
        with self.assertRaisesRegex(ValueError, 'legacy observation cache'):
            validate_cache_binding({}, cache, self.root)
        self.assertEqual(validate_cache_binding({}, cache, self.root, allow_legacy=True)['status'],
                         'legacy_unverified_generation_binding')
        np.savez(cache, all_frame=[5])
        with self.assertRaisesRegex(ValueError, 'NPZ content changed'):
            validate_cache_binding(identity, cache, self.root)
        identity['observation_cache_sha256'] = sha256(cache)
        self.write(result / 'mesh_meta.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            validate_cache_binding(identity, cache, self.root)

    def test_explicit_train_can_override_default_modulus(self):
        # Frame 25 is TRAIN under a custom split, not universally HELDOUT.
        scores = np.array([[3.], [2.], [1.]], np.float32)
        candidates = fallback_candidates(scores, np.array([0]), np.array([1, 25, 0]), 5, 2,
                                         train_allowed=np.array([True, True, False]))
        self.assertEqual(candidates[:, 0].tolist(), [1, -1])

    def test_no_native_and_incomplete_default_meta_needs_input(self):
        self.clip['native_archive'] = self.clip['native_manifest'] = None
        self.write(self.meta, {'frames': 6, 'fps': 25., 'vertices': 18439, 'faces': 1})
        self.clip['mesh_meta'] = self.ref(self.meta)
        report = self.build()
        self.assertEqual(report['status'], 'needs_input')
        self.assertIn('focal', report['clips'][0]['missing_meta_fields'])
        self.assertIsNone(report['clips'][0]['inputs']['native_archive'])
        with self.assertRaisesRegex(ValueError, 'needs_input'):
            verify_fusion_group(report, [{'folder': str(self.root)}])

    def test_geometry_without_native_parameters_needs_input(self):
        del self.arrays['mhr_model_params']
        self.refresh_archive()
        self.assertEqual(self.build()['status'], 'needs_input')

    def test_mixed_identity_rejected(self):
        for key in ('scene_id', 'person_id', 'outfit_id'):
            with self.subTest(key=key):
                other = copy.deepcopy(self.clip)
                other['clip_id'] = 'b'; other[key] = 'other'
                self.spec['clips'] = [self.clip, other]
                with self.assertRaisesRegex(ValueError, 'mixed'):
                    self.build()

    def test_input_hash_rejected(self):
        self.clip['video']['sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_native_source_binding_rejected(self):
        remote = json.loads(self.remote.read_text()); remote['source_sha256'] = 'f'*64
        self.write(self.remote, remote); self.clip['native_manifest'] = self.ref(self.remote)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_frame_count_rejected(self):
        self.arrays['vertices'] = self.arrays['vertices'][:5]
        self.refresh_archive()
        with self.assertRaisesRegex(ValueError, 'frame count'):
            self.build()

    def test_topology_rejected(self):
        self.arrays['faces'] = self.faces[:, ::-1]
        self.refresh_archive()
        with self.assertRaisesRegex(ValueError, 'topology mismatch'):
            self.build()

    def test_focal_mismatch_rejected(self):
        self.arrays['focal'] = np.full(6, 51.)
        self.refresh_archive()
        with self.assertRaisesRegex(ValueError, 'focal mismatch'):
            self.build()

    def test_native_parameter_shape_rejected(self):
        self.arrays['mhr_model_params'] = np.zeros((6, 203))
        self.refresh_archive()
        with self.assertRaisesRegex(ValueError, 'parameter shape'):
            self.build()

    def test_metadata_frame_count_rejected(self):
        self.metadata['frames'] = 5
        self.write(self.meta, self.metadata); self.clip['mesh_meta'] = self.ref(self.meta)
        with self.assertRaisesRegex(ValueError, 'metadata frames mismatch'):
            self.build()

    def test_mirror_view_rejected(self):
        self.clip['view'] = 'mirror'
        with self.assertRaisesRegex(ValueError, 'only real view'):
            self.build()

    def test_heldout_overlap_rejected(self):
        self.clip['train_frames'].append(0)
        with self.assertRaisesRegex(ValueError, 'TRAIN/HELDOUT overlap'):
            self.build()

    def test_alias_heldout_leakage_rejected(self):
        other = copy.deepcopy(self.clip); other['clip_id'] = 'b'
        other['train_frames'] = [0, 1]; other['heldout_frames'] = [2, 5]
        self.spec['clips'].append(other)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.build()

    def test_default_fixed_split(self):
        del self.clip['train_frames']; del self.clip['heldout_frames']
        frames = self.build()['clips'][0]['frames']
        self.assertEqual(frames['heldout_frames'], [0])
        self.assertEqual(frames['train_frames'], [5])

    def test_uv_invalid_rejected(self):
        with np.load(self.layout) as z:
            data = dict(z)
        data['uv'][0, 0] = np.nan
        np.savez(self.layout, **data); self.spec['layout_sha256'] = sha256(self.layout)
        with self.assertRaisesRegex(ValueError, 'UV layout'):
            self.build()

    def test_different_groups_cannot_fuse(self):
        report = self.build()
        other = copy.deepcopy(report['clips'][0]); other['clip_id'] = 'b'; other['group_id'] = 'other'
        second = self.root / 'second'; second.mkdir()
        other['inputs']['video']['path'] = str(second/'source.mp4')
        report['clips'].append(other)
        with self.assertRaisesRegex(ValueError, 'one explicit'):
            verify_fusion_group(report, [{'folder': str(self.root)}, {'folder': str(second)}])


if __name__ == '__main__':
    unittest.main()
