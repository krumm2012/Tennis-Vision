import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from export_pose_kinematics import angular_velocity, derivative, segment_frame, export
from run_records import sha256


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / 'result'
        self.result.mkdir()
        self.archive = self.root / 'reconstruction.npz'
        self.output = self.root / 'export'
        (self.result / 'video.mp4').write_bytes(b'synthetic-video-identity')
        self.vertices = np.full((5, 3, 3), 0.123456789, dtype=np.float64)
        self.joints = np.random.default_rng(0).normal(size=(5, 70, 3))
        self.roots = np.zeros((5, 3))
        self.meta = {'frames': 5, 'fps': 25, 'vertices': 3, 'image_size': [64, 48],
                     'video_sha256': sha256(self.result / 'video.mp4'),
                     'source_roots': self.roots.tolist()}
        self.racket = {**self.meta, 'model': {'grip_y_m': .09, 'throat_y_m': .2,
                       'length_m': .68, 'head_center_y_m': .5, 'head_half_width_m': .13},
                       'frames': [{'frame': i, 'status': 'fitted',
                                   'rotation_camera_columns': np.eye(3).tolist(),
                                   'translation_camera_m': [0, 0, 1]} for i in range(5)]}
        self.vertices.astype('<f4').tofile(self.result / 'mesh_local.bin')
        np.savez(self.archive, vertices=self.vertices, joints=self.joints,
                 source_roots=self.roots, video_sha256=self.meta['video_sha256'])

    def run_export(self):
        (self.result / 'mesh_meta.json').write_text(json.dumps(self.meta))
        (self.result / 'racket_poses.json').write_text(json.dumps(self.racket))
        return export(self.result, self.archive, self.output)

    def test_float64_archive_and_explicit_angular_time_axes(self):
        report = self.run_export()
        with np.load(self.output / 'pose3d.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['body_camera_m'], self.joints)
            for key, axis in report['derivatives']['angular_time_arrays'].items():
                self.assertEqual(len(data[key + '_angular_velocity_rad_s']), len(data[axis]))
            self.assertEqual(len(data['angular_interval_time_s']), 4)
            self.assertEqual(len(data['angular_frame_time_s']), 5)
        self.assertEqual(len(json.loads((self.output / 'pose3d.json').read_text())['frames']), 5)

    def test_invalid_fps_and_root_shape_rejected_before_output(self):
        for fps in [0, -25, float('nan'), float('inf')]:
            self.meta['fps'] = fps
            self.racket['fps'] = fps
            with self.assertRaises(ValueError):
                self.run_export()
            self.assertFalse(self.output.exists())
        self.meta['fps'] = self.racket['fps'] = 25
        self.meta['source_roots'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            self.run_export()

    def test_invalid_rigid_translation_cannot_broadcast(self):
        self.racket['frames'][0]['translation_camera_m'] = 1
        with self.assertRaises(ValueError):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_paired_preview_cannot_be_exported_with_raw_body(self):
        (self.result / 'joint_preview_manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'paired candidate archive'):
            self.run_export()
        self.assertFalse(self.output.exists())


class KinematicsTests(unittest.TestCase):
    def test_constant_translation_has_correct_velocity(self):
        fps = 25
        time = np.arange(10) / fps
        position = time[:, None] * np.array([[2., -3., 4.]])
        velocity = derivative(position, fps)
        np.testing.assert_allclose(velocity[1:-1], np.tile([2, -3, 4], (8, 1)))
        self.assertTrue(np.isnan(velocity[[0, -1]]).all())

    def test_hidden_frame_invalidates_neighbour_derivatives(self):
        positions = np.arange(21, dtype=float).reshape(7, 3)
        positions[3] = np.nan
        velocities = derivative(positions, 25)
        self.assertTrue(np.isnan(velocities[2:5]).all())
        self.assertTrue(np.isfinite(velocities[[1, 5]]).all())

    def test_known_rotation_and_hidden_frame(self):
        fps = 25
        angles = np.arange(7) / fps * 2
        matrices = Rotation.from_rotvec(angles[:, None] * np.array([[0., 0., 1.]])).as_matrix()
        np.testing.assert_allclose(angular_velocity(matrices, fps), np.tile([0, 0, 2], (6, 1)), atol=1e-12)
        matrices[3] = np.nan
        velocity = angular_velocity(matrices, fps)
        self.assertTrue(np.isnan(velocity[2:4]).all())
        self.assertTrue(np.isfinite(velocity[[0, 1, 4, 5]]).all())

    def test_proxy_frame_is_right_handed_and_degeneracy_is_missing(self):
        lateral = np.array([[1., 1., 0.], [0., 1., 0.]])
        up = np.array([[0., 1., 0.], [0., 1., 0.]])
        matrices = segment_frame(lateral, up)
        np.testing.assert_allclose(matrices[0], np.eye(3))
        self.assertTrue(np.isnan(matrices[1, :, 0]).all())


if __name__ == '__main__':
    unittest.main()
