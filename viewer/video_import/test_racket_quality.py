import json
import tempfile
import unittest
from pathlib import Path

from racket_quality import gate


class RacketPublicationTests(unittest.TestCase):
    def fixture(self, root, shift):
        row = {'status': 'fitted', 'quality': 'silhouette_fitted',
               'rotation_camera_columns': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               'translation_camera_m': [0, -.045, 5],
               'grip_target_camera_m': [0, 0, 5]}
        reference = {'video_sha256': 'same-video', 'model': {'head_center_y_m': .5},
                     'frames': [row for _ in range(3)]}
        candidate = json.loads(json.dumps(reference))
        for frame in candidate['frames']:
            frame['translation_camera_m'][0] += shift
        keypoints = {'video_sha256': 'same-video', 'frames': [
            {'real': {'points': {'head_center': [640, 469.2]}}} for _ in range(3)]}
        meta = {'video_sha256': 'same-video', 'image_size': [1280, 720], 'focal': [1200]*3}
        for name, data in [('racket_poses_reference.json', reference),
                           ('racket_poses.json', candidate),
                           ('racket_keypoints.json', keypoints), ('mesh_meta.json', meta)]:
            (root/name).write_text(json.dumps(data))

    def test_reprojection_regression_keeps_published_reference_and_reviewable_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, .5)
            before = (root/'racket_poses_reference.json').read_bytes()
            candidate = (root/'racket_poses.json').read_bytes()
            report = gate(root)
            self.assertEqual(report['status'], 'needs_review')
            self.assertIn('head_center_error_canonical_px_median', report['failed_gates'])
            self.assertEqual((root/'racket_poses.json').read_bytes(), before)
            self.assertEqual((root/'racket_poses_directional.json').read_bytes(), candidate)
            self.assertNotEqual(report['candidate_sha256'], report['published_sha256'])

    def test_matching_evidence_accepts_without_overwriting_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, 0)
            before = (root/'racket_poses_reference.json').read_bytes()
            report = gate(root)
            self.assertEqual(report['status'], 'accepted')
            self.assertEqual(report['failed_gates'], [])
            self.assertEqual((root/'racket_poses_reference.json').read_bytes(), before)
            self.assertEqual(report['candidate_sha256'], report['published_sha256'])
