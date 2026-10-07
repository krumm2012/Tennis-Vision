"""A fresh release must contain the scripts loaded by both Viewer pages."""
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class DeploymentTests(unittest.TestCase):
    def test_fresh_release_contains_viewer_script_dependencies(self):
        repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('prepare_release', repo / 'deploy/3dpose/prepare.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            for name in module.FILES + ('racket_poses.json', 'wilson_mesh.bin', 'wilson_model.json',
                                       'racket_poses_v1.json', 'racket_poses_v2.json', 'racket_poses_v3.json'):
                (data / name).write_bytes(b'synthetic-release-data')
            joint = data / 'joint_fit_v4'
            full = joint / 'full'
            full.mkdir(parents=True)
            for name in ('report.json', 'quality.json', 'viewer_data.json'):
                (joint / name).write_text('{}')
            for name in ('ground_calibration.json', 'racket_poses.json', 'wilson_mesh.bin', 'coaching_report.json'):
                (full / name).write_text('{}')
            (full / 'viewer.html').write_text((module.SOURCE / 'viewer.html').read_text())
            with patch.object(module, 'DATA', data), patch.object(module, 'TARGET', root / 'release'):
                public = module.prepare() / 'public'
            manifest = {row['name'] for row in json.loads((public / 'release_manifest.json').read_text())}
            for prefix in ('', 'joint_fit_v4/full/'):
                page = public / prefix / 'viewer.html'
                scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)', page.read_text())
                self.assertTrue(scripts)
                for script in scripts:
                    self.assertTrue((page.parent / script).is_file(), script)
                    self.assertIn(prefix + script, manifest)
