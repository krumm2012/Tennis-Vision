import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from apply_grip_collection import prepare


class GripCollectionTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'clip';(source/'result').mkdir(parents=True)
        video=b'normalized clip A';(source/'source.mp4').write_bytes(video)
        meta={'video_sha256':hashlib.sha256(video).hexdigest(),'image_size':[2560,1440],
              'frames':250,'fps':25,'mirror_available':True}
        (source/'result/mesh_meta.json').write_text(json.dumps(meta))
        (source/'record.json').write_text(json.dumps({'attempt':3}))
        work=source/'attempts/0003/work';work.mkdir(parents=True)
        (work/'reconstruction.npz').write_bytes(b'archive')
        (source/'result/mesh_local.bin').write_bytes(b'body')
        reference=root/'reference';reference.mkdir()
        accepted=reference/'manual_grip_face_acceptance.json';accepted.write_text('{}')
        (reference/'racket_poses.json').write_text(json.dumps({'model':{
            'length_m':.685,'head_half_width_m':.13,'head_outline':[[0,.35,0],[0,.685,0]]}}))
        return source,accepted,meta

    def test_setting_is_bound_to_target_clip_and_preserves_native_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source,reference,meta=self.fixture(root)
            target=root/'stage';prepare(source,target,reference)
            setting=json.loads((target/'result/racket_grip_settings.json').read_text())
            self.assertEqual(setting['video_sha256'],meta['video_sha256'])
            self.assertEqual(setting['grip_from_butt_m'],.09)
            self.assertFalse(setting['accepted_for_this_clip'])
            self.assertFalse(setting['distance_measured'])
            self.assertTrue((target/'attempts/0003/work/reconstruction.npz').exists())
            # A staging metadata change cannot change the existing displayed body.
            (target/'result/mesh_meta.json').write_text('{}')
            self.assertEqual(json.loads((source/'result/mesh_meta.json').read_text()),meta)
            self.assertEqual((source/'result/mesh_local.bin').read_bytes(),b'body')

    def test_wrong_source_video_fails_before_staging_or_fitting(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source,reference,_=self.fixture(root)
            (source/'source.mp4').write_bytes(b'wrong clip')
            with self.assertRaisesRegex(ValueError,'identity differs'):
                prepare(source,root/'stage',reference)
            self.assertFalse((root/'stage').exists())

    def test_incompatible_timeline_cannot_reuse_25fps_constraints(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source,reference,meta=self.fixture(root)
            meta['fps']=50
            (source/'result/mesh_meta.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'timeline'):
                prepare(source,root/'stage',reference)


if __name__=='__main__':unittest.main()
