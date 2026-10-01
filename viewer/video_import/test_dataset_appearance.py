import json,tempfile,unittest
from pathlib import Path
import numpy as np
from publish_dataset_appearance import publish
from run_records import sha256,write

class AppearanceBindingTests(unittest.TestCase):
    def test_same_capture_and_topology_required_and_pose_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'result';out.mkdir();fusion=root/'fusion';fusion.mkdir();layout=root/'uv.npz'
            faces=np.array([[0,1,2]],np.uint32);np.savez(layout,faces=faces,uv_faces=faces,uv=np.array([[0,0],[1,0],[0,1]],np.float32));faces.astype('<u4').tofile(out/'mesh_faces.bin')
            write(out/'mesh_meta.json',{'video_sha256':'a','vertices':3});(out/'racket_poses.json').write_text('preserve');(out/'viewer.html').write_text('old')
            (fusion/'body_texture_rgba.png').write_bytes(b'fixture');report={'inputs':[{'video_sha256':'b'}],'layout_sha256':sha256(layout),'artifact_sha256':{'body_texture_rgba.png':sha256(fusion/'body_texture_rgba.png')},'clips':9,'status':'candidate'};write(fusion/'texture_report.json',report)
            with self.assertRaisesRegex(ValueError,'not part'):publish(out,fusion,layout)
            self.assertFalse((out/'appearance_manifest.json').exists())
            report['inputs'][0]['video_sha256']='a';write(fusion/'texture_report.json',report)
            manifest=publish(out,fusion,layout);self.assertEqual(manifest['files']['body_texture_rgba.png'],sha256(out/'body_texture_rgba.png'));self.assertEqual((out/'racket_poses.json').read_text(),'preserve');self.assertEqual((out/'appearance_previous/viewer.html').read_text(),'old')
            np.array([2,1,0],dtype='<u4').tofile(out/'mesh_faces.bin')
            with self.assertRaisesRegex(ValueError,'topology'):publish(out,fusion,layout)

if __name__=='__main__':unittest.main()
