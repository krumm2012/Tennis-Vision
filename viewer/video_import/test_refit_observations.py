import json,tempfile,unittest
from pathlib import Path
from refit_observations import automatic_rows,assemble
from run_records import write,sha256

class ObservationTests(unittest.TestCase):
    def fixture(self,root):
        meta={'video_sha256':'own','fps':25.,'image_size':[1280,720],'frames':3};write(root/'mesh_meta.json',meta)
        point={'points':{'handle_end':[100,100],'tip':[200,100],'head_center':[175,100]},'confidence':.8,'uncertainty_px':{'handle_end':3,'tip':12}}
        write(root/'racket_keypoints.json',{**meta,'frames':[{'real':point,'mirror':point} for _ in range(3)]})
        write(root/'racket_review_frames.json',{'keypoints_sha256':sha256(root/'racket_keypoints.json')})
        write(root/'racket_landmarks.json',{**meta,'frames':[{'frame':0,'points':{'tip':[210,110]},'mirror_points':{}},{'frame':1,'points':{'handle_end':[100,100],'tip':[200,100]},'mirror_points':{}}]})
        return {'video_sha256':'own','heldout_frames':[1],'automatic_evidence_permitted':True,'validation_source':'reviewed_landmarks'}

    def test_manual_missing_points_and_views_never_inherit_auto_confidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ready=self.fixture(root);rows={r['frame']:r for r in assemble(root,ready)['frames']}
            self.assertEqual(rows[0]['points'],{'tip':[210.,110.]});self.assertEqual(rows[0]['mirror_points'],{})
            self.assertNotIn('head_center',rows[1]['points']);self.assertEqual(rows[0]['weights']['points'],{'tip':1.})
            self.assertEqual(rows[2]['source'],'automatic_contour_unverified')

    def test_automatic_frame_holdout_uncertainty_and_hash_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ready=self.fixture(root);rows=automatic_rows(root,'own',[1]);self.assertNotIn(1,rows)
            self.assertLess(rows[0]['weights']['points']['tip'],rows[0]['weights']['points']['handle_end'])
            with (root/'racket_keypoints.json').open('a') as f:f.write(' ')
            with self.assertRaises(ValueError):automatic_rows(root,'own')

if __name__=='__main__':unittest.main()
