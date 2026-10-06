from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from independent_validation import coach_agreement,DIMENSIONS,POLICY_SHA256,matched_prediction_source
from run_records import sha256


class IndependentValidationTests(unittest.TestCase):
    def packets(self):
        row=dict(id='shot',video_sha256='video',evidence_sha256={'pose':'hash'},video_frames=250,
                 start_frame=10,end_frame=30,stroke='Forehand',goal='consistent recovery',
                 machine={k:'fixed' for k in ['speed','frequency','direction','spin']},
                 ratings={k:3 for k in DIMENSIONS},confirmed=True)
        first=dict(policy_sha256=POLICY_SHA256,reviewer='Coach A',shots=[row])
        second=deepcopy(first);second['reviewer']='Coach B';return first,second

    def test_same_reviewer_cannot_claim_independent_agreement(self):
        a,b=self.packets();b['reviewer']=a['reviewer']
        self.assertEqual(coach_agreement(a,b)['paired_shots'],0)

    def test_missing_scores_are_not_zeroes(self):
        a,b=self.packets();b['shots'][0]['ratings']['contact']=None
        r=coach_agreement(a,b);self.assertEqual(r['paired_shots'],0);self.assertIsNone(r['dimension_mae'])

    def test_changed_frame_or_pose_is_not_paired(self):
        for field,value in [('start_frame',11),('evidence_sha256',{'pose':'changed'}),('goal','different')]:
            a,b=self.packets();b['shots'][0][field]=value
            self.assertEqual(coach_agreement(a,b)['paired_shots'],0)

    def test_per_dimension_errors_are_not_automatic_acceptance(self):
        a,b=self.packets();b['shots'][0]['ratings']['contact']=5
        r=coach_agreement(a,b);self.assertEqual(r['dimension_mae']['contact'],2)
        self.assertEqual(r['dimension_mae']['preparation'],0);self.assertFalse(r['accuracy_accepted'])

    def test_duplicate_and_policy_mismatch_are_rejected(self):
        a,b=self.packets();a['shots'].append(deepcopy(a['shots'][0]))
        with self.assertRaises(ValueError):coach_agreement(a,b)
        a,b=self.packets();b['policy_sha256']='old'
        with self.assertRaises(ValueError):coach_agreement(a,b)

    def test_matching_but_unbound_reviews_are_not_evidence(self):
        a,b=self.packets()
        for d in [a,b]:d['shots'][0]['evidence_sha256']={}
        self.assertEqual(coach_agreement(a,b)['paired_shots'],0)

    def test_derived_viewer_resolves_staged_source_only_with_matching_hashes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'stage';stage.mkdir()
            (stage/'source.mp4').write_bytes(b'normalized')
            (stage/'original.video').write_bytes(b'original')
            pred=dict(video_sha256=sha256(stage/'source.mp4'),
                      inputs_sha256={'video':sha256(stage/'original.video')})
            path=stage/'automatic_distance/predictions.json'
            self.assertEqual(matched_prediction_source(root/'viewer',path,pred),stage/'original.video')
            (stage/'source.mp4').write_bytes(b'changed')
            with self.assertRaises(ValueError):matched_prediction_source(root/'viewer',path,pred)


if __name__=='__main__':unittest.main()
