import copy,unittest
from prepare_racket_boundary_experiment import isolate

class FrozenAblationTests(unittest.TestCase):
    def setUp(self):
        self.kp={'frames':[{'frame':i,'real':{'points':{'tip':[i,0]}},'mirror':{'points':{'tip':[i,1]}},'stereo_shaft':{}} for i in range(12)]}
    def test_inputs_are_immutable_and_only_mirror_removed_in_region(self):
        before=copy.deepcopy(self.kp);out,events=isolate(self.kp,{'frames':[]},[1,2],[7,8])
        self.assertEqual(self.kp,before)
        for i in range(12):
            if i in [1,2]:self.assertNotIn('real',out['frames'][i]);self.assertNotIn('mirror',out['frames'][i])
            elif i in [7,8]:self.assertEqual(out['frames'][i]['real'],before['frames'][i]['real']);self.assertNotIn('mirror',out['frames'][i])
            else:self.assertEqual(out['frames'][i],before['frames'][i])
    def test_never_ablate_reviewed_or_reserved(self):
        with self.assertRaises(ValueError):isolate(self.kp,{'frames':[{'frame':7}]},[1],[7])
        with self.assertRaises(ValueError):isolate(self.kp,{'frames':[]},[1],[1])
    def test_prior_reviewed_frame_cannot_be_fresh(self):
        with self.assertRaises(ValueError):isolate(self.kp,{'frames':[{'frame':1}]},[1])
