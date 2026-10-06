import unittest
from evaluate_reserved_racket import reservation_gate,stats
class ReservedLabelTests(unittest.TestCase):
    def setUp(self):
        self.labels={'frames':[{'frame':30}],'video_sha256':'x'};self.protocol={'reserved_frames':[30,80],'video_sha256':'x'};self.report={'train_frames':[1,2],'video_sha256':'x'}
    def test_reject_training_overlap(self):
        self.report['train_frames'].append(30)
        with self.assertRaises(ValueError):reservation_gate(self.labels,self.protocol,self.report)
    def test_reject_unreserved_and_wrong_video(self):
        self.labels['frames'][0]['frame']=31
        with self.assertRaises(ValueError):reservation_gate(self.labels,self.protocol,self.report)
        self.labels['frames'][0]['frame']=30;self.labels['video_sha256']='y'
        with self.assertRaises(ValueError):reservation_gate(self.labels,self.protocol,self.report)
    def test_valid_and_absent_metrics(self):
        reservation_gate(self.labels,self.protocol,self.report)
        self.assertEqual(stats([])['samples'],0);self.assertIsNone(stats([])['median'])
