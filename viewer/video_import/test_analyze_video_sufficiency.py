import unittest
import numpy as np
from analyze_video_sufficiency import subset_support,smallest_subsets

class VideoSufficiencyTests(unittest.TestCase):
    def test_exhaustive_unions_preserve_overlap_and_uv_weights(self):
        support=np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1]],bool)
        face,uv,union=subset_support(support,np.array([1,2,7,10],np.int64))
        self.assertEqual(len(face),8)
        np.testing.assert_array_equal(face,[0,2,2,3,2,4,3,4])
        np.testing.assert_array_equal(uv,[0,3,9,10,17,20,19,20])
        self.assertEqual(union[0].sum(),0)
        self.assertEqual(union[-1].sum(),4)
    def test_recommended_subset_requires_both_face_and_uv_support(self):
        rows=[{'videos':1,'supported_faces':100,'supported_uv_proxy_texels':50,'fraction_of_9clip_face_support':1.,'fraction_of_9clip_uv_proxy_support':.5},
              {'videos':2,'supported_faces':99,'supported_uv_proxy_texels':99,'fraction_of_9clip_face_support':.99,'fraction_of_9clip_uv_proxy_support':.99}]
        result=smallest_subsets(rows,.99,100,100)
        self.assertEqual(result['minimum_k'],2)
        self.assertEqual(result['recommended']['videos'],2)

if __name__=='__main__':unittest.main()
