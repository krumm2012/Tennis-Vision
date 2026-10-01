import unittest
import numpy as np
from multivideo_texture import uv_lookup,depth_map,valid_points,coherent_labels,views
from texture_consistency import consistency_scores

class TextureTests(unittest.TestCase):
    def test_training_color_consistency_keeps_support_and_penalizes_outlier(self):
        scores=np.array([[1.,0],[1.,0],[1.1,0]],np.float32)
        colors=np.array([[[90,90,90],[0,0,0]],[[92,92,92],[0,0,0]],[[240,240,240],[0,0,0]]],np.float32)
        adjusted,consensus=consistency_scores(scores,colors,np.array([0,1,2]))
        np.testing.assert_array_equal(adjusted>0,scores>0)
        self.assertLess(adjusted[2,0],adjusted[0,0])
        np.testing.assert_allclose(consensus[0],92)
    def test_color_consensus_balances_clips_and_never_reads_heldout(self):
        colors=np.tile(np.array([[[10,10,10]]],np.float32),(12,1,1))
        colors[-2]=100;colors[-1]=110
        _,consensus=consistency_scores(np.ones((12,1)),colors,np.array([0]*10+[1,2]))
        np.testing.assert_allclose(consensus,100)

    def test_independent_mirror_uses_restored_axes_and_skips_missing_inference(self):
        data={'vertices':np.zeros((2,3,3)),'source_roots':np.tile([1,0,3],(2,1)),'masks':np.ones((2,4,4)),
              'mirror_vertices':np.zeros((2,3,3)),'mirror_roots':np.tile([-2,1,5],(2,1)),
              'mirror_valid':np.array([True,False]),'mirror_focal':np.array([100,0]),'masks_mirror_sam2':np.ones((2,4,4))}
        meta={'mirror_available':True,'focal':[100,100]};plane={'normal_camera':[0,0,1],'distance_camera_m':4}
        observed=list(views(data,meta,plane,0,'independent'))
        self.assertEqual(len(observed),2)
        np.testing.assert_allclose(observed[1][1],np.tile([-2,1,5],(3,1)))
        self.assertEqual(len(list(views(data,meta,plane,1,'independent'))),1)
        data['mirror_focal'][0]=99
        with self.assertRaises(ValueError):list(views(data,meta,plane,0,'independent'))
    def test_view_coherence_never_fills_unobserved_surfaces(self):
        faces=np.array([[0,1,2],[2,1,3],[3,1,4],[5,6,7]],np.int32)
        scores=np.array([[1,.9,1,0],[.95,1,.95,0]],np.float32)
        labels=coherent_labels(scores,faces)
        self.assertEqual(len(set(labels[:3])),1)
        self.assertEqual(scores[labels[3],3],0)
    def test_uv_lookup_preserves_barycentric_surface_correspondence(self):
        uv=np.array([[0,0],[1,0],[0,1]],np.float32);faces=np.array([[0,1,2]],np.int32)
        ids,b=uv_lookup(uv,faces,32);y,x=np.where(ids==0)
        self.assertGreater(len(y),400)
        np.testing.assert_allclose(b[y,x].sum(1),1,atol=1e-6)
        reconstructed=b[y,x]@uv
        np.testing.assert_allclose(reconstructed[:,0],(x+.5)/31,atol=1e-6)
        np.testing.assert_allclose(reconstructed[:,1],1-(y+.5)/31,atol=1e-6)
    def test_occluded_and_background_points_do_not_become_texture(self):
        uv=np.array([[8,8],[56,8],[8,56],[8,8],[56,8],[8,56]],np.float32)
        depth=depth_map(uv,np.array([2,2,2,3,3,3],np.float32),np.array([[3,4,5],[0,1,2]],np.int32),64,64)
        mask=np.full((64,64),255,np.uint8);points=np.array([[-.5,-.5,2],[-.75,-.75,3],[.5,.5,2]],float)
        valid,_,_=valid_points(points,mask,depth,32,np.array([64,64]),np.array([64,64]))
        self.assertTrue(valid[0]);self.assertFalse(valid[1]);self.assertFalse(valid[2])
        mask[:]=0;valid,_,_=valid_points(points,mask,depth,32,np.array([64,64]),np.array([64,64]));self.assertFalse(valid.any())

if __name__=='__main__':unittest.main()
