import unittest
import numpy as np
from multivideo_texture import uv_lookup,depth_map,valid_points,coherent_labels,views,fallback_candidates,sample_surface_texels,fill_pixel_fallback
from texture_consistency import consistency_scores

class TextureTests(unittest.TestCase):
    def test_pixel_fallback_real_sampling_preserves_primary_and_excludes_heldout(self):
        faces=np.array([[0,1,2]],np.int32)
        camera=np.array([[-1.5,-1.5,2],[1.5,-1.5,2],[-1.5,1.5,2]],np.float32)
        xy=np.array([[20,20],[28,20],[20,28],[28,28]],np.float32)
        weights=np.c_[1-(xy[:,0]-8)/48-(xy[:,1]-8)/48,(xy[:,0]-8)/48,(xy[:,1]-8)/48]
        masks=np.zeros((4,64,64),np.uint8)
        # Primary supports only texel 2. Alternative 1 supports texels 0 and 2.
        # Alternative 2 is geometrically occluded despite a full foreground mask.
        for label,points in [(0,[2]),(1,[0,2])]:
            for point in points:
                x,y=xy[point].astype(int);masks[label,y-4:y+5,x-4:x+5]=255
        masks[2:]=255
        images=[np.full((64,64,3),color,np.uint8) for color in [(10,20,30),(80,90,100),(120,130,140),(220,230,240)]]
        source_clip=np.full((1,4),-1,np.int16);source_frame=source_clip.copy();source_view=np.zeros((1,4),np.uint8)
        rgba=np.zeros((1,4,4),np.uint8);ys=np.zeros(4,int);xs=np.arange(4);face_ids=np.zeros(4,int)
        clips=np.array([0,1,2,3]);frames=np.array([5,10,15,25]);view_ids=np.array([0,1,0,1]);called=[]
        def sampler(label,locations):
            called.append(label)
            other=camera*.5 if label==2 else None
            return sample_surface_texels(camera,faces,face_ids[locations],weights[locations],masks[label],32,
                [64,64],images[label],other,np.ones(3))
        valid,rgb,alpha=sampler(0,xs);rgba[ys[valid],xs[valid],:3]=rgb[valid];rgba[ys[valid],xs[valid],3]=alpha[valid]
        source_clip[ys[valid],xs[valid]]=0;source_frame[ys[valid],xs[valid]]=5
        primary=rgba.copy();called.clear()
        candidates=fallback_candidates(np.array([[10],[9],[8],[100]],np.float32),np.array([0]),frames,5,3)
        np.testing.assert_array_equal(candidates[:,0],[1,2,-1])
        ranks=np.where(rgba[:,:,3]>0,0,-1).astype(np.int8)
        attempts=fill_pixel_fallback(candidates,face_ids,ys,xs,rgba,source_clip,source_frame,source_view,clips,frames,view_ids,sampler,ranks)
        np.testing.assert_array_equal(ranks,[[1,-1,0,-1]])
        np.testing.assert_array_equal(rgba[0,0],[80,90,100,255])
        np.testing.assert_array_equal(rgba[0,2],primary[0,2])
        np.testing.assert_array_equal(source_clip,[[1,-1,0,-1]])
        np.testing.assert_array_equal(source_frame,[[10,-1,5,-1]])
        np.testing.assert_array_equal(source_view,[[1,0,0,0]])
        self.assertEqual(attempts,5);self.assertNotIn(3,called)
        self.assertEqual(rgba[0,1,3],0);self.assertEqual(rgba[0,3,3],0)

    def test_occluded_primary_can_use_secondary_real_pixels(self):
        camera=np.array([[-1.5,-1.5,2],[1.5,-1.5,2],[-1.5,1.5,2]],np.float32)
        faces=np.array([[0,1,2]],np.int32);weights=np.array([[.5,.25,.25]],np.float32)
        mask=np.full((64,64),255,np.uint8);image=np.full((64,64,3),[45,65,85],np.uint8)
        args=(camera,faces,np.array([0]),weights,mask,32,[64,64],image)
        invalid,_,_=sample_surface_texels(*args,camera*.5,np.ones(3));self.assertFalse(invalid.any())
        rgba=np.zeros((1,1,4),np.uint8);clip=np.full((1,1),-1,np.int16);frame=clip.copy();view=np.zeros((1,1),np.uint8)
        candidates=fallback_candidates(np.array([[10],[9]],np.float32),np.array([0]),np.array([5,10]),5,1)
        fill_pixel_fallback(candidates,np.array([0]),np.array([0]),np.array([0]),rgba,clip,frame,view,
            np.array([0,1]),np.array([5,10]),np.array([0,0]),lambda label,locations:sample_surface_texels(*args,None,np.ones(3)))
        np.testing.assert_array_equal(rgba[0,0],[45,65,85,255]);self.assertEqual(clip[0,0],1)

    def test_fallback_ranking_is_bounded_stable_and_requires_face_evidence(self):
        scores=np.array([[5,0],[4,9],[4,8],[3,7]],np.float32)
        candidates=fallback_candidates(scores,np.array([0,0]),np.array([5,10,15,20]),5,2,face_chunk=1)
        np.testing.assert_array_equal(candidates,[[1,-1],[2,-1]])
        for limit in [0,9]:
            with self.assertRaises(ValueError):fallback_candidates(scores,np.array([0,0]),np.array([5,10,15,20]),5,limit)

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
