import unittest
import numpy as np
from refine_semantic_boundaries import boundary_guard


class BoundaryTests(unittest.TestCase):
    def scene(self):
        image=np.full((60,60,3),[65,95,130],np.uint8)
        labels=np.full((60,60),2,np.uint8); labels[45:]=3
        image[45:]=200
        confidence=np.full((60,60),255,np.uint8)
        return image,labels,confidence,np.ones((60,60),bool)

    def test_thin_white_strap_is_rejected_not_relabelled_skin(self):
        image,labels,conf,roi=self.scene(); image[15:45,29:32]=210
        out,certainty,flags=boundary_guard(image,labels,conf,roi)
        self.assertEqual(out[30,30],0);self.assertEqual(certainty[30,30],0)
        self.assertTrue(flags['white_ridge'][30,30]);self.assertEqual(out[30,10],2)
        self.assertEqual(out[50,30],3)

    def test_hair_band_keeps_hair_and_never_invents_material(self):
        image,labels,conf,roi=self.scene();labels[20:30,20:30]=1
        out,_,flags=boundary_guard(image,labels,conf,roi)
        self.assertEqual(out[25,25],1);self.assertEqual(out[25,30],0)
        self.assertTrue(np.all((out==labels)|(out==0)))

    def test_outside_roi_and_inputs_are_unchanged(self):
        image,labels,conf,roi=self.scene();image[:,29:32]=210;roi[:30]=False
        before=labels.copy(); c=conf.copy()
        out,certainty,_=boundary_guard(image,labels,conf,roi)
        np.testing.assert_array_equal(out[~roi],labels[~roi])
        np.testing.assert_array_equal(certainty[~roi],conf[~roi])
        np.testing.assert_array_equal(labels,before);np.testing.assert_array_equal(conf,c)

    def test_no_upper_clothing_does_not_guess_straps(self):
        image,labels,conf,roi=self.scene();labels[:]=2;image[:,29:32]=210
        out,_,flags=boundary_guard(image,labels,conf,roi)
        self.assertFalse(flags['white_ridge'].any());np.testing.assert_array_equal(out,labels)

    def test_coordinate_mismatch_is_rejected(self):
        image,labels,conf,roi=self.scene()
        with self.assertRaises(ValueError):boundary_guard(image[:30],labels,conf,roi)


if __name__ == '__main__':unittest.main()
