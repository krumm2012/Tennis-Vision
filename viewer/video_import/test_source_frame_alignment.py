import unittest
from fractions import Fraction

from source_frame_alignment import fps_filter_indices


class TimelineTests(unittest.TestCase):
    def test_vfr_drop_and_duplicate_follow_rounded_pts(self):
        # Frames at .080 and .086 both round to output tick 2: the latter wins.
        source = [Fraction(x, 1000) for x in [0, 80, 86, 129, 130, 169]]
        target = [Fraction(i, 25) for i in range(5)]
        self.assertEqual(fps_filter_indices(source, target, 25).tolist(), [0, 0, 2, 4, 5])

    def test_half_tick_rounds_up_not_bankers_rounding(self):
        source = [Fraction(0), Fraction(1, 50), Fraction(3, 50), Fraction(1, 10)]
        target = [Fraction(i, 25) for i in range(4)]
        self.assertEqual(fps_filter_indices(source, target, 25).tolist(), [0, 1, 2, 3])

    def test_wrong_normalization_or_nonmonotonic_source_rejected(self):
        for source, target in [([0, .04], [.01, .05]), ([0, .04], [0, .05]),
                               ([0, .04, .03], [0, .04]), ([0, .04], [0, .04, .08])]:
            with self.assertRaises(ValueError):
                fps_filter_indices(source, target, 25)

    def test_last_frame_duplicated_only_with_known_eof(self):
        source = [Fraction(0), Fraction(1, 10)]
        target = [Fraction(i, 25) for i in range(5)]
        self.assertEqual(fps_filter_indices(source, target, 25, Fraction(1, 5)).tolist(), [0, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            fps_filter_indices(source, target, 25, Fraction(4, 25))
