import unittest
from fractions import Fraction
from math_verify import attempt_metrics, enumerate_policies, threshold, totals
from ake_core import THRESHOLD

class ExactVerification(unittest.TestCase):
    def test_fixed_policy_probabilities(self):
        self.assertEqual(attempt_metrics(False)[0], Fraction(441,23188))
        self.assertEqual(attempt_metrics(True)[0], Fraction(21,1364))

    def test_totals_and_threshold(self):
        _, _, d, r = totals(False)
        self.assertAlmostEqual(float(d), 145.68253968)
        self.assertAlmostEqual(float(r), 51.58049887)
        self.assertEqual(threshold(), THRESHOLD)

    def test_policy_enumeration(self):
        for ratio, early in [(Fraction(1),True), (Fraction(2),True), (Fraction(3),False)]:
            _, _, d, r = totals(early)
            self.assertEqual(enumerate_policies(Fraction(1), ratio)[0][0], d + ratio*r)
        _, _, d1, r1 = totals(True)
        _, _, d2, r2 = totals(False)
        self.assertEqual(d1+THRESHOLD*r1, d2+THRESHOLD*r2)
        self.assertEqual(enumerate_policies(Fraction(1),THRESHOLD)[0][0], d1+THRESHOLD*r1)

if __name__ == '__main__': unittest.main()
