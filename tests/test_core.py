import unittest
from fractions import Fraction
from ake_core import Observation, Decision, chance, choose_early_restart, decide

def observe(slots):
    return Observation(tuple(slots), tuple(n - slots.count(i) for i,n in enumerate((6,6,6,6,10),1)), 1)

class Tests(unittest.TestCase):
    def test_probabilities(self):
        self.assertEqual(chance(0,0), Fraction(441,23188))
        self.assertEqual(chance(0,1), Fraction(7,1364))
        self.assertEqual(chance(2,2), 0)
    def test_decisions(self):
        self.assertEqual(decide(observe([5,5,5,5,0])), Decision.SUCCESS)
        self.assertEqual(decide(observe([5,1,5,5,5])), Decision.SUCCESS)
        self.assertEqual(decide(observe([4,4,4,4,0])), Decision.RESTART)
        self.assertEqual(decide(observe([1,0,0,0,0])), Decision.DRAW)
        self.assertEqual(decide(observe([1,0,0,0,0]), True), Decision.RESTART)
        self.assertEqual(decide(observe([1,2,0,0,0])), Decision.RESTART)
    def test_fail_closed(self):
        self.assertEqual(decide(Observation((5,5,5,5,0),(6,6,6,6,10),1)), Decision.WAIT)
        self.assertEqual(decide(Observation((5,5,5,5,None),(6,6,6,6,6),1)), Decision.WAIT)
        self.assertEqual(decide(Observation((0,0,0,0,0),(6,6,6,6,10),2)), Decision.WAIT)
    def test_threshold(self):
        self.assertFalse(choose_early_restart([1]*9,[0]*3))
        self.assertTrue(choose_early_restart([1]*10,[2]*3))
        self.assertFalse(choose_early_restart([1]*10,[3]*3))
        with self.assertRaises(ValueError):
            choose_early_restart([float('nan')]*10,[1]*3)
        with self.assertRaises(ValueError):
            choose_early_restart([1]*10,[-1]*3)

if __name__ == '__main__': unittest.main()
