import unittest
from timing import Timings

class TimingTests(unittest.TestCase):
    def test_change_only_at_attempt_boundary(self):
        t = Timings()
        self.assertFalse(t.begin_attempt())
        for _ in range(10): t.add('draw', 1, uninterrupted=True)
        for _ in range(3): t.add('restart', 1, uninterrupted=True)
        t.add('draw', 999, uninterrupted=False)
        self.assertFalse(t.early_restart)
        self.assertTrue(t.begin_attempt())
        self.assertEqual(t.summary()['draw']['count'], 10)
        self.assertEqual(t.summary()['draw']['stddev'], 0)
        with self.assertRaises(ValueError): t.add('draw', float('nan'), uninterrupted=True)

if __name__ == '__main__': unittest.main()
