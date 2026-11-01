import unittest
import numpy as np
from vision import classify, recognize_card

class VisionTests(unittest.TestCase):
    def test_synthetic_match_and_rejections(self):
        rng = np.random.default_rng(19)
        a = rng.integers(0,256,(32,32),dtype=np.uint8)
        b = rng.integers(0,256,(32,32),dtype=np.uint8)
        templates = {'a':[a], 'b':[b]}
        self.assertEqual(classify(a, templates, {'a':.99,'b':.99}, .1).label,'a')
        self.assertIsNone(classify(a, {'a':[a]}, {'a':.9}, .1).label)
        self.assertIsNone(classify(a, {'a':[a],'b':[a]}, {'a':.9,'b':.9}, .1).label)
        self.assertIsNone(classify(a, templates, {}, .1).label)
        self.assertIsNone(classify(np.zeros((32,32),dtype=np.uint8), templates, {'a':.9,'b':.9}, .1).label)
        self.assertIsNone(recognize_card(a, {'1':[a],'5':[b]}, {'1':.9,'5':.9}, .1).label)

if __name__ == '__main__': unittest.main()
