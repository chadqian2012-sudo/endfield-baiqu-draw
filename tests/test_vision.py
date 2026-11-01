import unittest
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from vision import classify, recognize_card, segment_digits, normalize_digit, recognize_stock, read_image, write_image

class VisionTests(unittest.TestCase):
    def test_chinese_absolute_image_path(self):
        with TemporaryDirectory() as folder:
            file = Path(folder)/'中文模板.png'
            image = np.arange(64,dtype=np.uint8).reshape(8,8)
            write_image(file,image)
            np.testing.assert_array_equal(read_image(file),image)
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

def image_of(text):
    img = np.full((64,120),235,dtype=np.uint8)
    cv2.putText(img,text,(8,49),cv2.FONT_HERSHEY_SIMPLEX,1.4,80,3,cv2.LINE_AA)
    return img

class StockTests(unittest.TestCase):
    def test_single_and_two_digits(self):
        self.assertEqual(len(segment_digits(image_of('6'))),1)
        self.assertEqual(len(segment_digits(image_of('10'))),2)
        self.assertEqual(normalize_digit(segment_digits(image_of('6'))[0]).shape,(48,32))
    def test_reject_missing_training_and_empty(self):
        self.assertIsNone(segment_digits(np.full((64,120),235,dtype=np.uint8)))
        self.assertIsNone(recognize_stock(image_of('6'),{}, {}, .1))
        img = image_of('6')
        img[2:4,100:105] = 0
        self.assertIsNone(segment_digits(img))

    def test_complete_inventory_pipeline(self):
        templates = {}
        for label in '0123456789':
            parts = segment_digits(image_of(label))
            self.assertIsNotNone(parts, label)
            self.assertEqual(len(parts), 1, label)
            templates[label] = [normalize_digit(parts[0])]
        thresholds = {label:.9 for label in templates}
        self.assertEqual(recognize_stock(image_of('10'),templates,thresholds,.05),10)
        self.assertEqual(recognize_stock(image_of('6'),templates,thresholds,.05),6)
        self.assertIsNone(recognize_stock(image_of('11'),templates,thresholds,.05))


if __name__ == '__main__': unittest.main()
