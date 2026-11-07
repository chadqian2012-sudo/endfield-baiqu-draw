import unittest
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
import json
from io import StringIO
from unittest.mock import Mock, patch
import numpy as np
import collect_observation
from ake import main, validate
from ake_core import Observation
from vision import VisionProfile, classify, recognize_card, segment_digits, normalize_digit, recognize_stock, read_image, write_image

class VisionTests(unittest.TestCase):
    def test_recording_preserves_calibration_geometry_and_masks_account_strip(self):
        binding = Mock(client=(20,30,100,100),hwnd=1)
        source = np.random.default_rng(7).integers(0,256,(100,100,4),dtype=np.uint8)
        frames = []
        writer = Mock()
        writer.isOpened.return_value = True
        writer.write.side_effect = lambda frame:frames.append(frame.copy())
        capture = Mock()
        capture.grab.return_value = source
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('sys.argv',['collect_observation.py','--seconds','1','--fps','5']), \
                 patch('sys.stdout',new_callable=StringIO), \
                 patch('collect_observation.find_unique',return_value=binding), \
                 patch('collect_observation.snapshot',return_value=binding), \
                 patch('collect_observation.foreground_matches',return_value=True), \
                 patch('collect_observation.Path',return_value=root/'captures'), \
                 patch('collect_observation.cv2.VideoWriter',return_value=writer) as create_writer, \
                 patch('collect_observation.mss.mss') as mss, \
                 patch('collect_observation.time.monotonic',side_effect=[0,0,.1,.2,1.1]), \
                 patch('collect_observation.time.sleep'):
                mss.return_value.__enter__.return_value = capture
                collect_observation.main()
            self.assertEqual(create_writer.call_args.args[-1],(100,100))
            capture.grab.assert_called_once_with({'left':20,'top':30,'width':100,'height':100})
            self.assertEqual(len(frames),1)
            self.assertEqual(frames[0].shape,(100,100,3))
            np.testing.assert_array_equal(frames[0][:95],source[:95,:,:3])
            self.assertFalse(np.any(frames[0][95:]))
            writer.release.assert_called_once()
            profile_path = root/'profile.json'
            profile_path.write_text(json.dumps({'frame_size':[100,100],
                'regions':{'slots':[[5,5,20,20]]*5},'groups':{},'anchors':{}}),encoding='utf-8')
            VisionProfile(profile_path).add_sample(frames[0],'cards','5',0,0,.95,.1,'fixture')
            self.assertIn('5',VisionProfile(profile_path).groups['cards'][0])

    def test_recording_rejects_odd_geometry_without_cropping(self):
        for size in ((101,100),(100,101)):
            with self.subTest(size=size), \
                 patch('sys.argv',['collect_observation.py','--seconds','1']), \
                 patch('collect_observation.find_unique',return_value=Mock(client=(0,0,*size))), \
                 patch('collect_observation.cv2.VideoWriter') as create_writer, \
                 patch('collect_observation.mss.mss') as mss:
                with self.assertRaisesRegex(SystemExit,'even client width and height'):
                    collect_observation.main()
                create_writer.assert_not_called()
                mss.assert_not_called()

    def test_uninformative_sample_preserves_existing_profile(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_image(root/'old.png',np.arange(64,dtype=np.uint8).reshape(8,8))
            config = {'frame_size':[8,8],'regions':{'hand':[0,0,8,8],'slots':[[0,0,8,8]]*5},
                      'anchors':{'hand':{'region':[0,0,8,8],'template':'old.png','threshold':.95}},
                      'groups':{'cards':{'templates':{'5':['old.png']},'thresholds':{'5':.95},'margin':.1}}}
            path = root/'profile.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            original = path.read_bytes()
            for group,label in [('hand',None),('cards','5')]:
                with self.subTest(group=group):
                    profile = VisionProfile(path)
                    with self.assertRaisesRegex(ValueError,'visual detail'):
                        profile.add_sample(np.full((8,8),235,dtype=np.uint8),group,label,0,0,.95,.1,'new-batch')
                    self.assertEqual(path.read_bytes(),original)
                    self.assertFalse((root/'logs').exists())
                    self.assertEqual(profile.config,config)

    def test_live_rejects_old_report_before_initializing_windows_input(self):
        profile = Mock()
        profile.issues.return_value = []
        with TemporaryDirectory() as folder:
            report = Path(folder)/'old-report.json'
            report.write_text(json.dumps({'passed':True,'profile_sha256':'fixture-profile'}),encoding='utf-8')
            with patch('vision.VisionProfile',return_value=profile), \
                 patch('ake.fingerprint',return_value='fixture-profile'), \
                 patch('windowing.find_unique') as find_window, \
                 patch('sys.stdout',new_callable=StringIO) as output:
                code = main(['run','--live','--profile','unused.json','--qualification',str(report)])
            self.assertEqual(code,2)
            self.assertEqual(json.loads(output.getvalue())['event'],'live_blocked')
            find_window.assert_not_called()

    def test_qualification_requires_correct_rejection_examples(self):
        # Semantic verifier fixture, not evidence that real screenshot recognition passed.
        def hand(slots, game_round=1):
            inventory = tuple(n-slots.count(v) for v,n in enumerate((6,6,6,6,10),1))
            return Observation(tuple(slots),inventory,game_round,
                               buttons=('disabled' if 0 not in slots else 'enabled',
                                        'disabled' if not any(slots) else 'enabled',None))
        empty = hand([0]*5)
        positive = [empty,hand([1,0,0,0,0]),
                    Observation((None,)*5,(None,)*5,None,page='abandon_dialog',buttons=(None,None,'enabled')),
                    empty,hand([1,2,3,4,5]),hand([1,1,1,1,0]),hand([1,1,1,0,0]),
                    hand([1,1,0,0,0]),hand([5,5,5,0,0]),hand([5,5,0,0,0]),hand([0]*5,2),
                    Observation((None,)*5,(None,)*5,None,page='abandon_dialog',buttons=(None,None,'disabled'))]
        rejected = [Observation((None,)*5,(None,)*5,None,page='unknown'),
                    Observation((None,0,0,0,0),empty.inventory,1,buttons=empty.buttons),
                    Observation(empty.slots,(None,6,6,6,10),1,buttons=empty.buttons)]
        predictions = []
        class Profile:
            CLASSES = VisionProfile.CLASSES
            config = {'training_batches':['calibration']}
            def read(self, image): return predictions[image],{}
            def issues(self): return []
        with TemporaryDirectory() as folder:
            dataset = Path(folder)/'labels.jsonl'
            def check(samples):
                rows = []
                for i,obs in enumerate(samples):
                    expected = {key:value for key,value in asdict(obs).items()
                                if key in ('slots','inventory','game_round','page','buttons')}
                    rows.append(json.dumps({'source':'real','batch':'held-out','image':str(i),'expected':expected}))
                dataset.write_text('\n'.join(rows),encoding='utf-8')
                return validate(Profile(),dataset)
            with patch('vision.read_image',side_effect=lambda path,flags:int(path.name)), \
                 patch('ake.fingerprint',return_value='fixture-profile'):
                for count in range(4):
                    samples = positive+rejected[:count]
                    predictions[:] = samples
                    report = check(samples)
                    with self.subTest(rejection_examples=count):
                        self.assertEqual(report['passed'],count == 3)
                        self.assertEqual(len(report['rejection_cases']),count)
                        self.assertEqual(len(report['errors']),3-count)
                for index,field in enumerate(('page','slots','inventory')):
                    predictions[:] = positive+rejected
                    predictions[len(positive)+index] = empty
                    report = check(positive+rejected)
                    with self.subTest(misrecognized_field=field):
                        self.assertFalse(report['passed'])
                        self.assertTrue(any(field+' mismatch' in error for error in report['errors']))

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
        self.assertIsNone(classify(a, {'a':[a],'b':[a]}, {'a':.9,'b':.9}, 0).label)
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
