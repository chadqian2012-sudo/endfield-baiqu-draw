import unittest
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict, replace
import json
import runpy
from io import StringIO
from unittest.mock import Mock, patch
import numpy as np
import collect_observation
from ake import main, validate
from ake_core import Observation
from vision import VisionProfile, classify, recognize_card, segment_digits, normalize_digit, recognize_stock, read_image, write_image
from windowing import WindowIdentity

class VisionTests(unittest.TestCase):
    def test_profile_rejects_inconsistent_card_geometry_before_sampling(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_image(root/'good.png',np.arange(64,dtype=np.uint8).reshape(8,8))
            write_image(root/'narrow.png',np.arange(56,dtype=np.uint8).reshape(8,7))
            config = {'frame_size':[8,8],'regions':{'slots':[[0,0,8,8] for _ in range(5)]},
                      'groups':{'cards':{'templates':{'5':['good.png']},'thresholds':{'5':.95},'margin':.1}},'anchors':{}}
            path = root/'profile.json'
            for case in ('slot','template'):
                invalid = json.loads(json.dumps(config))
                if case == 'slot': invalid['regions']['slots'][1][2] = 7
                else: invalid['groups']['cards']['templates']['5'].append('narrow.png')
                path.write_text(json.dumps(invalid),encoding='utf-8')
                original = path.read_bytes()
                with self.subTest(case=case), self.assertRaisesRegex(ValueError,'same width and height|template dimensions'):
                    VisionProfile(path)
                self.assertEqual(path.read_bytes(),original)
                self.assertFalse((root/'logs').exists())
            with self.assertRaisesRegex(ValueError,'same width and height'):
                VisionProfile.check_regions((8,8),{'slots':[[0,0,8,8],[0,0,7,8]]+[[0,0,8,8]]*3})

    def test_equal_slot_shapes_share_new_samples_without_poisoning_other_slots(self):
        rng = np.random.default_rng(23)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            images = {label:rng.integers(0,256,(8,8),dtype=np.uint8) for label in ('empty','1','2','3','4','5')}
            for label,image in images.items(): write_image(root/(label+'.png'),image)
            slots = [[0,0,8,8]]+[[8,0,8,8]]*4
            config = {'frame_size':[16,8],'regions':{'slots':slots},'anchors':{},'groups':{'cards':{
                'templates':{label:[label+'.png'] for label in images},'thresholds':{label:.9 for label in images},'margin':.1}}}
            path = root/'profile.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            frame = np.hstack([images['1'],images['5']])
            profile = VisionProfile(path)
            self.assertEqual(profile.match_group(frame,'cards',slots[0]).label,'1')
            profile.add_sample(frame,'cards','5',1,0,.9,.1,'second-slot')
            profile = VisionProfile(path)
            self.assertEqual(profile.match_group(frame,'cards',slots[0]).label,'1')
            self.assertEqual(profile.match_group(frame,'cards',slots[1]).label,'5')

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
            for source in (None,'source.jsonl'):
                cached = {'passed':True,'profile_sha256':'fixture-profile',
                          'rejection_cases':['unknown_page','unknown_slot','unknown_inventory']}
                if source: cached['dataset'] = source
                report.write_text(json.dumps(cached),encoding='utf-8')
                with self.subTest(source=source), \
                     patch('vision.VisionProfile',return_value=profile), \
                     patch('ake.validate',return_value={'errors':['source recheck failed']}) as recheck, \
                     patch('windowing.find_unique') as find_window, \
                     patch('sys.stdout',new_callable=StringIO) as output:
                    code = main(['run','--live','--profile','unused.json','--qualification',str(report)])
                self.assertEqual(code,2)
                self.assertEqual(json.loads(output.getvalue())['event'],'live_blocked')
                find_window.assert_not_called()
                if source: recheck.assert_called_once_with(profile,report.parent/source)
                else: recheck.assert_not_called()

    def test_live_preserves_success_when_evidence_or_notification_fails(self):
        binding = WindowIdentity(1,2,3,'fixture.exe',(0,0,8,8),96)
        success = Observation((5,5,5,5,0),(6,6,6,6,6),1)
        for failure in ('evidence','notification'):
            with self.subTest(failure=failure):
                profile = Mock(config={'window':json.loads(json.dumps(asdict(binding)))})
                profile.issues.return_value = []
                profile.read.return_value = success,{}
                now,events,images = [0.0],[],[object() for _ in range(3)]
                def frames():
                    for i,image in enumerate(images):
                        now[0] = i*.125
                        yield i,now[0],image,True
                def save(*_):
                    now[0] += .25
                    if failure == 'evidence': raise OSError('disk full')
                    return 'success-evidence'
                def notify():
                    now[0] += .375
                    if failure == 'notification': raise RuntimeError('sound failed')
                notifier = Mock(MessageBeep=Mock(side_effect=notify))
                with patch('vision.VisionProfile',return_value=profile), \
                     patch('pathlib.Path.read_text',return_value='{"dataset":"fixture.jsonl"}'), \
                     patch('pathlib.Path.mkdir'), \
                     patch('ake.validate',return_value={'errors':[]}), \
                     patch('ake.StrictLog') as log, \
                     patch('ake.frames_from_window',return_value=frames()), \
                     patch('windowing.find_unique',return_value=binding), \
                     patch('windowing.Hotkeys') as hotkeys, \
                     patch('windowing.click') as click, \
                     patch('ake.evidence',side_effect=save) as evidence, \
                     patch('ake.time.monotonic',side_effect=lambda:now[0]), \
                     patch('ake.dump',side_effect=events.append), \
                     patch.dict('sys.modules',{'winsound':notifier}):
                    hotkeys.return_value.poll.return_value = []
                    code = main(['run','--live','--profile','unused','--qualification','unused'])
                self.assertEqual(code,1)
                self.assertEqual(sum(e.get('event') == 'success' for e in events),1)
                summary = next(e for e in events if e.get('event') == 'summary')
                self.assertEqual(summary['termination_reason'],'success')
                if failure == 'evidence': self.assertIn('disk full',summary['evidence_error'])
                evidence.assert_called_once_with(profile,images[-1],'success')
                notifier.MessageBeep.assert_called_once_with()
                notification = next(e for e in events if e.get('event') in ('notification','notification_error'))
                self.assertEqual(notification['event'],'notification_error' if failure == 'notification' else 'notification')
                self.assertEqual(notification['elapsed'],.375)
                click.assert_not_called()
                hotkeys.return_value.close.assert_called_once_with()
                log.return_value.close.assert_called_once_with()

    def test_cli_preparation_failures_use_structured_exit_and_preserve_argparse(self):
        sample = ['sample','--profile','unused','--image','unused','--group','cards',
                  '--label','5','--score','.95','--margin','.1','--batch','fixture']
        for args,target,error,event,code in [
                (['calibrate'],'windowing.find_unique',RuntimeError('missing window'),'error',1),
                (sample,'vision.read_image',cv2.error('invalid image'),'error',1),
                (['calibrate'],'windowing.find_unique',KeyboardInterrupt(),'stopped',0)]:
            with self.subTest(error=type(error).__name__), \
                 patch('sys.argv',['ake.py',*args]), \
                 patch('vision.VisionProfile',return_value=Mock()), \
                 patch(target,side_effect=error), \
                 patch('sys.stdout',new_callable=StringIO) as output, \
                 patch('sys.stderr',new_callable=StringIO) as stderr:
                with self.assertRaises(BaseException) as stopped:
                    runpy.run_path(str(Path(__file__).resolve().parents[1]/'ake.py'),run_name='__main__')
                self.assertIsInstance(stopped.exception,SystemExit)
                self.assertEqual(stopped.exception.code,code)
                self.assertEqual(json.loads(output.getvalue())['event'],event)
                self.assertEqual(stderr.getvalue(),'')
        for args,code in [(['--version'],0),(['--invalid-option'],2)]:
            with self.subTest(args=args), patch('sys.argv',['ake.py',*args]), \
                 patch('sys.stdout',new_callable=StringIO), patch('sys.stderr',new_callable=StringIO):
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(Path(__file__).resolve().parents[1]/'ake.py'),run_name='__main__')
                self.assertEqual(stopped.exception.code,code)

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
        recording_images = [np.full((8,8),0 if i == 3 else i,dtype=np.uint8) for i in range(5)]
        closed_recordings = []
        consumed_frames = []
        tail_error = None
        def recording_frames(recording):
            try:
                for i,image in enumerate(recording_images):
                    consumed_frames.append(i)
                    yield i,i*.125,image,True
                if tail_error is not None: raise tail_error
            finally:
                closed_recordings.append(recording)
        class Profile:
            CLASSES = VisionProfile.CLASSES
            config = {'training_batches':['calibration']}
            def read(self, image):
                return predictions[int(image[0,0]) if isinstance(image,np.ndarray) else image],{}
            def issues(self): return []
        with TemporaryDirectory() as folder:
            dataset = Path(folder)/'labels.jsonl'
            def check(samples, recording='evidence'):
                rows = []
                for i,obs in enumerate(samples):
                    expected = {key:value for key,value in asdict(obs).items()
                                if key in ('slots','inventory','game_round','page','buttons')}
                    source = {'recording':recording,'frame':i} if recording and i < 4 else {'image':str(i)}
                    rows.append(json.dumps({'source':'real','batch':'held-out',**source,'expected':expected}))
                dataset.write_text('\n'.join(rows),encoding='utf-8')
                return validate(Profile(),dataset)
            with patch('vision.read_image',side_effect=lambda path,flags:int(path.name)), \
                 patch('ake.frames_from_recording',side_effect=recording_frames) as reader, \
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
                samples = positive+rejected
                predictions[:] = samples
                check(samples)
                duplicate = json.loads(dataset.read_text(encoding='utf-8').splitlines()[4])
                duplicate['image'] = './4'
                with dataset.open('a',encoding='utf-8') as file:
                    file.write('\n'+json.dumps(duplicate))
                report = validate(Profile(),dataset)
                self.assertFalse(report['passed'])
                self.assertEqual(report['frames'],len(samples))
                self.assertTrue(any('repeated validation image' in error for error in report['errors']))
                for row,button,missing_flow in ((0,0,'single-draw'),(1,1,'early-abandon'),(2,2,'early-abandon')):
                    samples = positive+rejected
                    buttons = list(samples[row].buttons)
                    buttons[button] = 'disabled'
                    samples[row] = replace(samples[row],buttons=tuple(buttons))
                    predictions[:] = samples
                    report = check(samples)
                    with self.subTest(unavailable_flow_button=(row,button)):
                        self.assertFalse(report['passed'])
                        self.assertTrue(any(missing_flow in error for error in report['errors']))
                predictions[:] = positive+rejected
                report = check(predictions,recording=None)
                self.assertFalse(report['passed'])
                self.assertTrue(any('single-draw' in error for error in report['errors']))
                self.assertTrue(any('early-abandon' in error for error in report['errors']))
                for case in ('cross-recording','duplicate-frame','reversed-frames','missing-frame'):
                    check(predictions)
                    rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
                    if case == 'cross-recording': rows[1]['recording'] = 'other-evidence'
                    elif case == 'duplicate-frame': rows[3]['frame'] = 0
                    elif case == 'reversed-frames': rows[0]['frame'],rows[3]['frame'] = 3,0
                    else: rows[3]['frame'] = 99
                    dataset.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
                    closed_recordings.clear()
                    with self.subTest(recording_evidence=case):
                        report = validate(Profile(),dataset)
                        self.assertFalse(report['passed'])
                        self.assertIn(dataset.parent/'evidence',closed_recordings)
                check(predictions)
                rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
                rows[2]['frame'],rows[3]['frame'] = 3,4
                dataset.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
                original_images = recording_images[:]
                recording_images[:] = [original_images[i] for i in (0,1,4,2,3)]
                report = validate(Profile(),dataset)
                recording_images[:] = original_images
                self.assertFalse(report['passed'])
                self.assertTrue(any('early-abandon' in error for error in report['errors']))
                self.assertFalse(any('mismatch' in error for error in report['errors']))
                recording_images[:] = [original_images[0],original_images[1],
                                       np.full((8,8),len(positive),dtype=np.uint8),
                                       original_images[2],original_images[3]]
                report = validate(Profile(),dataset)
                recording_images[:] = original_images
                self.assertTrue(report['passed'])
                # Independent recording frames may have identical pixels after a full reset.
                self.assertTrue(np.array_equal(recording_images[0],recording_images[3]))
                reader.reset_mock()
                closed_recordings.clear()
                consumed_frames.clear()
                self.assertTrue(check(predictions)['passed'])
                reader.assert_called_once_with(dataset.parent/'evidence')
                self.assertEqual(consumed_frames,[0,1,2,3,4])
                self.assertEqual(closed_recordings,[dataset.parent/'evidence'])
                tail_error = ValueError('invalid recording tail')
                reader.reset_mock()
                closed_recordings.clear()
                with self.assertRaisesRegex(ValueError,'invalid recording tail'):
                    check(predictions)
                reader.assert_called_once_with(dataset.parent/'evidence')
                self.assertEqual(closed_recordings,[dataset.parent/'evidence'])

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
