from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from capture import WindowCapture, _Display, _check_blt, _direct_dxgi


class CaptureTests(TestCase):
    def test_missing_frames_and_lost_focus_never_trigger_fallback(self):
        binding = SimpleNamespace(hwnd=1, client=(0,0,2,2))
        pixels = np.ones((2,2,3), dtype=np.uint8)
        for name,focus,frame,expected in [
            ('already blurred',[False],None,False),
            ('blur during open',[True,False],None,False),
            ('blur during capture',[True,True,False],pixels,False),
            ('error while blurred',[True,True,False],OSError('capture failed'),False),
            ('new frame unavailable',[True,True,True],None,True),
            ('black frame',[True,True,True],pixels*0,True),
        ]:
            with self.subTest(name=name):
                acquire = Mock(side_effect=frame if isinstance(frame,Exception) else None,
                               return_value=frame)
                @contextmanager
                def reader(backend, client):
                    self.assertEqual(backend,'mss')
                    yield acquire
                with patch('capture._reader',side_effect=reader) as open_reader, \
                     patch('capture.foreground_matches',side_effect=focus), WindowCapture(binding) as capture:
                    self.assertEqual(capture.grab(),(None,expected))
                self.assertEqual(open_reader.call_count,int(len(focus)>1))
                self.assertEqual(acquire.call_count,int(len(focus)>2))

    def test_backend_failure_switches_once_and_releases_both_readers(self):
        binding = SimpleNamespace(hwnd=1, client=(0,0,2,2))
        pixels = np.ones((2,2,3), dtype=np.uint8)
        closed, selected = [],[]
        @contextmanager
        def reader(backend, client):
            try:
                if backend == 'mss':
                    yield lambda discard: _check_blt(False,None,None)
                else:
                    yield lambda discard: pixels
            finally:
                closed.append(backend)
        with patch('capture._reader',side_effect=reader), patch('capture.foreground_matches',return_value=True):
            with WindowCapture(binding,on_backend=lambda name,why:selected.append((name,why))) as capture:
                for _ in range(2):
                    frame, focused = capture.grab()
                    self.assertTrue(focused)
                    np.testing.assert_array_equal(frame,pixels)
            self.assertEqual(closed,['mss','dxgi'])
            self.assertEqual([name for name,_ in selected],['mss','dxgi'])
            self.assertIn('BitBlt',selected[-1][1])
            with WindowCapture(binding,'mss') as capture, self.assertRaisesRegex(RuntimeError,'BitBlt'):
                capture.grab()
            self.assertEqual(closed,['mss','dxgi','mss'])

    def test_identity_change_and_slow_capture_discard_pixels(self):
        binding = SimpleNamespace(hwnd=1, client=(0,0,2,2))
        pixels = np.ones((2,2,3), dtype=np.uint8)
        @contextmanager
        def reader(backend, client): yield lambda discard:pixels
        with patch('capture._reader',side_effect=reader):
            with patch('capture.foreground_matches',side_effect=[True,True,RuntimeError('geometry')]), WindowCapture(binding) as capture:
                with self.assertRaisesRegex(RuntimeError,'geometry'): capture.grab()
            with patch('capture.foreground_matches',return_value=True), patch('capture.monotonic',side_effect=[1.,1.6]), WindowCapture(binding) as capture:
                self.assertEqual(capture.grab(),(None,True))

    def test_same_size_monitor_swap_discards_dxgi_frame(self):
        original = _Display(1,101,'DISPLAY1',(0,0,2,2))
        swapped = _Display(1,202,'DISPLAY2',original.rect)
        native = Mock()
        native.DxgiDuplicationSession.return_value.acquire_frame.return_value = Mock(width=2,height=2)
        with patch.dict('sys.modules',{'windows_capture':native}), \
             patch('capture._monitor_for',side_effect=[original,original,swapped]), \
             _direct_dxgi(original.rect) as read:
            with self.assertRaisesRegex(ValueError,'映射'): read()
        native.DxgiDuplicationSession.return_value.acquire_frame.return_value.to_bgr.assert_not_called()
