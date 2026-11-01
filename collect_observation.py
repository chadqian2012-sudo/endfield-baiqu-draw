"""Temporary read-only recording for manual game-flow validation on Windows.

Records only while the Endfield window is foreground. Saves the upper 95% of
the client area, excluding the UID visible at the bottom of the tested layout.
Local captures are private and excluded from Git. No input is sent.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
import time
import cv2
import mss
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300 or not 1 <= args.fps <= 10:
        raise SystemExit('duration or FPS out of bounds')
    u = ctypes.WinDLL('user32', use_last_error=True)
    try:
        u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        pass
    u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    u.FindWindowW.restype = wintypes.HWND
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    hwnd = u.FindWindowW(None, 'Endfield')
    if not hwnd:
        raise SystemExit('Endfield window not found')
    rect, point = wintypes.RECT(), wintypes.POINT(0, 0)
    if not u.GetClientRect(hwnd, ctypes.byref(rect)) or not u.ClientToScreen(hwnd, ctypes.byref(point)):
        raise SystemExit('Cannot read client geometry')
    width, height = rect.right - rect.left, rect.bottom - rect.top
    cropped_height = int(height * .95) // 2 * 2
    width = width // 2 * 2
    if width < 100 or cropped_height < 100:
        raise SystemExit('Invalid game window size')
    region = {'left': point.x, 'top': point.y, 'width': width, 'height': cropped_height}
    outdir = Path('captures') / datetime.now().strftime('%Y%m%d-%H%M%S')
    outdir.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(str(outdir / 'observation.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, cropped_height))
    if not writer.isOpened():
        raise SystemExit('Video writer unavailable')
    start = time.monotonic()
    frames = 0
    with (outdir / 'frames.csv').open('w', newline='', encoding='utf-8') as f, mss.mss() as capture:
        log = csv.writer(f)
        log.writerow(['frame', 'elapsed_seconds'])
        print(f'READY {outdir} duration={args.seconds}s; input disabled', flush=True)
        try:
            while time.monotonic() - start < args.seconds:
                if (outdir / 'STOP').exists():
                    print('Stop marker detected', flush=True)
                    break
                tick = time.monotonic()
                if u.GetForegroundWindow() == hwnd:
                    current_rect, current_point = wintypes.RECT(), wintypes.POINT(0, 0)
                    if not u.GetClientRect(hwnd, ctypes.byref(current_rect)) or not u.ClientToScreen(hwnd, ctypes.byref(current_point)):
                        break
                    if (current_point.x, current_point.y, current_rect.right-current_rect.left, current_rect.bottom-current_rect.top) != (point.x,point.y,rect.right-rect.left,rect.bottom-rect.top):
                        print('Window geometry changed; stopping', flush=True)
                        break
                    frame = np.asarray(capture.grab(region))
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
                    log.writerow([frames, f'{tick-start:.6f}'])
                    frames += 1
                time.sleep(max(0, 1/args.fps - (time.monotonic()-tick)))
        finally:
            writer.release()
    print(f'DONE {frames} frames {outdir}', flush=True)

if __name__ == '__main__': main()
