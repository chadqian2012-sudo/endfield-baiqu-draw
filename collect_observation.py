"""Temporary read-only recording for manual game-flow validation on Windows.

Records only while the Endfield window is foreground. Preserves client geometry
and masks its bottom 5%, containing the UID in the tested layout.
Local captures are private and excluded from Git. No input is sent.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import cv2
import mss
import numpy as np
from windowing import find_unique, snapshot, foreground_matches

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300 or not 1 <= args.fps <= 10:
        raise SystemExit('duration or FPS out of bounds')
    binding = find_unique()
    left,top,width,height = binding.client
    if width < 100 or height < 100:
        raise SystemExit('Invalid game window size')
    if width % 2 or height % 2:
        raise SystemExit('Recording requires even client width and height; resize and recalibrate')
    region = {'left': left, 'top': top, 'width': width, 'height': height}
    outdir = Path('captures') / datetime.now().strftime('%Y%m%d-%H%M%S')
    outdir.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(str(outdir / 'observation.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
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
                if snapshot(binding.hwnd) != binding:
                    raise RuntimeError('Window identity or geometry changed; stopping')
                if foreground_matches(binding):
                    frame = cv2.cvtColor(np.asarray(capture.grab(region)),cv2.COLOR_BGRA2BGR)
                    frame[int(height*.95):] = 0
                    writer.write(frame)
                    log.writerow([frames, f'{tick-start:.6f}'])
                    frames += 1
                time.sleep(max(0, 1/args.fps - (time.monotonic()-tick)))
        finally:
            writer.release()
    print(f'DONE {frames} frames {outdir}', flush=True)

if __name__ == '__main__': main()
