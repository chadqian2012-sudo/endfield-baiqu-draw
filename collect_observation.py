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

def record(seconds=120, fps=5, *, output_root='captures', on_start=None, stopped=None):
    """Record from the first foreground frame. Return the directory and frame count."""
    if not 1 <= seconds <= 300 or not 1 <= fps <= 10:
        raise ValueError('duration or FPS out of bounds')
    binding = find_unique()
    left,top,width,height = binding.client
    if width < 100 or height < 100:
        raise ValueError('Invalid game window size')
    if width % 2 or height % 2:
        raise ValueError('Recording requires even client width and height; resize and recalibrate')
    region = {'left': left, 'top': top, 'width': width, 'height': height}
    outdir = Path(output_root) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    outdir.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(str(outdir / 'observation.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    try:
        if not writer.isOpened():
            raise ValueError('Video writer unavailable')
        start = None
        frames = 0
        with (outdir / 'frames.csv').open('w', newline='', encoding='utf-8') as f, mss.mss() as capture:
            log = csv.writer(f)
            log.writerow(['frame', 'elapsed_seconds'])
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if (stopped and stopped()) or (outdir / 'STOP').exists():
                    if not frames:
                        raise ValueError('录制已取消，没有录到游戏画面。')
                    break
                tick = time.monotonic()
                if snapshot(binding.hwnd) != binding:
                    raise RuntimeError('Window identity or geometry changed; stopping')
                if foreground_matches(binding):
                    frame = cv2.cvtColor(np.asarray(capture.grab(region)),cv2.COLOR_BGRA2BGR)
                    frame[int(height*.95):] = 0
                    writer.write(frame)
                    if start is None:
                        start = tick
                        deadline = start + seconds
                    log.writerow([frames, f'{tick-start:.6f}'])
                    frames += 1
                    if frames == 1 and on_start:
                        on_start(outdir)
                time.sleep(max(0, 1/fps - (time.monotonic()-tick)))
    finally:
        writer.release()
    if not frames:
        raise ValueError('等待游戏前台超时，没有录到画面。请打开抽牌页再录制。')
    return outdir, frames

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()
    print('Switch to the game within 30 seconds; input disabled', flush=True)
    try:
        outdir,frames = record(args.seconds,args.fps,on_start=lambda path: print(
            f'READY {path} duration={args.seconds}s; input disabled', flush=True))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f'DONE {frames} frames {outdir}', flush=True)

if __name__ == '__main__': main()
