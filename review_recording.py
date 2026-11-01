"""Create a contact sheet from a local observation video and its timestamp log."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--times', type=float, nargs='+', required=True)
    args = p.parse_args()
    with (args.recording/'frames.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('No recorded frames')
    video = cv2.VideoCapture(str(args.recording/'observation.mp4'))
    if not video.isOpened():
        raise SystemExit('Cannot open recording')
    tiles = []
    for requested in args.times:
        row = min(rows, key=lambda r: abs(float(r['elapsed_seconds'])-requested))
        video.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame']))
        ok, frame = video.read()
        if not ok:
            raise SystemExit(f"Cannot read frame {row['frame']}")
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        tile = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"t={float(row['elapsed_seconds']):.2f}s frame={row['frame']}", (8,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 3)
        cv2.putText(tile, f"t={float(row['elapsed_seconds']):.2f}s frame={row['frame']}", (8,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        tiles.append(tile)
    video.release()
    columns = min(3,len(tiles))
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
    output = args.recording/'contact-sheet.png'
    if not cv2.imwrite(str(output), sheet):
        raise SystemExit('Could not write contact sheet')
    print(output)

if __name__ == '__main__': main()
