"""Create a contact sheet from a local observation video and its timestamp log."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from vision import write_image
from ake import frames_from_recording

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--times', type=float, nargs='+', required=True)
    args = p.parse_args()
    selected = [None]*len(args.times)
    distances = [float('inf')]*len(args.times)
    for sequence,stamp,frame,_ in frames_from_recording(args.recording):
        for i,requested in enumerate(args.times):
            distance = abs(stamp-requested)
            if distance < distances[i]:
                selected[i] = (sequence,stamp,frame)
                distances[i] = distance
    if any(frame is None for frame in selected):
        raise SystemExit('No recorded frames')
    tiles = []
    for sequence,stamp,frame in selected:
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        tile = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"t={stamp:.2f}s frame={sequence}", (8,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 3)
        cv2.putText(tile, f"t={stamp:.2f}s frame={sequence}", (8,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        tiles.append(tile)
    columns = min(3,len(tiles))
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
    output = args.recording/'contact-sheet.png'
    write_image(output,sheet)
    print(output)

if __name__ == '__main__': main()
