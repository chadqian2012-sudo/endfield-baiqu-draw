"""Experimental read-only card-crop replay; never creates an input backend.

Scores are diagnostic, not calibrated probabilities or game acceptance.
Region geometry is explicitly supplied and applies only to the recorded layout.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
from vision import recognize_card

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--times', type=float, nargs='+', required=True)
    p.add_argument('--min-score', type=float, required=True)
    p.add_argument('--min-margin', type=float, required=True)
    args = p.parse_args()
    config = json.loads(args.regions.read_text(encoding='utf-8'))
    regions = config['slots']
    width,height = config['frame_size']
    if len(regions) != 5 or any(len(r) != 4 or any(type(x) is not int for x in r) or
        r[0] < 0 or r[1] < 0 or r[2] <= 0 or r[3] <= 0 or r[0]+r[2] > width or r[1]+r[3] > height for r in regions):
        raise ValueError('invalid regions')
    with (args.samples/'manifest.csv').open(encoding='utf-8',newline='') as f:
        sample_rows = list(csv.DictReader(f))
    templates = {}
    for row in sample_rows:
        image = cv2.imread(str(args.samples/row['file']),cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError('sample missing: '+row['file'])
        templates.setdefault(row['label'],[]).append(image)
    with (args.recording/'frames.csv').open(encoding='utf-8',newline='') as f:
        frames = list(csv.DictReader(f))
    if not frames: raise ValueError('no frames')
    video = cv2.VideoCapture(str(args.recording/'observation.mp4'))
    if not video.isOpened(): raise ValueError('cannot open recording')
    try:
        for seconds in args.times:
            row = min(frames,key=lambda r:abs(float(r['elapsed_seconds'])-seconds))
            video.set(cv2.CAP_PROP_POS_FRAMES,int(row['frame']))
            ok,frame = video.read()
            if not ok or frame.shape[:2] != (height,width): raise ValueError('frame dimensions do not match calibration')
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            result = []
            for x,y,w,h in regions:
                match = recognize_card(gray[y:y+h,x:x+w],templates,
                                       {label:args.min_score for label in templates},args.min_margin)
                result.append({'label':match.label,'score':match.score,'margin':match.margin})
            print(json.dumps({'elapsed':float(row['elapsed_seconds']),'frame':int(row['frame']),
                              'slots':result,'mode':'experimental-replay-only'},ensure_ascii=False))
    finally:
        video.release()

if __name__ == '__main__': main()
