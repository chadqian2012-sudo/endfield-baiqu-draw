"""Evaluate local labelled crops using whole recording batches as folds.

This produces experimental results only; full-hand and game-control acceptance
require independent recordings of every class, buttons, stock and transitions.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from vision import classify, read_image

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', type=Path)
    p.add_argument('--min-score', type=float, required=True)
    p.add_argument('--min-margin', type=float, required=True)
    args = p.parse_args()
    with (args.samples/'manifest.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    groups = sorted({r['recording'] for r in rows})
    report = {}
    for heldout in groups:
        train = defaultdict(list)
        for row in rows:
            if row['recording'] != heldout:
                im = read_image(args.samples / row['file'])
                if im is None: raise ValueError('missing sample: '+row['file'])
                train[row['label']].append(im)
        counts = defaultdict(lambda: {'total':0,'correct':0,'rejected':0,'wrong':0})
        for row in rows:
            if row['recording'] != heldout: continue
            im = read_image(args.samples / row['file'])
            if im is None: raise ValueError('missing sample: '+row['file'])
            match = classify(im, train, {label:args.min_score for label in train}, args.min_margin)
            count = counts[row['label']]
            count['total'] += 1
            if match.label is None: count['rejected'] += 1
            elif match.label == row['label']: count['correct'] += 1
            else: count['wrong'] += 1
        report[heldout] = {'training_classes':sorted(train), 'classes':dict(counts)}
    print(json.dumps({'score':args.min_score,'margin':args.min_margin,'folds':report,
                      'note':'experimental crop-only results; samples within a recording are correlated'},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
