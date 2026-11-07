"""Conservative grayscale template classification; templates must be calibrated externally."""
from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import asdict
from ake_core import Observation
from uuid import uuid4

def read_image(path, flags=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(),dtype=np.uint8),flags)

def write_image(path, image):
    ok,data = cv2.imencode('.png',image)
    if not ok: raise RuntimeError('cannot encode evidence image')
    Path(path).write_bytes(data.tobytes())

@dataclass(frozen=True)
class Match:
    label: str | None
    score: float | None
    margin: float | None

def classify(image, templates, thresholds, minimum_margin):
    """Return unknown when no calibrated match clears both acceptance checks.

    Image and template dimensions must match exactly. This deliberately does not
    infer coordinates, rescale screenshots, or invent classification thresholds.
    """
    if image is None or image.ndim != 2 or image.size == 0 or not templates or np.std(image) < 1:
        return Match(None, None, None)
    if minimum_margin is None or not 0 < minimum_margin <= 2:
        return Match(None, None, None)
    scores = {}
    for label, samples in templates.items():
        if label not in thresholds or not 0 <= thresholds[label] <= 1 or not samples:
            return Match(None, None, None)
        results = []
        for sample in samples:
            if sample is None or sample.ndim != 2 or sample.shape != image.shape or np.std(sample) < 1:
                return Match(None, None, None)
            score = float(cv2.matchTemplate(image, sample, cv2.TM_CCOEFF_NORMED)[0,0])
            if not np.isfinite(score):
                return Match(None, None, None)
            results.append(score)
        scores[label] = max(results)
    if len(scores) < 2:
        return Match(None, None, None)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    label, score = ranked[0]
    margin = score - ranked[1][1]
    return Match(label if score >= thresholds[label] and margin >= minimum_margin else None, score, margin)

def recognize_card(image, templates, thresholds, minimum_margin):
    """Require calibrated examples for every supported card class and empty slot."""
    expected = {'empty', '1', '2', '3', '4', '5'}
    if set(templates) != expected or set(thresholds) != expected:
        return Match(None, None, None)
    return classify(image, templates, thresholds, minimum_margin)


def segment_digits(image):
    """Segment one tightly cropped inventory value; return None when ambiguous."""
    if image is None or image.ndim != 2 or image.size == 0 or min(image.shape) < 8:
        return None
    if np.std(image) < 5:
        return None
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for i in range(1,n):
        x,y,w,h,area = map(int,stats[i])
        if area < 3:
            continue
        if h < image.shape[0]*.35 or w < 2 or y == 0 or y+h >= image.shape[0]:
            return None
        components.append((x,y,w,h))
    if not 1 <= len(components) <= 2:
        return None
    components.sort()
    if len(components) == 2:
        a,b = components
        if b[0] < a[0]+a[2] or b[0]-(a[0]+a[2]) > image.shape[1]*.3:
            return None
        if abs(a[3]-b[3]) > max(a[3],b[3])*.3:
            return None
    return [mask[y:y+h,x:x+w] for x,y,w,h in components]

def normalize_digit(binary):
    if binary is None or binary.ndim != 2 or binary.size == 0:
        return None
    return cv2.resize(binary,(32,48),interpolation=cv2.INTER_NEAREST)

def recognize_stock(image, templates, thresholds, minimum_margin):
    """Return 0–10 or None; no result is inferred from the hand."""
    expected = set('0123456789')
    if set(templates) != expected or set(thresholds) != expected:
        return None
    parts = segment_digits(image)
    if parts is None:
        return None
    values = []
    for part in parts:
        matched = classify(normalize_digit(part),templates,thresholds,minimum_margin)
        if matched.label is None:
            return None
        values.append(matched.label)
    value = ''.join(values)
    if len(value) == 2 and value != '10':
        return None
    return int(value)


class VisionProfile:
    """One explicitly calibrated layout, shared by replay and live observation."""
    CLASSES = {'cards': {'empty','1','2','3','4','5'}, 'stock': set('0123456789'),
               'round': {'1','2'}, 'draw': {'enabled','disabled'},
               'abandon': {'enabled','disabled'}, 'confirm': {'enabled','disabled'}}

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.size = tuple(self.config['frame_size'])
        self.regions = self.config.get('regions', {})
        self.check_regions(self.size,self.regions)
        self.groups = {}
        for name, data in self.config.get('groups', {}).items():
            if name not in self.CLASSES:
                raise ValueError('unknown template group: '+name)
            bank = {}
            for label, paths in data.get('templates', {}).items():
                if label not in self.CLASSES[name]:
                    raise ValueError('unknown class: '+name+'/'+label)
                bank[label] = [self.image(p) for p in paths]
                if name == 'cards' and 'slots' in self.regions:
                    shape = tuple(reversed(self.regions['slots'][0][2:]))
                    if any(image.shape != shape for image in bank[label]):
                        raise ValueError('cards template dimensions differ from slot regions')
            self.groups[name] = (bank, data.get('thresholds', {}), data.get('margin'))
        self.anchors = {}
        for name, data in self.config.get('anchors', {}).items():
            self.check_region(self.size,data['region'])
            sample = self.image(data['template'])
            if sample.shape != (data['region'][3],data['region'][2]) or np.std(sample) < 1:
                raise ValueError('invalid anchor: '+name)
            if not 0 <= data['threshold'] <= 1:
                raise ValueError('invalid anchor threshold')
            self.anchors[name] = (data, sample)

    def image(self, path):
        image = read_image(self.path.parent / path)
        if image is None: raise ValueError('cannot read template: '+str(path))
        return image

    @staticmethod
    def save_config(path, config):
        text = json.dumps(config,ensure_ascii=False,indent=2)
        path = Path(path).resolve()
        temporary = path.with_name('.ake-config-'+uuid4().hex+'.tmp')
        stream = temporary.open('x',encoding='utf-8')
        try:
            with stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def check_region(size, roi):
        if (len(roi) != 4 or any(type(n) is not int for n in roi) or
            min(roi[:2]) < 0 or min(roi[2:]) <= 0 or
            roi[0]+roi[2] > size[0] or roi[1]+roi[3] > size[1]):
            raise ValueError('region outside calibrated frame')

    @staticmethod
    def check_regions(size, regions):
        if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError('invalid frame_size')
        for name, rois in regions.items():
            if name in ('slots','inventory'):
                if len(rois) != 5: raise ValueError(name+' must contain five regions')
                for roi in rois: VisionProfile.check_region(size,roi)
                if name == 'slots' and any(tuple(roi[2:]) != tuple(rois[0][2:]) for roi in rois[1:]):
                    raise ValueError('slot regions must have the same width and height')
            else:
                VisionProfile.check_region(size,rois)

    @staticmethod
    def crop(gray, roi):
        x,y,w,h = roi
        return gray[y:y+h,x:x+w]

    def issues(self):
        issues = []
        for name, expected in self.CLASSES.items():
            bank, thresholds, margin = self.groups.get(name, ({},{},None))
            missing = expected - {label for label, samples in bank.items() if samples}
            if missing: issues.append(name+': missing classes '+','.join(sorted(missing)))
            if set(thresholds) != expected or any(not 0 <= v <= 1 for v in thresholds.values()):
                issues.append(name+': incomplete score thresholds')
            if margin is None or not 0 < margin <= 2: issues.append(name+': missing positive margin')
        for name in ('slots','inventory','round','draw','abandon','confirm'):
            if name not in self.regions: issues.append('missing regions: '+name)
        for name in ('hand','abandon_dialog'):
            if name not in self.anchors: issues.append('missing anchor: '+name)
        return issues

    def match_group(self, gray, name, roi):
        bank, thresholds, margin = self.groups.get(name, ({},{},None))
        if set(bank) != self.CLASSES[name]: return Match(None,None,None)
        return classify(self.crop(gray,roi),bank,thresholds,margin)

    def add_sample(self, frame, group, label, index, digit, score, margin, batch):
        if frame.shape[:2] != (self.size[1],self.size[0]):
            raise ValueError('sample frame dimensions differ from calibration')
        if not 0 < score <= 1 or not 0 < margin <= 2:
            raise ValueError('explicit positive score and margin required')
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if group in ('hand','abandon_dialog'):
            roi = self.regions[group]
        else:
            if group not in self.CLASSES or label not in self.CLASSES[group]:
                raise ValueError('unknown sample class')
            name = 'slots' if group == 'cards' else 'inventory' if group == 'stock' else group
            roi = self.regions[name][index] if name in ('slots','inventory') else self.regions[name]
        image = self.crop(gray,roi)
        if group == 'stock':
            parts = segment_digits(image)
            if parts is None or not 0 <= digit < len(parts):
                raise ValueError('cannot segment requested inventory digit')
            image = normalize_digit(parts[digit])
        if np.std(image) < 1:
            raise ValueError('sample lacks visual detail')
        assets = self.path.parent/'logs'/'calibration-assets'
        assets.mkdir(parents=True,exist_ok=True)
        file = assets/(uuid4().hex+'.png')
        write_image(file,image)
        config = deepcopy(self.config)
        if group in ('hand','abandon_dialog'):
            config.setdefault('anchors',{})[group] = {'region':roi,'template':str(file),'threshold':score}
        else:
            data = config.setdefault('groups',{}).setdefault(group,{'templates':{},'thresholds':{}})
            data['templates'].setdefault(label,[]).append(str(file))
            data['thresholds'][label] = score
            data['margin'] = margin
        batches = config.setdefault('training_batches',[])
        if batch not in batches: batches.append(batch)
        self.save_config(self.path,config)
        self.config = config

    def read(self, frame):
        if frame.shape[:2] != (self.size[1],self.size[0]):
            raise ValueError('frame dimensions differ from calibration')
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        matched_pages = []
        quality = {}
        for name, (data, sample) in self.anchors.items():
            crop = self.crop(gray,data['region'])
            score = float(cv2.matchTemplate(crop,sample,cv2.TM_CCOEFF_NORMED)[0,0])
            quality['anchor_'+name] = score
            if np.isfinite(score) and np.std(crop) >= 1 and score >= data['threshold']:
                matched_pages.append(name)
        page = matched_pages[0] if len(matched_pages) == 1 else 'unknown'
        slots = [None]*5
        inventory = [None]*5
        game_round = None
        buttons = {name:None for name in ('draw','abandon','confirm')}
        if page == 'hand':
            for i,roi in enumerate(self.regions.get('slots', [])):
                match = self.match_group(gray,'cards',roi)
                quality['slot_'+str(i+1)] = asdict(match)
                slots[i] = 0 if match.label == 'empty' else int(match.label) if match.label else None
            bank, thresholds, margin = self.groups.get('stock', ({},{},None))
            for i,roi in enumerate(self.regions.get('inventory', [])):
                inventory[i] = recognize_stock(self.crop(gray,roi),bank,thresholds,margin)
            if 'round' in self.regions:
                label = self.match_group(gray,'round',self.regions['round']).label
                game_round = int(label) if label else None
            for name in ('draw','abandon'):
                if name in self.regions: buttons[name] = self.match_group(gray,name,self.regions[name]).label
        elif page == 'abandon_dialog' and 'confirm' in self.regions:
            buttons['confirm'] = self.match_group(gray,'confirm',self.regions['confirm']).label
        return Observation(tuple(slots),tuple(inventory),game_round,page=page,
                           buttons=tuple(buttons[name] for name in ('draw','abandon','confirm'))),quality
