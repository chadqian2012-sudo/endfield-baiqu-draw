"""Conservative grayscale template classification; templates must be calibrated externally."""
from dataclasses import dataclass
import cv2
import numpy as np

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
    if image is None or image.ndim != 2 or image.size == 0 or not templates:
        return Match(None, None, None)
    if minimum_margin is None or not 0 <= minimum_margin <= 2:
        return Match(None, None, None)
    scores = {}
    for label, samples in templates.items():
        if label not in thresholds or not 0 <= thresholds[label] <= 1 or not samples:
            return Match(None, None, None)
        results = []
        for sample in samples:
            if sample is None or sample.ndim != 2 or sample.shape != image.shape or np.std(sample) < 1 or np.std(image) < 1:
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
