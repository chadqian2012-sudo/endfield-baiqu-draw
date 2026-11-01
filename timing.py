"""Timing sample bookkeeping; caller excludes interrupted and paused cycles."""
from dataclasses import dataclass, field
from math import isfinite, sqrt
from ake_core import THRESHOLD

@dataclass
class Samples:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def summary(self):
        return {'count': self.count, 'mean': self.mean if self.count else None,
                'stddev': sqrt(max(0, self.m2)/(self.count-1)) if self.count >= 2 else None}

@dataclass
class Timings:
    draws: Samples = field(default_factory=Samples)
    restarts: Samples = field(default_factory=Samples)
    early_restart: bool = False

    def add(self, kind, seconds, *, uninterrupted):
        if kind not in ('draw', 'restart'):
            raise ValueError('unknown cycle')
        if not uninterrupted:
            return
        if not isfinite(seconds) or seconds < 0 or (kind == 'draw' and seconds == 0):
            raise ValueError('invalid duration')
        (self.draws if kind == 'draw' else self.restarts).add(seconds)

    def begin_attempt(self):
        self.early_restart = (self.draws.count >= 10 and self.restarts.count >= 3
                              and self.restarts.mean / self.draws.mean < float(THRESHOLD))
        return self.early_restart

    def summary(self):
        return {'draw': self.draws.summary(), 'restart': self.restarts.summary(),
                'early_restart': self.early_restart}
