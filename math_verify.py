"""Exact rational verification for the two fixed policies.

The model assumes uniform draws without replacement and independent resets.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
from ake_core import THRESHOLD

def attempt_metrics(early: bool):
    """Return (success probability, expected draws) for one attempt."""
    @lru_cache(None)
    def visit(k, j):
        if k >= 4:
            return Fraction(1), Fraction(0)
        if j >= 2 or (early and k == 0 and j == 1):
            return Fraction(0), Fraction(0)
        total = 34-k-j
        p5 = Fraction(10-k, total)
        pa, da = visit(k+1, j)
        pb, db = visit(k, j+1)
        return p5*pa+(1-p5)*pb, 1+p5*da+(1-p5)*db
    return visit(0, 0)

def totals(early: bool):
    p, draws = attempt_metrics(early)
    return p, draws, draws/p, (1-p)/p

def threshold():
    _, _, d1, r1 = totals(True)
    _, _, d2, r2 = totals(False)
    return (d2-d1)/(r1-r2)

def enumerate_policies(draw_cost, restart_cost):
    """Evaluate all 128 deterministic policies on reachable nonterminal states."""
    states = [(k,j) for k in range(4) for j in range(2) if (k,j) != (0,0)]
    out = []
    for bits in product((False, True), repeat=len(states)):
        restart = dict(zip(states,bits))
        @lru_cache(None)
        def visit(k,j):
            if k == 4:
                return Fraction(1), Fraction(0)
            if j == 2 or restart.get((k,j), False):
                return Fraction(0), Fraction(0)
            p = Fraction(10-k,34-k-j)
            p1,d1 = visit(k+1,j)
            p2,d2 = visit(k,j+1)
            return p*p1+(1-p)*p2, 1+p*d1+(1-p)*d2
        p, draws = visit(0,0)
        if p:
            cost = (draws*draw_cost+(1-p)*restart_cost)/p
            out.append((cost, restart))
    return sorted(out, key=lambda item: item[0])

if __name__ == '__main__':
    for label, early in [('first non-five', True), ('second non-five', False)]:
        p, draws, total_draws, restarts = totals(early)
        print(f'{label}: p={p} draws/attempt={float(draws):.8f} total draws={float(total_draws):.8f} restarts={float(restarts):.8f}')
    print(f'threshold={threshold()} ({float(threshold()):.8f}); expected={THRESHOLD}')
    assert threshold() == THRESHOLD
