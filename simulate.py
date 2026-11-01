"""Independent seeded simulation of whole attempts; never initializes input."""
import argparse
import math
import random
from math_verify import totals

def simulate(attempts, seed, early):
    if attempts <= 0:
        raise ValueError('attempts must be positive')
    rng = random.Random(seed)
    successes = draws = 0
    deck = [5]*10 + [0]*24
    for _ in range(attempts):
        k = j = 0
        for card in rng.sample(deck, 5):
            draws += 1
            if card == 5:
                k += 1
            else:
                j += 1
            if k == 4:
                successes += 1
                break
            if j == 2 or (early and k == 0 and j == 1):
                break
    return successes, draws

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    for early in (False, True):
        success, draws = simulate(args.attempts, args.seed, early)
        p = float(totals(early)[0])
        se = math.sqrt(p*(1-p)/args.attempts)
        print(f'early={early} seed={args.seed} attempts={args.attempts} successes={success} p={success/args.attempts:.8f} expected={p:.8f} z={(success/args.attempts-p)/se:.3f} draws/attempt={draws/args.attempts:.8f}')

if __name__ == '__main__': main()
