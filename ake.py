"""Offline diagnostics. No game input or screenshot backend is initialized."""
import argparse
import json
from ake_core import Observation, chance, decide
from math_verify import threshold, totals

def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('math', help='show exact fixed-policy model results')
    a = commands.add_parser('analyze', help='analyze a manually supplied observation')
    a.add_argument('--slots', type=int, nargs=5, required=True, metavar=('S1','S2','S3','S4','S5'))
    a.add_argument('--inventory', type=int, nargs=5, required=True, metavar=('I1','I2','I3','I4','I5'))
    a.add_argument('--round', type=int, required=True)
    a.add_argument('--early-restart', action='store_true')
    args = p.parse_args()
    if args.command == 'math':
        output = {}
        for name, early in [('baseline', False), ('early_restart', True)]:
            probability, draws, total_draws, restarts = totals(early)
            output[name] = {'attempt_probability': str(probability), 'draws_per_attempt': float(draws),
                            'expected_total_draws': float(total_draws), 'expected_restarts': float(restarts)}
        output['cost_ratio_threshold'] = str(threshold())
        output['model'] = 'uniform without replacement; independent identical resets'
    else:
        slots = tuple(args.slots)
        obs = Observation(slots, tuple(args.inventory), args.round)
        recommendation = decide(obs, args.early_restart)
        output = {'decision': recommendation.value, 'source': 'manual input; no game state verified'}
        if recommendation.value != 'wait':
            k = slots.count(5)
            j = sum(x != 0 and x != 5 for x in slots)
            output['continue_without_voluntary_restart_probability'] = str(chance(k,j))
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
