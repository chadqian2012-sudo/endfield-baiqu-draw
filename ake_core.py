"""Offline, fail-closed strategy primitives for the Endfield card task."""
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import comb, isfinite

INITIAL = (6, 6, 6, 6, 10)
THRESHOLD = Fraction(227759, 92752)

class Decision(Enum):
    WAIT = 'wait'
    SUCCESS = 'success'
    DRAW = 'draw'
    RESTART = 'restart'

def chance(k: int, j: int) -> Fraction:
    if not (0 <= k <= 5 and 0 <= j <= 5 and k + j <= 5):
        raise ValueError('invalid hand')
    if k >= 4:
        return Fraction(1)
    m = 5 - k - j
    if k + m < 4:
        return Fraction(0)
    return Fraction(sum(comb(10-k, x) * comb(24-j, m-x)
                        for x in range(max(0, 4-k, m-(24-j)), min(m, 10-k)+1)),
                    comb(34-k-j, m))

def choose_early_restart(draw_costs, restart_costs) -> bool:
    if any(not isfinite(x) or x <= 0 for x in draw_costs) or any(not isfinite(x) or x < 0 for x in restart_costs):
        raise ValueError('invalid timing samples')
    if len(draw_costs) < 10 or len(restart_costs) < 3:
        return False
    d = sum(draw_costs) / len(draw_costs)
    r = sum(restart_costs) / len(restart_costs)
    if d <= 0 or r < 0:
        raise ValueError('invalid timing samples')
    return r / d < float(THRESHOLD)

@dataclass(frozen=True)
class Observation:
    slots: tuple
    inventory: tuple
    game_round: int
    stable: bool = True
    fresh: bool = True

def decide(obs: Observation, early_restart: bool = False) -> Decision:
    if len(obs.slots) != 5 or not obs.stable or not obs.fresh:
        return Decision.WAIT
    if any(type(x) is not int or x not in range(6) for x in obs.slots):
        return Decision.WAIT
    if len(obs.inventory) != 5 or any(type(x) is not int or x < 0 for x in obs.inventory):
        return Decision.WAIT
    # Candidate target freezes input even while inventory is changing.
    if obs.slots.count(5) >= 4:
        if obs.game_round != 1 or any(x not in (0,1,2,3,4,5) for x in obs.slots):
            return Decision.WAIT
        if len(obs.inventory) == 5 and all(obs.inventory[v-1] + obs.slots.count(v) == INITIAL[v-1] for v in range(1,6)):
            return Decision.SUCCESS
        return Decision.WAIT
    if obs.game_round != 1 or any(x not in (0,1,2,3,4,5) for x in obs.slots):
        return Decision.WAIT
    if len(obs.inventory) != 5 or any(obs.inventory[v-1] + obs.slots.count(v) != INITIAL[v-1] for v in range(1,6)):
        return Decision.WAIT
    k = obs.slots.count(5)
    n = sum(x != 0 for x in obs.slots)
    j = n - k
    if k + 5 - n < 4 or (early_restart and n == 1 and j == 1):
        return Decision.RESTART
    return Decision.DRAW
