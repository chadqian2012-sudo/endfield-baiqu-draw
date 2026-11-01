"""Pure action-tracking and observation checks; no input backend is loaded."""
from dataclasses import dataclass
from collections import deque
from enum import Enum
from ake_core import INITIAL, Observation, Decision, decide

class Action(Enum):
    DRAW = 'draw'
    ABANDON = 'abandon'
    CONFIRM = 'confirm'

@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float
    observation: Observation

def stable(frames, now):
    """Three distinct, ordered captures, 100ms apart; latest <= 500ms old."""
    if len(frames) != 3:
        return False
    a, b, c = frames
    if not (a.sequence < b.sequence < c.sequence):
        return False
    if b.timestamp - a.timestamp < .1 or c.timestamp - b.timestamp < .1:
        return False
    if not 0 <= now - c.timestamp <= .5:
        return False
    return a.observation == b.observation == c.observation

def valid(obs):
    if not obs.stable or not obs.fresh or obs.game_round != 1:
        return False
    if len(obs.slots) != 5 or len(obs.inventory) != 5:
        return False
    if any(type(v) is not int or v not in range(6) for v in obs.slots):
        return False
    if any(type(v) is not int or v < 0 for v in obs.inventory):
        return False
    return all(obs.inventory[i-1] + obs.slots.count(i) == INITIAL[i-1] for i in range(1,6))

def draw_completed(before, after):
    if not valid(before) or not valid(after):
        return False
    changes = [(a,b) for a,b in zip(before.slots,after.slots) if a != b]
    return len(changes) == 1 and changes[0][0] == 0 and changes[0][1] in range(1,6)

def initial(obs):
    return valid(obs) and obs.slots == (0,0,0,0,0) and obs.inventory == INITIAL

class ObservationStabilizer:
    """Maintains only three independent captures, with an immediate success freeze."""
    def __init__(self):
        self.frames = deque(maxlen=3)
        self.candidate = False

    def add(self, frame, now):
        obs = frame.observation
        if len(obs.slots) == 5 and obs.slots.count(5) >= 4:
            self.candidate = True
        if self.frames and (frame.sequence <= self.frames[-1].sequence or frame.timestamp <= self.frames[-1].timestamp):
            self.frames.clear()
            return None
        self.frames.append(frame)
        if stable(list(self.frames), now) and valid(obs):
            return obs
        return None

class ActionTracker:
    """Tracks at most one sent action; pause never forgets an unconfirmed action."""
    def __init__(self):
        self.pending = None
        self.paused = True
        self.terminated = False
        self._next_id = 1
        self.phase = 'hand'
        self.candidate = False

    def resume(self):
        if self.terminated or self.pending is not None or self.candidate:
            return False
        self.paused = False
        return True

    def pause(self):
        self.paused = True

    def stop(self):
        self.paused = True
        self.terminated = True

    def observe(self, obs):
        # A possible fourth five freezes actions even before inventory settles.
        if len(obs.slots) == 5 and obs.slots.count(5) >= 4:
            self.candidate = True
            self.paused = True
            if valid(obs):
                self.terminated = True

    def sent(self, action, before, *, early_restart=False):
        if self.paused or self.terminated or self.candidate or self.pending is not None or not valid(before):
            raise RuntimeError('input blocked')
        if decide(before) == Decision.SUCCESS:
            raise RuntimeError('success freezes input')
        if action in (Action.DRAW, Action.ABANDON) and self.phase != 'hand':
            raise RuntimeError('unexpected action')
        if action == Action.CONFIRM and self.phase != 'dialog':
            raise RuntimeError('unexpected confirmation')
        if action == Action.DRAW and decide(before, early_restart) != Decision.DRAW:
            raise RuntimeError('draw blocked')
        if action == Action.ABANDON and decide(before, early_restart) != Decision.RESTART:
            raise RuntimeError('abandon blocked')
        action_id = self._next_id
        self._next_id += 1
        self.pending = (action_id, action, before)
        return action_id

    def reconcile(self, after, *, expected_abandon_dialog=False):
        if self.pending is None or self.terminated:
            return False
        self.observe(after)
        _, action, before = self.pending
        complete = ((action == Action.DRAW and draw_completed(before, after)) or
                    (action == Action.ABANDON and expected_abandon_dialog) or
                    (action == Action.CONFIRM and initial(after)))
        if complete:
            self.pending = None
            if action == Action.ABANDON:
                self.phase = 'dialog'
            elif action == Action.CONFIRM:
                self.phase = 'hand'
        return complete
