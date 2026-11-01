import unittest
from unittest.mock import patch
from ake import drive
from ake_core import Observation
from safety import Action, ActionTracker, Frame, ObservationStabilizer, draw_completed, stable

EMPTY = Observation((0,0,0,0,0),(6,6,6,6,10),1)
FIVE = Observation((5,0,0,0,0),(6,6,6,6,9),1)
STALE_STOCK = Observation((5,0,0,0,0),(6,6,6,6,10),1)

class SafetyTests(unittest.TestCase):
    def run_stream(self, observations, mode='live', stop_after_click=False):
        class Profile:
            regions = {'draw':[0,0,1,1],'abandon':[0,0,1,1],'confirm':[0,0,1,1]}
            def read(self, image): return image,{}
        now = [0.0]
        events,clicks = [],[]
        def frames():
            for i,obs in enumerate(observations):
                now[0] = i*.12
                yield i,now[0],obs,True
        started = [False]
        stopped = [False]
        def keys():
            if not started[0]:
                started[0] = True
                return [0x77]
            if stop_after_click and clicks and not stopped[0]:
                stopped[0] = True
                return [0x78]
            return []
        def perform(region, permitted):
            self.assertTrue(permitted())
            clicks.append(region)
            return True
        with patch('ake.time.monotonic',side_effect=lambda:now[0]):
            summary = drive(Profile(),frames(),mode=mode,emit=events.append,
                            keys=keys if mode=='live' else None,perform=perform)
        return events,clicks,summary

    def test_full_stream_stops_at_four_fives(self):
        snapshots = []
        for k in range(5):
            obs = Observation(tuple([5]*k+[0]*(5-k)),(6,6,6,6,10-k),1,
                              buttons=('enabled','enabled',None))
            snapshots.extend([obs]*4)
        events,clicks,summary = self.run_stream(snapshots)
        self.assertEqual(len(clicks),4)
        self.assertEqual(summary['draw_clicks_sent'],4)
        self.assertEqual(summary['cards_confirmed'],4)
        self.assertEqual(events[-1]['event'],'success')

    def test_dry_run_does_not_fake_execution(self):
        events,clicks,summary = self.run_stream([EMPTY]*40,mode='dry-run')
        self.assertEqual(clicks,[])
        self.assertEqual(summary['cards_confirmed'],0)
        self.assertEqual(sum(e['event']=='recommendation' for e in events),1)

    def test_f9_after_sent_click_is_terminal(self):
        obs = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','enabled',None))
        events,clicks,_ = self.run_stream([obs]*30,stop_after_click=True)
        self.assertEqual(len(clicks),1)
        self.assertEqual(events[-1],{'event':'stopped','reason':'F9'})

    def test_stability_requires_distinct_fresh_frames(self):
        frames = [Frame(1,0,EMPTY), Frame(2,.11,EMPTY), Frame(3,.22,EMPTY)]
        self.assertTrue(stable(frames,.3))
        self.assertFalse(stable([frames[0]]*3,.3))
        self.assertFalse(stable(frames,1))

    def test_stabilizer_freezes_candidate_before_inventory_settles(self):
        s = ObservationStabilizer()
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        self.assertIsNone(s.add(Frame(1,0,candidate), 0))
        self.assertTrue(s.candidate)
        self.assertIsNone(s.add(Frame(1,.11,candidate), .11))
        self.assertEqual(len(s.frames),0)

    def test_draw_requires_stock_and_exactly_one_new_card(self):
        self.assertTrue(draw_completed(EMPTY,FIVE))
        self.assertFalse(draw_completed(EMPTY,STALE_STOCK))
        self.assertFalse(draw_completed(EMPTY,Observation((5,5,0,0,0),(6,6,6,6,8),1)))

    def test_pause_preserves_pending_action(self):
        t = ActionTracker()
        self.assertTrue(t.resume())
        t.sent(Action.DRAW, EMPTY)
        t.pause()
        self.assertFalse(t.resume())
        self.assertFalse(t.reconcile(EMPTY))
        self.assertTrue(t.reconcile(FIVE))
        self.assertFalse(t.reconcile(FIVE))
        self.assertTrue(t.resume())
        t.stop()
        self.assertFalse(t.resume())

    def test_success_and_unexpected_confirmation_block_input(self):
        t = ActionTracker()
        t.resume()
        with self.assertRaises(RuntimeError):
            t.sent(Action.CONFIRM, EMPTY)
        with self.assertRaises(RuntimeError):
            t.sent(Action.ABANDON, EMPTY)
        success = Observation((5,5,5,5,0),(6,6,6,6,6),1)
        with self.assertRaises(RuntimeError):
            t.sent(Action.DRAW, success)

    def test_restart_requires_policy_and_matched_dialog(self):
        t = ActionTracker()
        t.resume()
        one_other = Observation((1,0,0,0,0),(5,6,6,6,10),1)
        with self.assertRaises(RuntimeError):
            t.sent(Action.ABANDON, one_other)
        t.sent(Action.ABANDON, one_other, early_restart=True)
        self.assertFalse(t.reconcile(one_other))
        self.assertTrue(t.reconcile(one_other, expected_abandon_dialog=True))
        t.sent(Action.CONFIRM, one_other)
        self.assertFalse(t.reconcile(one_other))
        self.assertTrue(t.reconcile(EMPTY))

    def test_candidate_freezes_during_inventory_lag(self):
        t = ActionTracker()
        t.resume()
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        t.observe(candidate)
        self.assertFalse(t.terminated)
        self.assertFalse(t.resume())
        with self.assertRaises(RuntimeError):
            t.sent(Action.DRAW, EMPTY)
        t.observe(Observation((5,5,5,5,0),(6,6,6,6,6),1))
        self.assertTrue(t.terminated)

    def test_timeout_pauses_without_resending_or_forgetting_action(self):
        now = [0.0]
        t = ActionTracker(clock=lambda: now[0])
        t.resume()
        action_id = t.sent(Action.DRAW, EMPTY)
        now[0] = 9.9
        self.assertFalse(t.check_timeout())
        now[0] = 10.0
        self.assertTrue(t.check_timeout())
        self.assertEqual(t.pending[0], action_id)
        self.assertFalse(t.resume())
        now[0] = 1000.0
        self.assertFalse(t.check_timeout())
        self.assertTrue(t.reconcile(FIVE))
        self.assertTrue(t.resume())

if __name__ == '__main__': unittest.main()
