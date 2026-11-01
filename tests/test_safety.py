import unittest
from unittest.mock import Mock, patch
from ake import drive
from ake_core import Observation
from safety import Action, ActionTracker, Frame, ObservationStabilizer, draw_completed, stable
from windowing import WindowIdentity, click

EMPTY = Observation((0,0,0,0,0),(6,6,6,6,10),1)
FIVE = Observation((5,0,0,0,0),(6,6,6,6,9),1)
STALE_STOCK = Observation((5,0,0,0,0),(6,6,6,6,10),1)

class SafetyTests(unittest.TestCase):
    def test_input_is_rejected_when_focus_changes_during_identity_check(self):
        binding = WindowIdentity(1,10,20,'game.exe',(0,0,100,100),96)
        foreground = [binding.hwnd]
        api = Mock()
        api.GetSystemMetrics.side_effect = {76:0,77:0,78:100,79:100}.__getitem__
        api.GetForegroundWindow.side_effect = lambda:foreground[0]
        api.SendInput.return_value = 3
        def switched_window(hwnd):
            foreground[0] = 2
            return binding
        with patch('windowing._api',return_value=(api,None)), patch('windowing.snapshot',side_effect=switched_window):
            self.assertFalse(click(binding,[10,10,10,10],lambda:True))
        api.SendInput.assert_not_called()

    def run_stream(self, observations, mode='live', stop_after_click=False,
                   stop_at_submission=False, step=.12, save=None, pause_at=None):
        class Profile:
            regions = {'draw':[0,0,1,1],'abandon':[0,0,1,1],'confirm':[0,0,1,1]}
            def read(self, image): return image,{}
        now = [0.0]
        events,clicks = [],[]
        def frames():
            for i,obs in enumerate(observations):
                now[0] = i*step
                yield i,now[0],obs,True
        started = [False]
        stopped = [False]
        def keys():
            nonlocal pause_at
            if not started[0]:
                started[0] = True
                return [0x77]
            if stop_at_submission and stopped[0]:
                return [0x78]
            if stop_after_click and clicks and not stopped[0]:
                stopped[0] = True
                return [0x78]
            if pause_at is not None and now[0] >= pause_at:
                pause_at = None
                return [0x77]
            return []
        def perform(region, permitted):
            if stop_at_submission: stopped[0] = True
            if not permitted(): return False
            clicks.append(region)
            return True
        with patch('ake.time.monotonic',side_effect=lambda:now[0]):
            summary = drive(Profile(),frames(),mode=mode,emit=events.append,
                            keys=keys if mode=='live' else None,perform=perform,save=save)
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

    def test_observe_reports_nonempty_initial_hand_without_taking_over(self):
        events,clicks,summary = self.run_stream([FIVE]*5,mode='observe')
        observed = [e for e in events if e['event']=='observation']
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0]['observation']['slots'],FIVE.slots)
        self.assertEqual(observed[0]['observation']['inventory'],FIVE.inventory)
        self.assertEqual(clicks,[])
        self.assertEqual(summary['attempts_confirmed'],0)

    def test_initial_success_candidate_without_pending_has_own_deadline(self):
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        save = Mock()
        events,clicks,summary = self.run_stream([candidate]*100,step=.125,save=save)
        self.assertEqual(summary['termination_reason'],'suspected success; confirmation timeout')
        self.assertEqual(clicks,[])
        self.assertEqual(sum(e['event']=='paused' for e in events),1)
        save.assert_called_once_with(candidate,'suspected success; confirmation timeout')

    def test_unstable_success_candidate_uses_own_deadline_after_sent_draw(self):
        snapshots = []
        for k in range(4):
            snapshots.extend([Observation(tuple([5]*k+[0]*(5-k)),(6,6,6,6,10-k),1,
                                          buttons=('enabled','enabled',None))]*4)
        snapshots.extend(Observation((5,5,5,5,0),(6,6,6,6,7+i%2),1) for i in range(100))
        save = Mock()
        events,clicks,summary = self.run_stream(snapshots,step=.125,save=save)
        self.assertEqual(summary['termination_reason'],'suspected success; confirmation timeout')
        self.assertEqual(len(clicks),4)
        self.assertEqual(summary['cards_confirmed'],3)
        self.assertEqual(sum(e['event']=='paused' for e in events),1)
        self.assertEqual(save.call_args.args[1],'suspected success; confirmation timeout')
        save.assert_called_once()

    def test_stable_success_on_deadline_takes_priority_over_candidate_timeout(self):
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        success = Observation(candidate.slots,(6,6,6,6,6),1)
        save = Mock()
        events,clicks,summary = self.run_stream([candidate]*78+[success]*3,step=.125,save=save)
        self.assertEqual(summary['termination_reason'],'success')
        self.assertEqual(events[-1]['event'],'success')
        self.assertFalse(any(e['event']=='paused' for e in events))
        self.assertEqual(clicks,[])
        save.assert_called_once_with(success,'success')

    def test_f8_pause_freezes_candidate_deadline_with_draw_pending(self):
        snapshots = []
        for k in range(4):
            snapshots.extend([Observation(tuple([5]*k+[0]*(5-k)),(6,6,6,6,10-k),1,
                                          buttons=('enabled','enabled',None))]*4)
        snapshots.extend([Observation((5,5,5,5,0),(6,6,6,6,7),1)]*100)
        save = Mock()
        events,clicks,summary = self.run_stream(snapshots,step=.125,save=save,pause_at=3)
        self.assertEqual(summary['termination_reason'],'source ended')
        self.assertEqual(len(clicks),4)
        self.assertEqual(summary['cards_confirmed'],3)
        self.assertEqual([e['reason'] for e in events if e['event']=='paused'],['F8'])
        self.assertGreater(summary['paused_seconds'],10)
        save.assert_not_called()

    def test_f9_after_sent_click_is_terminal(self):
        obs = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','enabled',None))
        events,clicks,_ = self.run_stream([obs]*30,stop_after_click=True)
        self.assertEqual(len(clicks),1)
        self.assertEqual(events[-1],{'event':'stopped','reason':'F9'})

    def test_f9_at_final_submission_check_does_not_append_pause(self):
        obs = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','enabled',None))
        events,clicks,summary = self.run_stream([obs]*8,stop_at_submission=True)
        self.assertEqual(summary['termination_reason'],'F9')
        self.assertEqual(events[-1],{'event':'stopped','reason':'F9'})
        self.assertFalse(any(e['event']=='paused' for e in events))
        self.assertEqual(summary['paused_seconds'],0)
        self.assertEqual(clicks,[])

    def test_unresponsive_click_pauses_once_and_counts_pause_time(self):
        obs = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','enabled',None))
        events,clicks,summary = self.run_stream([obs]*170)
        self.assertEqual(len(clicks),1)
        self.assertEqual(sum(e['event']=='paused' for e in events),1)
        self.assertGreater(summary['paused_seconds'],8)

    def test_stability_requires_distinct_fresh_frames(self):
        frames = [Frame(1,0,EMPTY), Frame(2,.11,EMPTY), Frame(3,.22,EMPTY)]
        self.assertTrue(stable(frames,.3))
        self.assertFalse(stable([frames[0]]*3,.3))
        self.assertFalse(stable(frames,1))

    def test_stabilizer_freezes_candidate_before_inventory_settles(self):
        s = ObservationStabilizer()
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        self.assertIsNone(s.add(Frame(1,0,candidate), 0))
        self.assertEqual(s.candidate_since,0)
        self.assertIsNone(s.add(Frame(1,.11,candidate), .11))
        self.assertEqual(len(s.frames),0)

    def test_stabilizer_candidate_deadline_excludes_paused_time(self):
        s = ObservationStabilizer()
        candidate = Observation((5,5,5,5,0),(6,6,6,6,7),1)
        s.add(Frame(1,2,candidate),2)
        s.pause(3)
        s.pause(50)
        self.assertFalse(s.candidate_timed_out(100,10))
        s.resume(100)
        self.assertEqual(len(s.frames),0)
        self.assertFalse(s.candidate_timed_out(108.875,10))
        self.assertTrue(s.candidate_timed_out(109,10))

    def test_draw_requires_stock_and_exactly_one_new_card(self):
        self.assertTrue(draw_completed(EMPTY,FIVE))
        self.assertFalse(draw_completed(EMPTY,STALE_STOCK))
        self.assertFalse(draw_completed(EMPTY,Observation((5,5,0,0,0),(6,6,6,6,8),1)))

    def test_pause_preserves_pending_action(self):
        t = ActionTracker()
        self.assertTrue(t.resume())
        t.send(Action.DRAW, EMPTY, lambda _: True)
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
            t.send(Action.CONFIRM, EMPTY, lambda _: True)
        with self.assertRaises(RuntimeError):
            t.send(Action.ABANDON, EMPTY, lambda _: True)
        success = Observation((5,5,5,5,0),(6,6,6,6,6),1)
        with self.assertRaises(RuntimeError):
            t.send(Action.DRAW, success, lambda _: True)

    def test_restart_requires_policy_and_matched_dialog(self):
        t = ActionTracker()
        t.resume()
        one_other = Observation((1,0,0,0,0),(5,6,6,6,10),1)
        with self.assertRaises(RuntimeError):
            t.send(Action.ABANDON, one_other, lambda _: True)
        t.send(Action.ABANDON, one_other, lambda _: True, early_restart=True)
        self.assertFalse(t.reconcile(one_other))
        self.assertTrue(t.reconcile(one_other, expected_abandon_dialog=True))
        t.send(Action.CONFIRM, one_other, lambda _: True)
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
            t.send(Action.DRAW, EMPTY, lambda _: True)
        t.observe(Observation((5,5,5,5,0),(6,6,6,6,6),1))
        self.assertTrue(t.terminated)

    def test_timeout_pauses_without_resending_or_forgetting_action(self):
        now = [0.0]
        t = ActionTracker(clock=lambda: now[0])
        t.resume()
        action_id = t.send(Action.DRAW, EMPTY, lambda _: True)
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

    def test_submission_is_validated_before_backend_and_uncertainty_is_retained(self):
        t = ActionTracker()
        t.resume()
        submit = Mock(return_value=True)
        with self.assertRaises(RuntimeError):
            t.send(Action.CONFIRM, EMPTY, submit)
        submit.assert_not_called()
        cancelled = Mock(return_value=False)
        self.assertIsNone(t.send(Action.DRAW, EMPTY, cancelled))
        self.assertIsNone(t.pending)
        failed = Mock(side_effect=OSError('partial submission'))
        with self.assertRaises(OSError):
            t.send(Action.DRAW, EMPTY, failed)
        self.assertIsNotNone(t.pending)
        with self.assertRaises(RuntimeError):
            t.send(Action.DRAW, EMPTY, failed)
        self.assertEqual(failed.call_count,1)

if __name__ == '__main__': unittest.main()
