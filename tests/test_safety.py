import unittest
import time
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
from ake import drive, frames_from_recording
from ake_core import Observation
from safety import Action, ActionTracker, Frame, ObservationStabilizer, draw_completed, stable
from windowing import WindowIdentity, click

EMPTY = Observation((0,0,0,0,0),(6,6,6,6,10),1)
FIVE = Observation((5,0,0,0,0),(6,6,6,6,9),1)
STALE_STOCK = Observation((5,0,0,0,0),(6,6,6,6,10),1)

class SafetyTests(unittest.TestCase):
    def test_recording_matches_csv_frames_and_releases_video(self):
        with TemporaryDirectory() as folder:
            recording = Path(folder)
            (recording/'frames.csv').write_text('frame,elapsed_seconds\n0,0\n1,0.12\n',encoding='utf-8')
            images = [object(),object()]
            video = Mock()
            video.isOpened.return_value = True
            video.read.side_effect = [(True,image) for image in images]+[(False,None)]
            with patch('cv2.VideoCapture',return_value=video):
                frames = list(frames_from_recording(recording))
            self.assertEqual([(f[0],f[1]) for f in frames],[(0,0),(1,.12)])
            self.assertIs(frames[0][2],images[0])
            self.assertIs(frames[1][2],images[1])
            video.release.assert_called_once()

    def test_recording_rejects_invalid_timeline_or_frame_count(self):
        cases = [('0,0\n2,0.2\n',2), ('0,0.2\n1,0.2\n',2),
                 ('0,0.2\n1,0.1\n',2), ('0,nan\n',1), ('0,inf\n',1),
                 ('0,0\n1,0.12\n2,0.24\n',2), ('0,0\n',2)]
        with TemporaryDirectory() as folder:
            recording = Path(folder)
            for rows,count in cases:
                with self.subTest(rows=rows,video_frames=count):
                    (recording/'frames.csv').write_text('frame,elapsed_seconds\n'+rows,encoding='utf-8')
                    video = Mock()
                    video.isOpened.return_value = True
                    video.read.side_effect = [(True,object()) for _ in range(count)]+[(False,None)]
                    with patch('cv2.VideoCapture',return_value=video), self.assertRaises(ValueError):
                        list(frames_from_recording(recording))
                    video.release.assert_called_once()

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
                   stop_at_submission=False, step=.12, save=None, pause_at=None,
                   sent_times=None, stop_after_confirmation=False, resume_at=None,
                   read_seconds=None, save_seconds=0):
        class Profile:
            regions = {'draw':[0,0,1,1],'abandon':[0,0,1,1],'confirm':[0,0,1,1]}
            def read(self, image):
                if read_seconds: now[0] += read_seconds(image)
                return image,{}
        now = [0.0]
        events,clicks = [],[]
        def frames():
            for i,obs in enumerate(observations):
                now[0] = i*step
                yield i,now[0],obs,True
        started = [False]
        stopped = [False]
        def keys():
            nonlocal pause_at,resume_at
            if not started[0]:
                started[0] = True
                return [0x77]
            if stop_at_submission and stopped[0]:
                return [0x78]
            if stop_after_click and clicks and not stopped[0]:
                stopped[0] = True
                return [0x78]
            if stop_after_confirmation and not stopped[0] and any(e['event']=='action_confirmed' for e in events):
                stopped[0] = True
                return [0x78]
            if pause_at is not None and now[0] >= pause_at:
                pause_at = None
                return [0x77]
            if resume_at is not None and now[0] >= resume_at:
                resume_at = None
                return [0x77]
            return []
        def perform(region, permitted):
            if stop_at_submission: stopped[0] = True
            if not permitted(): return False
            clicks.append(region)
            if sent_times is not None: sent_times.append(now[0])
            return True
        def persist(image,reason):
            now[0] += save_seconds
            return save(image,reason)
        with patch('ake.time.monotonic',side_effect=lambda:now[0]):
            summary = drive(Profile(),frames(),mode=mode,emit=events.append,
                            keys=keys if mode=='live' else None,perform=perform,save=persist if save else None)
        self.assertIs(events[-1],summary)
        self.assertEqual(sum(e['event']=='summary' for e in events),1)
        return events[:-1],clicks,summary

    def test_source_interruptions_preserve_summary_and_propagate(self):
        class Profile:
            def read(self,image): return EMPTY,{}
        for error,reason in [(KeyboardInterrupt(),'Ctrl+C'),(OSError('capture failed'),'error: capture failed')]:
            with self.subTest(error=type(error).__name__):
                events = []
                def frames():
                    for i in range(3): yield i,i*.125,None,True
                    raise error
                with patch('ake.time.monotonic',return_value=.25),self.assertRaises(type(error)):
                    drive(Profile(),frames(),mode='dry-run',emit=events.append)
                summaries = [e for e in events if e['event']=='summary']
                self.assertEqual(len(summaries),1)
                self.assertEqual(summaries[0]['termination_reason'],reason)
                self.assertEqual(summaries[0]['attempts_confirmed'],1)
                self.assertEqual(summaries[0]['draw_clicks_sent'],0)

    def test_capture_failure_preserves_unconfirmed_action_in_summary(self):
        obs = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','enabled',None))
        class Profile:
            regions = {'draw':[0,0,1,1]}
            def read(self,image): return obs,{}
        def frames():
            for i in range(3): yield i,i*.125,None,True
            raise OSError('capture failed')
        events = []
        keys = Mock(side_effect=[[0x77],[],[],[],[]])
        def perform(region,permitted): return permitted()
        with patch('ake.time.monotonic',return_value=.25),self.assertRaises(OSError):
            drive(Profile(),frames(),mode='live',emit=events.append,keys=keys,perform=perform)
        summary = events[-1]
        self.assertEqual(summary['event'],'summary')
        self.assertEqual(summary['draw_clicks_sent'],1)
        self.assertEqual(summary['cards_confirmed'],0)
        self.assertEqual(summary['pending_action'],{'id':1,'action':'draw'})

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

    def test_confirmed_draw_continues_on_the_same_stable_observation(self):
        empty = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','disabled',None))
        five = Observation(FIVE.slots,FIVE.inventory,1,buttons=('enabled','enabled',None))
        sent = []
        _,clicks,summary = self.run_stream([empty]*3+[five]*3,step=.125,sent_times=sent)
        self.assertEqual(sent,[.25,.625])
        self.assertEqual(len(clicks),2)
        self.assertEqual(summary['cards_confirmed'],1)
        self.assertEqual(summary['timings']['draw']['mean'],.375)

    def test_cycle_timing_uses_completed_observations_instead_of_capture_stamps(self):
        empty = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','disabled',None))
        five = Observation(FIVE.slots,FIVE.inventory,1,buttons=('enabled','enabled',None))
        snapshots = [empty]*3+[five]*3
        for before,after,expected in ((.02,.2,1.08),(.2,.02,.72)):
            with self.subTest(read_seconds=(before,after)):
                sent = []
                _,clicks,summary = self.run_stream(snapshots,step=.3,sent_times=sent,
                    read_seconds=lambda obs:before if obs == empty else after)
                self.assertEqual(len(clicks),2)
                self.assertAlmostEqual(sent[1]-sent[0],expected)
                self.assertAlmostEqual(summary['timings']['draw']['mean'],expected)
                self.assertAlmostEqual(summary['elapsed'],expected)
        _,clicks,summary = self.run_stream(snapshots,mode='replay',step=.3,
            read_seconds=lambda obs:.02 if obs == empty else .2)
        self.assertEqual(clicks,[])
        self.assertAlmostEqual(summary['elapsed'],.9)

    def test_success_evidence_time_and_failures_preserve_terminal_result(self):
        snapshots = []
        for k in range(5):
            snapshots.extend([Observation(tuple([5]*k+[0]*(5-k)),(6,6,6,6,10-k),1,
                                          buttons=('enabled','enabled',None))]*3)
        for error in (None,OSError('disk full'),KeyboardInterrupt()):
            with self.subTest(save_error=type(error).__name__):
                save = Mock(return_value='fixture-evidence',side_effect=error)
                events,clicks,summary = self.run_stream(snapshots,step=.3,save=save,
                    save_seconds=2,read_seconds=lambda _: .02)
                self.assertEqual(summary['termination_reason'],'success')
                self.assertEqual(len(clicks),4)
                self.assertEqual(summary['cards_confirmed'],4)
                self.assertAlmostEqual(summary['elapsed'],3.6)
                self.assertAlmostEqual(summary['evidence_seconds'],2)
                save.assert_called_once()
                successes = [event for event in events if event['event']=='success']
                self.assertEqual(len(successes),1)
                if error is None:
                    self.assertIsNone(summary['evidence_error'])
                    self.assertEqual(successes[0]['evidence'],'fixture-evidence')
                else:
                    self.assertIsNotNone(summary['evidence_error'])
                    self.assertIsNone(successes[0]['evidence'])

    def test_confirmed_dialog_and_reset_continue_without_unmeasured_gaps(self):
        empty = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','disabled',None))
        one = Observation((1,0,0,0,0),(5,6,6,6,10),1,buttons=('enabled','enabled',None))
        failed = Observation((1,2,0,0,0),(5,5,6,6,10),1,buttons=('enabled','enabled',None))
        dialog = Observation((None,)*5,(None,)*5,None,page='abandon_dialog',buttons=(None,None,'enabled'))
        draw_wait = Observation(one.slots,one.inventory,1,buttons=('disabled','enabled',None))
        reset_wait = Observation(failed.slots,failed.inventory,1,buttons=('enabled','disabled',None))
        cases = [('ready',(empty,one,failed,dialog,empty),[.25,.625,1,1.375,1.75],.75,.75),
                 ('draw wait',(empty,draw_wait,one,failed,dialog,empty),[.25,1,1.375,1.75,2.125],1.125,.75),
                 ('reset wait',(empty,one,reset_wait,failed,dialog,empty),[.25,.625,1.375,1.75,2.125],.75,1.125)]
        for name,states,expected_times,draw_time,reset_time in cases:
            with self.subTest(readiness=name):
                snapshots = [obs for obs in states for _ in range(3)]
                sent = []
                events,_,summary = self.run_stream(snapshots,step=.125,sent_times=sent)
                self.assertEqual(sent,expected_times)
                self.assertEqual([e['action'] for e in events if e['event']=='input_sent'],
                                 ['draw','draw','abandon','confirm','draw'])
                self.assertEqual(summary['restarts_completed'],1)
                draw,reset = summary['timings']['draw'],summary['timings']['restart']
                self.assertEqual(draw['count'],2)
                self.assertEqual(reset['count'],1)
                self.assertEqual(draw['mean']*draw['count'],draw_time)
                self.assertEqual(reset['mean']*reset['count'],reset_time)
                self.assertEqual(draw_time+reset_time,summary['elapsed'])

    def test_f9_after_confirmation_prevents_same_frame_followup(self):
        empty = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','disabled',None))
        five = Observation(FIVE.slots,FIVE.inventory,1,buttons=('enabled','enabled',None))
        events,clicks,summary = self.run_stream([empty]*3+[five]*3,step=.125,stop_after_confirmation=True)
        self.assertEqual(len(clicks),1)
        self.assertEqual(summary['cards_confirmed'],1)
        self.assertEqual(summary['termination_reason'],'F9')
        self.assertEqual(events[-1],{'event':'stopped','reason':'F9'})

    def test_readiness_timeout_covers_unknown_unstable_and_disabled_states(self):
        unknown = Observation((None,)*5,(None,)*5,None,page='unknown')
        disabled = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('disabled','disabled',None))
        incomplete = Observation((None,0,0,0,0),EMPTY.inventory,1,buttons=disabled.buttons)
        for name,snapshots in [('unknown',[unknown]*90),('unstable',[incomplete,disabled]*45),('disabled',[disabled]*90)]:
            with self.subTest(state=name):
                saved_at = []
                save = Mock(side_effect=lambda *_:saved_at.append(time.monotonic()))
                events,clicks,_ = self.run_stream(snapshots,step=.125,save=save)
                self.assertEqual(clicks,[])
                self.assertEqual([e['reason'] for e in events if e['event']=='paused'],
                                 ['observation or action readiness timed out'])
                self.assertEqual(saved_at,[10])
                save.assert_called_once()

    def test_readiness_deadline_restarts_after_explicit_pause_and_resume(self):
        unknown = Observation((None,)*5,(None,)*5,None,page='unknown')
        saved_at = []
        save = Mock(side_effect=lambda *_:saved_at.append(time.monotonic()))
        events,clicks,_ = self.run_stream([unknown]*250,step=.125,save=save,pause_at=3,resume_at=20)
        self.assertEqual(clicks,[])
        self.assertEqual([e['reason'] for e in events if e['event']=='paused'],
                         ['F8','observation or action readiness timed out'])
        self.assertEqual(saved_at,[30])
        save.assert_called_once()

    def test_confirmed_draw_refreshes_readiness_deadline_for_disabled_button(self):
        empty = Observation(EMPTY.slots,EMPTY.inventory,1,buttons=('enabled','disabled',None))
        disabled = Observation(FIVE.slots,FIVE.inventory,1,buttons=('disabled','enabled',None))
        saved_at = []
        save = Mock(side_effect=lambda *_:saved_at.append(time.monotonic()))
        events,clicks,summary = self.run_stream([empty]*3+[disabled]*100,step=.125,save=save)
        self.assertEqual(len(clicks),1)
        self.assertEqual(summary['cards_confirmed'],1)
        self.assertEqual([e['reason'] for e in events if e['event']=='paused'],
                         ['observation or action readiness timed out'])
        self.assertEqual(saved_at,[10.625])
        save.assert_called_once()

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
