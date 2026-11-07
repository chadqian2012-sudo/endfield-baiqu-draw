"""Windows CLI for calibrated observation, replay and qualified automation."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time
from math import isfinite
from ake_core import Observation, Decision, chance, decide, valid_observation
from math_verify import threshold, totals
from safety import Action, ActionTracker, Frame, ObservationStabilizer, initial, draw_completed
from timing import Timings

VERSION = '0.2.0'
REJECTION_CASES = {'unknown_page','unknown_slot','unknown_inventory'}

def fingerprint(profile):
    digest = hashlib.sha256(profile.path.read_bytes())
    for group in sorted(profile.groups):
        for label, samples in sorted(profile.groups[group][0].items()):
            digest.update((group+'/'+label).encode())
            for image in samples:
                digest.update(str(image.shape).encode())
                digest.update(image.tobytes())
    for name, (_, image) in sorted(profile.anchors.items()):
        digest.update(name.encode())
        digest.update(image.tobytes())
    return digest.hexdigest()

def dump(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)

class StrictLog(RotatingFileHandler):
    def handleError(self, record):
        raise RuntimeError('critical log write failed') from sys.exc_info()[1]

def evidence(profile, image, reason):
    from vision import write_image
    target = Path('logs')/'evidence'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    target.mkdir(parents=True)
    for name in ('slots','inventory'):
        for index,roi in enumerate(profile.regions.get(name, [])):
            write_image(target/f'{name}-{index+1}.png',profile.crop(image,roi))
    (target/'reason.txt').write_text(reason,encoding='utf-8')
    return str(target)

def frames_from_recording(recording):
    import cv2
    video = cv2.VideoCapture(str(recording/'observation.mp4'))
    if not video.isOpened(): raise ValueError('cannot open recording')
    try:
        with (recording/'frames.csv').open(encoding='utf-8',newline='') as f:
            previous_time = None
            for index,row in enumerate(csv.DictReader(f)):
                sequence,stamp = int(row['frame']),float(row['elapsed_seconds'])
                if sequence != index or not isfinite(stamp) or stamp < 0 or (previous_time is not None and stamp <= previous_time):
                    raise ValueError('invalid recording frame index or timestamp')
                ok,image = video.read()
                if not ok: raise ValueError('video and timestamp log differ')
                previous_time = stamp
                yield sequence,stamp,image,True
            if video.read()[0]: raise ValueError('video and timestamp log differ')
    finally:
        video.release()

def frames_from_window(binding, seconds):
    import mss
    import numpy as np
    from windowing import snapshot, foreground_matches
    start = time.monotonic()
    x,y,w,h = binding.client
    with mss.mss() as capture:
        sequence = 0
        while seconds is None or time.monotonic()-start < seconds:
            tick = time.monotonic()
            if snapshot(binding.hwnd) != binding:
                raise RuntimeError('window identity, geometry or DPI changed; recalibrate')
            focused = foreground_matches(binding)
            image = np.asarray(capture.grab({'left':x,'top':y,'width':w,'height':h}))[:,:,:3].copy() if focused else None
            yield sequence,tick,image,focused
            sequence += 1
            time.sleep(max(0,.12-(time.monotonic()-tick)))

def drive(profile, frames, *, mode, emit, keys=None, perform=None, save=None):
    """Dry modes never call perform. The main thread owns stop and input order."""
    now = [0.0]
    tracker = ActionTracker(clock=lambda:now[0])
    stabilizer = ObservationStabilizer()
    timings = Timings()
    last_hand = None
    last_observation = None
    last_output = None
    entered = False
    cycle_started = reset_started = None
    cycle_clean = True
    paused_started = None
    pause_total = 0.0
    active_started = None
    last_policy = None
    reason = 'source ended'
    counters = {'draw_clicks_sent':0,'cards_confirmed':0,'attempts_confirmed':0,'restarts_completed':0}

    def pause(reason):
        nonlocal cycle_clean,paused_started
        tracker.pause()
        stabilizer.pause(now[0])
        cycle_clean = False
        if paused_started is None and active_started is not None:
            paused_started = now[0]
        emit({'event':'paused','reason':reason})

    def handle_keys():
        nonlocal cycle_clean,paused_started,pause_total,reason
        pressed = keys() if keys else []
        if 0x78 in pressed:
            tracker.stop()
            reason = 'F9'
            emit({'event':'stopped','reason':'F9'})
        elif 0x77 in pressed:
            cycle_clean = False
            if tracker.paused:
                if tracker.resume():
                    if paused_started is not None:
                        pause_total += now[0]-paused_started
                        paused_started = None
                    stabilizer.resume(now[0])
                    emit({'event':'resumed'})
                else:
                    emit({'event':'paused','reason':'pending action or success candidate requires reconciliation'})
            else:
                pause('F8')

    def new_attempt():
        nonlocal last_policy
        counters['attempts_confirmed'] += 1
        timings.begin_attempt()
        if last_policy != timings.early_restart:
            emit({'event':'strategy','attempt':counters['attempts_confirmed'],'early_restart':timings.early_restart})
            last_policy = timings.early_restart

    if mode == 'live': emit({'event':'armed','message':'F8 start/pause, F9 terminal stop'})
    try:
        for sequence,stamp,image,focused in frames:
            now[0] = stamp
            handle_keys()
            if tracker.terminated: break
            if not focused:
                stabilizer.frames.clear()
                if mode == 'live': stabilizer.pause(stamp)
                if mode == 'live' and not tracker.paused:
                    pause('focus lost; F8 required after reconciliation')
                continue
            obs,quality = profile.read(image)
            last_observation = obs
            stable_obs = stabilizer.add(Frame(sequence,stamp,obs),stamp if mode == 'replay' else time.monotonic())
            tracker.candidate = stabilizer.candidate_since is not None
            decision = decide(stable_obs,timings.early_restart) if stable_obs is not None else Decision.WAIT
            if decision == Decision.SUCCESS:
                if tracker.pending and tracker.pending[1] == Action.DRAW and draw_completed(tracker.pending[2],obs):
                    tracker.reconcile(obs)
                    counters['cards_confirmed'] += 1
                    timings.add('draw',stamp-cycle_started,uninterrupted=cycle_clean)
                tracker.stop()
                reason = 'success'
                path = save(image,'success') if save else None
                emit({'event':'success','observation':asdict(obs),'evidence':path})
                break
            if mode == 'live' and stabilizer.candidate_timed_out(stamp,tracker.timeout):
                reason = 'suspected success; confirmation timeout'
                pause(reason)
                if save: save(image,reason)
                break
            if stable_obs is None:
                if mode == 'live' and not tracker.candidate and tracker.check_timeout():
                    pause(tracker.error)
                    if save: save(image,tracker.error)
                continue
            if tracker.candidate:
                if last_output != 'candidate':
                    emit({'event':'suspected_success','reason':'input frozen; awaiting consistent inventory'})
                    last_output = 'candidate'
                continue
            if not entered:
                if initial(obs):
                    entered = True
                    last_hand = obs
                    active_started = stamp
                    if mode == 'live' and tracker.paused: paused_started = stamp
                    new_attempt()
            if mode != 'live':
                semantic = (obs,decision,entered)
                if semantic != last_output:
                    emit({'event':'observation' if mode == 'observe' else 'recommendation',
                          'observation':asdict(obs),'decision':decision.value if entered else 'wait','quality':quality})
                    last_output = semantic
                continue
            if not entered and valid_observation(obs):
                if last_output != 'takeover':
                    emit({'event':'takeover_refused','reason':'nonempty initial hand; return manually to empty round 1'})
                    last_output = 'takeover'
                tracker.pause()
                continue
            if tracker.pending:
                _,action,_ = tracker.pending
                dialog = obs.page == 'abandon_dialog' and obs.buttons[2] == 'enabled'
                if tracker.reconcile(obs,expected_abandon_dialog=dialog):
                    emit({'event':'action_confirmed','action':action.value,'observation':asdict(obs)})
                    if action == Action.DRAW:
                        counters['cards_confirmed'] += 1
                        timings.add('draw',stamp-cycle_started,uninterrupted=cycle_clean)
                        last_hand = obs
                    elif action == Action.CONFIRM:
                        counters['restarts_completed'] += 1
                        timings.add('restart',stamp-reset_started,uninterrupted=cycle_clean)
                        last_hand = obs
                        new_attempt()
                elif tracker.check_timeout():
                    pause(tracker.error)
                    if save: save(image,tracker.error)
                continue
            if tracker.paused or not entered: continue
            if tracker.phase == 'dialog':
                if obs.page != 'abandon_dialog' or obs.buttons[2] != 'enabled':
                    pause('expected abandon dialog no longer aligned')
                    continue
                action,context,button = Action.CONFIRM,last_hand,'confirm'
            else:
                if (not valid_observation(obs) or obs.slots != last_hand.slots or obs.inventory != last_hand.inventory):
                    pause('unexplained or unknown state; manual reconciliation required')
                    continue
                action = Action.DRAW if decision == Decision.DRAW else Action.ABANDON
                context,button = obs,'draw' if action == Action.DRAW else 'abandon'
                if obs.buttons[0 if action == Action.DRAW else 1] != 'enabled': continue
            handle_keys()
            if tracker.paused or tracker.terminated or time.monotonic()-stamp > .5: continue
            cycle_started = stamp
            if action == Action.ABANDON:
                reset_started = stamp
                cycle_clean = True
            elif action == Action.DRAW:
                cycle_clean = True
            def permitted():
                handle_keys()
                return not tracker.paused and not tracker.terminated and not tracker.candidate and time.monotonic()-stamp <= .5
            def submit(action_id):
                emit({'event':'input_intent','id':action_id,'action':action.value,
                      'attempt':counters['attempts_confirmed'],'observation':asdict(obs),'timings':timings.summary()})
                return perform(profile.regions[button],permitted)
            action_id = tracker.send(action,context,submit,early_restart=timings.early_restart)
            if action_id is None:
                if tracker.terminated: break
                pause('input cancelled: stop, pause, stale observation or focus change')
                continue
            if action == Action.DRAW: counters['draw_clicks_sent'] += 1
            emit({'event':'input_sent','id':action_id,'action':action.value})
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
        raise
    except Exception as error:
        reason = 'error: '+str(error)
        raise
    finally:
        if paused_started is not None: pause_total += now[0]-paused_started
        result = {'event':'summary','termination_reason':reason,**counters,'timings':timings.summary(),
                  'elapsed':None if active_started is None else now[0]-active_started,'paused_seconds':pause_total,
                  'last_observation':asdict(last_observation) if last_observation is not None else None,
                  'pending_action':{'id':tracker.pending[0],'action':tracker.pending[1].value} if tracker.pending else None}
        emit(result)
    return result

def validate(profile, dataset):
    """Evaluate independently labelled batches and observed single-draw/early-reset paths."""
    from vision import read_image
    import cv2
    coverage = {name:set() for name in profile.CLASSES}
    results = {name:{} for name in profile.CLASSES}
    results['stock_values'] = {}
    errors = []
    history = []
    total = 0
    rejection_cases = set()
    training = set(profile.config.get('training_batches',[]))
    if not training: errors.append('profile must name training_batches')
    with dataset.open(encoding='utf-8-sig') as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if not row.get('batch') or row['batch'] in training:
                errors.append(f'row {total}: missing or overlapping batch')
            if row.get('source') != 'real':
                errors.append(f'row {total}: real-client evidence required for live qualification')
            image = read_image(dataset.parent/row['image'],cv2.IMREAD_COLOR)
            if image is None: raise ValueError('missing validation image')
            actual,_ = profile.read(image)
            expected = row['expected']
            if set(expected) != {'slots','inventory','game_round','page','buttons'}:
                errors.append(f'row {total}: incomplete annotation')
                continue
            observed = asdict(actual)
            for key,value in expected.items():
                if json.loads(json.dumps(observed[key])) != value:
                    errors.append(f'row {total}: {key} mismatch')
            pairs = [('cards','empty' if v == 0 else str(v),actual.slots[i]) for i,v in enumerate(expected['slots']) if v is not None]
            pairs += [('stock_values',str(v),actual.inventory[i]) for i,v in enumerate(expected['inventory']) if v is not None]
            pairs += [('round',str(expected['game_round']),actual.game_round)] if expected['game_round'] is not None else []
            pairs += [(name,v,actual.buttons[i]) for i,(name,v) in enumerate(zip(('draw','abandon','confirm'),expected['buttons'])) if v is not None]
            for group,label,value in pairs:
                count = results[group].setdefault(label,{'total':0,'correct':0,'rejected':0,'wrong':0})
                predicted = 'empty' if group == 'cards' and value == 0 else str(value) if value is not None else None
                count['total'] += 1
                count['correct' if predicted == label else 'rejected' if predicted is None else 'wrong'] += 1
            obs = Observation(tuple(expected['slots']),tuple(expected['inventory']),expected['game_round'],
                              page=expected['page'],buttons=tuple(expected['buttons']))
            history.append((row.get('batch'),obs))
            if obs.page == 'unknown': rejection_cases.add('unknown_page')
            if obs.page == 'hand':
                if None in obs.slots: rejection_cases.add('unknown_slot')
                if None in obs.inventory: rejection_cases.add('unknown_inventory')
                coverage['cards'].update('empty' if v == 0 else str(v) for v in obs.slots if v is not None)
                coverage['stock'].update(''.join(str(v) for v in obs.inventory if v is not None))
                if obs.game_round is not None: coverage['round'].add(str(obs.game_round))
            for i,name in enumerate(('draw','abandon','confirm')):
                if obs.buttons[i] is not None: coverage[name].add(obs.buttons[i])
    early_reset = any(ba == bb == bc and valid_observation(a) and 0 < sum(v != 0 for v in a.slots) < 5
                      and b.page == 'abandon_dialog' and initial(c)
                      for (ba,a),(bb,b),(bc,c) in zip(history,history[1:],history[2:]))
    single_draw = any(ba == bb and draw_completed(a,b) for (ba,a),(bb,b) in zip(history,history[1:]))
    for name,classes in profile.CLASSES.items():
        missing = classes-coverage[name]
        if missing: errors.append(name+': validation missing '+','.join(sorted(missing)))
    if not early_reset: errors.append('no labelled early-abandon/dialog/full-reset sequence')
    if not single_draw: errors.append('no labelled single-draw transition')
    for case in sorted(REJECTION_CASES-rejection_cases):
        errors.append('validation missing rejection case: '+case)
    errors.extend(profile.issues())
    return {'passed':not errors,'profile_sha256':fingerprint(profile),'frames':total,'errors':errors,
            'coverage':{name:sorted(values) for name,values in coverage.items()},'class_results':results,
            'rejection_cases':sorted(rejection_cases)}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version=VERSION)
    commands = p.add_subparsers(dest='command',required=True)
    commands.add_parser('math')
    a = commands.add_parser('analyze')
    a.add_argument('--slots',type=int,nargs=5,required=True)
    a.add_argument('--inventory',type=int,nargs=5,required=True)
    a.add_argument('--round',type=int,required=True)
    a.add_argument('--early-restart',action='store_true')
    c = commands.add_parser('calibrate')
    c.add_argument('--layout',type=Path)
    c.add_argument('--output',type=Path,default=Path('calibration.local.json'))
    c = commands.add_parser('sample')
    c.add_argument('--profile',type=Path,required=True)
    c.add_argument('--group',choices=('cards','stock','round','draw','abandon','confirm','hand','abandon_dialog'),required=True)
    c.add_argument('--label')
    c.add_argument('--index',type=int,choices=range(5),default=0)
    c.add_argument('--digit',type=int,choices=range(2),default=0)
    c.add_argument('--score',type=float,required=True)
    c.add_argument('--margin',type=float,required=True)
    c.add_argument('--batch',required=True)
    sources = c.add_mutually_exclusive_group()
    sources.add_argument('--image',type=Path)
    sources.add_argument('--recording',type=Path)
    c.add_argument('--at',type=float,default=0)
    c = commands.add_parser('validate')
    c.add_argument('--profile',type=Path,required=True)
    c.add_argument('--dataset',type=Path,required=True)
    c.add_argument('--output',type=Path,default=Path('logs/qualification.json'))
    for name in ('observe','run','replay'):
        c = commands.add_parser(name)
        c.add_argument('--profile',type=Path,required=True)
        if name == 'replay': c.add_argument('recording',type=Path)
        else: c.add_argument('--seconds',type=float)
        if name == 'run':
            group = c.add_mutually_exclusive_group()
            group.add_argument('--dry-run',action='store_true')
            group.add_argument('--live',action='store_true')
            c.add_argument('--qualification',type=Path)
    args = p.parse_args(argv)
    if args.command == 'math':
        for name,early in [('baseline',False),('early_restart',True)]:
            probability,draws,total_draws,restarts = totals(early)
            dump({'strategy':name,'probability':str(probability),'draws_per_attempt':float(draws),
                  'expected_draws':float(total_draws),'expected_restarts':float(restarts)})
        dump({'threshold':str(threshold()),'model':'uniform without replacement; independent identical resets'})
        return 0
    if args.command == 'analyze':
        obs = Observation(tuple(args.slots),tuple(args.inventory),args.round)
        decision = decide(obs,args.early_restart)
        output = {'decision':decision.value,'source':'manual input; no game state verified'}
        if decision != Decision.WAIT:
            output['continue_without_voluntary_restart_probability'] = str(chance(obs.slots.count(5),sum(v not in (0,5) for v in obs.slots)))
        dump(output)
        return 0
    from vision import VisionProfile
    if args.command == 'calibrate':
        from windowing import find_unique
        binding = find_unique()
        if args.layout is None:
            import cv2
            dump({'event':'awaiting_foreground','message':'switch to game within 10 seconds'})
            frame = next((image for _,_,image,focused in frames_from_window(binding,10) if focused),None)
            if frame is None: raise ValueError('foreground game capture timed out')
            config = {'frame_size':list(binding.client[2:]),'regions':{},'groups':{},'anchors':{},'training_batches':[]}
            try:
                for name,count in [('slots',5),('inventory',5),('round',1),('draw',1),('abandon',1),('confirm',1),('hand',1),('abandon_dialog',1)]:
                    regions = []
                    for i in range(count):
                        title = f'{name} {i+1}: select region, ENTER accept, ESC cancel'
                        roi = tuple(map(int,cv2.selectROI(title,frame,showCrosshair=True)))
                        cv2.destroyAllWindows()
                        if min(roi[2:]) <= 0: raise ValueError('calibration cancelled')
                        regions.append(roi)
                    config['regions'][name] = regions if count == 5 else regions[0]
            finally:
                cv2.destroyAllWindows()
            config['window'] = asdict(binding)
            args.output.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
            dump({'calibration':str(args.output),'issues':VisionProfile(args.output).issues()})
            return 0
        profile = VisionProfile(args.layout)
        if profile.size != binding.client[2:]: raise ValueError('layout size does not match game client')
        config = profile.config
        for group in config.get('groups',{}).values():
            for label,paths in group.get('templates',{}).items():
                group['templates'][label] = [str((profile.path.parent/path).resolve()) for path in paths]
        for anchor in config.get('anchors',{}).values():
            anchor['template'] = str((profile.path.parent/anchor['template']).resolve())
        config['window'] = asdict(binding)
        args.output.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        dump({'calibration':str(args.output),'issues':profile.issues()})
        return 0
    profile = VisionProfile(args.profile)
    if args.command == 'sample':
        import cv2
        from vision import read_image
        if args.image:
            image = read_image(args.image,cv2.IMREAD_COLOR)
            if image is None: raise ValueError('cannot read sample image')
        elif args.recording:
            image = None
            distance = float('inf')
            for _,stamp,frame,_ in frames_from_recording(args.recording):
                if abs(stamp-args.at) < distance:
                    image,distance = frame,abs(stamp-args.at)
            if image is None: raise ValueError('recording has no frames')
        else:
            from windowing import find_unique
            binding = find_unique()
            if json.loads(json.dumps(asdict(binding))) != profile.config.get('window'):
                raise ValueError('sample requires calibrated window')
            dump({'event':'awaiting_foreground','message':'switch to game within 10 seconds'})
            image = next((image for _,_,image,focused in frames_from_window(binding,10) if focused),None)
            if image is None: raise ValueError('foreground game capture timed out')
        profile.add_sample(image,args.group,args.label,args.index,args.digit,args.score,args.margin,args.batch)
        dump({'sample_saved':True,'issues':VisionProfile(args.profile).issues()})
        return 0
    if args.command == 'validate':
        report = validate(profile,args.dataset)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        dump(report)
        return 0 if report['passed'] else 2
    live = args.command == 'run' and args.live
    if live:
        problems = profile.issues()
        if args.qualification is None:
            problems.append('independent validation report required')
        else:
            report = json.loads(args.qualification.read_text(encoding='utf-8-sig'))
            if report.get('passed') is not True or report.get('profile_sha256') != fingerprint(profile):
                problems.append('validation failed or profile/templates changed')
            if not REJECTION_CASES.issubset(report.get('rejection_cases',[])):
                problems.append('validation report lacks rejection cases; rerun validate')
        if problems:
            dump({'event':'live_blocked','reasons':problems})
            return 2
    keys = handler = last_image = None
    try:
        Path('logs').mkdir(exist_ok=True)
        handler = StrictLog('logs/events.jsonl',maxBytes=2_000_000,backupCount=3,encoding='utf-8')
        def emit(event):
            event = {'time':datetime.now().astimezone().isoformat(),'monotonic':time.monotonic(),**event}
            handler.emit(logging.LogRecord('ake',logging.INFO,'',0,json.dumps(event,ensure_ascii=False),(),None))
            dump(event)
        if args.command == 'replay':
            frames = frames_from_recording(args.recording)
            mode,perform = 'replay',None
        else:
            from windowing import find_unique, Hotkeys, click
            binding = find_unique()
            expected = profile.config.get('window')
            if not expected or json.loads(json.dumps(asdict(binding))) != expected:
                raise ValueError('window is not the calibrated process; run calibrate again')
            frames = frames_from_window(binding,args.seconds)
            mode = 'live' if live else 'observe' if args.command == 'observe' else 'dry-run'
            if live: keys = Hotkeys()
            perform = (lambda region,permitted:click(binding,region,permitted)) if live else None
        def remember():
            nonlocal last_image
            for frame in frames:
                if frame[2] is not None: last_image = frame[2]
                yield frame
        drive(profile,remember(),mode=mode,emit=emit,keys=keys.poll if keys else None,
                        perform=perform,save=(lambda image,reason:evidence(profile,image,reason)) if live else None)
        if live:
            import winsound
            winsound.MessageBeep()
        return 0
    except KeyboardInterrupt:
        dump({'event':'stopped','reason':'Ctrl+C'})
        return 0
    except Exception as error:
        dump({'event':'error','reason':str(error)})
        if live and last_image is not None:
            try: dump({'evidence':evidence(profile,last_image,str(error))})
            except Exception as save_error: dump({'evidence_error':str(save_error)})
        return 1
    finally:
        if keys: keys.close()
        if handler: handler.close()

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,KeyError) as error:
        dump({'event':'error','reason':str(error)})
        sys.exit(1)
