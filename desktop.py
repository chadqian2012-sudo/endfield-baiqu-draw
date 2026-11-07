"""Double-click entry for the same validated controller used by the CLI."""
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, ttk


def launch(main):
    home = Path(sys.executable if getattr(sys, 'frozen', False) else __file__).resolve().parent
    os.chdir(home)
    root = tk.Tk()
    root.title('百趣演武 · 抽牌助手')
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=20)
    panel.grid()

    settings = Path('logs/desktop-settings.json')
    bound = Path('logs/desktop-profile.json')
    report = Path('logs/qualification.json')
    paths = {'profile': Path('calibration.local.json'),
             'validation': report if report.is_file() else Path('independent.jsonl')}
    settings_error = ''
    if settings.is_file():
        try:
            saved = json.loads(settings.read_text(encoding='utf-8'))
            paths = {key: Path(saved[key]) for key in paths}
        except (OSError, ValueError, KeyError, TypeError) as error:
            settings_error = f'无法读取上次的选择，请重新选择资料。\n{error}'

    title = tk.StringVar(value='准备开始')
    detail = tk.StringVar()
    files = tk.StringVar()
    ttk.Label(panel, textvariable=title, font=('Microsoft YaHei UI', 16, 'bold')).grid(
        row=0, column=0, sticky='w')
    ttk.Label(panel, textvariable=detail, wraplength=460, justify='left').grid(
        row=1, column=0, sticky='w', pady=(6, 12))
    controls = ttk.Frame(panel)
    controls.grid(row=2, column=0, sticky='w')
    ttk.Separator(panel).grid(row=3, column=0, sticky='ew', pady=12)
    ttk.Label(panel, textvariable=files, wraplength=460, justify='left').grid(
        row=4, column=0, sticky='w')
    setup = ttk.Frame(panel)
    setup.grid(row=5, column=0, sticky='w', pady=(6, 0))
    ttk.Label(panel, text='游戏内 F8 开始或暂停　·　F9 结束', foreground='#555555').grid(
        row=6, column=0, sticky='w', pady=(14, 0))

    messages = Queue()
    stop = Event()
    working = None
    closing = False
    success_text = ''

    def refresh():
        profile_ok, validation_ok = (paths[key].is_file() for key in ('profile', 'validation'))
        files.set('识别配置　' + (paths['profile'].name if profile_ok else '待选择') +
                  '\n验收素材　' + (paths['validation'].name if validation_ok else '待选择'))
        for button in setup.winfo_children():
            button.configure(state='disabled' if working else 'normal')
        rebind.configure(state='normal' if profile_ok and not working else 'disabled')
        start.configure(state='normal' if profile_ok and validation_ok and not working else 'disabled')
        capture.configure(state='disabled' if working else 'normal')
        end.configure(state='normal' if working and not stop.is_set() else 'disabled')
        return profile_ok and validation_ok

    def remember(key, path):
        from vision import VisionProfile
        selection = {**paths, key: path}
        settings.parent.mkdir(exist_ok=True)
        VisionProfile.save_config(settings, {k: str(v) for k, v in selection.items()})
        paths.update(selection)

    def ready():
        if refresh():
            title.set('可以检查抽牌页')
            detail.set('打开游戏第一轮的空牌面。点「开始抽牌」，通过检查后切回游戏按 F8。')
        elif paths['profile'].is_file():
            title.set('缺少独立验证素材')
            detail.set('已有标注集或验收报告，请点「选择验收素材」。需要补录时，点「录制素材」。录像完成后还需标注并核对。')
        else:
            title.set('首次使用需要游戏识别资料')
            detail.set('已有资料，请在下方选择。缺少资料，请点「录制素材」，手动演示抽牌和放弃重开。录像需完成标注与核验后才能抽牌。')

    def choose(key):
        formats = ([('识别配置', '*.json')] if key == 'profile' else
                   [('标注集或报告', '*.jsonl *.json')])
        name = filedialog.askopenfilename(parent=root, title='选择识别配置' if key == 'profile' else '选择独立标注集或验收报告',
                                          filetypes=formats)
        if name:
            try:
                remember(key, Path(name))
                ready()
            except Exception as error:
                title.set('选择未保存')
                detail.set(str(error))

    def run(operation):
        nonlocal working, success_text
        if working:
            return
        working = operation
        success_text = ''
        stop.clear()
        refresh()
        if operation == 'record':
            title.set('请在 30 秒内切回游戏')
            detail.set('从第一轮空牌面开始，手动抽一张，再演示放弃、确认和回到空牌面。每步等画面稳定。出现四张 5 时保留牌面。')
        elif operation == 'bind':
            title.set('正在核对新的游戏窗口')
            detail.set('使用已选配置核对游戏程序、分辨率与缩放。')
        else:
            title.set('正在检查识别资料')
            detail.set('请准备好第一轮空牌面。检查结束后会提示按 F8。')

        def worker():
            try:
                if operation == 'record':
                    from collect_observation import record
                    result = record(on_start=lambda path: messages.put(('recording', path)),
                                    stopped=stop.is_set)
                elif operation == 'bind':
                    bound.parent.mkdir(exist_ok=True)
                    result = main(['calibrate', '--layout', str(paths['profile']), '--rebind',
                                   '--output', str(bound)], output=lambda event: messages.put(('event', event)))
                else:
                    source = paths['validation']
                    flag = '--dataset' if source.suffix.lower() == '.jsonl' else '--qualification'
                    result = main(['run', '--live', '--profile', str(paths['profile']), flag, str(source)],
                                  output=lambda event: messages.put(('event', event)), stopped=stop.is_set)
                messages.put(('done', result))
            except Exception as error:
                messages.put(('error', str(error)))

        Thread(target=worker, daemon=False).start()

    def request_stop():
        stop.set()
        end.configure(state='disabled')
        title.set('正在结束')

    def close():
        nonlocal closing
        if working:
            closing = True
            request_stop()
        else:
            root.destroy()

    def failure(reason):
        if 'expected one matching window' in reason:
            return '请打开唯一的终末地游戏窗口，进入第一轮空牌面，再点「开始抽牌」。'
        if 'window is not the calibrated process' in reason:
            return '游戏窗口已变化。确认游戏及显示设置后，点「重新绑定游戏」，再开始抽牌。'
        if 'hotkey registration failed' in reason:
            return 'F8 或 F9 已被占用。关闭另一个抽牌助手或占用快捷键的程序，再重试。'
        return reason

    def show(event):
        nonlocal success_text
        kind = event.get('event')
        if kind == 'armed':
            title.set('可以开始抽牌')
            detail.set('切回游戏的第一轮空牌面，按 F8 开始。')
        elif kind == 'resumed':
            title.set('正在观察游戏并抽牌')
            detail.set('已确认四张 5 时会保留牌面。切换到其他窗口会暂停。')
        elif kind == 'action_confirmed':
            slots = event['observation']['slots']
            detail.set('当前手牌　' + '　'.join('空' if v == 0 else '?' if v is None else str(v) for v in slots))
        elif kind == 'paused':
            reason = event.get('reason', '')
            if reason.startswith('focus lost'):
                title.set('已暂停，游戏离开了前台')
                detail.set('切回原来的游戏窗口，确认牌面后按 F8 恢复。')
            elif reason == 'action confirmation timed out':
                title.set('已暂停，上一次点击的结果还未确认')
                detail.set('先核对牌面或放弃弹窗。等待游戏稳定后，按 F8 重新核对。')
            elif reason == 'observation or action readiness timed out':
                title.set('已暂停，牌面识别或按钮等待超时')
                detail.set('检查游戏画面是否被遮挡，等动画结束、按钮可用后按 F8。')
            else:
                title.set('已暂停')
                detail.set('确认游戏画面后按 F8 恢复，或按 F9 结束。')
        elif kind == 'takeover_refused':
            title.set('请手动回到第一轮空牌面')
            detail.set('回到空牌面后按 F8 开始。')
        elif kind == 'suspected_success':
            title.set('发现疑似第四张 5，正在复核')
            detail.set('请保留当前牌面。')
        elif kind == 'live_blocked':
            title.set('识别资料未通过检查')
            reasons = event['reasons']
            detail.set('请补齐资料再开始。\n' + str(reasons[0] if reasons else '')[:180])
        elif kind == 'summary':
            reason = event['termination_reason']
            if reason == 'success':
                title.set('已确认至少四张 5，抽牌已结束')
                success_text = ('牌面已保留。证据保存失败：' + str(event['evidence_error'])
                                if event.get('evidence_error') else '牌面已保留。证据目录：' + str(event.get('evidence')))
                detail.set(success_text)
            elif not reason.startswith('error:'):
                title.set('抽牌已结束')
                detail.set('已停止游戏输入。' if reason in ('F9', 'source ended') else reason)
        elif kind in ('error', 'notification_error'):
            if not success_text:
                title.set('运行已停止')
            detail.set((success_text + '\n' if success_text else '') + failure(str(event.get('reason', ''))))

    def poll():
        nonlocal working
        while True:
            try:
                kind, value = messages.get_nowait()
            except Empty:
                break
            if kind == 'event':
                show(value)
            elif kind == 'recording':
                title.set('正在录制游戏画面')
                detail.set('录满两分钟自动保存。也可以点「结束」提前保存。')
            else:
                operation, working = working, None
                if closing:
                    root.destroy()
                    return
                if kind == 'error':
                    if not success_text:
                        title.set('操作已结束')
                    detail.set((success_text + '\n' if success_text else '') + failure(value))
                elif operation == 'record':
                    folder, count = value
                    title.set('游戏录像已保存')
                    detail.set(f'已保存 {count} 张游戏画面。请将这个目录用于制作或核验识别资料。\n{folder.resolve()}')
                elif operation == 'bind' and value == 0:
                    try:
                        remember('profile', bound)
                        ready()
                    except Exception as error:
                        title.set('无法保存游戏绑定')
                        detail.set(str(error))
                elif operation == 'live' and stop.is_set() and value == 0 and not success_text:
                    title.set('抽牌已结束')
                    detail.set('已停止游戏输入。')
                refresh()
        root.after(80, poll)

    def button(parent, label, action):
        widget = ttk.Button(parent, text=label, command=action)
        widget.pack(side='left', padx=(0, 6))
        return widget

    start = button(controls, '开始抽牌', lambda: run('live'))
    capture = button(controls, '录制素材', lambda: run('record'))
    end = button(controls, '结束', request_stop)
    button(setup, '选择识别配置', lambda: choose('profile'))
    button(setup, '选择验收素材', lambda: choose('validation'))
    rebind = button(setup, '重新绑定游戏', lambda: run('bind'))
    root.protocol('WM_DELETE_WINDOW', close)
    ready()
    if settings_error:
        title.set('请重新选择资料')
        detail.set(settings_error)
    elif paths['profile'].is_file() and paths['validation'].is_file():
        root.after_idle(lambda: run('live'))
    root.after(80, poll)
    root.mainloop()
    return 0
