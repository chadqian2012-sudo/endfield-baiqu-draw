"""Foreground client capture with fresh, caller-owned BGR frames."""
from contextlib import ExitStack, contextmanager
import ctypes
from ctypes import wintypes
import multiprocessing
from time import monotonic
from typing import NamedTuple

import numpy as np
from windowing import foreground_matches


BACKENDS = ('auto', 'mss', 'dxgi')


class _Display(NamedTuple):
    index: int
    handle: int
    device: str
    rect: tuple


class _MonitorInfo(ctypes.Structure):
    _fields_ = [('size',wintypes.DWORD),('monitor',wintypes.RECT),('work',wintypes.RECT),
                ('flags',wintypes.DWORD),('device',wintypes.WCHAR*32)]


class _DevMode(ctypes.Structure):
    # DEVMODEW display union and fixed-size fields; only rotation is read.
    _fields_ = [('device',wintypes.WCHAR*32),('spec',wintypes.WORD),('driver',wintypes.WORD),
                ('size',wintypes.WORD),('extra',wintypes.WORD),('fields',wintypes.DWORD),
                ('position',wintypes.POINT),('orientation',wintypes.DWORD),('fixed_output',wintypes.DWORD),
                ('print_fields',ctypes.c_short*5),('form',wintypes.WCHAR*32),('log_pixels',wintypes.WORD),
                ('display_and_icm_fields',wintypes.DWORD*13)]


def _monitor_for(client):
    """windows-capture uses the one-based EnumDisplayMonitors order."""
    monitors = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                      ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    @callback_type
    def collect(handle, _, rect, __):
        area = rect.contents
        monitors.append((handle,(area.left, area.top, area.right, area.bottom)))
        return True
    user = ctypes.WinDLL('user32', use_last_error=True)
    enum = user.EnumDisplayMonitors
    enum.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), callback_type, wintypes.LPARAM]
    enum.restype = wintypes.BOOL
    if not enum(None, None, collect, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    x, y, w, h = client
    for index, (handle,rect) in enumerate(monitors, 1):
        left, top, right, bottom = rect
        if left <= x and top <= y and x+w <= right and y+h <= bottom:
            info, mode = _MonitorInfo(), _DevMode()
            info.size, mode.size = ctypes.sizeof(info), ctypes.sizeof(mode)
            user.GetMonitorInfoW.argtypes = [wintypes.HANDLE,ctypes.POINTER(_MonitorInfo)]
            user.EnumDisplaySettingsW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,ctypes.POINTER(_DevMode)]
            if not user.GetMonitorInfoW(handle,ctypes.byref(info)) or not user.EnumDisplaySettingsW(
                    info.device,0xFFFFFFFF,ctypes.byref(mode)):
                raise OSError('无法核对显示器标识和旋转方向。')
            if mode.orientation != 0:
                raise ValueError('DXGI 不支持旋转显示器，请改用 MSS。')
            return _Display(index,handle,info.device,rect)
    raise ValueError('DXGI 要求整个游戏窗口在同一台显示器内。')


def _check_blt(result, function, args):
    # MSS 10.1 checks GetDIBits but ignores BitBlt failure on its reused bitmap.
    if not result:
        raise OSError('BitBlt failed to refresh the desktop bitmap')
    return result


@contextmanager
def _direct_dxgi(client):
    from windows_capture import DxgiDuplicationSession
    display = _monitor_for(client)
    left, top, right, bottom = display.rect
    x, y, w, h = client
    x, y = x-left, y-top
    session = DxgiDuplicationSession(display.index)
    def grab(discard=False):
        if _monitor_for(client) != display:
            raise ValueError('DXGI 显示器映射已改变，请重新核对窗口。')
        if discard:
            session.acquire_frame(timeout_ms=0)
        frame = session.acquire_frame(timeout_ms=24)
        if _monitor_for(client) != display:
            raise ValueError('DXGI 显示器映射已改变，请重新核对窗口。')
        if frame is None:
            return None
        if (frame.width,frame.height) != (right-left,bottom-top):
            raise ValueError('显示器尺寸或方向已改变，请重新校准。')
        return frame.to_bgr()[y:y+h,x:x+w]
    try:
        yield grab
    finally:
        session = None  # Drop the owning native DXGI object even if grab is still referenced.


class _LostFocus(Exception):
    pass


def _dxgi_worker(pipe, buffer, binding, region):
    """One native session per helper. Windows reclaims driver handles on exit."""
    output = np.frombuffer(buffer,np.uint8).reshape(region[3],region[2],3)
    try:
        # Establish the worker thread's calibrated DPI coordinate space before enumeration.
        foreground_matches(binding,require_identity=True)
        with _direct_dxgi(region) as read:
            pipe.send(('ready',''))
            while True:
                discard = pipe.recv()
                if not foreground_matches(binding,require_identity=True):
                    pipe.send(('blur',''))
                    continue
                image = read(discard)
                if not foreground_matches(binding,require_identity=True):
                    pipe.send(('blur',''))
                elif image is None:
                    pipe.send(('empty',''))
                else:
                    output[:] = image
                    pipe.send(('frame',''))
                del image
    except (EOFError,BrokenPipeError):
        pass
    except Exception as error:
        try:
            pipe.send(('error',str(error)))
        except (OSError,EOFError):
            pass
    finally:
        pipe.close()


@contextmanager
def _dxgi_process(binding,region=None):
    region = binding.client if region is None else region
    x,y,w,h = binding.client
    rx,ry,rw,rh = region
    if not (w>0 and h>0 and x<=rx<rx+rw<=x+w and y<=ry<ry+rh<=y+h):
        raise ValueError('DXGI 截图区域超出游戏客户区。')
    context = multiprocessing.get_context('spawn')
    buffer = context.RawArray(ctypes.c_ubyte,rw*rh*3)
    pixels = np.frombuffer(buffer,np.uint8).reshape(rh,rw,3)
    parent, child = context.Pipe()
    process = context.Process(target=_dxgi_worker,args=(child,buffer,binding,region),daemon=True)
    def receive():
        try:
            kind, detail = parent.recv()
        except EOFError as error:
            raise RuntimeError('DXGI 截图子进程已意外终止。') from error
        if kind == 'error':
            raise RuntimeError(detail)
        return kind
    try:
        process.start()
        child.close()
        deadline = monotonic()+12
        ready = False
        pending_discard = True  # Discard desktop updates queued during helper startup.
        def grab(discard=False):
            nonlocal ready,pending_discard
            pending_discard |= discard
            if not ready:
                if not parent.poll(.024):
                    if monotonic() >= deadline:
                        raise TimeoutError('DXGI 截图进程启动超时。')
                    return None
                if receive() != 'ready': raise RuntimeError('DXGI 启动响应异常。')
                ready = True
            parent.send(pending_discard)
            pending_discard = False
            if not parent.poll(.45):
                raise TimeoutError('DXGI 截图进程未及时返回。')
            kind = receive()
            if kind == 'blur': raise _LostFocus()
            if kind == 'empty': return None
            if kind != 'frame': raise RuntimeError('DXGI 帧响应异常。')
            return pixels.copy()
        yield grab
    finally:
        child.close()
        parent.close()
        if process.pid is not None:
            process.join(.2)
            if process.is_alive():
                process.kill()
                process.join(2)
            if process.is_alive(): raise RuntimeError('DXGI 截图进程未退出。')
        process.close()


@contextmanager
def _reader(backend, binding):
    if backend == 'mss':
        import mss
        x,y,w,h = binding.client
        region = {'left':x, 'top':y, 'width':w, 'height':h}
        with mss.mss() as screen:
            screen.gdi32.BitBlt.errcheck = _check_blt
            yield lambda discard=False: np.asarray(screen.grab(region))[:, :, :3].copy()
    else:
        with _dxgi_process(binding) as grab:
            yield grab


class WindowCapture(ExitStack):
    """Capture on the calling thread. No image means no independent observation."""
    def __init__(self, binding, backend='auto', on_backend=None):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError('unknown capture backend: ' + str(backend))
        self.binding = binding
        self.backend = backend
        self.active = 'mss' if backend == 'auto' else backend
        self.on_backend = on_backend
        self.reader = None
        self.reason = ''
        self.discard = False
        self.reported = None

    def _focused(self):
        focused = foreground_matches(self.binding, require_identity=True)
        if not focused: self.discard = True
        return focused

    def grab(self):
        while self._focused():
            if self.reader is None:
                try:
                    self.reader = self.enter_context(_reader(self.active, self.binding))
                except Exception as error:
                    if not self._unavailable(error):
                        return None, False
                    continue
                # Opening a duplication session can take longer than a frame.
                if not self._focused():
                    return None, False
            started = monotonic()
            try:
                image = self.reader(self.discard)
                self.discard = False
            except _LostFocus:
                self.discard = True
                return None, False
            except Exception as error:
                if not self._unavailable(error):
                    return None, False
                continue
            if not self._focused():
                return None, False
            if image is None or monotonic()-started > .5:
                return None, True
            width, height = self.binding.client[2:]
            if image.dtype != np.uint8 or image.shape != (height,width,3):
                raise ValueError('截图与已校准的游戏尺寸不符。')
            # An unchanged hand can be legitimate. A fresh all-black surface
            # carries no evidence and alone does not prove the backend failed.
            if not image.any():
                return None, True
            if self.reported != self.active:
                self.reported = self.active
                if self.on_backend: self.on_backend(self.active,self.reason)
            return image, True
        return None, False

    def _unavailable(self, error):
        # Recheck outside the backend: lost focus/identity never triggers fallback.
        if not self._focused():
            return False
        if self.backend != 'auto' or self.active != 'mss':
            raise RuntimeError(f'{self.active} 截图失败：{error}') from error
        if not isinstance(error, (ImportError, OSError)):
            from mss.exception import ScreenShotError
            if not isinstance(error, ScreenShotError):
                raise error
        self.close()
        self.reason = f'MSS 截图失败：{error}'
        self.active = 'dxgi'
        self.reader = None
        return True
