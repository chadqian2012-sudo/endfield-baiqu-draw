"""Windows window identity, hotkeys and foreground-only mouse delivery."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class WindowIdentity:
    hwnd: int
    pid: int
    process_started: int
    process_path: str
    client: tuple[int, int, int, int]
    dpi: int

def _api():
    if os.name != 'nt': raise RuntimeError('Windows required')
    u = ctypes.WinDLL('user32', use_last_error=True)
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    u.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    u.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    if not u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
        raise ctypes.WinError(ctypes.get_last_error(),'cannot set Per-Monitor V2 DPI awareness for current thread')
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.IsWindow.argtypes = [wintypes.HWND]
    u.GetDpiForWindow.argtypes = [wintypes.HWND]
    u.GetDpiForWindow.restype = wintypes.UINT
    k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    return u,k

def snapshot(hwnd):
    u,k = _api()
    if not u.IsWindow(hwnd): raise RuntimeError('window disappeared')
    pid = wintypes.DWORD()
    if not u.GetWindowThreadProcessId(hwnd,ctypes.byref(pid)): raise ctypes.WinError(ctypes.get_last_error())
    process = k.OpenProcess(0x1000, False, pid.value)
    if not process: raise ctypes.WinError(ctypes.get_last_error())
    try:
        created,exited,kernel,user = (wintypes.FILETIME() for _ in range(4))
        if not k.GetProcessTimes(process,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        buf = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buf))
        if not k.QueryFullProcessImageNameW(process,0,buf,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        process_path = buf.value
    finally:
        k.CloseHandle(process)
    rect, pt = wintypes.RECT(), wintypes.POINT(0,0)
    if not u.GetClientRect(hwnd,ctypes.byref(rect)) or not u.ClientToScreen(hwnd,ctypes.byref(pt)):
        raise ctypes.WinError(ctypes.get_last_error())
    dpi = u.GetDpiForWindow(hwnd)
    if not dpi: raise RuntimeError('unable to read window DPI')
    return WindowIdentity(int(hwnd),pid.value,(created.dwHighDateTime<<32)|created.dwLowDateTime,
                          process_path,(pt.x,pt.y,rect.right-rect.left,rect.bottom-rect.top),dpi)

def find_unique(title='Endfield'):
    u,_ = _api()
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    found = []
    def visit(hwnd, _):
        if u.IsWindowVisible(hwnd):
            length = u.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length+1)
            u.GetWindowTextW(hwnd,buf,len(buf))
            if buf.value == title: found.append(hwnd)
        return True
    callback = callback_type(visit)
    u.EnumWindows.argtypes = [callback_type,wintypes.LPARAM]
    u.EnumWindows.restype = wintypes.BOOL
    if not u.EnumWindows(callback,0): raise ctypes.WinError(ctypes.get_last_error())
    if len(found) != 1: raise RuntimeError(f'expected one matching window, found {len(found)}')
    return snapshot(found[0])

def foreground_matches(binding, *, require_identity=False):
    u,_ = _api()
    if snapshot(binding.hwnd) != binding:
        if require_identity:
            raise RuntimeError('window identity, geometry or DPI changed; recalibrate')
        return False
    return u.GetForegroundWindow() == binding.hwnd

class Hotkeys:
    """Thread-owned F8/F9 registrations. Construction must precede live input."""
    def __init__(self):
        self.u,_ = _api()
        self.u.RegisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int,wintypes.UINT,wintypes.UINT]
        self.u.UnregisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int]
        self.u.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG),wintypes.HWND,wintypes.UINT,wintypes.UINT,wintypes.UINT]
        self.registered = []
        for key in (0x77,0x78):
            if not self.u.RegisterHotKey(None,key,0x4000,key):
                self.close()
                raise RuntimeError('F8/F9 hotkey registration failed')
            self.registered.append(key)

    def poll(self):
        message = wintypes.MSG()
        keys = []
        while self.u.PeekMessageW(ctypes.byref(message),None,0x0312,0x0312,1):
            keys.append(int(message.wParam))
        return keys

    def close(self):
        for key in self.registered:
            self.u.UnregisterHotKey(None,key)
        self.registered.clear()

class _MouseInput(ctypes.Structure):
    _fields_ = [('dx',wintypes.LONG),('dy',wintypes.LONG),('data',wintypes.DWORD),
                ('flags',wintypes.DWORD),('time',wintypes.DWORD),('extra',ctypes.c_size_t)]


class _Input(ctypes.Structure):
    _fields_ = [('type',wintypes.DWORD),('mouse',_MouseInput)]


class MouseExecutor:
    """on_backend reports the selected delivery method, never a game acknowledgement."""
    BACKENDS = ('auto','sendinput','window-message')

    def __init__(self,binding,*,backend='auto',on_backend=None):
        if backend not in self.BACKENDS:
            raise ValueError('unknown mouse backend: '+str(backend))
        self.binding = binding
        self.automatic = backend == 'auto'
        self.backend = 'sendinput' if self.automatic else backend
        self.on_backend = on_backend
        self.u,_ = _api()
        u = self.u
        u.GetSystemMetrics.argtypes = [ctypes.c_int]
        u.WindowFromPoint.argtypes = [wintypes.POINT]
        u.WindowFromPoint.restype = wintypes.HWND
        u.IsChild.argtypes = [wintypes.HWND,wintypes.HWND]
        u.ScreenToClient.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
        u.PostMessageW.argtypes = [wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        u.SendInput.argtypes = [wintypes.UINT,ctypes.POINTER(_Input),ctypes.c_int]
        u.SendInput.restype = wintypes.UINT
        if on_backend:
            on_backend(self.backend,'preferred' if self.automatic else 'selected')

    def _owns(self,hwnd):
        pid = wintypes.DWORD()
        return bool(hwnd and (hwnd == self.binding.hwnd or self.u.IsChild(self.binding.hwnd,hwnd))
                    and self.u.GetWindowThreadProcessId(hwnd,ctypes.byref(pid)) and pid.value == self.binding.pid)

    def _target(self,point,permitted):
        if not foreground_matches(self.binding):
            return None
        target = self.u.WindowFromPoint(point)
        if not self._owns(target):
            return None
        client,rect = wintypes.POINT(point.x,point.y),wintypes.RECT()
        if not (self.u.ScreenToClient(target,ctypes.byref(client)) and self.u.GetClientRect(target,ctypes.byref(rect))):
            return None
        if not (0 <= client.x < rect.right and 0 <= client.y < rect.bottom):
            return None
        if permitted() and self.u.GetForegroundWindow() == self.binding.hwnd and self.u.WindowFromPoint(point) == target:
            return target,client.x,client.y
        return None

    def __call__(self,region,permitted):
        x,y,w,h = region
        left,top,width,height = self.binding.client
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
            raise ValueError('click region outside client')
        point = wintypes.POINT(left+x+w//2,top+y+h//2)
        if self.backend == 'sendinput':
            left,top,sw,sh = (self.u.GetSystemMetrics(i) for i in (76,77,78,79))
            if sw <= 1 or sh <= 1 or not left <= point.x < left+sw or not top <= point.y < top+sh:
                raise RuntimeError('invalid virtual screen geometry')
            events = (_Input*3)()
            events[0].mouse = _MouseInput(((point.x-left)*65536+32768)//sw,((point.y-top)*65536+32768)//sh,
                                          0,0xC001,0,0)
            events[1].mouse.flags = 0x0002
            events[2].mouse.flags = 0x0004
            if not self._target(point,permitted):
                return False
            ctypes.set_last_error(0)
            count = self.u.SendInput(3,events,ctypes.sizeof(_Input))
            error = ctypes.get_last_error()
            if count == 3:
                return True
            if count == 2:
                if self.u.SendInput(1,ctypes.byref(events[2]),ctypes.sizeof(_Input)) != 1:
                    raise RuntimeError('SendInput accepted down but release failed; stop')
            # A zero count alone does not distinguish UIPI from other failures.
            if not (count == 0 and self.automatic and error in (50,120)):
                raise RuntimeError(f'SendInput inserted {count}/3 events (WinError {error}); do not retry')
            self.backend = 'window-message'
            if self.on_backend:
                self.on_backend(self.backend,f'SendInput unsupported (WinError {error}); zero events inserted')

        hit = self._target(point,permitted)
        if not hit:
            return False
        target,cx,cy = hit
        if cx > 32767 or cy > 32767:
            raise ValueError('mouse message outside signed client coordinates')
        lparam = cx | cy << 16
        release_required = True
        try:
            if not self.u.PostMessageW(target,0x0201,1,lparam):
                release_required = False
                raise ctypes.WinError(ctypes.get_last_error(),'mouse down was not queued')
            aligned = self._target(point,permitted) == hit
        finally:
            # An interruption in PostMessageW can leave the down outcome unknown.
            if release_required and (not self._owns(target) or not self.u.PostMessageW(target,0x0202,0,lparam)):
                raise RuntimeError('mouse down was queued but release could not be confirmed; do not retry')
        if not aligned:
            raise RuntimeError('target changed or stopped after mouse down; released without retry')
        return True


def click(binding,region,permitted):
    return MouseExecutor(binding,backend='sendinput')(region,permitted)

if __name__ == '__main__':
    binding = find_unique()
    print(binding)
    print('foreground:', foreground_matches(binding))
