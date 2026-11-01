"""Windows window identity, hotkeys and the single mouse-input backend."""
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
    try:
        u.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        pass
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

def foreground_matches(binding):
    u,_ = _api()
    return u.GetForegroundWindow() == binding.hwnd and snapshot(binding.hwnd) == binding

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

def click(binding, region, permitted):
    """Send one complete click after the caller's final state/stop check.

    Foreground checking and OS submission are not atomic. A partial submission
    is an error; only an inserted left-button-down may be followed by release.
    """
    u,_ = _api()
    x,y,w,h = region
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > binding.client[2] or y+h > binding.client[3]:
        raise ValueError('click region outside client')
    class Mouse(ctypes.Structure):
        _fields_ = [('dx',wintypes.LONG),('dy',wintypes.LONG),('data',wintypes.DWORD),
                    ('flags',wintypes.DWORD),('time',wintypes.DWORD),('extra',ctypes.c_size_t)]
    class Payload(ctypes.Union):
        _fields_ = [('mouse',Mouse)]
    class Input(ctypes.Structure):
        _fields_ = [('type',wintypes.DWORD),('payload',Payload)]
    u.SendInput.argtypes = [wintypes.UINT,ctypes.POINTER(Input),ctypes.c_int]
    u.SendInput.restype = wintypes.UINT
    u.GetSystemMetrics.argtypes = [ctypes.c_int]
    left,top,sw,sh = (u.GetSystemMetrics(i) for i in (76,77,78,79))
    sx,sy = binding.client[0]+x+w//2,binding.client[1]+y+h//2
    if sw <= 1 or sh <= 1 or not left <= sx < left+sw or not top <= sy < top+sh:
        raise RuntimeError('invalid virtual screen geometry')
    events = (Input*3)()
    events[0].payload.mouse = Mouse(round((sx-left)*65535/(sw-1)),round((sy-top)*65535/(sh-1)),0,0xC001,0,0)
    events[1].payload.mouse.flags = 0x0002
    events[2].payload.mouse.flags = 0x0004
    if not foreground_matches(binding) or not permitted():
        return False
    count = u.SendInput(3,events,ctypes.sizeof(Input))
    if count == 2:
        release = Input()
        release.payload.mouse.flags = 0x0004
        u.SendInput(1,ctypes.byref(release),ctypes.sizeof(Input))
    if count != 3:
        raise RuntimeError(f'partial or blocked input: {count}/3 events; do not retry')
    return True

if __name__ == '__main__':
    binding = find_unique()
    print(binding)
    print('foreground:', foreground_matches(binding))
