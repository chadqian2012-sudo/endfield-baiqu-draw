"""Read-only Windows window discovery and identity checks; no activation or input."""
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

if __name__ == '__main__':
    binding = find_unique()
    print(binding)
    print('foreground:', foreground_matches(binding))
