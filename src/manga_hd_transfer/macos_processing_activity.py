"""Keep user-requested processing active when a macOS window is backgrounded.

No dependency on PyObjC, no permanent sleep assertion, no change to renderers.
The Foundation activity is released even on cancellation or processing failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import logging
import platform

logger = logging.getLogger(__name__)


class _FoundationActivity:
    def __init__(self):
        self.foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.address = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
        self.process = None
        self.token = None

    def send(self, target, selector: bytes, result, argument_types=(), arguments=()):
        function = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *argument_types)(self.address)
        return function(target, self.objc.sel_registerName(selector), *arguments)

    def begin(self):
        self.process = self.send(self.objc.objc_getClass(b"NSProcessInfo"), b"processInfo", ctypes.c_void_p)
        reason = self.send(self.objc.objc_getClass(b"NSString"), b"stringWithUTF8String:",
                           ctypes.c_void_p, (ctypes.c_char_p,), (b"Folirina page processing",))
        # SDK: NSActivityUserInitiatedAllowingIdleSystemSleep. Do not keep the
        # display or machine awake after the user leaves it idle.
        options = 0x00FFFFFF & ~(1 << 20)
        self.token = self.send(self.process, b"beginActivityWithOptions:reason:", ctypes.c_void_p,
                               (ctypes.c_uint64, ctypes.c_void_p), (options, reason))
        if not self.token:
            raise RuntimeError("Foundation returned no processing activity")
        self.send(self.token, b"retain", ctypes.c_void_p)

    def end(self):
        if self.token:
            try:
                self.send(self.process, b"endActivity:", None, (ctypes.c_void_p,), (self.token,))
            finally:
                self.send(self.token, b"release", None)
                self.token = None


@contextmanager
def processing_activity():
    """Best-effort App Nap protection scoped strictly to active processing."""
    activity = None
    state = {"app_nap_protected": False, "platform": platform.system()}
    if state["platform"] == "Darwin":
        try:
            activity = _FoundationActivity()
            activity.begin()
            state["app_nap_protected"] = True
        except (OSError, AttributeError, RuntimeError) as exc:
            logger.warning("macOS processing activity unavailable: %s", exc)
    try:
        yield state
    finally:
        if activity is not None:
            try:
                activity.end()
            except (OSError, AttributeError, RuntimeError) as exc:
                logger.warning("macOS processing activity release failed: %s", exc)
