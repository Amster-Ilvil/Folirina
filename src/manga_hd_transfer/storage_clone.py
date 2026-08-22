from __future__ import annotations

"""Independent copy-on-write publication for immutable page inputs.

Page workspaces need their own SOURCE/TARGET copies for offline restore and
manual review.  Re-encoding an already-PNG input wastes CPU and disk.  Prefer a
filesystem clone (APFS clonefile / Linux FICLONE), which is independent but
shares physical blocks until either side changes.  Never use a hardlink to user
inputs: source edits must not mutate an already-processed workspace.
"""

import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(8) == PNG_SIGNATURE
    except OSError:
        return False


def _clone_macos(source: Path, destination: Path) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        clonefile = libc.clonefile
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clonefile.restype = ctypes.c_int
        rc = clonefile(os.fsencode(source), os.fsencode(destination), 0)
        return rc == 0
    except Exception:
        return False


def _clone_linux(source: Path, destination: Path) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        import fcntl
        # linux/fs.h: #define FICLONE _IOW(0x94, 9, int)
        FICLONE = 0x40049409
        with source.open("rb") as src, destination.open("wb") as dst:
            fcntl.ioctl(dst.fileno(), FICLONE, src.fileno())
        return True
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def publish_independent_png(source: str | Path, destination: str | Path) -> str | None:
    """Atomically publish an independent copy of an existing PNG.

    Returns ``reflink``, ``copy`` or ``None`` when the source is not a PNG and
    the caller should encode its decoded pixels instead.
    """
    src = Path(source)
    dst = Path(destination)
    if not src.is_file() or not _is_png(src):
        return None
    try:
        if src.resolve() == dst.resolve():
            return "existing"
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink(missing_ok=True)
        cloned = _clone_macos(src, tmp) or _clone_linux(src, tmp)
        method = "reflink" if cloned else "copy"
        if not cloned:
            shutil.copyfile(src, tmp)
        try:
            with tmp.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        # A reflink/copy must be a distinct directory entry/inode, never a hardlink.
        try:
            if os.path.samefile(src, tmp):
                tmp.unlink(missing_ok=True)
                shutil.copyfile(src, tmp)
                method = "copy"
        except OSError:
            pass
        os.replace(tmp, dst)
        return method
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
