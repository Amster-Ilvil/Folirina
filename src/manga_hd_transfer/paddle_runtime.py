from __future__ import annotations

"""Isolated PaddleOCR runtime management.

PaddlePaddle does not publish wheels for every CPython/architecture combination
used by the desktop GUI.  In particular, the GUI may run on Python 3.14 or an
Intel/Rosetta interpreter while current macOS Paddle wheels are CPython 3.9-3.13
and arm64 only.  Keep Paddle in its own venv, like Novel Formatter does, so the
main application Python can evolve independently.
"""

from dataclasses import dataclass
import json
import functools
import os
import platform
import queue
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable

from .storage_paths import model_home
from .tls_support import apply_runtime_tls_environment, ssl_failure_hint

ProgressFn = Callable[[str], None]

PADDLE_MIN_MINOR = 9
PADDLE_MAX_MINOR = 13
PADDLE_PACKAGE_SPEC = "paddlepaddle>=3.0,<4"
PADDLEOCR_PACKAGE_SPEC = "paddleocr>=3.7,<4"

PIP_INDEXES: tuple[tuple[str, str], ...] = (
    ("清华大学 TUNA", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("华为云", "https://repo.huaweicloud.com/repository/pypi/simple"),
    ("PyPI 官方", "https://pypi.org/simple"),
)



@dataclass(frozen=True, slots=True)
class PaddleRuntimeStatus:
    ready: bool
    python: str | None
    detail: str
    version: str = ""
    architecture: str = ""


def runtime_root() -> Path:
    return model_home().parent / "runtime" / "paddle-ocr"


def venv_dir() -> Path:
    return runtime_root() / ".venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _emit(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _candidate_paths() -> Iterable[str]:
    seen: set[str] = set()

    def add(value: str | None):
        if not value:
            return
        text = str(value)
        if text in seen:
            return
        seen.add(text)
        yield text

    for env_name in ("MHD_PADDLE_PYTHON", "MHD_OCR_PYTHON", "NOVEL_FORMATTER_OCR_PYTHON", "NOVEL_FORMATTER_PYTHON313"):
        yield from add(os.environ.get(env_name))

    try:
        from .standalone_python import runtime_python as standalone_runtime_python
        yield from add(str(standalone_runtime_python()))
    except Exception:
        pass

    # Keep the current interpreter as a candidate only; compatibility is
    # validated below.  This is important when the GUI itself already runs on
    # a supported arm64 Python 3.9-3.13.
    yield from add(sys.executable)

    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"):
        yield from add(shutil.which(name))

    for path in (
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
        "/opt/local/bin/python3.13",
        "/opt/local/bin/python3.12",
        "/opt/local/bin/python3.11",
    ):
        yield from add(path)

    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("3.13", "3.12", "3.11", "3.10", "3.9"):
            try:
                proc = subprocess.run(
                    [py_launcher, f"-{version}", "-c", "import sys;print(sys.executable)"],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:
                continue
            if proc.returncode == 0 and proc.stdout.strip():
                yield from add(proc.stdout.strip().splitlines()[-1])


def _probe_python(path: str | Path) -> tuple[bool, str, dict[str, str]]:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return False, "文件不存在", {}
    script = (
        "import json,platform,sys,venv;"
        "print(json.dumps({'version':'.'.join(map(str,sys.version_info[:3])),"
        "'minor':str(sys.version_info.minor),'machine':platform.machine().lower(),"
        "'bits':platform.architecture()[0],'exe':sys.executable}))"
    )
    try:
        proc = subprocess.run([str(candidate), "-c", script], capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return False, f"无法启动：{exc}", {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return False, (proc.stderr or "无法读取解释器信息").strip()[-500:], {}
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        minor = int(info.get("minor", -1))
    except Exception:
        return False, "解释器信息格式无效", {}
    if not (PADDLE_MIN_MINOR <= minor <= PADDLE_MAX_MINOR):
        return False, f"Python {info.get('version')} 不兼容（需要 3.{PADDLE_MIN_MINOR}～3.{PADDLE_MAX_MINOR}）", info
    if info.get("bits") != "64bit":
        return False, "PaddlePaddle 需要 64 位 Python", info
    # Current PaddlePaddle macOS wheels are arm64 only.  Reject a Rosetta/x86
    # interpreter even when the physical machine is Apple Silicon, otherwise pip
    # misleadingly reports "from versions: none".
    if platform.system() == "Darwin" and info.get("machine") not in {"arm64", "aarch64"}:
        return False, f"macOS 当前 Paddle wheel 仅支持 arm64；该 Python 是 {info.get('machine') or 'unknown'}", info
    return True, f"Python {info.get('version')} · {info.get('machine')}", info


def find_compatible_python() -> Path:
    failures: list[str] = []
    for candidate in _candidate_paths():
        ok, detail, info = _probe_python(candidate)
        if ok:
            return Path(info.get("exe") or candidate).resolve()
        failures.append(f"- {candidate}: {detail}")
    detail = "\n".join(failures[-14:]) or "（没有发现候选解释器）"
    if platform.system() == "Darwin":
        hint = (
            "PaddleOCR 需要 arm64 Python 3.9～3.13。当前 GUI 可以继续使用自己的 Python；"
            "只需另外安装 Python 3.13（推荐 python.org 或 `brew install python@3.13`）。"
        )
    else:
        hint = "PaddleOCR 需要 Python 3.9～3.13。请安装兼容 Python，或设置 MHD_PADDLE_PYTHON。"
    raise RuntimeError(hint + "\n可手动指定：MHD_PADDLE_PYTHON=/完整路径/python3.13\n\n已检查：\n" + detail)


def _install_env(progress: ProgressFn | None = None) -> dict[str, str]:
    env = os.environ.copy()
    proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = proxy
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    return apply_runtime_tls_environment(env, runtime_root(), progress=progress)


def _run(cmd: list[str], progress: ProgressFn | None = None, *, timeout: int = 3600) -> tuple[int, str]:
    """Run a dependency command with a real wall-clock timeout.

    Iterating directly over ``proc.stdout`` can block forever when a downloader
    stops producing output.  Pump stdout on a daemon thread and let the main
    thread enforce the deadline so a dead mirror cannot freeze the GUI worker.
    """
    _emit(progress, "执行：" + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=_install_env(progress),
        bufsize=1,
    )
    tail: list[str] = []
    lines: queue.Queue[str | None] = queue.Queue()
    assert proc.stdout is not None

    def pump() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw)
        finally:
            lines.put(None)

    threading.Thread(target=pump, name="mhd-paddle-pip-output", daemon=True).start()
    deadline = time.monotonic() + max(10, int(timeout))
    eof = False
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and proc.poll() is None:
            timed_out = True
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try: proc.kill()
                except Exception: pass
            break
        try:
            item = lines.get(timeout=min(0.25, max(0.01, remaining)))
        except queue.Empty:
            if proc.poll() is not None and eof:
                break
            continue
        if item is None:
            eof = True
            if proc.poll() is not None:
                break
            continue
        line = item.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 100:
            tail = tail[-100:]
        low = line.lower()
        if any(token in low for token in (
            "error", "failed", "collecting", "installing", "successfully",
            "looking in indexes", "requirement already satisfied", "warning", "retrying",
        )):
            _emit(progress, line)
    try:
        code = int(proc.wait(timeout=10))
    except Exception:
        try: proc.kill()
        except Exception: pass
        code = int(proc.wait())
    if timed_out:
        tail.append(f"安装进程超过 {timeout}s，已终止并切换到下一个下载源。")
        if code == 0:
            code = 124
    return code, "\n".join(tail)


def _pip_install(py: Path, specs: list[str], progress: ProgressFn | None, *, extra_args: list[str] | None = None) -> tuple[int, str]:
    cmd = [
        str(py), "-m", "pip", "install", "--disable-pip-version-check",
        "--prefer-binary", "--timeout", "45", "--retries", "2", "--upgrade",
        *(extra_args or []), *specs,
    ]
    return _run(cmd, progress)


def _pip_source_attempts(*, paddle_package: bool = False) -> list[tuple[str, list[str]]]:
    """Build a deterministic mirror retry list without duplicating configured URLs."""
    attempts: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    configured = str(os.environ.get("MHD_PIP_INDEX", "") or os.environ.get("PIP_INDEX_URL", "") or "").strip().rstrip("/")
    if configured:
        seen.add(configured)
        attempts.append(("用户/环境指定 PyPI", ["--index-url", configured]))

    # Do not preflight all mirrors here. DNS resolution can ignore urllib's
    # socket timeout on some corporate/VPN networks and freeze before pip starts.
    # pip itself has bounded connect/read retries, so deterministic fallback is
    # safer. The GUI's explicit Network Diagnosis remains available separately.
    for name, url in PIP_INDEXES:
        clean = url.rstrip("/")
        if clean in seen:
            continue
        seen.add(clean)
        attempts.append((name, ["--index-url", clean]))
    if paddle_package:
        url = "https://www.paddlepaddle.org.cn/packages/stable/cpu/".rstrip("/")
        if url not in seen:
            attempts.append(("Paddle 官方 CPU wheel 源", ["--index-url", url]))
    return attempts


def _pip_install_with_sources(py: Path, specs: list[str], progress: ProgressFn | None, *, paddle_package: bool = False) -> str:
    failures: list[str] = []
    attempts = _pip_source_attempts(paddle_package=paddle_package)
    for index, (label, args) in enumerate(attempts, start=1):
        _emit(progress, f"依赖源 {index}/{len(attempts)}：{label}")
        code, tail = _pip_install(py, specs, progress, extra_args=args)
        if code == 0:
            _emit(progress, f"依赖安装成功：{label}")
            return label
        failures.append(f"[{label}]\n{tail[-2200:]}")
    detail = "\n\n".join(failures[-5:])
    if "certificate_verify_failed" in detail.lower() or "ssl" in detail.lower():
        detail = ssl_failure_hint(py) + "\n\n" + detail
    raise RuntimeError("所有 PaddleOCR 依赖安装源均失败。\n" + detail)


def _runtime_marker(py: Path) -> tuple[bool, str]:
    script = (
        "import json,platform; import paddle,paddleocr; "
        "from importlib.metadata import version; "
        "print(json.dumps({'paddle':version('paddlepaddle'),'paddleocr':version('paddleocr'),'machine':platform.machine()}))"
    )
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=60, env=_install_env())
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "runtime import failed").strip()[-2000:]
    return True, proc.stdout.strip().splitlines()[-1]


@functools.lru_cache(maxsize=1)
def runtime_status() -> PaddleRuntimeStatus:
    py = venv_python()
    if not py.exists():
        return PaddleRuntimeStatus(False, None, "独立 PaddleOCR 运行环境尚未创建。")
    ok_py, detail_py, info = _probe_python(py)
    if not ok_py:
        return PaddleRuntimeStatus(False, str(py), "现有 PaddleOCR venv 已失效：" + detail_py, architecture=info.get("machine", ""))
    ok, detail = _runtime_marker(py)
    if not ok:
        return PaddleRuntimeStatus(False, str(py), "独立 PaddleOCR 运行环境缺包/损坏：" + detail, architecture=info.get("machine", ""))
    version = ""
    try:
        payload = json.loads(detail)
        version = f"Paddle {payload.get('paddle','')} / PaddleOCR {payload.get('paddleocr','')}"
    except Exception:
        version = detail
    return PaddleRuntimeStatus(True, str(py), f"独立运行环境已就绪：{version} · {info.get('machine','')}", version=version, architecture=info.get("machine", ""))


def _write_install_state(py: Path, *, dependency_sources: list[str]) -> None:
    root = runtime_root(); root.mkdir(parents=True, exist_ok=True)
    ok, marker = _runtime_marker(py)
    payload = {
        "schema": "mhd.paddle_runtime.v2",
        "ready": bool(ok),
        "python": str(py),
        "marker": marker if ok else "",
        "dependency_sources": list(dependency_sources),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temp = root / "install-state.json.tmp"
    final = root / "install-state.json"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(final)


def ensure_runtime(progress: ProgressFn | None = None) -> PaddleRuntimeStatus:
    root = runtime_root(); root.mkdir(parents=True, exist_ok=True)
    dependency_sources: list[str] = []
    py = venv_python()

    # An old venv created from Python 3.14 or Rosetta x86_64 cannot be repaired
    # by pip because no compatible Paddle wheel exists. Rebuild it from a valid
    # arm64 3.9-3.13 interpreter instead.
    if py.exists():
        ok_py, detail, _ = _probe_python(py)
        if not ok_py:
            _emit(progress, "现有 PaddleOCR venv 不兼容，将重建：" + detail)
            shutil.rmtree(venv_dir(), ignore_errors=True)

    if not py.exists():
        try:
            base = find_compatible_python()
        except RuntimeError as original_error:
            # Match the deployment pattern used by Novel Formatter's macOS
            # launcher: when the user explicitly requested dependency setup and
            # no compatible local Python exists, prepare a pinned standalone
            # runtime instead of forcing them to alter the GUI Python.
            bootstrap_enabled = str(os.environ.get("MHD_PADDLE_BOOTSTRAP_PYTHON", "1") or "1").strip().lower() not in {"0", "false", "no"}
            if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} and bootstrap_enabled:
                from .standalone_python import ensure_standalone_python
                _emit(progress, "未发现可用的 arm64 Python 3.9～3.13，自动准备独立 OCR Python…")
                base = ensure_standalone_python(progress)
            else:
                raise original_error
        ok, detail, _ = _probe_python(base)
        if not ok:
            raise RuntimeError(detail)
        _emit(progress, f"使用兼容解释器创建独立 PaddleOCR 环境：{base}（{detail}）")
        code, tail = _run([str(base), "-m", "venv", str(venv_dir())], progress, timeout=300)
        if code != 0:
            raise RuntimeError("创建 PaddleOCR 独立 venv 失败。\n" + tail[-3000:])
        py = venv_python()
        code, tail = _run([str(py), "-m", "ensurepip", "--upgrade"], progress, timeout=300)
        if code != 0:
            raise RuntimeError("初始化 PaddleOCR venv 的内置 pip 失败。\n" + tail[-3000:])

    ok, detail = _runtime_marker(py)
    if not ok:
        # Also repair a half-created venv left by an earlier TLS failure.
        code, tail = _run([str(py), "-m", "ensurepip", "--upgrade"], progress, timeout=300)
        if code != 0:
            raise RuntimeError("修复 PaddleOCR venv 的内置 pip 失败。\n" + tail[-3000:])
        _emit(progress, "初始化/修复 PaddleOCR 独立环境的 pip 构建工具…")
        try:
            dependency_sources.append(_pip_install_with_sources(py, ["pip", "setuptools", "wheel"], progress, paddle_package=False))
        except RuntimeError as exc:
            if "certificate" in str(exc).lower() or "ssl" in str(exc).lower():
                raise RuntimeError("初始化 PaddleOCR venv 的 pip 失败。\n" + ssl_failure_hint(py) + "\n\n" + str(exc)[-5000:]) from exc
            raise
        _emit(progress, "安装 PaddlePaddle 到独立运行环境（不再安装进 GUI Python）…")
        dependency_sources.append(_pip_install_with_sources(py, [PADDLE_PACKAGE_SPEC], progress, paddle_package=True))
        _emit(progress, "安装 PaddleOCR 到独立运行环境…")
        dependency_sources.append(_pip_install_with_sources(py, [PADDLEOCR_PACKAGE_SPEC], progress, paddle_package=False))

    ok, detail = _runtime_marker(py)
    if not ok:
        raise RuntimeError("PaddleOCR 独立环境安装后仍无法导入。\n" + detail[-4000:])
    runtime_status.cache_clear()
    status = runtime_status()
    if not status.ready:
        raise RuntimeError(status.detail)
    _write_install_state(py, dependency_sources=dependency_sources)
    _emit(progress, status.detail)
    return status


def _dist_version(py: Path, distribution: str) -> str:
    script = (
        "from importlib.metadata import version; "
        f"print(version({distribution!r}))"
    )
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=30, env=_install_env())
    except Exception:
        return ""
    return proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 and proc.stdout.strip() else ""


def _version_tuple(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for token in str(value or "").replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def ensure_pipeline_dependencies(pipeline: str, progress: ProgressFn | None = None, *, force_repair: bool = False) -> PaddleRuntimeStatus:
    """Backward-compatible dispatcher for document parser runtimes.

    v2.0.48 moved VL/PP-StructureV3 into a dedicated venv so repairing their
    extras can no longer destabilise classic PP-OCRv6.
    """
    name = str(pipeline or "ocr").strip().lower()
    if name not in {"vl", "structure"}:
        return ensure_runtime(progress)
    # Import at call time only.  ``paddle_doc_runtime`` reuses the classic
    # runtime's install primitives, so a module-level/static import here would
    # recreate a paddle_runtime <-> paddle_doc_runtime cycle.
    import importlib
    ensure_doc_runtime = importlib.import_module(f"{__package__}.paddle_doc_runtime").ensure_runtime
    return ensure_doc_runtime(progress, force_repair=force_repair)


def repair_pipeline_dependencies(pipeline: str, progress: ProgressFn | None = None) -> PaddleRuntimeStatus:
    return ensure_pipeline_dependencies(pipeline, progress, force_repair=True)


def require_runtime_python() -> Path:
    status = runtime_status()
    if not status.ready or not status.python:
        raise RuntimeError(
            "PaddleOCR 依赖尚未就绪。请在“模型下载与接入状态”中点击 PaddleOCR 的“安装依赖”。\n" + status.detail
        )
    return Path(status.python)


def worker_script() -> Path:
    return Path(__file__).with_name("paddle_worker.py")
