from __future__ import annotations

"""Isolated LightGlue / LoFTR runtime.

The desktop GUI may run on a newer CPython than the current PyTorch ecosystem
supports reliably.  In particular a partially-compatible Torch build can import
far enough to be discoverable and still fail during ``torch.distributed``
initialisation.  Deep page registration therefore runs in a dedicated Python
3.10-3.13 virtual environment and communicates with the GUI through a tiny JSONL
worker.  The worker is persistent so models are loaded once per application
session, preserving the original deep-model cache behaviour without importing
Torch into the Qt process.
"""

from dataclasses import dataclass
import atexit
import json
import functools
import os
import platform
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Iterable, Any

import numpy as np

from .model_downloads import model_home
from .tls_support import apply_runtime_tls_environment, ssl_failure_hint

ProgressFn = Callable[[str], None]
DEEP_MIN_MINOR = 10
DEEP_MAX_MINOR = 13


@dataclass(frozen=True, slots=True)
class DeepRegistrationRuntimeStatus:
    ready: bool
    python: str | None
    detail: str
    version: str = ""
    architecture: str = ""
    mps_available: bool = False


def runtime_root() -> Path:
    return model_home().parent / "runtime" / "deep-registration"


def venv_dir() -> Path:
    return runtime_root() / ".venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker_script() -> Path:
    return Path(__file__).with_name("deep_registration_worker.py")


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

    for env_name in (
        "MHD_DEEP_PYTHON",
        "MHD_TORCH_PYTHON",
        "MHD_OCR_PYTHON",
        "NOVEL_FORMATTER_OCR_PYTHON",
        "NOVEL_FORMATTER_PYTHON313",
    ):
        yield from add(os.environ.get(env_name))

    yield from add(sys.executable)
    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3"):
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
        "/opt/local/bin/python3.10",
    ):
        yield from add(path)

    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("3.13", "3.12", "3.11", "3.10"):
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
    if not (DEEP_MIN_MINOR <= minor <= DEEP_MAX_MINOR):
        return False, f"Python {info.get('version')} 不作为深度配准运行时使用（需要 3.{DEEP_MIN_MINOR}～3.{DEEP_MAX_MINOR}）", info
    if info.get("bits") != "64bit":
        return False, "深度配准运行时需要 64 位 Python", info
    # On Apple Silicon, a native arm64 runtime is strongly preferred.  Rosetta
    # can install a different Torch wheel family and defeats MPS acceleration.
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} and info.get("machine") not in {"arm64", "aarch64"}:
        return False, f"Apple Silicon 需要原生 arm64 Python；该解释器是 {info.get('machine') or 'unknown'}", info
    return True, f"Python {info.get('version')} · {info.get('machine')}", info


def find_compatible_python() -> Path:
    failures: list[str] = []
    for candidate in _candidate_paths():
        ok, detail, info = _probe_python(candidate)
        if ok:
            return Path(info.get("exe") or candidate).resolve()
        failures.append(f"- {candidate}: {detail}")
    details = "\n".join(failures[-14:]) or "（没有发现候选解释器）"
    raise RuntimeError(
        "LightGlue / LoFTR 需要独立 Python 3.10～3.13 运行环境。\n"
        "Mac 推荐安装 Python 3.13（python.org 或 `brew install python@3.13`），"
        "主 GUI 可以继续使用 Python 3.14。\n"
        "也可设置 MHD_DEEP_PYTHON=/完整路径/python3.13\n\n已检查：\n" + details
    )


def _install_env(progress: ProgressFn | None = None) -> dict[str, str]:
    env = os.environ.copy()
    proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = proxy
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    if platform.system() == "Darwin":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return apply_runtime_tls_environment(env, runtime_root(), progress=progress)


def _run(cmd: list[str], progress: ProgressFn | None = None, *, timeout: int = 3600) -> tuple[int, str]:
    _emit(progress, "执行：" + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=_install_env(progress),
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 100:
            tail = tail[-100:]
        low = line.lower()
        if any(token in low for token in (
            "error", "failed", "collecting", "installing", "successfully",
            "looking in indexes", "requirement already satisfied", "warning",
        )):
            _emit(progress, line)
    try:
        code = int(proc.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        proc.kill(); code = int(proc.wait())
        tail.append("进程超时并已终止。")
    return code, "\n".join(tail)


def _pip_install(py: Path, specs: list[str], progress: ProgressFn | None, *, extra: list[str] | None = None) -> None:
    attempts = [
        ("当前/默认 PyPI", []),
        ("PyPI 官方", ["-i", "https://pypi.org/simple"]),
        ("清华 PyPI", ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]),
        ("阿里云 PyPI", ["-i", "https://mirrors.aliyun.com/pypi/simple"]),
    ]
    failures: list[str] = []
    for label, index_args in attempts:
        _emit(progress, f"尝试 {label}…")
        cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", *index_args, *(extra or []), *specs]
        code, tail = _run(cmd, progress)
        if code == 0:
            return
        failures.append(f"[{label}]\n{tail[-2200:]}")
    detail = "\n\n".join(failures[-4:])
    if "certificate_verify_failed" in detail.lower() or "ssl" in detail.lower():
        detail = ssl_failure_hint(py) + "\n\n" + detail
    raise RuntimeError("所有可用安装源均失败。\n" + detail)


def _runtime_marker(py: Path) -> tuple[bool, str, dict[str, Any]]:
    script = (
        "import json,platform; import torch,torchvision,kornia,lightglue; "
        "from importlib.metadata import version; "
        "mps=bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()); "
        "print(json.dumps({'torch':version('torch'),'torchvision':version('torchvision'),"
        "'kornia':version('kornia'),'lightglue':getattr(lightglue,'__version__','source'),"
        "'machine':platform.machine(),'mps':mps}))"
    )
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=90, env=_install_env())
    except Exception as exc:
        return False, str(exc), {}
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "runtime import failed").strip()[-4000:], {}
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {}
    return True, proc.stdout.strip().splitlines()[-1], payload


@functools.lru_cache(maxsize=1)
def runtime_status() -> DeepRegistrationRuntimeStatus:
    py = venv_python()
    if not py.exists():
        return DeepRegistrationRuntimeStatus(False, None, "独立 LightGlue / LoFTR 运行环境尚未创建。")
    ok_py, detail_py, info = _probe_python(py)
    if not ok_py:
        return DeepRegistrationRuntimeStatus(False, str(py), "现有深度配准 venv 已失效：" + detail_py, architecture=info.get("machine", ""))
    ok, detail, payload = _runtime_marker(py)
    if not ok:
        return DeepRegistrationRuntimeStatus(False, str(py), "独立深度配准环境缺包/损坏：" + detail, architecture=info.get("machine", ""))
    version = f"Torch {payload.get('torch','')} / Kornia {payload.get('kornia','')} / LightGlue"
    return DeepRegistrationRuntimeStatus(
        True, str(py), f"独立配准环境已就绪：{version} · {info.get('machine','')}" + (" · MPS" if payload.get("mps") else " · CPU"),
        version=version, architecture=info.get("machine", ""), mps_available=bool(payload.get("mps")),
    )


def ensure_runtime(progress: ProgressFn | None = None) -> DeepRegistrationRuntimeStatus:
    runtime_status.cache_clear()
    root = runtime_root(); root.mkdir(parents=True, exist_ok=True)
    py = venv_python()
    if py.exists():
        ok_py, detail, _ = _probe_python(py)
        if not ok_py:
            _emit(progress, "现有深度配准 venv 不兼容，将重建：" + detail)
            shutil.rmtree(venv_dir(), ignore_errors=True)
    if not py.exists():
        base = find_compatible_python()
        ok, detail, _ = _probe_python(base)
        if not ok:
            raise RuntimeError(detail)
        _emit(progress, f"使用兼容解释器创建独立 LightGlue / LoFTR 环境：{base}（{detail}）")
        code, tail = _run([str(base), "-m", "venv", str(venv_dir())], progress, timeout=300)
        if code != 0:
            raise RuntimeError("创建深度配准独立 venv 失败。\n" + tail[-3000:])
        py = venv_python()
        # venv/ensurepip is local and must not require HTTPS.  Only after pip is
        # seeded do we access package indexes through the repaired CA bundle.
        code, tail = _run([str(py), "-m", "ensurepip", "--upgrade"], progress, timeout=300)
        if code != 0:
            raise RuntimeError("初始化深度配准 venv 的内置 pip 失败。\n" + tail[-3000:])

    ok, detail, _payload = _runtime_marker(py)
    if not ok:
        # Also repair a half-created venv left by an earlier TLS failure.
        code, tail = _run([str(py), "-m", "ensurepip", "--upgrade"], progress, timeout=300)
        if code != 0:
            raise RuntimeError("修复深度配准 venv 的内置 pip 失败。\n" + tail[-3000:])
        _emit(progress, "初始化/修复独立环境的 pip 构建工具…")
        try:
            _pip_install(py, ["pip", "setuptools", "wheel"], progress)
        except RuntimeError as exc:
            if "certificate" in str(exc).lower() or "ssl" in str(exc).lower():
                raise RuntimeError("初始化深度配准 venv 的 pip 失败。\n" + ssl_failure_hint(py) + "\n\n" + str(exc)[-5000:]) from exc
            raise
        _emit(progress, "安装 LightGlue / LoFTR 独立运行依赖…")
        _pip_install(py, ["numpy<3", "opencv-python-headless>=4.8", "torch", "torchvision", "kornia>=0.8"], progress)
        # LightGlue is installed from its official source package.  --no-deps
        # prevents pip from replacing the already isolated OpenCV/Torch stack.
        urls = [
            "https://github.com/cvg/LightGlue/archive/refs/heads/main.zip",
            "https://codeload.github.com/cvg/LightGlue/zip/refs/heads/main",
        ]
        failures: list[str] = []
        for url in urls:
            code, tail = _run([str(py), "-m", "pip", "install", "--upgrade", "--no-deps", url], progress, timeout=3600)
            if code == 0:
                break
            failures.append(tail[-2500:])
        else:
            raise RuntimeError("LightGlue 官方源码包安装失败。\n" + "\n".join(failures[-2:]))

    ok, detail, _ = _runtime_marker(py)
    if not ok:
        raise RuntimeError("独立 LightGlue / LoFTR 环境安装后仍无法导入。\n" + detail[-5000:])
    runtime_status.cache_clear()
    status = runtime_status()
    if not status.ready:
        raise RuntimeError(status.detail)
    _emit(progress, status.detail)
    return status


def require_runtime_python() -> Path:
    status = runtime_status()
    if not status.ready or not status.python:
        raise RuntimeError(
            "LightGlue / LoFTR 独立运行环境尚未就绪。请在模型中心点击 LightGlue 或 LoFTR 的“安装依赖”。\n"
            + status.detail
        )
    return Path(status.python)


class _PersistentWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._counter = 0

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        py = require_runtime_python()
        env = _install_env()
        self._responses = queue.Queue()
        self._stderr_tail = []
        self._proc = subprocess.Popen(
            [str(py), str(worker_script())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

    def _read_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                self._responses.put(row)

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.rstrip()
            if text:
                self._stderr_tail.append(text)
                if len(self._stderr_tail) > 80:
                    del self._stderr_tail[:40]

    def request(self, payload: dict[str, Any], *, timeout: float = 360.0) -> dict[str, Any]:
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None
            self._counter += 1
            request_id = self._counter
            row = dict(payload); row["request_id"] = request_id
            try:
                self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except Exception:
                self.stop(); self._start()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n"); self._proc.stdin.flush()
            while True:
                try:
                    result = self._responses.get(timeout=timeout)
                except queue.Empty as exc:
                    detail = "\n".join(self._stderr_tail[-20:])
                    self.stop()
                    raise RuntimeError("深度配准 Worker 超时。\n" + detail[-4000:]) from exc
                if int(result.get("request_id", -1)) != request_id:
                    continue
                if not result.get("ok"):
                    raise RuntimeError(str(result.get("error") or "深度配准 Worker 执行失败"))
                return result

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"op": "shutdown", "request_id": -1}) + "\n"); proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate(); proc.wait(timeout=3)
            except Exception:
                try: proc.kill()
                except Exception: pass


_WORKER = _PersistentWorker()
atexit.register(_WORKER.stop)


def run_deep_registration(
    kind: str,
    source: np.ndarray,
    target: np.ndarray,
    *,
    feature: str = "sift",
    max_features: int = 4096,
    deep_max_side: int = 1800,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    if kind not in {"lightglue", "loftr"}:
        raise ValueError(kind)
    runtime_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mhd-deep-reg-", dir=str(runtime_root())) as td:
        root = Path(td)
        src_path = root / "source.npy"; tgt_path = root / "target.npy"
        np.save(src_path, np.asarray(source), allow_pickle=False)
        np.save(tgt_path, np.asarray(target), allow_pickle=False)
        result = _WORKER.request({
            "op": "match", "kind": kind,
            "source": str(src_path), "target": str(tgt_path),
            "feature": str(feature), "max_features": int(max_features),
            "deep_max_side": int(deep_max_side), "device": str(device),
        })
    sp = np.asarray(result.get("source_points", []), dtype=np.float32).reshape((-1, 2))
    tp = np.asarray(result.get("target_points", []), dtype=np.float32).reshape((-1, 2))
    return sp, tp, str(result.get("method") or kind), dict(result.get("diagnostics") or {})
