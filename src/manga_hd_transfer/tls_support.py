from __future__ import annotations

"""TLS/CA support for isolated model runtimes.

Python.org macOS interpreters can be installed before their optional
``Install Certificates.command`` has been run.  A venv created from such an
interpreter inherits the incomplete OpenSSL CA configuration, so pip may fail
with ``CERTIFICATE_VERIFY_FAILED`` even though Safari/system networking works.

Do not disable certificate verification.  Instead build a process-local PEM
bundle from trusted sources already present on the machine and point pip,
Requests and Python/OpenSSL at that bundle.
"""

import os
import json
from pathlib import Path
import platform
import re
import ssl
import subprocess
import sys
from typing import Callable

ProgressFn = Callable[[str], None]
_CERT_RE = re.compile(
    rb"-----BEGIN CERTIFICATE-----\s+.*?-----END CERTIFICATE-----\s*",
    re.DOTALL,
)


def _emit(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _read_pem_certificates(path: Path) -> list[bytes]:
    try:
        data = path.expanduser().read_bytes()
    except Exception:
        return []
    return [m.group(0).strip() + b"\n" for m in _CERT_RE.finditer(data)]


def _security_certificates(keychain: str | None = None) -> list[bytes]:
    security = Path("/usr/bin/security")
    if not security.exists():
        return []
    cmd = [str(security), "find-certificate", "-a", "-p"]
    if keychain:
        cmd.append(str(keychain))
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [m.group(0).strip() + b"\n" for m in _CERT_RE.finditer(proc.stdout or b"")]


def _candidate_ca_files() -> list[Path]:
    out: list[Path] = []
    # Explicit user/administrator CA settings take priority and are also merged
    # with public roots below.  MHD_CA_BUNDLE is controlled by the model center.
    for name in ("MHD_CA_BUNDLE", "PIP_CERT", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            p = Path(value).expanduser()
            if p.is_file():
                out.append(p)
    try:
        paths = ssl.get_default_verify_paths()
        for value in (paths.cafile,):
            if value:
                p = Path(value)
                if p.is_file():
                    out.append(p)
    except Exception:
        pass
    # Common macOS/Homebrew/system locations.  Only existing PEM files are read.
    for value in (
        "/etc/ssl/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
    ):
        p = Path(value)
        if p.is_file():
            out.append(p)
    # The GUI environment may already contain certifi even when the isolated
    # Python's OpenSSL symlink is missing.
    try:
        import certifi  # type: ignore
        p = Path(certifi.where())
        if p.is_file():
            out.append(p)
    except Exception:
        pass
    # Stable order, no duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key); unique.append(p)
    return unique


def build_runtime_ca_bundle(runtime_root: str | Path, progress: ProgressFn | None = None) -> Path | None:
    """Build a PEM CA bundle without downloading anything.

    On macOS, roots are exported from the system root and system keychains.  We
    deliberately do not scrape arbitrary user/login certificates by default.
    If a corporate proxy CA lives only in a user keychain, the model center can
    point ``MHD_CA_BUNDLE`` at the administrator-provided PEM file explicitly.
    """
    root = Path(runtime_root).expanduser()
    cert_dir = root / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    target = cert_dir / "runtime-ca-bundle.pem"
    meta_path = cert_dir / "runtime-ca-bundle.json"
    explicit = str(os.environ.get("MHD_CA_BUNDLE", "") or "").strip()
    signature = {"platform": platform.system(), "explicit": explicit}
    if explicit:
        try:
            st = Path(explicit).expanduser().stat()
            signature.update({"explicit_size": int(st.st_size), "explicit_mtime_ns": int(st.st_mtime_ns)})
        except Exception:
            pass
    try:
        if target.is_file() and target.stat().st_size > 100 and meta_path.is_file():
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
            if prior == signature:
                return target
    except Exception:
        pass

    certs: list[bytes] = []
    for path in _candidate_ca_files():
        certs.extend(_read_pem_certificates(path))

    if platform.system() == "Darwin":
        for keychain in (
            "/System/Library/Keychains/SystemRootCertificates.keychain",
            "/Library/Keychains/System.keychain",
        ):
            if Path(keychain).exists():
                certs.extend(_security_certificates(keychain))

    # Deduplicate exact PEM blocks while preserving deterministic order.
    unique: list[bytes] = []
    seen: set[bytes] = set()
    for cert in certs:
        normalized = cert.strip() + b"\n"
        if normalized not in seen:
            seen.add(normalized); unique.append(normalized)
    if not unique:
        return None

    payload = b"\n".join(unique)
    tmp = target.with_suffix(".pem.part")
    tmp.write_bytes(payload)
    os.replace(tmp, target)
    try:
        meta_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        target.chmod(0o600)
    except Exception:
        pass
    _emit(progress, f"TLS：已准备本机 CA 证书链（{len(unique)} 张），用于模型/依赖安装。")
    return target


def apply_runtime_tls_environment(
    env: dict[str, str],
    runtime_root: str | Path,
    progress: ProgressFn | None = None,
) -> dict[str, str]:
    """Return *env* with verified-HTTPS CA settings; verification stays enabled."""
    out = dict(env)
    bundle = build_runtime_ca_bundle(runtime_root, progress=progress)
    if bundle is not None:
        value = str(bundle)
        # pip explicitly documents PIP_CERT; Requests honors REQUESTS_CA_BUNDLE;
        # Python/OpenSSL and urllib honor SSL_CERT_FILE.
        out["PIP_CERT"] = value
        out["REQUESTS_CA_BUNDLE"] = value
        out["SSL_CERT_FILE"] = value
        out["CURL_CA_BUNDLE"] = value
    return out


def ssl_failure_hint(python_executable: str | Path | None = None) -> str:
    py = Path(python_executable or sys.executable)
    version = ""
    try:
        proc = subprocess.run(
            [str(py), "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=10,
        )
        version = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 and proc.stdout.strip() else ""
    except Exception:
        pass
    extra = ""
    if platform.system() == "Darwin" and version:
        command = Path(f"/Applications/Python {version}/Install Certificates.command")
        if command.exists():
            extra = f"\n检测到 Python 官方证书修复脚本：{command}（也可手动运行一次）。"
    return (
        "HTTPS 证书链校验失败。程序不会关闭 SSL 校验；已尝试 macOS 系统 CA/自定义 CA。"
        "若使用公司/校园网或 HTTPS 代理，请在模型中心选择代理签发机构提供的 PEM CA 证书。"
        + extra
    )
