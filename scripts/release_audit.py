#!/usr/bin/env python3
from __future__ import annotations

"""Release-tree hygiene audit for local ZIP builds.

Inspired by Novel-formatter/KCC release privacy checks, but intentionally works
without a git checkout. It rejects common runtime caches, model weights, user
outputs, credentials and private absolute home paths before packaging.
"""

import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/release_audit.py"

FORBIDDEN_NAMES = {".env", ".env.local", ".env.production"}
FORBIDDEN_PARTS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "models", "outputs", "output", "dist", "build",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".db", ".sqlite", ".sqlite3"}
MODEL_SUFFIXES = {".pth", ".pt", ".onnx", ".ckpt", ".safetensors", ".bin"}

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"(?<![A-Za-z0-9])" + "sk" + r"-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])" + "github_pat_" + r"[A-Za-z0-9_]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])" + "gh" + r"[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"(?<![A-Z0-9])" + "AKIA" + r"[0-9A-Z]{16}(?![A-Z0-9])")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])" + "AIza" + r"[0-9A-Za-z_-]{30,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])" + "hf_" + r"[A-Za-z0-9]{20,}")),
]
MAC_HOME = re.compile(r"/Users/(?!Shared(?:/|$)|runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$))[^/\s\"']+")
LINUX_HOME = re.compile(r"/home/(?!runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$))[^/\s\"']+")
WIN_HOME = re.compile(r"(?i)[A-Z]:\\Users\\(?!Public\\|runneradmin\\|<|USER\\|username\\|yourname\\)[^\\\r\n\"']+")


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file():
            yield path


def main() -> int:
    problems: list[str] = []
    checked = 0
    for path in iter_files():
        rel = path.relative_to(ROOT).as_posix()
        p = PurePosixPath(rel)
        lower_parts = {part.lower() for part in p.parts}
        if p.name.lower() in FORBIDDEN_NAMES:
            problems.append(f"forbidden environment file: {rel}")
        if any(part.lower() in FORBIDDEN_PARTS for part in p.parts):
            problems.append(f"forbidden runtime/generated path: {rel}")
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden runtime/generated file: {rel}")
        if p.suffix.lower() in MODEL_SUFFIXES or ("models" in lower_parts and path.stat().st_size > 100_000):
            problems.append(f"model/binary weight must not be packaged: {rel}")
        if rel == SELF or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        checked += 1
        text = data.decode("utf-8", "ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible {label}: {rel}")
        if MAC_HOME.search(text):
            problems.append(f"possible private macOS home path: {rel}")
        if LINUX_HOME.search(text):
            problems.append(f"possible private Linux home path: {rel}")
        if WIN_HOME.search(text):
            problems.append(f"possible private Windows home path: {rel}")
    # Architectural release invariants: Direct and Mask must remain distinct
    # namespaces and artifacts. This catches accidental regressions where Direct
    # becomes a hidden alias of Mask during refactors.
    try:
        config_text = (ROOT / "src/manga_hd_transfer/config.py").read_text(encoding="utf-8")
        pipeline_text = (ROOT / "src/manga_hd_transfer/pipeline.py").read_text(encoding="utf-8")
        result_state_text = (ROOT / "src/manga_hd_transfer/result_state.py").read_text(encoding="utf-8")
        manual_service_text = (ROOT / "src/manga_hd_transfer/manual_review_service.py").read_text(encoding="utf-8")
        gui_text = (ROOT / "src/manga_hd_transfer/gui_qt.py").read_text(encoding="utf-8")
        version_text = (ROOT / "src/manga_hd_transfer/version.py").read_text(encoding="utf-8")
        pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version_match = re.search(r'^__version__\s*=\s*"([^"]+)"', version_text, re.MULTILINE)
        package_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)
        runtime_version = version_match.group(1) if version_match else ""
        package_version = package_match.group(1) if package_match else ""
        aligned_text = (ROOT / "src/manga_hd_transfer/aligned_overlay_reveal.py").read_text(encoding="utf-8")
        required = [
            ("DirectPatchConfig namespace", "class DirectPatchConfig" in config_text),
            ("Pipeline direct_patch config", "direct_patch: DirectPatchConfig" in config_text),
            ("Pipeline mask_replace config", "mask_replace: MaskReplaceConfig" in config_text),
            ("Direct artifact namespace", '"direct_patch_layer.png"' in pipeline_text),
            ("Mask artifact namespace", '"mask_transfer_layer.png"' in pipeline_text),
            ("Direct JSON schema", "manga_hd_translation_transfer.direct_patch.v1" in pipeline_text),
            ("Shared result-state contract", "def commit_reviewed_result" in result_state_text and "review_sync.v3" in result_state_text),
            ("Qt-free manual review service", "def commit_manual_effect" in manual_service_text and "PySide6" not in manual_service_text),
            ("GUI delegates manual commit", "commit_manual_effect(" in gui_text),
            ("Single runtime version source", "VERSION = __version__" in gui_text and bool(runtime_version)),
            ("Package metadata version", bool(package_version) and package_version == runtime_version),
            ("Aligned overlay config namespace", "class AlignedOverlayRevealConfig" in config_text and "aligned_overlay_reveal: AlignedOverlayRevealConfig" in config_text),
            ("Aligned overlay defaults OFF", "enabled: bool = False" in config_text and "allow_in_auto: bool = False" in config_text and "require_explicit_mode: bool = True" in config_text),
            ("Aligned overlay pixel module is state-free", "def build_aligned_overlay_plan" in aligned_text and "def execute_aligned_overlay" in aligned_text and "final.png" not in aligned_text),
            ("Aligned overlay artifact namespace", '"aligned_overlay_reveal_layer.png"' in pipeline_text and '"aligned_overlay_reveal_mask.png"' in pipeline_text),
            ("Aligned automatic result uses result_state", "commit_automatic_result(" in pipeline_text and "def commit_automatic_result" in result_state_text),
        ]
        for label, ok in required:
            if not ok:
                problems.append(f"architecture invariant missing: {label}")
    except OSError as exc:
        problems.append(f"architecture invariant audit failed to read source: {exc}")

    if problems:
        print("RELEASE AUDIT FAILED", file=sys.stderr)
        for item in sorted(set(problems)):
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Release audit passed: {checked} text files checked; no caches, model weights, credentials or private home paths detected.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
