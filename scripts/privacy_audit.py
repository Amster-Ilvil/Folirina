#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail CI/release builds when tracked files look like private runtime data.

The release builders package only ``git archive`` output, so untracked files can
never enter a release. This audit is the second boundary: it rejects accidentally
tracked credentials, developer home paths, generated workspaces, logs, databases,
model weights and user documents before a release can be published.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/privacy_audit.py"

FORBIDDEN_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "credentials.json", "secrets.json", "id_rsa", "id_ed25519",
}
FORBIDDEN_PARTS = {
    ".runtime", ".venv", ".venv-app-windows", ".venv-app-linux",
    ".model-cache", ".ocr-runtimes", ".ocr-runtime-state",
    ".manual-model-updates", ".windows-runtime-state",
    "venv", "debug", "output", "outputs", "results", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
FORBIDDEN_SUFFIXES = {
    ".log", ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx",
    ".epub", ".docx", ".pdf", ".cbz", ".cbr", ".psd", ".ora", ".7z", ".rar",
}
MODEL_SUFFIXES = {".pth", ".pt", ".onnx", ".ckpt", ".safetensors", ".pdparams", ".pdmodel"}

# Construct token prefixes in pieces so the audit source does not match itself.
SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"(?<![A-Za-z0-9])" + "sk" + r"-[A-Za-z0-9_-]{20,}")),
    ("GitHub fine-grained token", re.compile(r"(?<![A-Za-z0-9])" + "github_pat_" + r"[A-Za-z0-9_]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])" + "gh" + r"[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"(?<![A-Z0-9])" + "AKIA" + r"[0-9A-Z]{16}(?![A-Z0-9])")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])" + "AIza" + r"[0-9A-Za-z_-]{30,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])" + "hf_" + r"[A-Za-z0-9]{20,}")),
]

MAC_HOME = re.compile(r"/Users/(?!Shared(?:/|$)|runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$)|path(?:/|$))[^/\s\"']+")
LINUX_HOME = re.compile(r"/home/(?!runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$)|path(?:/|$))[^/\s\"']+")
WIN_HOME = re.compile(r"(?i)[A-Z]:\\Users\\(?!Public\\|runneradmin\\|<|USER\\|username\\|yourname\\|path\\)[^\\\r\n\"']+")
EMAIL = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})(?![\w.-])", re.I)
ALLOWED_EMAIL_DOMAINS = {"example.com", "users.noreply.github.com"}


def tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("privacy audit requires a git checkout")
    return [p.decode("utf-8", "surrogateescape") for p in proc.stdout.split(b"\0") if p]


def path_violations(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for raw in paths:
        p = PurePosixPath(raw)
        parts = {part.lower() for part in p.parts}
        if p.name.lower() in FORBIDDEN_NAMES:
            problems.append(f"forbidden tracked environment/credential file: {raw}")
        if parts & FORBIDDEN_PARTS:
            problems.append(f"forbidden tracked runtime/generated path: {raw}")
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden tracked private/generated file: {raw}")
        if ("models" in parts or "weights" in parts) and p.suffix.lower() in MODEL_SUFFIXES:
            problems.append(f"model weight must not be tracked: {raw}")
    return problems


def text_violations(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for raw in paths:
        if raw == SELF:
            continue
        path = ROOT / raw
        try:
            if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", "ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible {label} in tracked file: {raw}")
        if MAC_HOME.search(text):
            problems.append(f"possible private macOS home path in tracked file: {raw}")
        if LINUX_HOME.search(text):
            problems.append(f"possible private Linux home path in tracked file: {raw}")
        if WIN_HOME.search(text):
            problems.append(f"possible private Windows home path in tracked file: {raw}")
        for match in EMAIL.finditer(text):
            domain = match.group(2).lower()
            if domain not in ALLOWED_EMAIL_DOMAINS:
                problems.append(f"possible personal email address in tracked file: {raw}")
                break
    return problems


def main() -> int:
    paths = tracked_files()
    problems = sorted(set(path_violations(paths) + text_violations(paths)))
    if problems:
        print("PRIVACY AUDIT FAILED", file=sys.stderr)
        for item in problems:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Privacy audit passed: {len(paths)} tracked files checked; no blocked private artifacts detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
