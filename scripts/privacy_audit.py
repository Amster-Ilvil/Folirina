#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/privacy_audit.py"
ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", "credentials.json", "secrets.json", "service-account.json"}
FORBIDDEN_DIRS = {".runtime", ".model-cache", ".ocr-runtimes", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache", "logs", "backups", "debug", "output", "outputs", "exports", "workspaces"}
FORBIDDEN_SUFFIXES = {".log", ".db", ".sqlite", ".sqlite3", ".pem", ".p12", ".pfx", ".key"}
SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("OpenAI key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}")),
    ("Anthropic key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")),
    ("AWS access key", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
]
MAC_HOME = re.compile(r"/Users/(?!Shared(?:/|$)|runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$))[^/\s\"']+")
WIN_HOME = re.compile(r"(?i)[A-Z]:\\Users\\(?!Public\\|runneradmin\\|<|USER\\|username\\|yourname\\)[^\\\r\n\"']+")


def tracked_files() -> list[str]:
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise SystemExit("privacy audit requires a git checkout")
    return [x.decode("utf-8", "surrogateescape") for x in proc.stdout.split(b"\0") if x]


def main() -> int:
    problems: list[str] = []
    paths = tracked_files()
    for raw in paths:
        p = PurePosixPath(raw)
        name = p.name.lower()
        parts = {part.lower() for part in p.parts}
        if name.startswith(".env") and name not in ALLOWED_ENV_TEMPLATES:
            problems.append(f"forbidden environment file: {raw}")
        if name in FORBIDDEN_NAMES or parts & FORBIDDEN_DIRS or p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden private/runtime path: {raw}")
        if raw == SELF:
            continue
        path = ROOT / raw
        try:
            if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", "ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible {label}: {raw}")
        for label, pattern in (("macOS home path", MAC_HOME), ("Windows home path", WIN_HOME)):
            if pattern.search(text):
                problems.append(f"possible private {label}: {raw}")
    problems = sorted(set(problems))
    if problems:
        print("PRIVACY AUDIT FAILED", file=sys.stderr)
        for item in problems:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Privacy audit passed: {len(paths)} tracked files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
