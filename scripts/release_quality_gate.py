from __future__ import annotations

"""One command that must pass before a Folirina release is publishable."""

import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "tools") + (os.pathsep + ENV["PYTHONPATH"] if ENV.get("PYTHONPATH") else "")
ENV.setdefault("FOLIRINA_SELFTEST_TIMEOUT", "60")

STEP_TIMEOUT_SECONDS = max(60, int(ENV.get("FOLIRINA_QUALITY_GATE_STEP_TIMEOUT", "600")))

COMMANDS = (
    ("compileall", [sys.executable, "-m", "compileall", "-q", "src", "tests", "tools", "scripts"]),
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
    ("architecture_audit", [sys.executable, "-m", "manga_hd_transfer.architecture_audit"]),
    ("gui_interaction_audit", [sys.executable, "-m", "manga_hd_transfer.gui_interaction_audit"]),
    ("undefined_name_audit", [sys.executable, "tools/undefined_name_audit.py", "--strict"]),
    # Run filesystem/architecture health before importing the entire module
    # surface. On some native OpenCV builds, the all-module import probe can
    # delay a following filesystem-heavy audit even though both pass alone.
    ("project_health", [sys.executable, "tools/project_health_audit.py", "--strict"]),
    ("deep_audit", [sys.executable, "tools/deep_audit.py"]),
    ("import_surface_audit", [sys.executable, "tools/import_surface_audit.py", "--strict"]),
    # Native runtime probing is last and has its own trusted timeout/receipt.
    ("selftest", [sys.executable, "scripts/run_runtime_selftest.py"]),
)



def _command_line(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def main() -> int:
    # Replace the gate process with one native shell and run the checks in the
    # empirically stable dependency order above.  No Python gate parent remains
    # around native OpenCV/import-audit children.
    lines: list[str] = []
    if os.name == "nt":
        for name, cmd in COMMANDS:
            line = _command_line(cmd)
            lines.append(f'{line} >NUL 2>&1')
            lines.append(f'if errorlevel 1 (echo [{name}] FAIL & exit /b 1) else (echo [{name}] PASS)')
        lines.append('echo {"pass":true}')
        script = "\r\n".join(lines)
        shell = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
        os.execve(shell, [shell, "/d", "/s", "/c", script], ENV)
    else:
        for name, cmd in COMMANDS:
            line = _command_line(cmd)
            lines.append(
                f'if {line} >/dev/null 2>&1; then printf "[{name}] PASS\\n"; '
                f'else rc=$?; printf "[{name}] FAIL\\n"; exit $rc; fi'
            )
        lines.append("printf '{\"pass\":true}\\n'")
        script = "\n".join(lines)
        shell = "/bin/bash" if Path("/bin/bash").is_file() else "/bin/sh"
        os.execve(shell, [shell, "-c", script], ENV)
    return 1  # os.execve never returns on success

if __name__ == "__main__":
    raise SystemExit(main())
