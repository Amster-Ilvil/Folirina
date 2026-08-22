from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = max(30, int(os.environ.get("FOLIRINA_SELFTEST_TIMEOUT", "90")))
POLL_SECONDS = 0.05


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _read_text(path: Path, limit: int | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:] if limit else text


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="folirina-selftest-gate-") as td:
        td_path = Path(td)
        stage = td_path / "stage.txt"
        output_path = td_path / "selftest.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + (env.get("PYTHONPATH") or "")
        env["FOLIRINA_SELFTEST_STAGE_FILE"] = str(stage)
        cmd = [sys.executable, "-m", "manga_hd_transfer.selftest"]
        popen_kwargs = {
            "cwd": ROOT,
            "env": env,
            "text": False,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        deadline = time.monotonic() + TIMEOUT_SECONDS
        marker = ""
        returncode: int | None = None
        with output_path.open("wb") as output_fh:
            proc = subprocess.Popen(cmd, stdout=output_fh, **popen_kwargs)
            while True:
                if stage.exists():
                    marker = _read_text(stage).strip()
                    if marker in {"complete_pass", "complete_fail"}:
                        break
                returncode = proc.poll()
                if returncode is not None:
                    break
                if time.monotonic() >= deadline:
                    _terminate_process_tree(proc)
                    print(_read_text(output_path, 12000), end="")
                    print(f"\n[selftest-runner] FAIL: timeout at stage={marker or 'unknown'}")
                    return 124
                time.sleep(POLL_SECONDS)

            # A completion receipt is authoritative only because selftest writes
            # it after printing+flushing the complete JSON report. Reclaim the
            # isolated process group immediately instead of waiting for native
            # library atexit/teardown.
            if marker in {"complete_pass", "complete_fail"}:
                _terminate_process_tree(proc)
            elif returncode is None:
                returncode = proc.poll()

        output = _read_text(output_path)
        if marker == "complete_fail":
            print(output[-12000:], end="")
            print("\n[selftest-runner] FAIL: selftest reported complete_fail.")
            return 2
        if marker != "complete_pass":
            print(output[-12000:], end="")
            print(f"\n[selftest-runner] FAIL: process exited without complete_pass receipt (stage={marker or 'missing'}, rc={returncode}).")
            return int(returncode or 3)

        try:
            report = json.loads(output)
        except Exception as exc:
            print(output[-12000:], end="")
            print(f"\n[selftest-runner] FAIL: complete_pass receipt but JSON report is invalid: {exc}")
            return 4
        summary = {
            "pass": bool(report.get("pass")),
            "qa": report.get("qa", {}),
            "stage": marker,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
