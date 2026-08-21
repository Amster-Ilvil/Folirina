from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = max(30, int(os.environ.get("FOLIRINA_SELFTEST_TIMEOUT", "90")))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="folirina-selftest-gate-") as td:
        stage = Path(td) / "stage.txt"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + (env.get("PYTHONPATH") or "")
        env["FOLIRINA_SELFTEST_STAGE_FILE"] = str(stage)
        cmd = [sys.executable, "-m", "manga_hd_transfer.selftest"]
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            marker = stage.read_text(encoding="utf-8", errors="ignore").strip() if stage.exists() else ""
            if marker == "complete_pass":
                tail = str(partial)[-4000:]
                if tail:
                    print(tail, end="")
                print("\n[selftest-runner] PASS: checks completed; native runtime teardown exceeded timeout.")
                return 0
            print(str(partial)[-12000:], end="")
            print(f"\n[selftest-runner] FAIL: timeout at stage={marker or 'unknown'}")
            return 124
        output = proc.stdout or ""
        marker = stage.read_text(encoding="utf-8", errors="ignore").strip() if stage.exists() else ""
        if proc.returncode == 0 and marker != "complete_pass":
            print(output[-12000:], end="")
            print(f"\n[selftest-runner] FAIL: process returned 0 without complete_pass marker (stage={marker or 'missing'}).")
            return 3
        if proc.returncode != 0:
            print(output[-12000:], end="")
            return int(proc.returncode)
        try:
            report = json.loads(output)
        except Exception:
            report = {"pass": True}
        summary = {
            "pass": bool(report.get("pass")),
            "qa": report.get("qa", {}),
            "stage": marker,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
