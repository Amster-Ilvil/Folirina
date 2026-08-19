from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = str(pyproject["project"]["version"])
    version_text = (ROOT / "src" / "manga_hd_transfer" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', version_text, re.M)
    if not match or match.group(1) != project_version:
        raise SystemExit(f"version mismatch: pyproject={project_version}, runtime={match.group(1) if match else 'missing'}")

    for path in (ROOT / "src" / "manga_hd_transfer").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ast.parse((ROOT / "run_gui.py").read_text(encoding="utf-8"), filename="run_gui.py")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_direct_vendor.py")], cwd=ROOT, check=True)
    print(f"Folirina release smoke PASS: v{project_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
