#!/usr/bin/env python3
"""Repository-level license and attribution consistency audit."""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"LICENSE AUDIT FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(name: str) -> str:
    path = ROOT / name
    if not path.is_file():
        fail(f"missing required file: {name}")
    return path.read_text(encoding="utf-8")


def _package_name(spec: str) -> str:
    raw = str(spec).strip()
    if " @ " in raw:
        raw = raw.split(" @ ", 1)[0]
    raw = raw.split(";", 1)[0]
    raw = re.split(r"[<>=!~\[\s]", raw, maxsplit=1)[0]
    return raw.strip().lower().replace("_", "-")


def main() -> int:
    license_text = read("LICENSE")
    notices = read("THIRD_PARTY_NOTICES.md")
    references = read("REFERENCES.md")
    citation = read("CITATION.cff")
    version = read("VERSION").strip()

    if not license_text.startswith("MIT License\n"):
        fail("LICENSE must remain the standard MIT license text")
    if "Copyright (c) 2026 Amster-Ilvil" not in license_text:
        fail("project MIT copyright notice is missing")

    with (ROOT / "pyproject.toml").open("rb") as fh:
        metadata = tomllib.load(fh)
    project = metadata.get("project", {})
    build_system = metadata.get("build-system", {})

    build_requires = [str(x) for x in build_system.get("requires", [])]
    setuptools_ok = False
    for req in build_requires:
        match = re.match(r"^setuptools\s*>=\s*(\d+)", req, flags=re.I)
        if match and int(match.group(1)) >= 77:
            setuptools_ok = True
            break
    if not setuptools_ok:
        fail("PEP 639 metadata requires build-system setuptools>=77")

    if project.get("license") != "MIT":
        fail("pyproject.toml must use PEP 639 license = \"MIT\"")
    license_files = project.get("license-files") or []
    for required in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if required not in license_files:
            fail(f"pyproject.toml license-files must include {required}")
    if str(project.get("version", "")).strip() != version:
        fail("VERSION and pyproject.toml version differ")

    match = re.search(r"(?m)^version:\s*['\"]?([^'\"\s]+)", citation)
    if not match or match.group(1) != version:
        fail("CITATION.cff version must match VERSION")
    if not re.search(r"(?m)^license:\s*MIT\s*$", citation):
        fail("CITATION.cff must declare MIT")

    required_notice_tokens = {
        "Novel Formatter": "implementation lineage",
        "KCC-Kindle-CHS": "UI design attribution",
        "manga-image-translator": "development research attribution",
        "comic-translate": "development research attribution",
        "NumPy": "core dependency",
        "OpenCV": "core dependency",
        "Pillow": "core dependency",
        "SciPy": "core dependency",
        "Pydantic": "core dependency",
        "Typer": "core dependency",
        "PySide6": "GUI dependency",
        "PaddleOCR": "OCR dependency",
        "PaddlePaddle": "OCR runtime",
        "LightGlue": "registration dependency",
        "PyTorch": "ML runtime",
        "Kornia": "registration dependency",
        "Transformers": "RT-DETR dependency",
        "Ultralytics": "optional AGPL-sensitive dependency",
        "Spandrel": "optional dependency",
        "setuptools": "build dependency",
        "wheel": "build dependency",
        "pytest": "development dependency",
        "pytest-cov": "development dependency",
        "python-build-standalone": "downloaded Python runtime",
    }
    for token, purpose in required_notice_tokens.items():
        if token.lower() not in notices.lower():
            fail(f"THIRD_PARTY_NOTICES.md missing {purpose}: {token}")

    # Ensure every package declared in the runtime/optional dependency groups is
    # represented by a human-readable notice. Aliases account for PyPI names.
    notice_aliases = {
        "opencv-python-headless": "opencv",
        "pyside6": "pyside6",
        "paddleocr": "paddleocr",
        "paddlepaddle": "paddlepaddle",
        "torch": "pytorch",
        "transformers": "transformers",
        "pytest-cov": "pytest-cov",
    }
    declared = list(project.get("dependencies", []))
    for group in (project.get("optional-dependencies", {}) or {}).values():
        declared.extend(group or [])
    for spec in declared:
        package = _package_name(spec)
        if not package:
            continue
        token = notice_aliases.get(package, package)
        if token.lower() not in notices.lower():
            fail(f"THIRD_PARTY_NOTICES.md does not mention declared dependency: {package}")

    # Keep licensing-sensitive dependencies explicit. A generic third-party
    # disclaimer is not enough for these integration choices.
    if "AGPL-3.0" not in notices or "Ultralytics Enterprise License" not in notices:
        fail("Ultralytics AGPL/commercial licensing must be called out explicitly")
    if "LGPLv3/GPLv3" not in notices or "Qt commercial" not in notices:
        fail("PySide6/Qt community/commercial licensing must be called out explicitly")
    if "model checkpoint's license" not in notices:
        fail("model-weight license separation notice is missing")

    for token in ("LightGlue", "LoFTR", "RT-DETR", "PaddleOCR 3.0"):
        if token not in references:
            fail(f"REFERENCES.md missing academic reference: {token}")
    if "```bibtex" not in references:
        fail("REFERENCES.md must retain BibTeX citations")

    print("License audit passed: MIT metadata, PEP 639 files, dependency notices, citations, and sensitive-license callouts are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
