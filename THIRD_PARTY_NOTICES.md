# Third-Party Notices

This file documents third-party software, upstream projects, implementation lineage, and research references relevant to **Manga HD Transfer Studio**.

## Scope of the project license

The original source code and documentation in this repository are licensed under the MIT License in [`LICENSE`](LICENSE), unless a file explicitly states otherwise.

**The MIT License does not relicense third-party software, frameworks, model weights, datasets, system frameworks, or downloaded runtimes.** Those components remain subject to their own upstream licenses and terms.

The public release workflow is intentionally source-first: release archives are built from Git-tracked repository files. OCR/ML model weights, model caches, virtual environments, and optional ML runtimes are not intended to be bundled in the repository Release artifacts. Some launchers or user-invoked features can download/install third-party components later; their upstream licenses apply at that time.

This notice is a practical attribution inventory, not a substitute for the full upstream license texts. When redistributing third-party binaries, wheels, model files, or modified upstream code, retain all notices and license files required by the exact versions you distribute.

---

## 1. Explicit implementation/design lineage

### Amster-Ilvil/Novel-formatter

- Upstream: https://github.com/Amster-Ilvil/Novel-formatter
- License: MIT
- Relationship: implementation/design lineage.
- This project explicitly mirrors the Novel Formatter Apple OCR architecture in `src/manga_hd_transfer/apple_live_text.py`.
- The multi-platform bootstrap/release/privacy-audit design introduced for Manga HD Transfer v1.3.14 also follows the same source-first, deferred-runtime approach used by Novel Formatter.
- Code that belongs to this repository remains under this repository's MIT license; the upstream MIT copyright/license notice remains applicable to any material actually adapted from Novel Formatter.

### Amster-Ilvil/KCC-Kindle-CHS

- Upstream: https://github.com/Amster-Ilvil/KCC-Kindle-CHS
- License: ISC
- Relationship: visual/design inspiration only.
- `src/manga_hd_transfer/gui_qt.py` explicitly identifies the KCC-Kindle-CHS macOS palette as inspiration for the UI palette.
- No KCC/Kindle Comic Converter source tree is vendored into this repository by that palette reference.

---

## 2. Development research references (not vendored dependencies)

The following projects were used as conceptual/research references for manga text detection, masking, inpainting, rendering, review, and workflow decomposition. Listing a project here does **not** mean its source code is included in this repository.

### manga-image-translator

- Upstream: https://github.com/zyddnys/manga-image-translator
- Upstream license: GPL-3.0
- Relationship here: conceptual/workflow research reference only; no GPL source is intentionally vendored into this MIT repository.

### comic-translate

- Upstream: https://github.com/ogkalu2/comic-translate
- Upstream license: Apache-2.0
- Relationship here: conceptual/workflow research reference only; no upstream source tree is intentionally vendored into this repository.

See [`REFERENCES.md`](REFERENCES.md) for research/paper citation details.

---

## 3. Direct Python/runtime dependencies

The package ranges in `pyproject.toml` can resolve to different versions over time. The exact license files shipped by the installed version control. The summary below reflects the upstream licensing of the projects currently referenced by this repository and highlights cases where binary wheels may contain additional third-party notices.

| Component | How used | Upstream license / licensing model | Distribution note |
| --- | --- | --- | --- |
| NumPy | arrays/numerics | BSD-3-Clause core; current wheels/source can include additional permissive/copyleft runtime notices | Installed as a dependency; preserve licenses shipped in the actual wheel/build |
| OpenCV (`opencv-python-headless`) | image processing/registration/masks | Apache-2.0 | Installed as a dependency |
| Pillow | image I/O | MIT-CMU (Pillow/PIL license text) | Installed as a dependency |
| SciPy | numerical algorithms | BSD-3-Clause core; binary builds can include additional bundled licenses | Installed as a dependency |
| Pydantic | configuration/data models | MIT | Installed as a dependency |
| Typer | CLI | MIT | Installed as a dependency |
| PySide6 / Qt for Python | desktop GUI | Community Edition: LGPLv3/GPLv3; alternatively Qt commercial licensing | Installed at first-run/source setup; Qt/PySide license obligations remain separate from this project's MIT license |

### OCR / optional ML dependencies

| Component | How used | Upstream license / licensing model | Distribution note |
| --- | --- | --- | --- |
| PaddleOCR | optional OCR | Apache-2.0 | Installed only when the OCR feature is requested/confirmed |
| PaddlePaddle | PaddleOCR runtime | Apache-2.0 | Installed only when required |
| LightGlue | optional learned feature matching | Apache-2.0 for LightGlue code and LightGlue pretrained weights | Different feature extractors/weights may have different terms; check upstream notices for the exact feature/weights used |
| PyTorch | optional ML runtime | BSD-style project license; current distributions include multiple third-party license notices | Preserve license files supplied with the installed wheel/distribution |
| Kornia | optional LoFTR/vision functions | Apache-2.0 | Installed only for relevant optional paths |
| Hugging Face Transformers | optional RT-DETR/model runtime | Apache-2.0 | Model weights may have separate model-card licenses |
| Ultralytics | optional detection/acceleration paths | **AGPL-3.0 or Ultralytics Enterprise License** | Not relicensed by MIT. Users/distributors enabling this dependency must satisfy the applicable Ultralytics license for their use/distribution |
| Spandrel | optional model loading/upscale support | MIT | Installed only for relevant optional paths |

### Important model-weight rule

A library's software license and a model checkpoint's license are not necessarily the same. Model files obtained from Hugging Face, Paddle, Ultralytics, Torch Hub, GitHub Releases, or other model hosts must be checked against their own model card, repository, or distribution terms before redistribution. Manga HD Transfer's repository license does not grant rights to third-party model weights.

---

## 4. Standalone Python runtime used by launchers

### astral-sh/python-build-standalone

- Upstream: https://github.com/astral-sh/python-build-standalone
- Project license: MPL-2.0.
- The Windows/macOS/Linux launchers may download a verified Python standalone distribution at first run when no supported system Python is available.
- These Python distributions include CPython and other components with their own license metadata. The downloaded distribution's included license inventory controls those components.
- The runtime is downloaded to the user's local runtime directory and is not intended to be committed to or bundled from this repository.

---

## 5. Apple system APIs

The Apple Live Text route can use Apple's VisionKit/ImageAnalyzer and other macOS system frameworks through a small Swift helper. Apple system frameworks are provided by macOS/Xcode and are not redistributed under this project's MIT license. Apple's SDK/platform terms apply to those system components.

---

## 6. Trademarks and project names

Third-party project names and trademarks are used only for factual attribution/compatibility description. The MIT license for Manga HD Transfer Studio does not grant trademark rights in Python, Qt, Apple, PaddlePaddle, Ultralytics, Hugging Face, PyTorch, OpenCV, or other third-party names.

---

## 7. Upstream license/source links

- Novel Formatter: https://github.com/Amster-Ilvil/Novel-formatter
- KCC-Kindle-CHS: https://github.com/Amster-Ilvil/KCC-Kindle-CHS
- manga-image-translator: https://github.com/zyddnys/manga-image-translator
- comic-translate: https://github.com/ogkalu2/comic-translate
- NumPy: https://github.com/numpy/numpy
- OpenCV: https://github.com/opencv/opencv
- Pillow: https://github.com/python-pillow/Pillow
- SciPy: https://github.com/scipy/scipy
- Pydantic: https://github.com/pydantic/pydantic
- Typer: https://github.com/fastapi/typer
- Qt for Python / PySide6: https://doc.qt.io/qtforpython-6/
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- PaddlePaddle: https://github.com/PaddlePaddle/Paddle
- LightGlue: https://github.com/cvg/LightGlue
- PyTorch: https://github.com/pytorch/pytorch
- Kornia: https://github.com/kornia/kornia
- Transformers: https://github.com/huggingface/transformers
- Ultralytics: https://github.com/ultralytics/ultralytics
- Spandrel: https://github.com/chaiNNer-org/spandrel
- python-build-standalone: https://github.com/astral-sh/python-build-standalone
