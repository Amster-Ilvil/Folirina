from __future__ import annotations

"""macOS Apple Live Text OCR adapters.

This module mirrors Novel Formatter's Apple OCR architecture:

1. A small Swift helper calls VisionKit ImageAnalyzer (system Live Text) and is
   kept alive as a JSON-lines subprocess for batch work.
2. If that helper is unavailable or fails, the backend falls back to the user's
   macOS Shortcut (normally ``ExtractText`` -> "Extract Text from Image").

Both routes are transcript/text-only APIs.  Manga-HD-Transfer supplies geometry
from its own paired bubble detector and uses this backend only to answer "what
Chinese text is inside this already-known region?".
"""

import json
import os
import platform
import select
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .geometry import bbox_polygon
from .models import TextBlock

if TYPE_CHECKING:  # pragma: no cover
    from .config import OCRConfig

ROOT = Path(__file__).resolve().parents[2]
HELPER_SOURCE = ROOT / "tools" / "apple_live_text_helper" / "AppleLiveTextOCRHelper.swift"
HELPER_BINARY = ROOT / "tools" / "apple_live_text_helper" / "bin" / "apple_live_text_helper"
HELPER_BUILD = ROOT / "build_apple_live_text_helper.command"


class AppleLiveTextError(RuntimeError):
    pass


class AppleLiveTextInfrastructureError(AppleLiveTextError):
    pass


def _mac_version_major() -> int:
    try:
        return int((platform.mac_ver()[0] or "0").split(".")[0])
    except Exception:
        return 0


def ensure_live_text_helper() -> Path:
    if platform.system() != "Darwin":
        raise AppleLiveTextInfrastructureError("Apple Live Text 仅支持 macOS")
    # Keep the same conservative availability policy as Novel-formatter's
    # LiveTextHelperBackend. Older macOS versions still use the Shortcut route.
    if _mac_version_major() < 15:
        raise AppleLiveTextInfrastructureError("Swift VisionKit Live Text Helper 需要 macOS 15 或更高版本")
    if not HELPER_SOURCE.exists():
        raise AppleLiveTextInfrastructureError(f"缺少 Swift Live Text Helper 源码：{HELPER_SOURCE}")
    needs_build = not HELPER_BINARY.exists()
    if HELPER_BINARY.exists():
        try:
            needs_build = HELPER_BINARY.stat().st_mtime < HELPER_SOURCE.stat().st_mtime
        except OSError:
            needs_build = True
    if not needs_build:
        return HELPER_BINARY
    if not shutil.which("xcrun"):
        raise AppleLiveTextInfrastructureError("未找到 xcrun，请先安装 Xcode Command Line Tools")
    if not HELPER_BUILD.exists():
        raise AppleLiveTextInfrastructureError(f"缺少编译脚本：{HELPER_BUILD}")
    try:
        result = subprocess.run(
            [str(HELPER_BUILD)], cwd=str(ROOT), capture_output=True, text=True,
            timeout=180,
        )
    except Exception as exc:
        raise AppleLiveTextInfrastructureError(f"Swift Live Text Helper 编译失败：{exc}") from exc
    if result.returncode != 0 or not HELPER_BINARY.exists():
        detail = (result.stderr or result.stdout or "Swift Helper 编译失败").strip()
        raise AppleLiveTextInfrastructureError(f"Swift Live Text Helper 编译失败：{detail}")
    return HELPER_BINARY


class _JSONLineClient:
    def __init__(self, binary: Path) -> None:
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )

    def _stderr_tail(self) -> str:
        try:
            self._stderr.flush(); self._stderr.seek(0)
            return self._stderr.read()[-4000:].strip()
        except Exception:
            return ""

    def request(self, payload: dict, timeout: float) -> dict:
        if self.process.poll() is not None:
            raise AppleLiveTextInfrastructureError(
                f"Swift Live Text Helper 已退出：{self._stderr_tail() or self.process.returncode}"
            )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + max(1.0, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise AppleLiveTextInfrastructureError("Swift Live Text Helper OCR 超时，已终止 Helper")
            ready, _, _ = select.select([self.process.stdout], [], [], min(0.25, remaining))
            if ready:
                break
            if self.process.poll() is not None:
                raise AppleLiveTextInfrastructureError(
                    f"Swift Live Text Helper 异常退出：{self._stderr_tail() or self.process.returncode}"
                )
        line = self.process.stdout.readline()
        if not line:
            raise AppleLiveTextInfrastructureError(
                f"Swift Live Text Helper 未返回结果：{self._stderr_tail()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise AppleLiveTextInfrastructureError(f"Swift Live Text Helper 返回了无效 JSON：{line[:300]}") from exc

    def close(self) -> None:
        proc = getattr(self, "process", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            self._stderr.close()
        except Exception:
            pass


class _TextOnlyBase:
    """Marker API consumed by Pipeline._recognize_paired_regions_text_only.

    v2.3.63 also exposes a real ``recognize_region`` contract.  The Studio's
    manual OCR editor works on user-selected ROI crops, while Apple Live Text
    historically implemented only whole-image ``recognize``.  That mismatch made
    manual OCR silently return no text on the default macOS backend.
    """

    region_text_only = True
    supports_crop_recognition = True

    def __init__(self, lang: str, config: "OCRConfig") -> None:
        self.lang = lang
        self.config = config

    def _languages(self) -> list[str]:
        name = (self.lang or "").lower()
        if name in {"ch", "chi_sim", "zh", "zh-hans", "zh-cn"}:
            return ["zh-Hans", "zh-Hant", "ja-JP"]
        if name in {"japan", "ja", "ja-jp"}:
            return ["ja-JP", "zh-Hans", "zh-Hant"]
        return ["zh-Hans", "ja-JP", "en-US"]

    def recognize_region(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int],
        *,
        image_path: str | Path | None = None,
    ) -> list[TextBlock]:
        """Recognize one ROI crop using the same Apple backend.

        Returned polygons intentionally live in crop-local coordinates.  The
        manual OCR editor offsets TARGET polygons back into page coordinates and
        treats SOURCE output as text-only content, so no page-global geometry is
        lost here.
        """
        if image is None or getattr(image, "ndim", 0) < 2:
            return []
        h, w = image.shape[:2]
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in list(bbox)[:4]]
        except Exception as exc:
            raise AppleLiveTextError(f"Apple OCR ROI 无效：{bbox!r}") from exc
        x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
        if x1 <= x0 or y1 <= y0:
            return []
        crop=image[y0:y1,x0:x1].copy()
        # Never pass the original page path for an ROI: the backend must OCR the
        # actual crop rather than accidentally re-reading the full page.
        return self.recognize(crop, image_path=None)

    def _block(self, image: np.ndarray, text: str, backend: str) -> list[TextBlock]:
        text = str(text or "").strip()
        if not text:
            return []
        h, w = image.shape[:2]
        confidence = float(getattr(self.config, "apple_live_text_assumed_confidence", 0.88))
        return [TextBlock(
            id=f"{backend}-{uuid.uuid4().hex[:8]}",
            polygon=bbox_polygon((0, 0, w, h)),
            text=text,
            confidence=confidence,
            kind="unknown",
            reading_order=0,
            meta={
                "backend": backend,
                "text_only": True,
                "geometry_source": "caller_region",
                "languages": self._languages(),
            },
        )]

    @staticmethod
    def _temp_image(image: np.ndarray) -> str:
        fh = tempfile.NamedTemporaryFile(prefix="mhd_live_text_", suffix=".png", delete=False)
        path = fh.name
        fh.close()
        if not cv2.imwrite(path, image):
            Path(path).unlink(missing_ok=True)
            raise AppleLiveTextError("无法创建 Apple Live Text 临时图片")
        return path


class AppleVisionKitLiveTextBackend(_TextOnlyBase):
    """Swift VisionKit ImageAnalyzer transcript backend."""

    def __init__(self, lang: str, config: "OCRConfig") -> None:
        super().__init__(lang, config)
        if platform.system() != "Darwin":
            raise AppleLiveTextInfrastructureError("Apple VisionKit Live Text 仅支持 macOS")
        self._client: _JSONLineClient | None = None
        self._lock = threading.RLock()

    def _client_for_request(self) -> _JSONLineClient:
        with self._lock:
            if self._client is None:
                self._client = _JSONLineClient(ensure_live_text_helper())
            return self._client

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        temp_path: str | None = None
        try:
            if image_path is not None and Path(image_path).is_file():
                path = str(Path(image_path).resolve())
            else:
                temp_path = self._temp_image(image)
                path = temp_path
            payload = {
                "id": uuid.uuid4().hex,
                "api": "live_text",
                "image": path,
                "languages": self._languages(),
            }
            try:
                response = self._client_for_request().request(
                    payload, float(getattr(self.config, "apple_live_text_timeout", 45.0))
                )
            except AppleLiveTextError:
                self.close()
                raise
            if not bool(response.get("success")):
                error = str(response.get("error") or "Swift VisionKit Live Text OCR 失败")
                raise AppleLiveTextError(error)
            return self._block(image, str(response.get("text") or ""), "apple_live_text")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


class AppleShortcutLiveTextBackend(_TextOnlyBase):
    """macOS Shortcuts 'Extract Text from Image' backend, matching Novel-formatter."""

    def __init__(self, lang: str, config: "OCRConfig") -> None:
        super().__init__(lang, config)
        if platform.system() != "Darwin":
            raise AppleLiveTextInfrastructureError("Apple 快捷指令 OCR 仅支持 macOS")
        if not shutil.which("shortcuts"):
            raise AppleLiveTextInfrastructureError("找不到 shortcuts 命令，请确认 macOS 12 或更高版本")

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        temp_path: str | None = None
        try:
            if image_path is not None and Path(image_path).is_file():
                path = str(Path(image_path).resolve())
            else:
                temp_path = self._temp_image(image)
                path = temp_path
            shortcut_name = str(getattr(self.config, "apple_shortcut_name", "ExtractText") or "ExtractText").strip()
            timeout = max(1.0, float(getattr(self.config, "apple_live_text_timeout", 45.0)))
            try:
                result = subprocess.run(
                    ["shortcuts", "run", shortcut_name, "-i", path],
                    capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise AppleLiveTextError(f"Apple OCR 快捷指令 {shortcut_name!r} 超时") from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "未知错误").strip()
                raise AppleLiveTextError(
                    f"Apple OCR 快捷指令 {shortcut_name!r} 调用失败：{detail}"
                )
            return self._block(image, result.stdout, "apple_shortcut")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)


class AppleAutoLiveTextBackend(_TextOnlyBase):
    """VisionKit first, then Shortcuts; once helper fails, keep fallback for the job."""

    def __init__(self, lang: str, config: "OCRConfig") -> None:
        super().__init__(lang, config)
        self._helper: AppleVisionKitLiveTextBackend | None = None
        self._shortcut: AppleShortcutLiveTextBackend | None = None
        self._helper_disabled = False
        self._shortcut_disabled = False
        self._last_fallback_reason = ""
        self._all_unavailable_reason = ""

    def _get_helper(self) -> AppleVisionKitLiveTextBackend:
        if self._helper is None:
            self._helper = AppleVisionKitLiveTextBackend(self.lang, self.config)
        return self._helper

    def _get_shortcut(self) -> AppleShortcutLiveTextBackend:
        if self._shortcut is None:
            self._shortcut = AppleShortcutLiveTextBackend(self.lang, self.config)
        return self._shortcut

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if not self._helper_disabled:
            try:
                blocks = self._get_helper().recognize(image, image_path=image_path)
                for b in blocks:
                    b.meta["apple_auto_route"] = "visionkit_live_text"
                return blocks
            except Exception as exc:
                self._helper_disabled = True
                self._last_fallback_reason = f"{type(exc).__name__}: {exc}"
                try:
                    if self._helper is not None:
                        self._helper.close()
                finally:
                    self._helper = None
        if self._shortcut_disabled:
            if bool(getattr(self.config, "apple_live_text_soft_fail", True)):
                return []
            raise AppleLiveTextInfrastructureError(self._all_unavailable_reason or "Apple OCR 后端均不可用")
        try:
            blocks = self._get_shortcut().recognize(image, image_path=image_path)
        except Exception as exc:
            self._shortcut_disabled = True
            self._all_unavailable_reason = (
                f"VisionKit: {self._last_fallback_reason or 'disabled'}; "
                f"Shortcut: {type(exc).__name__}: {exc}"
            )
            if bool(getattr(self.config, "apple_live_text_soft_fail", True)):
                return []
            raise
        for b in blocks:
            b.meta["apple_auto_route"] = "shortcut"
            if self._last_fallback_reason:
                b.meta["apple_live_text_fallback_reason"] = self._last_fallback_reason
        return blocks

    def close(self) -> None:
        if self._helper is not None:
            self._helper.close()
        self._helper = None
        self._shortcut = None


def apple_live_text_probe(config: "OCRConfig" | None = None) -> dict[str, object]:
    """Side-effect-light readiness probe used by the GUI; does not compile helper."""
    is_mac = platform.system() == "Darwin"
    return {
        "platform": platform.system(),
        "visionkit_source": HELPER_SOURCE.exists(),
        "visionkit_binary": HELPER_BINARY.exists(),
        "xcrun": bool(shutil.which("xcrun")) if is_mac else False,
        "shortcuts": bool(shutil.which("shortcuts")) if is_mac else False,
        "shortcut_name": str(getattr(config, "apple_shortcut_name", "ExtractText") if config is not None else "ExtractText"),
    }
