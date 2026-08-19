from __future__ import annotations

"""Thin desktop launcher that starts logging before importing Qt."""

import logging
import sys

from .app_logging import configure_application_logging, install_exception_hooks


def main() -> int:
    paths = configure_application_logging(component="gui-launcher", level=logging.DEBUG)
    install_exception_hooks()
    logger = logging.getLogger(__name__)
    try:
        from .gui_qt import main as gui_main
        return int(gui_main())
    except BaseException as exc:
        logger.critical("GUI startup failed", exc_info=True)
        try:
            print(
                f"Folirina 启动失败。运行日志：{paths.directory}\n{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
